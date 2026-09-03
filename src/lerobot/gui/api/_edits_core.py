# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared business logic for dataset edits.

The FastAPI handlers in ``edits.py`` and the MCP tools in ``mcp/server.py``
both drive the same in-memory PendingEdit queue. Rather than have each
re-implement validation + queue mutation (or worse — have MCP self-call
FastAPI, see the README's "Don't auto-bind" anti-pattern), the
substantive logic lives here as sync free functions that take
``AppState`` and raise typed exceptions on failure.

Mapping:

- ``DatasetNotFoundError``  → FastAPI 404 / MCP error
- ``EditValidationError``   → FastAPI 400 / MCP error
- ``EditConflictError``     → FastAPI 409 with the carried ``detail`` /
                              MCP returns the structured detail to the AI
                              so it can retry with the appropriate confirm
                              flag.
- ``DatasetBusyError``      → FastAPI 423 / MCP error

Each function returns a plain dict that the FastAPI handler or MCP tool
forwards as the response. Persistence to disk (the per-dataset
``.lerobot_gui_edits.json``) is handled inside these functions so neither
caller has to remember to save.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.gui.state import AppState, PendingEdit

logger = logging.getLogger(__name__)


# ── Typed exceptions ──────────────────────────────────────────────────────


class DatasetNotFoundError(KeyError):
    """Dataset id not present in ``AppState.datasets``."""


class EditValidationError(ValueError):
    """Request arguments fail validation (bad range, unknown feature, etc.)."""


class EditConflictError(ValueError):
    """Edit conflicts with prior state (overlap, large-edit threshold).

    The ``detail`` attribute carries a structured dict the caller can
    surface verbatim — frontend / AI can read it to decide how to retry
    (typically with ``confirm_overlap=True`` or ``confirm_large=True``).
    """

    def __init__(self, detail: dict[str, Any]):
        self.detail = detail
        super().__init__(detail.get("message", "Edit conflicts with prior state"))


class DatasetBusyError(RuntimeError):
    """Dataset is locked by an in-progress operation."""


# ── Internal helpers ──────────────────────────────────────────────────────


def _require_dataset(app_state: AppState, dataset_id: str):
    if dataset_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")
    return app_state.datasets[dataset_id]


def _require_unlocked(app_state: AppState, dataset_id: str) -> None:
    if app_state.is_locked(dataset_id):
        raise DatasetBusyError(f"Dataset {dataset_id} is busy (operation in progress)")


def _save_edits(app_state: AppState, dataset_id: str) -> None:
    from lerobot.gui.state import save_edits_to_file

    if dataset_id not in app_state.datasets:
        return
    dataset = app_state.datasets[dataset_id]
    save_edits_to_file(dataset.root, app_state.get_edits_for_dataset(dataset_id))


def _edit_info(index: int, edit: PendingEdit) -> dict[str, Any]:
    return {
        "index": index,
        "edit_type": edit.edit_type,
        "dataset_id": edit.dataset_id,
        "episode_index": edit.episode_index,
        "params": edit.params,
        "created_at": edit.created_at.isoformat(),
    }


# ── Public helpers — called by both FastAPI handlers and MCP tools ────────


def list_pending(app_state: AppState, dataset_id: str | None = None) -> dict[str, Any]:
    """List pending edits, optionally scoped to one dataset.

    Returns ``{"edits": [...], "total": N}``.
    """
    if dataset_id is not None:
        edits = app_state.get_edits_for_dataset(dataset_id)
        # Preserve original indices so a caller can remove by index later.
        infos = [
            _edit_info(i, e) for i, e in enumerate(app_state.pending_edits) if e.dataset_id == dataset_id
        ]
    else:
        edits = app_state.pending_edits
        infos = [_edit_info(i, e) for i, e in enumerate(app_state.pending_edits)]
    return {"edits": infos, "total": len(edits)}


def propose_delete(app_state: AppState, dataset_id: str, episode_index: int) -> dict[str, Any]:
    """Mark an episode for deletion. Idempotent only on the failure side —
    re-marking the same episode raises ``EditValidationError``.
    """
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)
    if not (0 <= episode_index < dataset.meta.total_episodes):
        raise EditValidationError(
            f"Invalid episode index {episode_index} for dataset {dataset_id} "
            f"(total_episodes={dataset.meta.total_episodes})"
        )
    if app_state.is_episode_deleted(dataset_id, episode_index):
        raise EditValidationError(f"Episode {episode_index} is already marked for deletion in {dataset_id}")
    edit = PendingEdit(
        edit_type="delete",
        dataset_id=dataset_id,
        episode_index=episode_index,
    )
    app_state.add_edit(edit)
    _save_edits(app_state, dataset_id)
    logger.info(f"Marked episode {episode_index} for deletion in {dataset_id}")
    return {
        "status": "ok",
        "message": f"Episode {episode_index} marked for deletion",
        "dataset_id": dataset_id,
        "episode_index": episode_index,
    }


def propose_trim(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    """Set the trim range ``[start_frame, end_frame)`` for an episode.

    Replaces any prior trim on the same episode. A full-range trim
    (``start=0, end=ep_length``) clears the existing trim without
    adding one (= "untrim"); the response message still reads as a set.
    """
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)
    if not (0 <= episode_index < dataset.meta.total_episodes):
        raise EditValidationError(
            f"Invalid episode index {episode_index} for dataset {dataset_id} "
            f"(total_episodes={dataset.meta.total_episodes})"
        )

    episode = dataset.meta.episodes[episode_index]
    episode_length = int(episode["length"])

    if start_frame < 0 or end_frame > episode_length:
        raise EditValidationError(
            f"Invalid trim range [{start_frame}, {end_frame}) for episode "
            f"{episode_index} (length={episode_length})"
        )
    if start_frame >= end_frame:
        raise EditValidationError(
            f"Invalid trim range [{start_frame}, {end_frame}): the keep window must have positive length"
        )

    # Drop any prior trim on this episode — replace semantics.
    app_state.pending_edits = [
        e
        for e in app_state.pending_edits
        if not (e.dataset_id == dataset_id and e.episode_index == episode_index and e.edit_type == "trim")
    ]

    # Only stage if it's not the full range; full-range = "untrim".
    if start_frame > 0 or end_frame < episode_length:
        edit = PendingEdit(
            edit_type="trim",
            dataset_id=dataset_id,
            episode_index=episode_index,
            params={"start_frame": start_frame, "end_frame": end_frame},
        )
        app_state.add_edit(edit)
        logger.info(f"Set trim for episode {episode_index}: frames [{start_frame}, {end_frame})")

    _save_edits(app_state, dataset_id)
    kept = end_frame - start_frame
    dropped = episode_length - kept
    return {
        "status": "ok",
        # The message states kept/dropped explicitly so a caller who got
        # the semantics backwards (thinking it means "remove this range")
        # sees the actual outcome in the response and can self-correct.
        "message": (
            f"Episode {episode_index}: keep frames [{start_frame}, {end_frame}) "
            f"({kept} of {episode_length} frames; dropping {dropped})"
        ),
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "kept_frames": kept,
        "dropped_frames": dropped,
        "episode_length_before": episode_length,
    }


#: A treatment edit describes the dataset, not one episode. The queue's
#: records are episode-shaped, so it carries this sentinel rather than a real
#: episode number that would badge an unrelated episode in the tree.
MASK_TREATMENTS_EPISODE = -1


def _mask_edit_is_a_noop(dataset, episode_index, camera, label, lo, hi, action) -> bool:
    """Would this edit leave every frame in ``[lo, hi)`` as it already is?

    Read from the STORED rows, which is what the edit is lowered against.
    Cheap: ``states`` reports presence and mutedness per frame without decoding
    any RLE.
    """
    from lerobot.datasets.mask_store import states

    try:
        per_frame = states(dataset, episode_index, camera)
    except Exception:
        return False  # never suppress an edit because the check itself failed
    want = {"enable": True, "disable": False}.get(action)
    for f in range(lo, min(hi, len(per_frame))):
        carried = label in per_frame[f]
        if action == "delete":
            if carried:
                return False
        elif carried and per_frame[f][label] is not want:
            return False
    return True


def propose_mask_range(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    camera: str,
    label: str,
    from_frame: int,
    to_frame: int,
    action: str,
) -> dict[str, Any]:
    """Stage one segment-level mask edit. Returns the pending summary.

    Pre: ``dataset_id`` is open, ``camera`` has a mask column, ``label`` is in
    its vocabulary, and ``[from_frame, to_frame)`` is a non-empty range inside
    the episode. ``action`` is ``disable``, ``enable`` or ``delete``.
    Post: exactly one pending edit covers the union of this span and any
    already-staged span for the same (camera, label, action) that touches it.

    Coalescing at stage time is what keeps the queue readable: dragging across
    three adjacent segments, or toggling a run frame by frame, would otherwise
    produce an entry each, in a queue meant for changes a person reviews before
    saving. Only identical (camera, label, action) triples merge -- a disable
    and a delete over the same frames are different intents and both survive,
    to be applied in the order they were made.
    """
    from lerobot.datasets.mask_store import labels_of, spec_of
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)
    if action not in ("disable", "enable", "delete"):
        raise EditValidationError(f"unknown mask action {action!r}")
    if spec_of(dataset, camera) is None:
        raise EditValidationError(f"{camera} has no mask column")
    if label not in labels_of(dataset, camera):
        raise EditValidationError(f"{label!r} is not in {camera}'s vocabulary")
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise EditValidationError(f"no episode {episode_index}")
    length = int(dataset.meta.episodes["length"][episode_index])
    lo, hi = int(from_frame), int(to_frame)
    if not (0 <= lo < hi <= length):
        raise EditValidationError(f"frames [{lo}, {hi}) outside episode {episode_index}, which has {length}")

    def _for_label(e) -> bool:
        return (
            e.edit_type == "mask_range"
            and e.dataset_id == dataset_id
            and e.episode_index == episode_index
            and e.params.get("camera") == camera
            and e.params.get("label") == label
        )

    # The queue records the intended END STATE, not the click history -- the
    # same rule the flag edits follow, where an edit collapses against its own
    # opposite. So a later edit SUPERSEDES an earlier one over the frames they
    # share: disabling a segment and then re-enabling it leaves the queue as it
    # started, rather than two entries that cancel at save time and read to the
    # operator as two changes to review.
    superseded: list = []
    for e in list(app_state.pending_edits):
        if not _for_label(e):
            continue
        e_lo, e_hi = int(e.params["from_frame"]), int(e.params["to_frame"])
        if e_hi < lo or hi < e_lo:
            continue  # disjoint, and not even touching
        if e.params.get("action") == action:
            # Same intent: absorb it, so touching spans are one entry. [0,4) and
            # [4,8) are one span to the eye and must not be two rows.
            lo, hi = min(lo, e_lo), max(hi, e_hi)
            superseded.append(e)
        elif e_hi <= lo or hi <= e_lo:
            continue  # different intent, merely adjacent: both stand
        elif lo <= e_lo and e_hi <= hi:
            superseded.append(e)  # fully overwritten by this one
        elif e_lo < lo:
            e.params["to_frame"] = lo  # trim rather than drop what this misses
        else:
            e.params["from_frame"] = hi
    for e in superseded:
        app_state.pending_edits.remove(e)

    # If what remains would change nothing on disk, stage nothing. The end state
    # already matches, and an entry that applies to zero frames is a change to
    # review that is not a change.
    if _mask_edit_is_a_noop(dataset, episode_index, camera, label, lo, hi, action):
        logger.info(
            "MASK_RANGE_STAGE noop ep=%d camera=%s label=%r action=%s frames=[%d,%d) "
            "superseded=%d (the stored rows already match)",
            episode_index,
            camera,
            label,
            action,
            lo,
            hi,
            len(superseded),
        )
        _save_edits(app_state, dataset_id)
        return {
            "status": "ok",
            "pending": False,
            "frames": 0,
            "merged": len(superseded),
            "action": action,
            "label": label,
        }

    app_state.pending_edits.append(
        PendingEdit(
            edit_type="mask_range",
            dataset_id=dataset_id,
            episode_index=episode_index,
            params={
                "camera": camera,
                "label": label,
                "from_frame": lo,
                "to_frame": hi,
                "action": action,
            },
        )
    )
    # Asserted rather than assumed: superseding rewrites lo/hi, and an inverted
    # or out-of-episode span here would be lowered against the wrong frames at
    # save time, when the operator is no longer watching.
    assert 0 <= lo < hi <= length, f"staged span [{lo}, {hi}) escaped episode {episode_index} ({length})"
    logger.info(
        "MASK_RANGE_STAGE ep=%d camera=%s label=%r action=%s frames=[%d,%d) superseded=%d",
        episode_index,
        camera,
        label,
        action,
        lo,
        hi,
        len(superseded),
    )
    _save_edits(app_state, dataset_id)
    return {
        "status": "ok",
        "pending": True,
        "frames": hi - lo,
        "merged": len(superseded),
        "action": action,
        "label": label,
    }


def propose_mask_run(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    rows: list[dict],
) -> dict[str, Any]:
    """Extend the run's single pending edit with the frames just segmented.

    Pre: ``dataset_id`` is open; every row is
    ``{"camera", "frame", "rle": {label_name: counts}}`` for a camera with a
    mask column and a frame inside ``episode_index``. Post: exactly one
    ``mask_run`` edit exists for this (dataset, episode), holding the union of
    what was already staged and what is passed here; a frame sent twice keeps
    the FIRST masks, because the write rule fills a gap once and a re-send is a
    repeat of the same run, not a correction.

    One edit rather than one per frame: a 1,440-frame episode would otherwise
    produce 1,440 entries in a queue meant for human-sized changes. Cancel
    discards the run whole; Save commits it.

    Labels are declared here, on the first flush that carries them -- not when
    Apply is ticked. The vocabulary is positional and can never shrink, so a
    label declared on tick would outlive a run cancelled a second later with no
    way to take the slot back. Declaring on the first flush also makes the new
    label's lane appear and fill while the operator watches, which is why the
    declaration is a metadata write now rather than part of the commit.
    """
    from lerobot.datasets.mask_store import append_labels, mask_columns, spec_of
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise EditValidationError(f"no episode {episode_index}")
    length = int(dataset.meta.episodes["length"][episode_index])

    seen_labels: list[str] = []
    clean: dict[str, dict[str, str]] = {}
    for row in rows:
        camera = str(row.get("camera") or "")
        if spec_of(dataset, camera) is None:
            raise EditValidationError(f"{camera} has no mask column")
        frame = int(row.get("frame", -1))
        if not 0 <= frame < length:
            raise EditValidationError(f"frame {frame} outside episode {episode_index}, which has {length}")
        rle = {str(k): str(v) for k, v in (row.get("rle") or {}).items()}
        for name in rle:
            if name not in seen_labels:
                seen_labels.append(name)
        clean.setdefault(f"{camera}:{frame}", {}).update(rle)

    if not clean:
        raise EditValidationError("a run flush carried no rows")

    # Declare on the first flush that names them, into EVERY mask column: a
    # label names an object, and the same object seen from three cameras is one
    # label. Rows still land only where the run looked.
    if mask_columns(dataset) and seen_labels:
        append_labels(dataset, seen_labels)

    def _is_run(e) -> bool:
        return e.edit_type == "mask_run" and e.dataset_id == dataset_id and e.episode_index == episode_index

    existing = next((e for e in app_state.pending_edits if _is_run(e)), None)
    if existing is None:
        edit = PendingEdit(
            edit_type="mask_run",
            dataset_id=dataset_id,
            episode_index=episode_index,
            params={"rows": clean},
        )
        app_state.pending_edits.append(edit)
    else:
        merged = dict(existing.params.get("rows") or {})
        for key, rle in clean.items():
            # First write wins: the write rule fills a gap once, so a frame
            # arriving twice in one run is a repeat, not a correction.
            merged.setdefault(key, {}).update({k: v for k, v in rle.items() if k not in merged.get(key, {})})
        existing.params["rows"] = merged
    _save_edits(app_state, dataset_id)
    held = (existing or edit).params["rows"]
    logger.info(
        "MASK_RUN_STAGE ep=%d flushed=%d frames=%d labels=%s",
        episode_index,
        len(clean),
        len(held),
        seen_labels,
    )
    return {
        "status": "ok",
        "pending": True,
        "staged": len(clean),
        "frames": len(held),
        "masks": sum(len(v) for v in held.values()),
    }


def apply_mask_run(dataset, episode_index: int, params: dict) -> int:
    """Lower a run's staged rows onto the dataset. Frames changed.

    Follows the write rule per (frame, label): a label is filled only where it is
    ABSENT, so a detected mask and a disabled one are both left alone, and the
    flags already on a row are carried across. The client drops pairs it knows
    are taken before sending them, but the rule is enforced here too -- this is
    the only place that sees the rows as they are at SAVE time, which is not
    necessarily how they were when the run passed over them.
    """
    from lerobot.datasets.feature_value_edits import set_feature_values
    from lerobot.datasets.mask_codec import decode_frame, decode_mask, encode_frame, frame_states
    from lerobot.datasets.mask_compositing import mask_feature_of
    from lerobot.datasets.mask_store import labels_of, spec_of

    rows = params.get("rows") or {}
    by_camera: dict[str, dict[int, dict[str, str]]] = {}
    for key, rle in rows.items():
        camera, _, frame = key.rpartition(":")
        by_camera.setdefault(camera, {})[int(frame)] = rle

    start = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    length = int(dataset.meta.episodes["length"][episode_index])
    batch: list[dict] = []
    changed = 0
    for camera, per_frame in by_camera.items():
        spec = spec_of(dataset, camera)
        if spec is None:
            continue
        labels = labels_of(dataset, camera)
        shape = tuple(spec.get("mask_size") or (0, 0))
        key = mask_feature_of(camera)
        column = dataset.hf_dataset[key][start : start + length]

        def _stored(f: int, _col=column) -> str:
            cell = _col[f]
            return str(cell[0] if isinstance(cell, (list, tuple)) and cell else (cell or ""))

        for frame, rle in sorted(per_frame.items()):
            # Bounds again, not only at propose time. A pending edit outlives the
            # proposal that made it, and the episode it names can be shorter by
            # the time Save runs -- and an out-of-range frame here is not an
            # error, it is a wrong write: `column[-1]` reads the episode's last
            # row and `start + frame` addresses the PREVIOUS episode's frames.
            if not 0 <= frame < length:
                raise EditValidationError(
                    f"frame {frame} is outside episode {episode_index}, which has {length} frames; "
                    "the pending edit was made against a different version of this dataset"
                )
            current = _stored(frame)
            states = frame_states(current, labels) if current else {}
            # The write rule, per (frame, label): absent only.
            fresh = {n: c for n, c in rle.items() if n in labels and n not in states}
            if not fresh:
                continue
            merged = decode_frame(current, labels, shape, include_disabled=True) if current else {}
            muted = [n for n, on in states.items() if not on]
            for name, counts in fresh.items():
                merged[name] = decode_mask(counts, shape)
            batch.append(
                {
                    "feature": key,
                    "from_index": start + frame,
                    "to_index": start + frame + 1,
                    "value": encode_frame(merged, labels, disabled=muted),
                }
            )
            changed += 1
    if batch:
        set_feature_values(dataset, batch, in_place=True)
    return changed


def apply_mask_range(dataset, episode_index: int, params: dict) -> int:
    """Lower one staged mask edit onto the rows as they now are. Frames changed.

    Lowered here rather than at stage time because the value is a different RLE
    per frame and depends on what else that frame carries -- two labels edited
    over overlapping ranges must compose, not overwrite.
    """
    from lerobot.datasets.mask_store import delete_label_range, set_label_enabled

    span = range(int(params["from_frame"]), int(params["to_frame"]))
    action = params["action"]
    if action == "delete":
        return delete_label_range(dataset, episode_index, params["camera"], params["label"], frames=span)
    return set_label_enabled(
        dataset,
        episode_index,
        params["camera"],
        params["label"],
        action == "enable",
        frames=span,
    )


def propose_mask_treatments(
    app_state: AppState,
    dataset_id: str,
    treatments: dict[str, dict],
    background: dict,
) -> dict[str, Any]:
    """Stage a change to how the stored masks are rendered.

    Pre: ``dataset_id`` is open and has at least one adopted mask feature;
    ``treatments`` maps a label of that feature's vocabulary to an effect
    ``{key, params}``; ``background`` is the effect for everything outside
    every mask. Post: exactly one staged treatment edit exists for the dataset
    (a later one replaces it — the queue records the intended end state, not
    each click), and it is on disk with the rest of the pending edits.

    Raises ``EditValidationError`` when a label is not in the vocabulary: a
    treatment for an object the masks do not contain would silently never
    apply.
    """
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)

    mask_features = {
        name: ft
        for name, ft in dataset.meta.features.items()
        if isinstance(ft, dict) and ft.get("mask_encoding") == "coco_rle"
    }
    if not mask_features:
        raise EditValidationError(f"{dataset_id} has no saved masks to treat")

    vocabulary = {label for ft in mask_features.values() for label in ft.get("mask_labels", [])}
    unknown = sorted(set(treatments) - vocabulary)
    if unknown:
        raise EditValidationError(
            f"no mask is labelled {unknown[0]!r} in this dataset "
            f"(labels: {sorted(vocabulary)}) — a treatment for it would never apply"
        )

    # One pending treatment edit per dataset: clicking through four effects is
    # one decision, not four, and Discard should return to the saved recipe
    # rather than to whichever click came before.
    for existing in list(app_state.get_edits_for_dataset(dataset_id)):
        if existing.edit_type == "mask_treatments":
            app_state.pending_edits.remove(existing)

    edit = PendingEdit(
        edit_type="mask_treatments",
        dataset_id=dataset_id,
        episode_index=MASK_TREATMENTS_EPISODE,
        params={"treatments": dict(treatments), "background": dict(background)},
    )
    app_state.add_edit(edit)
    _save_edits(app_state, dataset_id)
    return {"status": "staged", "features": sorted(mask_features), "treatments": treatments}


def apply_mask_treatments(dataset, params: dict) -> list[str]:
    """Write a staged treatment edit into every mask feature's spec.

    Post: ``meta/info.json`` carries the new recipe and the in-memory metadata
    agrees. No parquet is touched — the rows are the masks, and how they are
    rendered is not stored per frame.
    """
    from lerobot.datasets.dataset_postprocess import _update_mask_feature_info

    keys = [
        name
        for name, ft in dataset.meta.features.items()
        if isinstance(ft, dict) and ft.get("mask_encoding") == "coco_rle"
    ]
    updates = {
        key: {
            "mask_treatments": params.get("treatments", {}),
            "mask_background": params.get("background", {}),
        }
        for key in keys
    }
    _update_mask_feature_info(Path(dataset.root), updates)
    for key, fields in updates.items():
        dataset.meta.features[key].update(fields)
    return keys


def staged_mask_treatments(app_state: AppState, dataset_id: str) -> dict | None:
    """The staged recipe for ``dataset_id``, or None when nothing is staged.

    Playback consults this so the operator sees the edit being judged; the
    training path deliberately does not, because an unsaved edit is not what
    the dataset says yet.
    """
    for edit in app_state.get_edits_for_dataset(dataset_id):
        if edit.edit_type == "mask_treatments":
            return dict(edit.params)
    return None


def discard_pending(app_state: AppState, dataset_id: str | None = None) -> dict[str, Any]:
    """Drop pending edits without touching disk. Scope is per-dataset
    when ``dataset_id`` is given, otherwise every dataset's queue is
    cleared (and every dataset checked for the busy guard first).
    """
    from lerobot.gui.state import clear_edits_file

    if dataset_id is not None:
        _require_unlocked(app_state, dataset_id)
        count = len(app_state.get_edits_for_dataset(dataset_id))
        if dataset_id in app_state.datasets:
            clear_edits_file(app_state.datasets[dataset_id].root)
        app_state.clear_edits(dataset_id)
    else:
        for ds_id in app_state.datasets:
            _require_unlocked(app_state, ds_id)
        count = len(app_state.pending_edits)
        for dataset in app_state.datasets.values():
            clear_edits_file(dataset.root)
        app_state.clear_edits(None)

    logger.info(f"Discarded {count} pending edits")
    return {
        "status": "ok",
        "message": f"Discarded {count} pending edits",
        "discarded": count,
    }


# ── Merge dataset-into-dataset helpers ────────────────────────────────────


def check_merge_compat(app_state: AppState, source_id: str, target_id: str) -> dict[str, Any]:
    """Compare schemas of two opened datasets; return a structured compat report.

    Pure read — no disk writes, no locks taken. Used by both the FastAPI
    ``/api/edits/merge-into/validate`` endpoint and the MCP
    ``validate_dataset_merge`` tool. Returns ``{"compatible": bool,
    "mismatches": [...]}`` where each mismatch dict names the field and
    carries the conflicting values.

    Raises ``DatasetNotFoundError`` if either id isn't opened in the GUI.
    """
    if source_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Source dataset not found: {source_id}")
    if target_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Target dataset not found: {target_id}")

    # Local import to avoid pulling FastAPI handlers into _edits_core.
    from lerobot.gui.api.edits import _validate_merge_compat

    mismatches = _validate_merge_compat(
        app_state.datasets[source_id].meta,
        app_state.datasets[target_id].meta,
    )
    return {"compatible": len(mismatches) == 0, "mismatches": mismatches}


async def merge_dataset_into(
    app_state: AppState,
    source_id: str,
    target_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Merge ``source_id``'s episodes into ``target_id`` (canonical write).

    Source is read-only; target's parquet + videos grow in place. Both
    dataset locks are acquired for the duration of the merge so concurrent
    edits / saves on either side can't race. The underlying
    ``dataset_tools.merge_into`` runs synchronously and can take minutes
    for large datasets; it's pushed onto the default executor so the
    event loop stays responsive.

    Outcome transparency: the response carries before/after counts on the
    target so the caller can see exactly how much grew. ``source_*``
    values describe what was copied.

    Raises:
        DatasetNotFoundError: source or target not opened in the GUI.
        EditValidationError: source == target (cannot self-merge) or
            ``merge_into`` raised ``ValueError`` (schema mismatch and
            ``force=False``).
        DatasetBusyError: source or target lock already held.
    """
    import asyncio

    if source_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Source dataset not found: {source_id}")
    if target_id not in app_state.datasets:
        raise DatasetNotFoundError(f"Target dataset not found: {target_id}")
    if source_id == target_id:
        raise EditValidationError("Cannot merge a dataset into itself")

    _require_unlocked(app_state, source_id)
    _require_unlocked(app_state, target_id)

    target_lock = app_state.get_lock(target_id)
    source_lock = app_state.get_lock(source_id)
    if target_lock.locked() or source_lock.locked():
        raise DatasetBusyError("One or both datasets are busy")

    source_ds = app_state.datasets[source_id]
    target_ds = app_state.datasets[target_id]
    source_eps = source_ds.num_episodes
    source_frames = source_ds.num_frames
    target_eps_before = target_ds.num_episodes
    target_frames_before = target_ds.num_frames

    logger.info(
        f"Merging {source_eps} episodes from {source_id} into {target_id} "
        f"({target_eps_before} episodes) force={force}"
    )

    async with target_lock, source_lock:
        try:
            from lerobot.datasets.dataset_tools import merge_into

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: merge_into(target_ds, source_ds, skip_validation=force)
            )
        except ValueError as e:
            raise EditValidationError(str(e)) from e

        # Invalidate target caches: new episodes added, the cumulative-sum
        # cache must be dropped so subsequent frame lookups pick up growth.
        from lerobot.gui.api.datasets import _invalidate_episode_start_indices
        from lerobot.gui.cache_invalidation import invalidate_caches

        invalidate_caches(app_state, target_id, invalidate_episode_indices=_invalidate_episode_start_indices)

    logger.info(
        f"Merge complete: {target_ds.num_episodes} episodes, "
        f"{target_ds.num_frames} frames in target {target_id}"
    )

    return {
        "status": "ok",
        "source_id": source_id,
        "target_id": target_id,
        "source_episodes_merged": source_eps,
        "source_frames_merged": source_frames,
        "target_episodes_before": target_eps_before,
        "target_episodes_after": target_ds.num_episodes,
        "target_frames_before": target_frames_before,
        "target_frames_after": target_ds.num_frames,
        "force_used": force,
    }


# ── Feature-set helpers (heavier — validation + overlap resolution) ───────


_DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
_READONLY_DTYPES = {"image", "video"}
_LARGE_SAVE_FRAME_THRESHOLD = 10_000


def _resolve_synthetic_feature(dataset, requested_feature: str) -> str:
    """Map a user-facing feature name to its storage feature name.

    Special case for the LeRobot 3.0 subtask format: ``subtask`` (string)
    → ``subtask_index`` (int64) when the dataset has a subtask lookup.
    Returns the input unchanged otherwise.
    """
    from lerobot.gui.api.datasets import (
        SUBTASK_DISPLAY_FEATURE,
        SUBTASK_STORAGE_FEATURE,
        _has_subtask_lookup,
    )

    if (
        requested_feature == SUBTASK_DISPLAY_FEATURE
        and SUBTASK_STORAGE_FEATURE in dataset.meta.features
        and _has_subtask_lookup(dataset)
    ):
        return SUBTASK_STORAGE_FEATURE
    return requested_feature


def _validate_value_against_declared_bounds(feature_name: str, feature_info: dict, value: Any) -> str:
    """Return error string (empty when valid) for declared-bounds / categorical check."""
    from lerobot.datasets.feature_utils import (
        is_categorical_feature,
        validate_feature_numeric_bounds,
    )

    has_bounds = feature_info.get("min") is not None or feature_info.get("max") is not None
    if not has_bounds and not is_categorical_feature(feature_info):
        return ""

    import numpy as np

    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return f"Could not interpret value {value!r} as numeric for bounds check"
    return validate_feature_numeric_bounds(feature_name, feature_info, arr)


def _validate_feature_edit(
    dataset,
    dataset_id: str,
    episode_index: int,
    feature: str,
    frame_from: int,
    frame_to: int,
    value: Any,
    confirm_large: bool,
) -> tuple[str, int, int, int, int, dict[str, Any]]:
    """Validate a feature-set request against the dataset schema + envelope.

    Returns ``(storage_feature, frame_from, frame_to, global_from, global_to,
    feature_info)``. ``frame_from``/``frame_to`` may be coerced to the full
    episode range when the feature is detected as per-episode-broadcast.

    Raises:
        EditValidationError: for schema / range failures (400-class).
        EditConflictError: for the large-edit threshold (409-class with
            structured detail so the caller knows to retry with
            ``confirm_large=True``).
    """
    feature_dict = dataset.meta.features
    storage_feature = _resolve_synthetic_feature(dataset, feature)

    if storage_feature not in feature_dict:
        raise EditValidationError(f"Unknown feature: {feature!r}")

    feature_info = feature_dict[storage_feature]
    dtype = feature_info.get("dtype", "")

    if storage_feature in _DEFAULT_FEATURES:
        raise EditValidationError(f"Feature {feature!r} is auto-managed and not editable")
    if dtype in _READONLY_DTYPES:
        raise EditValidationError(f"Feature {feature!r} has dtype {dtype!r} and is not editable in V1")
    if storage_feature == "action" or storage_feature.startswith("observation."):
        raise EditValidationError(
            f"Feature {feature!r} is recorded sensor / control data and is read-only in V1"
        )

    if not (0 <= episode_index < dataset.meta.total_episodes):
        raise EditValidationError(f"Invalid episode index: {episode_index}")

    ep = dataset.meta.episodes[episode_index]
    ep_length = int(ep["length"])
    if frame_from < 0 or frame_to > ep_length or frame_from >= frame_to:
        raise EditValidationError(
            f"Invalid range [{frame_from}, {frame_to}) for episode {episode_index} (length={ep_length})"
        )

    # Per-episode broadcast features: silently coerce sub-range to full episode.
    from lerobot.gui.api.datasets import _detect_per_episode_features, _get_episode_start_index

    per_episode = _detect_per_episode_features(dataset_id, dataset)
    if storage_feature in per_episode:
        frame_from, frame_to = 0, ep_length

    n_frames = frame_to - frame_from
    if n_frames > _LARGE_SAVE_FRAME_THRESHOLD and not confirm_large:
        raise EditConflictError(
            {
                "code": "large_edit_confirmation_required",
                "message": (
                    f"This edit touches {n_frames} frames (> {_LARGE_SAVE_FRAME_THRESHOLD}). "
                    "Re-send with confirm_large=true to proceed."
                ),
                "frames": n_frames,
            }
        )

    bounds_error = _validate_value_against_declared_bounds(feature, feature_info, value)
    if bounds_error:
        raise EditValidationError(bounds_error)

    episode_start = _get_episode_start_index(dataset_id, episode_index)
    global_from = episode_start + frame_from
    global_to = episode_start + frame_to
    return storage_feature, frame_from, frame_to, global_from, global_to, feature_info


def _find_overlapping_feature_edits(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    feature: str,
    frame_from: int,
    frame_to: int,
) -> list[tuple[int, PendingEdit]]:
    """Return ``(index_in_pending_edits, edit)`` for prior edits that overlap."""
    overlaps: list[tuple[int, Any]] = []
    for i, e in enumerate(app_state.pending_edits):
        if (
            e.edit_type != "feature_set"
            or e.dataset_id != dataset_id
            or e.episode_index != episode_index
            or e.params.get("feature") != feature
        ):
            continue
        a = int(e.params.get("frame_from", 0))
        b = int(e.params.get("frame_to", 0))
        if frame_from < b and a < frame_to:
            overlaps.append((i, e))
    return overlaps


def _clip_overlapping_edits(
    app_state: AppState,
    overlaps: list[tuple[int, Any]],
    new_from: int,
    new_to: int,
) -> int:
    """Last-write-wins resolution: clip prior edits' ranges to the non-overlap.

    Iterates in reverse index order so removals don't shift indices we still
    need. Returns the count of fully-contained-and-removed edits. Mutates
    ``app_state.pending_edits`` in place.
    """
    from lerobot.gui.state import PendingEdit

    removed = 0
    for i, e in sorted(overlaps, key=lambda x: x[0], reverse=True):
        a = int(e.params["frame_from"])
        b = int(e.params["frame_to"])
        if new_from <= a and new_to >= b:
            app_state.pending_edits.pop(i)
            removed += 1
            continue
        left = (a, min(b, new_from))
        right = (max(a, new_to), b)
        left_keep = left[0] < left[1]
        right_keep = right[0] < right[1]
        if left_keep and right_keep:
            e.params["frame_from"] = left[0]
            e.params["frame_to"] = left[1]
            episode_start = e.params["global_from_index"] - a
            e.params["global_from_index"] = episode_start + left[0]
            e.params["global_to_index"] = episode_start + left[1]
            split = PendingEdit(
                edit_type="feature_set",
                dataset_id=e.dataset_id,
                episode_index=e.episode_index,
                params={
                    **e.params,
                    "frame_from": right[0],
                    "frame_to": right[1],
                    "global_from_index": episode_start + right[0],
                    "global_to_index": episode_start + right[1],
                },
            )
            app_state.pending_edits.append(split)
        elif left_keep:
            episode_start = e.params["global_from_index"] - a
            e.params["frame_from"] = left[0]
            e.params["frame_to"] = left[1]
            e.params["global_from_index"] = episode_start + left[0]
            e.params["global_to_index"] = episode_start + left[1]
        elif right_keep:
            episode_start = e.params["global_from_index"] - a
            e.params["frame_from"] = right[0]
            e.params["frame_to"] = right[1]
            e.params["global_from_index"] = episode_start + right[0]
            e.params["global_to_index"] = episode_start + right[1]
        else:
            app_state.pending_edits.pop(i)
            removed += 1
    return removed


def _episode_containing(dataset, global_index: int) -> tuple[int, int]:
    """``(episode_index, episode_start)`` for a global frame index.

    Bit edits are staged per feature rather than per episode, so a collapsed
    set can contain edits from several episodes at once. Each one's
    episode-local frames have to come from its own episode, which is what this
    answers.

    Pre: ``global_index`` lies inside the dataset. Post: ``episode_start <=
    global_index < `` the episode's end.
    """
    episodes = dataset.meta.episodes
    for index in range(dataset.meta.total_episodes):
        episode = episodes[index]
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        if start <= global_index < end:
            return index, start
    raise EditValidationError(f"frame {global_index} lies in no episode of {dataset.repo_id!r}")


def propose_feature_bits(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    feature: str,
    frame_from: int,
    frame_to: int,
    set_flags: Sequence[str] = (),
    clear_flags: Sequence[str] = (),
    confirm_large: bool = False,
) -> dict[str, Any]:
    """Stage "tick / untick these flags over this range".

    Unlike :func:`propose_feature_set` this never raises a conflict. Constant
    values contest -- two of them on one frame disagree about the whole cell,
    and clipping the older one discards something the operator asked for, which
    is why that path needs consent. Bit edits do not: the newest edit takes the
    bits it names away from older edits over the shared frames, which discards
    nothing still meant, since the operator just re-specified exactly those
    bits.

    Keeping the pending set per-bit disjoint that way is also what makes apply
    order irrelevant, the same guarantee clipping gives the constant-value path.

    An edit that would change no frame is not staged at all, so re-ticking a
    flag a range already carries leaves nothing pending rather than something
    that saves as a no-op.

    Raises:
        DatasetNotFoundError, DatasetBusyError: as for feature-set edits.
        EditValidationError: not a bitset column, an undeclared flag, or a
            flag named in both lists.
        EditConflictError: the large-edit threshold, retried with
            ``confirm_large=True``.
    """
    from lerobot.datasets.dataset_reader import _int_column
    from lerobot.datasets.feature_bit_edits import BitEdit, is_effective, stage
    from lerobot.datasets.feature_utils import flag_bit, is_flags_feature
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)

    spec = dataset.meta.features.get(feature)
    if not isinstance(spec, dict) or not is_flags_feature(spec):
        raise EditValidationError(f"{feature!r} is not a flags column")

    both = set(set_flags) & set(clear_flags)
    if both:
        raise EditValidationError(
            f"flag(s) {sorted(both)} are both ticked and unticked; an edit must not contradict itself"
        )

    def mask_for(flags: Sequence[str]) -> int:
        mask = 0
        for flag in flags:
            try:
                mask |= 1 << flag_bit(spec, flag)
            except ValueError as e:
                raise EditValidationError(str(e)) from e
        return mask

    set_bits, clear_bits = mask_for(set_flags), mask_for(clear_flags)
    if not (set_bits or clear_bits):
        raise EditValidationError("no flags given to tick or untick")

    # Reuse the value path's range resolution, per-episode coercion and
    # large-edit guard. Zero is a legal value for a bitset column, so it passes
    # bounds checking without standing for anything.
    (storage_feature, eff_from, eff_to, global_from, global_to, _) = _validate_feature_edit(
        dataset, dataset_id, episode_index, feature, frame_from, frame_to, 0, confirm_large
    )

    proposed = BitEdit(
        feature=storage_feature,
        from_index=global_from,
        to_index=global_to,
        set_bits=set_bits,
        clear_bits=clear_bits,
    )

    values = _int_column(dataset.reader.hf_dataset, storage_feature)
    existing = [
        BitEdit(
            feature=e.params["feature"],
            from_index=e.params["global_from_index"],
            to_index=e.params["global_to_index"],
            set_bits=int(e.params.get("set_bits", 0)),
            clear_bits=int(e.params.get("clear_bits", 0)),
        )
        for e in app_state.get_edits_for_dataset(dataset_id)
        if e.edit_type == "feature_bits" and e.params["feature"] == storage_feature
    ]
    collapsed = [edit for edit in stage(existing, proposed) if is_effective(values, edit)]

    # Replace this feature's staged bit edits wholesale: `stage` returns the
    # whole set, and rewriting it is simpler than reconciling in place.
    app_state.pending_edits = [
        e
        for e in app_state.pending_edits
        if not (
            e.dataset_id == dataset_id
            and e.edit_type == "feature_bits"
            and e.params["feature"] == storage_feature
        )
    ]
    for edit in collapsed:
        # Derived from this edit's own global range, not the new one's. The
        # pending set is keyed by feature and therefore spans episodes, so
        # collapsing can hand back an edit belonging to a different episode --
        # and stamping it with the caller's episode put it on the wrong row.
        owner_episode, episode_start = _episode_containing(dataset, edit.from_index)
        app_state.add_edit(
            PendingEdit(
                edit_type="feature_bits",
                dataset_id=dataset_id,
                episode_index=owner_episode,
                params={
                    "feature": edit.feature,
                    # Episode-local frames alongside the global ones, as
                    # feature_set carries: the GUI merges pending edits into the
                    # row it is drawing, and that row is episode-local.
                    "frame_from": edit.from_index - episode_start,
                    "frame_to": edit.to_index - episode_start,
                    "global_from_index": edit.from_index,
                    "global_to_index": edit.to_index,
                    "set_bits": edit.set_bits,
                    "clear_bits": edit.clear_bits,
                },
            )
        )
    _save_edits(app_state, dataset_id)
    logger.info(
        f"Staged flag edit: feature={storage_feature} ep={episode_index} "
        f"global=[{global_from}, {global_to}) set={set_bits:#b} clear={clear_bits:#b} "
        f"-> {len(collapsed)} pending"
    )
    return {
        "status": "ok",
        "message": "Flag edit staged" if collapsed else "Nothing to change",
        "pending": len(collapsed),
        "frame_from": eff_from,
        "frame_to": eff_to,
    }


def propose_feature_set(
    app_state: AppState,
    dataset_id: str,
    episode_index: int,
    feature: str,
    frame_from: int,
    frame_to: int,
    value: Any,
    confirm_large: bool = False,
    confirm_overlap: bool = False,
) -> dict[str, Any]:
    """Stage a per-frame feature-value edit.

    Raises:
        DatasetNotFoundError: dataset id not loaded.
        DatasetBusyError: dataset is locked by another operation.
        EditValidationError: schema / range / bounds failure.
        EditConflictError: large-edit threshold (retry with
            ``confirm_large=True``) or overlap with prior staged edits
            (retry with ``confirm_overlap=True``); ``detail`` carries
            structured info so the caller can present a useful prompt.
    """
    from lerobot.gui.state import PendingEdit

    _require_unlocked(app_state, dataset_id)
    dataset = _require_dataset(app_state, dataset_id)
    (
        storage_feature,
        eff_from,
        eff_to,
        global_from,
        global_to,
        _,
    ) = _validate_feature_edit(
        dataset, dataset_id, episode_index, feature, frame_from, frame_to, value, confirm_large
    )

    overlaps = _find_overlapping_feature_edits(
        app_state, dataset_id, episode_index, storage_feature, eff_from, eff_to
    )
    if overlaps and not confirm_overlap:
        raise EditConflictError(
            {
                "code": "overlapping_edit",
                "message": (
                    f"You already have {len(overlaps)} staged edit(s) on "
                    f"{feature!r} (episode {episode_index}) overlapping "
                    f"frames {eff_from}…{eff_to - 1}. "
                    "Re-send with confirm_overlap=true to clip the prior edit(s)."
                ),
                "feature": storage_feature,
                "episode_index": episode_index,
                "new_range": [eff_from, eff_to],
                "overlapping": [
                    {
                        "edit_index": i,
                        "frame_from": int(e.params["frame_from"]),
                        "frame_to": int(e.params["frame_to"]),
                        "value": e.params.get("value"),
                    }
                    for i, e in overlaps
                ],
            }
        )
    if overlaps and confirm_overlap:
        removed = _clip_overlapping_edits(app_state, overlaps, eff_from, eff_to)
        logger.info(
            f"Resolved {len(overlaps)} overlapping edit(s) on "
            f"{storage_feature} ep={episode_index}: {removed} removed, "
            f"{len(overlaps) - removed} clipped"
        )

    edit = PendingEdit(
        edit_type="feature_set",
        dataset_id=dataset_id,
        episode_index=episode_index,
        params={
            "feature": storage_feature,
            "frame_from": eff_from,
            "frame_to": eff_to,
            "global_from_index": global_from,
            "global_to_index": global_to,
            "value": value,
        },
    )
    app_state.add_edit(edit)
    _save_edits(app_state, dataset_id)
    logger.info(
        f"Staged feature-set edit: feature={storage_feature} ep={episode_index} "
        f"frames=[{eff_from}, {eff_to}) global=[{global_from}, {global_to})"
    )
    response: dict[str, Any] = {
        "status": "ok",
        "message": "Feature-set edit staged",
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "feature": storage_feature,
        "frame_from": eff_from,
        "frame_to": eff_to,
    }
    if eff_from != frame_from or eff_to != frame_to:
        response["coerced_range"] = [eff_from, eff_to]
        response["coerce_reason"] = "per_episode_broadcast"
    return response
