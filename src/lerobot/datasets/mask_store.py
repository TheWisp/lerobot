# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Read, write and remove a dataset's saved masks, without a segmenter.

The column is defined in :mod:`mask_codec` and named by :mod:`mask_compositing`;
this is the layer that puts rows into a dataset and takes them out again. It
deliberately knows nothing about SAM3: it accepts masks as arrays and returns
them as arrays, so a producer is a caller rather than a dependency. Anything
that can make a boolean array per object -- a model, a colour key, a hand-drawn
region, a test -- can write here.

The four operations, and what each costs:

* :func:`adopt` adds the column. A dataset-wide schema change, so callers with
  a user in front of them must confirm it first.
* :func:`write_episode` replaces one episode's rows. Touches parquet.
* :func:`read_frame` / :func:`read_episode` decode rows back to arrays.
* :func:`retire_label` and :func:`remove` are the delete half. Retiring is a
  metadata write; removing drops the column.

Positions are the contract. A row is ``[[label_id, rle], …]`` where
``label_id`` indexes ``mask_labels``, so a label can be appended or renamed in
place, and can never be moved or deleted -- see :func:`retire_label`.

**The vocabulary is one per DATASET, mirrored into every mask column.** A label
names a physical object, and the same object seen from three cameras is one
label, not three. So :func:`append_labels`, :func:`rename_label` and
:func:`retire_label` act on every column at once, and :func:`vocabulary_of`
refuses a dataset whose columns disagree rather than picking one.

It is mirrored rather than stored once because ``info.json`` has nowhere to put
it. ``DatasetInfo`` is a closed dataclass: ``from_dict`` drops keys it does not
declare and ``to_dict`` never writes them back, so a top-level
``mask_vocabulary`` section would survive until the next metadata write and
then vanish -- silently, since every row still decodes right up until the
labels are gone. Only ``features[...]`` is free-form. Hoisting it would mean
adding a field to a core upstream type for something only mask datasets use.

Because every column is appended to in the same order, a label id means the
same thing in all of them. That is a consequence of the mirroring, not a
separate guarantee, and :func:`unify_vocabulary` repairs datasets written
before it held.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.datasets.mask_codec import decode_frame, encode_frame, feature_spec, frame_states
from lerobot.datasets.mask_compositing import (
    MASK_NAMESPACE,
    camera_feature_of,
    mask_feature_of,
    mask_keys_for,
)

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

#: Reserved: positions retired from the vocabulary. See :func:`retire_label`.
RETIRED_KEY = "mask_labels_retired"

#: The fill for a freshly adopted column: NEVER WRITTEN. Distinct from
#: ``mask_codec.EMPTY`` ("[]"), which means segmented and nothing found.
NEVER_WRITTEN = ""


def _episode_bounds(dataset: LeRobotDataset, episode: int) -> tuple[int, int]:
    start = int(dataset.meta.episodes["dataset_from_index"][episode])
    return start, int(dataset.meta.episodes["length"][episode])


def _update_info(root: Path, updates: dict[str, dict]) -> None:
    """Merge fields into mask features' info.json entries, atomically."""
    info_path = Path(root) / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key, fields in updates.items():
        if key in info.get("features", {}):
            info["features"][key].update(fields)
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=4))
    # safe-destruct: info.json replaced by rename after a complete write
    os.replace(tmp, info_path)


def _as_bool(mask: np.ndarray) -> np.ndarray:
    """A mask as booleans, thresholding a probability map at 0.5.

    Segmenters emit floats. ``astype(bool)`` would threshold at 0 instead,
    turning every pixel the model gave any weight at all into object -- twice
    the area on a typical SAM output, which reads as poor segmentation rather
    than as a bug. Done here so a producer cannot forget it; the 0.5 matches
    what the SAM3 batch pass already applies at its own call site.
    """
    a = np.asarray(mask)
    return a > 0.5 if a.dtype.kind == "f" else a.astype(bool)


def _refresh(dataset: LeRobotDataset) -> None:
    """Re-read the rows this process holds after writing them.

    ``set_feature_values`` rewrites parquet; the in-memory ``hf_dataset`` was
    loaded at construction and does not notice. Without this a read straight
    after a write returns the old rows, which for a CRUD API is the wrong
    contract -- and is a trap, because the data on disk is correct and only
    this process disagrees.
    """
    reader = getattr(dataset, "reader", None)
    if reader is not None and hasattr(reader, "load_and_activate"):
        # `hf_dataset` is a read-only property over the reader, so reloading the
        # reader is the whole of it.
        reader.load_and_activate()


def spec_of(dataset: LeRobotDataset, camera_key: str) -> dict | None:
    """This camera's mask feature spec, or None when it has no mask column."""
    ft = dataset.meta.features.get(mask_feature_of(camera_key))
    return ft if ft is not None and ft.get("mask_encoding") == "coco_rle" else None


def labels_of(dataset: LeRobotDataset, camera_key: str) -> list[str]:
    """The vocabulary, positionally. Empty when there is no mask column."""
    spec = spec_of(dataset, camera_key)
    return list(spec.get("mask_labels", [])) if spec else []


def mask_columns(dataset: LeRobotDataset) -> dict[str, str]:
    """``{camera_key: mask_key}`` for every camera that has a mask column."""
    cams = [k for k in dataset.meta.features if k.startswith(f"{MASK_NAMESPACE}.")]
    out = {}
    for key in cams:
        # `mask_encoding` is what marks a column as one of ours; the namespace
        # prefix alone would also match a same-named feature added by hand.
        if dataset.meta.features[key].get("mask_encoding"):
            out[camera_feature_of(key, list(dataset.meta.features))] = key
    return out


def vocabulary_of(dataset: LeRobotDataset) -> list[str]:
    """The vocabulary shared by every mask column. Empty if there are none.

    Post: raises when two columns disagree, rather than picking one.

    The vocabulary is a property of the DATASET -- the same physical object
    seen from several cameras is one label -- but ``info.json`` cannot hold it
    at the top level: ``DatasetInfo`` is a closed dataclass whose ``from_dict``
    drops unknown keys and whose ``to_dict`` never writes them back, so a
    dataset-level section would be deleted by the next metadata write. Only
    ``features[...]`` is free-form, so the list is mirrored into each column
    and the invariant enforced here instead of being structurally impossible.

    Divergence is worth an exception rather than a merge: two columns naming
    the same object differently is not a state anything downstream can act on
    sensibly, and silently choosing one would keep a treatment pointed at a
    name that only some cameras use.
    """
    seen: dict[str, list[str]] = {}
    for cam, key in mask_columns(dataset).items():
        seen[cam] = list(dataset.meta.features[key].get("mask_labels", []))
    if not seen:
        return []
    first = next(iter(seen.values()))
    if any(v != first for v in seen.values()):
        raise ValueError(
            "mask columns declare different vocabularies, so the same object may be named "
            f"differently per camera: {seen}. Run `mask_store.unify_vocabulary` to union them."
        )
    return list(first)


def append_labels(dataset: LeRobotDataset, names: Iterable[str]) -> list[str]:
    """Declare new labels on EVERY mask column. Returns the new vocabulary.

    Pre: at least one camera has a mask column.
    Post: every mask column declares the same list, ending with the names that
    were not already present, in the order given and each at most once. No rows
    change, and no camera gains rows -- a label is declared here, detected
    elsewhere. Idempotent: appending what is already declared, or the same name
    twice in one call, is a no-op for that name.

    Appending to only the camera a detection came from is what lets the
    vocabularies drift apart, after which a rename or a treatment applied by
    name reaches some cameras and not others. Since positions are what rows
    reference, appending in the same order everywhere also keeps ids equal
    across columns, which is what makes a label id meaningful dataset-wide.
    """
    cols = mask_columns(dataset)
    if not cols:
        raise ValueError("no camera has a mask column")
    labels = vocabulary_of(dataset)
    # Deduped against the vocabulary AND against itself. Skipping names already
    # present makes this idempotent -- "ensure these exist" -- and a name
    # repeated inside one call means the same thing, so it is added once. The
    # filter used to compare only against the stored list, so a name new to the
    # dataset and given twice was appended twice, putting a duplicate straight
    # into a vocabulary whose ids rows already reference.
    added: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if name and name not in labels and name not in added:
            added.append(name)
    if not added:
        return labels
    labels = labels + added
    for key in cols.values():
        _update_info(dataset.root, {key: {"mask_labels": labels}})
        dataset.meta.features[key]["mask_labels"] = list(labels)
    return labels


def unify_vocabulary(dataset: LeRobotDataset) -> list[str]:
    """Union diverged vocabularies back into one. Returns it.

    For datasets written before appends became dataset-wide. Each column keeps
    its own labels at their existing POSITIONS -- moving one would re-point
    every row it already wrote -- so the union only ever appends, and columns
    that are already identical are left untouched.

    Post: every mask column declares the same list; no row changes.
    """
    cols = mask_columns(dataset)
    if not cols:
        return []
    merged: list[str] = []
    for key in cols.values():
        for name in dataset.meta.features[key].get("mask_labels", []):
            if name not in merged:
                merged.append(name)
    for key in cols.values():
        have = list(dataset.meta.features[key].get("mask_labels", []))
        if have == merged:
            continue
        if merged[: len(have)] != have:
            raise ValueError(
                f"{key} declares {have}, which is not a prefix of the union {merged}; "
                "unioning would move a label its rows already reference"
            )
        _update_info(dataset.root, {key: {"mask_labels": merged}})
        dataset.meta.features[key]["mask_labels"] = list(merged)
    return merged


def adopt(
    dataset: LeRobotDataset,
    camera_keys: list[str],
    labels: list[str],
    shape: tuple[int, int],
    *,
    treatments: dict | None = None,
    background: dict | None = None,
) -> dict[str, str]:
    """Add a mask column for each camera. Returns ``{camera: mask_key}``.

    Pre: no camera already has a mask column; ``labels`` is non-empty, holds
    each name at most once, and is ordered as it will be stored forever after.
    A repeat is refused by ``feature_spec`` -- rows reference labels by
    position, so a name may hold exactly one id.
    Post: every camera in ``camera_keys`` has an empty mask column declaring
    ``labels``.

    Adding a dataset-wide column is a schema change and cannot be undone by
    writing rows, so a caller with a user in front of it should confirm before
    calling. Refuses when two cameras would share a column.
    """
    from lerobot.datasets.dataset_tools import add_features_inplace

    if not labels:
        raise ValueError("a mask column needs at least one label; the vocabulary is positional")
    keys = mask_keys_for(camera_keys)
    already = [k for k in keys.values() if k in dataset.meta.features]
    if already:
        raise ValueError(f"already adopted: {already}; use write_episode to add rows")

    new = {}
    for key in keys.values():
        spec = feature_spec(labels, shape)
        spec["mask_treatments"] = dict(treatments or {n: {"key": "none"} for n in labels})
        spec["mask_background"] = dict(background or {"key": "none"})
        new[key] = (NEVER_WRITTEN, spec)
    add_features_inplace(dataset, new, recompute_stats=False)
    logger.info("adopted %s with labels %s", sorted(keys.values()), labels)
    return keys


def write_episode(
    dataset: LeRobotDataset,
    episode: int,
    camera_key: str,
    masks_per_frame: list[dict[str, np.ndarray]],
    disabled_per_frame: list[Iterable[str]] | None = None,
) -> int:
    """Replace one episode's rows for one camera. Returns frames written.

    Pre: the camera has a mask column; ``masks_per_frame`` has one entry per
    frame of the episode, each mapping a label already in the vocabulary to a
    boolean array of ``mask_size``. An empty dict means "segmented, found
    nothing", which is stored distinctly from "never written".
    Post: exactly this episode's rows are replaced; every other episode, and
    every other column, is untouched.

    Names outside the vocabulary are dropped rather than silently appended,
    because appending here would change what every other episode's rows mean.
    """
    from lerobot.datasets.feature_value_edits import set_feature_values

    spec = spec_of(dataset, camera_key)
    if spec is None:
        raise ValueError(f"{camera_key} has no mask column; call adopt first")
    labels = list(spec["mask_labels"])
    start, length = _episode_bounds(dataset, episode)
    if len(masks_per_frame) != length:
        raise ValueError(
            f"episode {episode} has {length} frames but {len(masks_per_frame)} were supplied; "
            "a partial episode would leave rows describing frames they do not belong to"
        )

    known = set(labels)
    key = mask_feature_of(camera_key)
    edits = []
    dropped: set[str] = set()
    for i, by_label in enumerate(masks_per_frame):
        dropped |= set(by_label or {}) - known
        usable = {n: _as_bool(m) for n, m in (by_label or {}).items() if n in known}
        asked_muted = list(disabled_per_frame[i]) if disabled_per_frame else []
        dropped |= set(asked_muted) - known
        muted = [n for n in asked_muted if n in known]
        edits.append(
            {
                "feature": key,
                "from_index": start + i,
                "to_index": start + i + 1,
                "value": encode_frame(usable, labels, disabled=muted),
            }
        )
    if dropped:
        # Not appended, because that re-points every other episode's rows -- but
        # not silent either: a typo'd label would otherwise write nothing for
        # that object and still report a full episode written.
        logger.warning(
            "%s: dropped %s -- not in the vocabulary %s; nothing was stored for them",
            key,
            sorted(dropped),
            labels,
        )
    set_feature_values(dataset, edits, in_place=True)
    _refresh(dataset)
    return length


def states(dataset: LeRobotDataset, episode: int, camera_key: str) -> list[dict[str, bool]]:
    """``{label: enabled}`` per frame of the episode, without decoding pixels.

    What the timeline draws: which labels a frame carries, and which of those
    are muted. Decoding the RLE to answer that would cost more than the track.
    """
    spec = spec_of(dataset, camera_key)
    if spec is None:
        return []
    labels = list(spec["mask_labels"])
    start, length = _episode_bounds(dataset, episode)
    col = dataset.hf_dataset[mask_feature_of(camera_key)][start : start + length]
    out = []
    for cell in col:
        row = cell[0] if isinstance(cell, (list, tuple)) and cell else cell
        out.append(frame_states(str(row) if row else "", labels))
    return out


def set_label_enabled(
    dataset: LeRobotDataset,
    episode: int,
    camera_key: str,
    label: str,
    enabled: bool,
    *,
    frames: range | None = None,
) -> int:
    """Mute or unmute one label over a frame range. Returns frames changed.

    Pre: the camera has a mask column and ``label`` is in its vocabulary;
    ``frames`` are episode-relative, defaulting to the whole episode.
    Post: every frame in range that CARRIES the label has its flag set. Frames
    where the label is absent are untouched -- there is nothing to mute.

    A muted mask keeps its pixels and stops reaching training, and no write
    will fill over it. That is what separates "this detection is wrong here"
    from "nothing was ever found here", which are otherwise identical in
    storage and so are refilled by the next gap-filling pass.
    """
    from lerobot.datasets.feature_value_edits import set_feature_values

    spec = spec_of(dataset, camera_key)
    if spec is None:
        raise ValueError(f"{camera_key} has no mask column")
    labels = list(spec["mask_labels"])
    if label not in labels:
        raise ValueError(f"{label!r} is not in the vocabulary {labels}")
    start, length = _episode_bounds(dataset, episode)
    span = frames if frames is not None else range(length)
    bad = [f for f in (span.start, span.stop - 1) if not 0 <= f < length]
    if bad:
        raise IndexError(f"frames {span} outside episode {episode}, which has {length}")

    key = mask_feature_of(camera_key)
    shape = tuple(spec["mask_size"])
    edits = []
    for f in span:
        cell = dataset.hf_dataset[start + f][key]
        row = cell[0] if isinstance(cell, (list, tuple)) and cell else cell
        if not row:
            continue
        st = frame_states(str(row), labels)
        if label not in st or st[label] == enabled:
            continue
        masks = decode_frame(str(row), labels, shape, include_disabled=True)
        muted = [n for n, on in st.items() if not on and n != label]
        if not enabled:
            muted.append(label)
        edits.append(
            {
                "feature": key,
                "from_index": start + f,
                "to_index": start + f + 1,
                "value": encode_frame(masks, labels, disabled=muted),
            }
        )
    if edits:
        set_feature_values(dataset, edits, in_place=True)
        _refresh(dataset)
    return len(edits)


def delete_label_range(
    dataset: LeRobotDataset,
    episode: int,
    camera_key: str,
    label: str,
    *,
    frames: range | None = None,
) -> int:
    """Remove one label's masks over a frame range. Returns frames changed.

    Pre: as :func:`set_label_enabled`.
    Post: the label is absent from those frames, so a later gap-filling write
    MAY fill it again -- which is the difference from muting, and the reason
    both exist. Other labels in those frames, and the same label outside the
    range, are untouched.
    """
    from lerobot.datasets.feature_value_edits import set_feature_values

    spec = spec_of(dataset, camera_key)
    if spec is None:
        raise ValueError(f"{camera_key} has no mask column")
    labels = list(spec["mask_labels"])
    if label not in labels:
        raise ValueError(f"{label!r} is not in the vocabulary {labels}")
    start, length = _episode_bounds(dataset, episode)
    span = frames if frames is not None else range(length)
    key = mask_feature_of(camera_key)
    shape = tuple(spec["mask_size"])
    edits = []
    for f in span:
        cell = dataset.hf_dataset[start + f][key]
        row = cell[0] if isinstance(cell, (list, tuple)) and cell else cell
        if not row:
            continue
        st = frame_states(str(row), labels)
        if label not in st:
            continue
        masks = decode_frame(str(row), labels, shape, include_disabled=True)
        masks.pop(label, None)
        muted = [n for n, on in st.items() if not on and n != label]
        edits.append(
            {
                "feature": key,
                "from_index": start + f,
                "to_index": start + f + 1,
                "value": encode_frame(masks, labels, disabled=muted),
            }
        )
    if edits:
        set_feature_values(dataset, edits, in_place=True)
        _refresh(dataset)
    return len(edits)


def read_frame(
    dataset: LeRobotDataset, episode: int, frame: int, camera_key: str
) -> dict[str, np.ndarray] | None:
    """Decode one frame's masks, as ``{label: bool array}``.

    Post: ``None`` when the camera has no mask column; ``{}`` when the frame
    was segmented and nothing was found. The two are different answers and are
    stored differently.
    """
    spec = spec_of(dataset, camera_key)
    if spec is None:
        return None
    start, _ = _episode_bounds(dataset, episode)
    cell = dataset.hf_dataset[start + frame][mask_feature_of(camera_key)]
    row = cell[0] if isinstance(cell, (list, tuple)) and cell else cell
    if not row:
        return {}
    return decode_frame(str(row), list(spec["mask_labels"]), tuple(spec["mask_size"]))


def read_episode(
    dataset: LeRobotDataset, episode: int, camera_key: str
) -> list[dict[str, np.ndarray]] | None:
    """Every frame's masks for one episode, in order."""
    spec = spec_of(dataset, camera_key)
    if spec is None:
        return None
    _, length = _episode_bounds(dataset, episode)
    return [read_frame(dataset, episode, i, camera_key) or {} for i in range(length)]


def coverage(dataset: LeRobotDataset, episode: int, camera_key: str) -> tuple[int, int]:
    """``(frames carrying at least one mask, frames in the episode)``."""
    spec = spec_of(dataset, camera_key)
    if spec is None:
        return (0, 0)
    start, length = _episode_bounds(dataset, episode)
    col = dataset.hf_dataset[mask_feature_of(camera_key)][start : start + length]
    n = 0
    for cell in col:
        row = cell[0] if isinstance(cell, (list, tuple)) and cell else cell
        if row and str(row) not in ("", "[]"):
            n += 1
    return (n, length)


def rename_label(dataset: LeRobotDataset, index: int, new_name: str) -> None:
    """Rename the label at ``index``, in place, on EVERY mask column.

    Pre: ``index`` is a position in the shared vocabulary; ``new_name`` is not
    already used.
    Post: every stored row, in every camera, now reads as the new name. No rows
    change, and the treatment follows the name in each column.

    Dataset-wide because a label is one physical object: renaming it in the one
    camera that happens to be selected leaves the same thing called ``apple``
    in two views and ``fruit`` in a third, with nothing recording that they are
    the same -- and a treatment keyed on the new name then reaches only some
    cameras.

    Safe precisely because rows reference positions: changing the string at a
    position re-points every row that used it, which is what a rename means.
    Stated as an index rather than inferred from a changed name list, because
    from a list alone a rename and a drop-plus-add are indistinguishable.
    """
    cols = mask_columns(dataset)
    if not cols:
        raise ValueError("no camera has a mask column")
    labels = vocabulary_of(dataset)
    if not 0 <= index < len(labels):
        raise IndexError(f"no label at position {index}; vocabulary is {labels}")
    new_name = str(new_name).strip()
    if not new_name:
        raise ValueError("a label needs a name")
    if new_name in labels and labels[index] != new_name:
        raise ValueError(f"{new_name!r} is already at position {labels.index(new_name)}")
    old, labels[index] = labels[index], new_name
    for key in cols.values():
        treatments = dict(dataset.meta.features[key].get("mask_treatments") or {})
        if old in treatments:
            treatments[new_name] = treatments.pop(old)
        _update_info(dataset.root, {key: {"mask_labels": labels, "mask_treatments": treatments}})
        dataset.meta.features[key]["mask_labels"] = list(labels)
        dataset.meta.features[key]["mask_treatments"] = treatments


def retire_label(dataset: LeRobotDataset, index: int) -> list[int]:
    """Mark the label at ``index`` obsolete everywhere. Returns the positions.

    Pre: ``index`` is a position in the shared vocabulary.
    Post: the position is listed in ``mask_labels_retired`` on every mask
    column; the entry itself stays, and no row changes.

    Dataset-wide for the same reason as a rename: retiring an object in one
    camera and not the others leaves it live where it was not selected.

    The vocabulary cannot be compacted: removing an entry shifts every later
    label down and re-points every stored row. So retirement is a tombstone,
    the way protobuf reserves field numbers and COCO keeps category ids for
    removed categories. Nothing is stored until the first retirement.
    """
    cols = mask_columns(dataset)
    if not cols:
        raise ValueError("no camera has a mask column")
    labels = vocabulary_of(dataset)
    if not 0 <= index < len(labels):
        raise IndexError(f"no label at position {index}; vocabulary is {labels}")
    retired: list[int] = []
    for key in cols.values():
        retired = sorted(set(dataset.meta.features[key].get(RETIRED_KEY, [])) | {int(index)})
        _update_info(dataset.root, {key: {RETIRED_KEY: retired}})
        dataset.meta.features[key][RETIRED_KEY] = retired
    return retired


def active_labels(dataset: LeRobotDataset, camera_key: str) -> list[str]:
    """The vocabulary minus retired positions, for callers offering a choice.

    Positions are unchanged -- this is a display filter, not a compaction.
    """
    spec = spec_of(dataset, camera_key)
    if spec is None:
        return []
    retired = set(spec.get(RETIRED_KEY, []))
    return [n for i, n in enumerate(spec["mask_labels"]) if i not in retired]


def remove(dataset: LeRobotDataset, camera_keys: list[str] | None = None) -> list[str]:
    """Drop the mask column(s) entirely. Returns the keys removed.

    Pre: nothing else is mid-write on the dataset.
    Post: the columns and their specs are gone; the frames are untouched.

    This is the destructive one, and the counterpart to :func:`adopt`: it takes
    away a dataset-wide column and the treatments that ride in its spec, and it
    cannot be undone by anything short of re-segmenting. A caller with a user in
    front of it should confirm, naming what it deletes.
    """
    from lerobot.datasets.dataset_tools import remove_features_inplace

    cams = list(camera_keys) if camera_keys is not None else list(dataset.meta.camera_keys)
    keys = [mask_feature_of(c) for c in cams]
    present = [k for k in keys if k in dataset.meta.features]
    if not present:
        return []
    remove_features_inplace(dataset, present)
    _refresh(dataset)
    logger.info("removed mask columns %s", present)
    return present


def describe(dataset: LeRobotDataset) -> dict[str, Any]:
    """Everything stored about masks, for a caller that wants to show it."""
    out: dict[str, Any] = {}
    for cam in dataset.meta.camera_keys:
        spec = spec_of(dataset, cam)
        if spec is None:
            continue
        out[mask_feature_of(cam)] = {
            "camera": cam,
            "labels": list(spec["mask_labels"]),
            "retired": list(spec.get(RETIRED_KEY, [])),
            "size": list(spec["mask_size"]),
            "treatments": dict(spec.get("mask_treatments") or {}),
            "background": dict(spec.get("mask_background") or {"key": "none"}),
        }
    return out
