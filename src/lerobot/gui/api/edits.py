# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""REST API endpoints for dataset editing operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState, PendingEdit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/edits", tags=["edits"])

# Will be set by server.py
_app_state: AppState = None  # type: ignore


def set_app_state(state: AppState) -> None:
    """Set the application state for edit handlers."""
    global _app_state
    _app_state = state


def _require_unlocked(dataset_id: str) -> None:
    """Raise HTTP 423 if the dataset is locked by an in-progress operation."""
    if _app_state.is_locked(dataset_id):
        raise HTTPException(status_code=423, detail="Dataset is busy (operation in progress)")


def _save_edits_for_dataset(dataset_id: str) -> None:
    """Persist pending edits for a dataset to disk."""
    from lerobot.gui.state import save_edits_to_file

    if dataset_id not in _app_state.datasets:
        return

    dataset = _app_state.datasets[dataset_id]
    edits = _app_state.get_edits_for_dataset(dataset_id)
    save_edits_to_file(dataset.root, edits)


class DeleteRequest(BaseModel):
    """Request to mark an episode for deletion."""

    dataset_id: str
    episode_index: int


class TrimRequest(BaseModel):
    """Request to trim an episode."""

    dataset_id: str
    episode_index: int
    start_frame: int
    end_frame: int


class EditInfo(BaseModel):
    """Information about a pending edit."""

    index: int
    edit_type: str
    dataset_id: str
    episode_index: int
    params: dict
    created_at: str


class PendingEditsResponse(BaseModel):
    """Response containing all pending edits."""

    edits: list[EditInfo]
    total: int


@router.get("", response_model=PendingEditsResponse)
async def list_pending_edits(dataset_id: str | None = None):
    """List all pending edits, optionally filtered by dataset."""
    edits = _app_state.get_edits_for_dataset(dataset_id) if dataset_id else _app_state.pending_edits

    return PendingEditsResponse(
        edits=[
            EditInfo(
                index=i,
                edit_type=e.edit_type,
                dataset_id=e.dataset_id,
                episode_index=e.episode_index,
                params=e.params,
                created_at=e.created_at.isoformat(),
            )
            for i, e in enumerate(_app_state.pending_edits)
            if dataset_id is None or e.dataset_id == dataset_id
        ],
        total=len(edits),
    )


@router.post("/delete")
async def mark_episode_deleted(request: DeleteRequest):
    """Mark an episode for deletion."""
    from lerobot.gui.state import PendingEdit

    _require_unlocked(request.dataset_id)

    if request.dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    dataset = _app_state.datasets[request.dataset_id]
    if request.episode_index < 0 or request.episode_index >= dataset.meta.total_episodes:
        raise HTTPException(status_code=400, detail=f"Invalid episode index: {request.episode_index}")

    # Check if already marked for deletion
    if _app_state.is_episode_deleted(request.dataset_id, request.episode_index):
        raise HTTPException(status_code=400, detail="Episode already marked for deletion")

    edit = PendingEdit(
        edit_type="delete",
        dataset_id=request.dataset_id,
        episode_index=request.episode_index,
    )
    _app_state.add_edit(edit)
    _save_edits_for_dataset(request.dataset_id)

    logger.info(f"Marked episode {request.episode_index} for deletion in {request.dataset_id}")
    return {"status": "ok", "message": f"Episode {request.episode_index} marked for deletion"}


@router.post("/trim")
async def set_episode_trim(request: TrimRequest):
    """Set trim range for an episode."""
    from lerobot.gui.state import PendingEdit

    _require_unlocked(request.dataset_id)

    if request.dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    dataset = _app_state.datasets[request.dataset_id]
    if request.episode_index < 0 or request.episode_index >= dataset.meta.total_episodes:
        raise HTTPException(status_code=400, detail=f"Invalid episode index: {request.episode_index}")

    # Get episode length
    episode = dataset.meta.episodes[request.episode_index]
    episode_length = episode["length"]

    if request.start_frame < 0 or request.end_frame > episode_length:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid trim range: {request.start_frame}-{request.end_frame} (length={episode_length})",
        )

    if request.start_frame >= request.end_frame:
        raise HTTPException(status_code=400, detail="Start frame must be less than end frame")

    # Remove existing trim for this episode
    _app_state.pending_edits = [
        e
        for e in _app_state.pending_edits
        if not (
            e.dataset_id == request.dataset_id
            and e.episode_index == request.episode_index
            and e.edit_type == "trim"
        )
    ]

    # Only add trim if it's not the full range
    if request.start_frame > 0 or request.end_frame < episode_length:
        edit = PendingEdit(
            edit_type="trim",
            dataset_id=request.dataset_id,
            episode_index=request.episode_index,
            params={"start_frame": request.start_frame, "end_frame": request.end_frame},
        )
        _app_state.add_edit(edit)
        logger.info(
            f"Set trim for episode {request.episode_index}: frames {request.start_frame}-{request.end_frame}"
        )

    _save_edits_for_dataset(request.dataset_id)

    return {
        "status": "ok",
        "message": f"Episode {request.episode_index} will be trimmed to frames {request.start_frame}-{request.end_frame}",
    }


# ───────────────────────────────────────────────────────────────────────────
# Feature-value editing (Phase B3 + B5)
# ───────────────────────────────────────────────────────────────────────────


# Read-only feature names / dtypes — editing is blocked.
_DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
_READONLY_DTYPES = {"image", "video"}
_LARGE_SAVE_FRAME_THRESHOLD = 10_000


class FeatureSetRequest(BaseModel):
    """Stage a per-frame feature value edit for a contiguous range.

    The frontend sends one of these per Inspector card change. The range
    uses the episode's local frame numbers; the backend translates to the
    global ``index`` column at apply time.
    """

    dataset_id: str
    episode_index: int
    feature: str
    frame_from: int  # inclusive
    frame_to: int  # exclusive
    value: Any  # JSON-serializable; coerced to feature dtype on apply
    confirm_large: bool = False  # set True to acknowledge a > 10k frame Save
    confirm_overlap: bool = False
    # set True to clip prior staged edits that overlap this one's range. Without
    # this flag, an overlap returns 409 so the frontend can prompt the user.


def _validate_feature_edit(dataset, request: FeatureSetRequest) -> tuple[int, int, int, int, dict[str, Any]]:
    """Validate the request against the dataset schema and trim envelope.

    Returns ``(frame_from, frame_to, global_from, global_to, feature_info)``.
    The returned frame_from/frame_to may be coerced to ``[0, episode_length)``
    if the feature is detected as per-episode-broadcast (e.g. ``success``) —
    sub-range edits would silently break the broadcast invariant otherwise.
    Raises HTTPException with appropriate 4xx codes on failure.
    """
    feature_dict = dataset.meta.features
    if request.feature not in feature_dict:
        raise HTTPException(status_code=400, detail=f"Unknown feature: {request.feature!r}")

    feature_info = feature_dict[request.feature]
    dtype = feature_info.get("dtype", "")

    if request.feature in _DEFAULT_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Feature {request.feature!r} is auto-managed and not editable",
        )
    if dtype in _READONLY_DTYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Feature {request.feature!r} has dtype {dtype!r} and is not editable in V1",
        )
    if request.feature == "action" or request.feature.startswith("observation."):
        raise HTTPException(
            status_code=400,
            detail=f"Feature {request.feature!r} is recorded sensor / control data and is read-only in V1",
        )

    if request.episode_index < 0 or request.episode_index >= dataset.meta.total_episodes:
        raise HTTPException(status_code=400, detail=f"Invalid episode index: {request.episode_index}")

    ep = dataset.meta.episodes[request.episode_index]
    ep_length = int(ep["length"])
    if request.frame_from < 0 or request.frame_to > ep_length or request.frame_from >= request.frame_to:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid range [{request.frame_from}, {request.frame_to}) for episode "
                f"{request.episode_index} (length={ep_length})"
            ),
        )

    # Per-episode broadcast features: silently coerce sub-range edits to the
    # full episode. The frontend visualizes this; the staging endpoint enforces
    # it as the source of truth so a stale frontend can't break the invariant.
    from lerobot.gui.api.datasets import _detect_per_episode_features, _get_episode_start_index

    per_episode = _detect_per_episode_features(request.dataset_id, dataset)
    if request.feature in per_episode:
        frame_from, frame_to = 0, ep_length
    else:
        frame_from, frame_to = request.frame_from, request.frame_to

    n_frames = frame_to - frame_from
    if n_frames > _LARGE_SAVE_FRAME_THRESHOLD and not request.confirm_large:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "large_edit_confirmation_required",
                "message": (
                    f"This edit touches {n_frames} frames (> {_LARGE_SAVE_FRAME_THRESHOLD}). "
                    "Re-send with confirm_large=true to proceed."
                ),
                "frames": n_frames,
            },
        )

    episode_start = _get_episode_start_index(request.dataset_id, request.episode_index)
    global_from = episode_start + frame_from
    global_to = episode_start + frame_to
    return frame_from, frame_to, global_from, global_to, feature_info


def _find_overlapping_feature_edits(
    dataset_id: str, episode_index: int, feature: str, frame_from: int, frame_to: int
) -> list[tuple[int, PendingEdit]]:
    """Return ``(index_in_pending_edits, edit)`` pairs that overlap the given range.

    Half-open `[a, b)` overlaps `[c, d)` iff `a < d and c < b`. Only considers
    feature_set edits on the same (dataset, episode, feature) tuple.
    """
    from lerobot.gui.state import PendingEdit  # noqa: F401  (referenced in type hint only)

    overlaps: list[tuple[int, Any]] = []
    for i, e in enumerate(_app_state.pending_edits):
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
    overlaps: list[tuple[int, Any]],
    new_from: int,
    new_to: int,
) -> int:
    """Last-write-wins resolution: clip prior edits' ranges to the non-overlap.

    Pre: ``overlaps`` is the result of :func:`_find_overlapping_feature_edits` for
    the same ``(dataset, episode, feature)`` tuple. Iterates in reverse index
    order so removals don't shift indices we still need.
    Post: returns the count of removed (fully-contained) edits.

    Cases per overlapping prior edit ``[a, b)``:

    * ``new`` fully contains prior (``new_from <= a`` and ``new_to >= b``) → remove it.
    * ``new`` cuts off the right tail (``a < new_from <= b <= new_to``) → clip to ``[a, new_from)``.
    * ``new`` cuts off the left tail (``new_from <= a < new_to < b``) → clip to ``[new_to, b)``.
    * ``new`` is strictly inside (``a < new_from`` and ``new_to < b``) → split into two pieces.
      We keep the left piece in place and append a new edit for the right piece.
    """
    removed = 0
    for i, e in sorted(overlaps, key=lambda x: x[0], reverse=True):
        a = int(e.params["frame_from"])
        b = int(e.params["frame_to"])
        if new_from <= a and new_to >= b:
            _app_state.pending_edits.pop(i)
            removed += 1
            continue
        # Compute remaining piece(s).
        left = (a, min(b, new_from))  # may be empty
        right = (max(a, new_to), b)  # may be empty
        left_keep = left[0] < left[1]
        right_keep = right[0] < right[1]
        if left_keep and right_keep:
            # Strictly inside — split. Keep left in place, append right.
            e.params["frame_from"] = left[0]
            e.params["frame_to"] = left[1]
            # Recompute global indices.
            episode_start = e.params["global_from_index"] - a
            e.params["global_from_index"] = episode_start + left[0]
            e.params["global_to_index"] = episode_start + left[1]
            from lerobot.gui.state import PendingEdit

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
            _app_state.pending_edits.append(split)
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
            # Both empty — should be caught by the "fully contains" branch above.
            _app_state.pending_edits.pop(i)
            removed += 1
    return removed


@router.post("/feature-set")
async def stage_feature_set(request: FeatureSetRequest):
    """Stage a feature-set edit for later apply on Save.

    On overlap with an existing staged edit on the same
    ``(dataset, episode, feature)``: returns 409 with structured detail
    unless ``request.confirm_overlap=True``, in which case prior edits are
    clipped (or removed if fully contained) and the new edit is staged.
    """
    from lerobot.gui.state import PendingEdit

    _require_unlocked(request.dataset_id)

    if request.dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_id}")

    dataset = _app_state.datasets[request.dataset_id]
    frame_from, frame_to, global_from, global_to, _ = _validate_feature_edit(dataset, request)

    overlaps = _find_overlapping_feature_edits(
        request.dataset_id, request.episode_index, request.feature, frame_from, frame_to
    )
    if overlaps and not request.confirm_overlap:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "overlapping_edit",
                "message": (
                    f"You already have {len(overlaps)} staged edit(s) on "
                    f"{request.feature!r} (episode {request.episode_index}) overlapping "
                    f"frames {frame_from}…{frame_to - 1}. "
                    "Re-send with confirm_overlap=true to clip the prior edit(s)."
                ),
                "feature": request.feature,
                "episode_index": request.episode_index,
                "new_range": [frame_from, frame_to],
                "overlapping": [
                    {
                        "edit_index": i,
                        "frame_from": int(e.params["frame_from"]),
                        "frame_to": int(e.params["frame_to"]),
                        "value": e.params.get("value"),
                    }
                    for i, e in overlaps
                ],
            },
        )
    if overlaps and request.confirm_overlap:
        removed = _clip_overlapping_edits(overlaps, frame_from, frame_to)
        logger.info(
            f"Resolved {len(overlaps)} overlapping edit(s) on "
            f"{request.feature} ep={request.episode_index}: {removed} removed, "
            f"{len(overlaps) - removed} clipped"
        )

    edit = PendingEdit(
        edit_type="feature_set",
        dataset_id=request.dataset_id,
        episode_index=request.episode_index,
        params={
            "feature": request.feature,
            "frame_from": frame_from,
            "frame_to": frame_to,
            "global_from_index": global_from,
            "global_to_index": global_to,
            "value": request.value,
        },
    )
    _app_state.add_edit(edit)
    _save_edits_for_dataset(request.dataset_id)

    logger.info(
        f"Staged feature-set edit: feature={request.feature} ep={request.episode_index} "
        f"frames=[{frame_from}, {frame_to}) global=[{global_from}, {global_to})"
    )
    response: dict[str, Any] = {"status": "ok", "message": "Feature-set edit staged"}
    if frame_from != request.frame_from or frame_to != request.frame_to:
        # Surface coercion (per-episode broadcast) so the GUI can update its
        # selection band rather than show stale ranges.
        response["coerced_range"] = [frame_from, frame_to]
        response["coerce_reason"] = "per_episode_broadcast"
    return response


@router.delete("/{edit_index}")
async def remove_edit(edit_index: int):
    """Remove a pending edit by index."""
    if edit_index < 0 or edit_index >= len(_app_state.pending_edits):
        raise HTTPException(status_code=404, detail=f"Edit not found: {edit_index}")

    edit = _app_state.pending_edits[edit_index]
    dataset_id = edit.dataset_id
    _require_unlocked(dataset_id)
    _app_state.remove_edit(edit_index)
    _save_edits_for_dataset(dataset_id)

    logger.info(f"Removed {edit.edit_type} edit for episode {edit.episode_index}")
    return {"status": "ok", "message": "Edit removed"}


@router.post("/discard")
async def discard_edits(dataset_id: str | None = None):
    """Discard all pending edits, optionally for a specific dataset."""
    from lerobot.gui.state import clear_edits_file

    if dataset_id:
        _require_unlocked(dataset_id)
    else:
        # Discarding all — check no dataset is locked
        for ds_id in _app_state.datasets:
            _require_unlocked(ds_id)

    count = len(
        _app_state.pending_edits if dataset_id is None else _app_state.get_edits_for_dataset(dataset_id)
    )

    # Clear the edits file(s)
    if dataset_id:
        if dataset_id in _app_state.datasets:
            clear_edits_file(_app_state.datasets[dataset_id].root)
    else:
        # Clear all datasets' edit files
        for dataset in _app_state.datasets.values():
            clear_edits_file(dataset.root)

    _app_state.clear_edits(dataset_id)

    logger.info(f"Discarded {count} pending edits")
    return {"status": "ok", "message": f"Discarded {count} pending edits"}


@router.post("/apply")
async def apply_edits(dataset_id: str):
    """Apply all pending edits for a dataset to disk."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    lock = _app_state.get_lock(dataset_id)
    if lock.locked():
        raise HTTPException(status_code=423, detail="Dataset is busy (operation in progress)")

    async with lock:
        return await _apply_edits_locked(dataset_id)


async def _apply_edits_locked(dataset_id: str):
    """Apply edits while holding the dataset lock."""
    from pathlib import Path

    from lerobot.datasets.dataset_tools import (
        delete_episodes_virtual,
        reaggregate_dataset_stats,
        set_feature_values,
        trim_episode_virtual,
    )
    from lerobot.datasets.io_utils import load_episodes

    edits = _app_state.get_edits_for_dataset(dataset_id)
    if not edits:
        return {"status": "ok", "message": "No edits to apply", "applied": 0}

    dataset = _app_state.datasets[dataset_id]
    original_root = Path(dataset.root)
    applied = 0
    errors = []

    # Sort edits: apply value-edits first (they modify cells in existing rows),
    # then trims (modify episode bounds), then deletes (remove rows). Doing
    # value-edits first means their global indices haven't been shifted by trims
    # / deletes yet — staged global_from / global_to remain valid.
    feature_set_edits = [e for e in edits if e.edit_type == "feature_set"]
    trim_edits = [e for e in edits if e.edit_type == "trim"]
    delete_edits = sorted(
        [e for e in edits if e.edit_type == "delete"], key=lambda e: e.episode_index, reverse=True
    )

    # Apply staged feature_set edits in one pass.
    if feature_set_edits:
        try:
            payload = [
                {
                    "feature": e.params["feature"],
                    "from_index": e.params["global_from_index"],
                    "to_index": e.params["global_to_index"],
                    "value": e.params["value"],
                }
                for e in feature_set_edits
            ]
            for e in feature_set_edits:
                logger.info(
                    f"FEATURE_SET dataset={original_root} feature={e.params['feature']} "
                    f"ep={e.episode_index} global=[{e.params['global_from_index']}, "
                    f"{e.params['global_to_index']})"
                )
            set_feature_values(dataset, payload, in_place=True)
            applied += len(feature_set_edits)
            logger.info(f"Applied {len(feature_set_edits)} feature-set edits")
        except Exception as e:
            errors.append(f"Feature-set edits: {e}")
            logger.exception("Failed to apply feature-set edits")

    # Apply trims (these modify in-place)
    for edit in trim_edits:
        try:
            start_frame = edit.params["start_frame"]
            end_frame = edit.params["end_frame"]

            # Log original state before trim
            original_length = dataset.meta.episodes[edit.episode_index]["length"]
            logger.info(
                f"TRIM dataset={original_root} episode={edit.episode_index} "
                f"original_length={original_length} keep_frames=[{start_frame}, {end_frame})"
            )

            trim_episode_virtual(
                dataset=dataset,
                episode_index=edit.episode_index,
                start_frame=start_frame,
                end_frame=end_frame,
                recompute_stats=False,
            )

            # Reload metadata to see changes
            dataset.meta.episodes = load_episodes(original_root)
            applied += 1
            logger.info(
                f"Applied trim to episode {edit.episode_index}: keeping frames {start_frame}-{end_frame - 1}"
            )
        except Exception as e:
            errors.append(f"Trim episode {edit.episode_index}: {e}")
            logger.exception(f"Failed to trim episode {edit.episode_index}")

    # Apply deletes - virtual delete works in-place (no video re-encoding)
    if delete_edits:
        try:
            episode_indices = [e.episode_index for e in delete_edits]

            # Log episode info before deletion
            for ep_idx in episode_indices:
                ep_length = dataset.meta.episodes[ep_idx]["length"]
                logger.info(f"DELETE dataset={original_root} episode={ep_idx} length={ep_length}")

            # Reload metadata to pick up any trim changes
            dataset.meta.episodes = load_episodes(original_root)

            # Virtual delete modifies metadata and parquet in-place (no video re-encoding)
            delete_episodes_virtual(dataset, episode_indices=episode_indices, recompute_stats=False)

            logger.info(f"Deleted episodes {episode_indices} (virtual - no video re-encoding)")

            applied += len(delete_edits)
        except Exception as e:
            errors.append(f"Delete episodes: {e}")
            logger.exception("Failed to delete episodes")

    # Re-aggregate stats once after all edits (O(E) instead of O(N*E))
    if applied > 0:
        try:
            reaggregate_dataset_stats(dataset)
            logger.info("Re-aggregated dataset stats after all edits")
        except Exception as e:
            logger.exception(f"Failed to re-aggregate stats: {e}")
            errors.append(f"Stats re-aggregation failed: {e}")

    # Clear applied edits (both in memory and on disk)
    from lerobot.gui.state import clear_edits_file

    _app_state.clear_edits(dataset_id)
    clear_edits_file(original_root)

    # Invalidate all dataset-scoped caches. Edits change episode lengths,
    # so the cumulative-sum cache (_episode_start_indices) must also be
    # dropped — otherwise subsequent frame lookups use stale offsets.
    from lerobot.gui.api.datasets import _invalidate_episode_start_indices
    from lerobot.gui.cache_invalidation import invalidate_caches

    invalidate_caches(_app_state, dataset_id, invalidate_episode_indices=_invalidate_episode_start_indices)

    # Reload dataset from disk to get updated metadata
    # We reload in-place to avoid LeRobotDataset constructor trying to access HuggingFace Hub
    try:
        from lerobot.gui.dataset_reload import reload_dataset_from_disk

        old_dataset = _app_state.datasets[dataset_id]
        logger.info(f"Reloading dataset from {old_dataset.root}")
        reload_dataset_from_disk(old_dataset)
        logger.info(
            f"Reloaded dataset {dataset_id}: {old_dataset.meta.total_episodes} episodes, {old_dataset.meta.total_frames} frames"
        )

    except Exception as e:
        logger.exception(f"Failed to reload dataset after edits: {e}")
        errors.append(f"Dataset reload failed: {e}")

    # Verify dataset integrity after edits
    warnings = []
    try:
        from lerobot.datasets.dataset_tools import verify_dataset

        verification = verify_dataset(old_dataset.root, check_videos=False, verbose=False)
        if not verification.is_valid:
            for err in verification.errors:
                logger.warning(f"Post-edit verification: {err.message}")
                warnings.append(err.message)
        for warn in verification.warnings:
            logger.warning(f"Post-edit verification warning: {warn.message}")
            warnings.append(warn.message)
    except Exception as e:
        logger.warning(f"Post-edit verification failed: {e}")

    if errors:
        return {
            "status": "partial",
            "message": f"Applied {applied} edits with {len(errors)} errors",
            "applied": applied,
            "errors": errors,
            "warnings": warnings,
        }

    return {"status": "ok", "message": f"Applied {applied} edits", "applied": applied, "warnings": warnings}


class MergeIntoRequest(BaseModel):
    source_dataset_id: str
    target_dataset_id: str
    force: bool = False


def _validate_merge_compat(source_meta, target_meta) -> list[dict]:
    """Compare two dataset metas and return a list of mismatches (may be empty)."""
    mismatches = []
    if target_meta.fps != source_meta.fps:
        mismatches.append({"field": "fps", "target": target_meta.fps, "source": source_meta.fps})
    if target_meta.robot_type != source_meta.robot_type:
        mismatches.append(
            {
                "field": "robot_type",
                "target": target_meta.robot_type,
                "source": source_meta.robot_type,
            }
        )
    tf = target_meta.features
    sf = source_meta.features
    if tf != sf:
        target_only = sorted(set(tf.keys()) - set(sf.keys()))
        source_only = sorted(set(sf.keys()) - set(tf.keys()))
        shared_diff = {}
        for k in sorted(set(tf.keys()) & set(sf.keys())):
            if tf[k] != sf[k]:
                shared_diff[k] = {"target": tf[k], "source": sf[k]}
        mismatches.append(
            {
                "field": "features",
                "target_only": target_only,
                "source_only": source_only,
                "shared_diff": shared_diff,
            }
        )
    return mismatches


@router.post("/merge-into/validate")
async def validate_merge(request: MergeIntoRequest):
    """Check compatibility between source and target datasets for merge."""
    source_id = request.source_dataset_id
    target_id = request.target_dataset_id

    if source_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Source dataset not found: {source_id}")
    if target_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Target dataset not found: {target_id}")

    mismatches = _validate_merge_compat(
        _app_state.datasets[source_id].meta,
        _app_state.datasets[target_id].meta,
    )
    return {"compatible": len(mismatches) == 0, "mismatches": mismatches}


@router.post("/merge-into")
async def merge_into_dataset(request: MergeIntoRequest):
    """Merge all episodes from source dataset into target dataset in-place."""

    source_id = request.source_dataset_id
    target_id = request.target_dataset_id

    if source_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Source dataset not found: {source_id}")
    if target_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Target dataset not found: {target_id}")
    if source_id == target_id:
        raise HTTPException(status_code=400, detail="Cannot merge a dataset into itself")

    _require_unlocked(source_id)
    _require_unlocked(target_id)

    target_lock = _app_state.get_lock(target_id)
    source_lock = _app_state.get_lock(source_id)

    if target_lock.locked() or source_lock.locked():
        raise HTTPException(status_code=423, detail="One or both datasets are busy")

    async with target_lock, source_lock:
        return await _merge_into_locked(source_id, target_id, force=request.force)


async def _merge_into_locked(source_id: str, target_id: str, *, force: bool = False):
    """Execute merge while holding both dataset locks."""
    source_ds = _app_state.datasets[source_id]
    target_ds = _app_state.datasets[target_id]

    source_eps = source_ds.num_episodes
    source_frames = source_ds.num_frames
    target_eps_before = target_ds.num_episodes

    logger.info(
        f"Merging {source_eps} episodes from {source_id} into {target_id} "
        f"({target_eps_before} episodes) force={force}"
    )

    try:
        from lerobot.datasets.dataset_tools import merge_into

        merge_into(target_ds, source_ds, skip_validation=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Merge failed: {e}")
        raise HTTPException(status_code=500, detail=f"Merge failed: {e}") from e

    # Invalidate caches for the merge target (new episodes added — the
    # cumulative-sum cache must be dropped so subsequent frame lookups
    # pick up the grown dataset).
    from lerobot.gui.api.datasets import _invalidate_episode_start_indices
    from lerobot.gui.cache_invalidation import invalidate_caches

    invalidate_caches(_app_state, target_id, invalidate_episode_indices=_invalidate_episode_start_indices)

    logger.info(f"Merge complete: {target_ds.num_episodes} episodes, {target_ds.num_frames} frames in target")

    return {
        "status": "ok",
        "message": f"Merged {source_eps} episodes ({source_frames} frames) into {target_id}",
        "target_episodes": target_ds.num_episodes,
        "target_frames": target_ds.num_frames,
    }
