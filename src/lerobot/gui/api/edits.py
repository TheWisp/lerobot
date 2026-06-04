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
    from lerobot.gui.state import AppState

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


def _map_core_exception(exc: Exception) -> HTTPException:
    """Translate a typed exception from ``_edits_core`` into HTTPException.

    Keeps the FastAPI layer free of business logic while preserving the
    existing status-code contract.
    """
    from lerobot.gui.api._edits_core import (
        DatasetBusyError,
        DatasetNotFoundError,
        EditConflictError,
        EditValidationError,
    )

    if isinstance(exc, DatasetNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, EditConflictError):
        return HTTPException(status_code=409, detail=exc.detail)
    if isinstance(exc, EditValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, DatasetBusyError):
        return HTTPException(status_code=423, detail=str(exc))
    raise exc  # not ours


@router.get("", response_model=PendingEditsResponse)
async def list_pending_edits(dataset_id: str | None = None):
    """List all pending edits, optionally filtered by dataset."""
    from lerobot.gui.api._edits_core import list_pending

    result = list_pending(_app_state, dataset_id)
    return PendingEditsResponse(
        edits=[EditInfo(**e) for e in result["edits"]],
        total=result["total"],
    )


@router.post("/delete")
async def mark_episode_deleted(request: DeleteRequest):
    """Mark an episode for deletion."""
    from lerobot.gui.api._edits_core import propose_delete

    try:
        return propose_delete(_app_state, request.dataset_id, request.episode_index)
    except Exception as e:
        raise _map_core_exception(e) from e


@router.post("/trim")
async def set_episode_trim(request: TrimRequest):
    """Set trim range for an episode."""
    from lerobot.gui.api._edits_core import propose_trim

    try:
        return propose_trim(
            _app_state,
            request.dataset_id,
            request.episode_index,
            request.start_frame,
            request.end_frame,
        )
    except Exception as e:
        raise _map_core_exception(e) from e


# ───────────────────────────────────────────────────────────────────────────
# Feature-value editing (Phase B3 + B5)
# ───────────────────────────────────────────────────────────────────────────


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


@router.post("/feature-set")
async def stage_feature_set(request: FeatureSetRequest):
    """Stage a feature-set edit for later apply on Save.

    On overlap with an existing staged edit on the same
    ``(dataset, episode, feature)``: returns 409 with structured detail
    unless ``request.confirm_overlap=True``, in which case prior edits are
    clipped (or removed if fully contained) and the new edit is staged.
    """
    from lerobot.gui.api._edits_core import propose_feature_set

    try:
        result = propose_feature_set(
            _app_state,
            request.dataset_id,
            request.episode_index,
            request.feature,
            request.frame_from,
            request.frame_to,
            request.value,
            confirm_large=request.confirm_large,
            confirm_overlap=request.confirm_overlap,
        )
    except Exception as e:
        raise _map_core_exception(e) from e

    # FastAPI clients expect the slim response (no dataset_id / feature
    # echoes) — preserved from the original handler for compat with
    # frontend code that pattern-matches on key presence.
    out: dict[str, Any] = {"status": "ok", "message": result["message"]}
    if "coerced_range" in result:
        out["coerced_range"] = result["coerced_range"]
        out["coerce_reason"] = result["coerce_reason"]
    return out


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
    from lerobot.gui.api._edits_core import discard_pending

    try:
        result = discard_pending(_app_state, dataset_id)
    except Exception as e:
        raise _map_core_exception(e) from e
    return {"status": "ok", "message": result["message"]}


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
    from lerobot.datasets.feature_value_edits import StatsRecomputationError
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
        except StatsRecomputationError as e:
            # Data writes succeeded, only stats recompute failed — increment
            # applied (the user's edits ARE on disk) but surface the staleness
            # so the frontend shows partial. The user can re-run stats from
            # the dataset reload path or via a manual aggregation pass.
            applied += len(feature_set_edits)
            errors.append(f"Feature-set edits applied, but stats recompute failed: {e}")
            logger.warning(f"Feature-set edits applied with stale stats: {e}")
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
        import asyncio

        from lerobot.datasets.dataset_tools import merge_into

        # `merge_into` copies parquet data + videos — gigabytes for a
        # full dataset, easily minutes of synchronous I/O. Push it to
        # the default executor so SSE keepalives + other tabs'
        # WebSockets / HTTP requests stay responsive during the merge.
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: merge_into(target_ds, source_ds, skip_validation=force)
        )
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
