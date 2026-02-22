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
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/edits", tags=["edits"])

# Will be set by server.py
_app_state: "AppState" = None  # type: ignore


def set_app_state(state: "AppState") -> None:
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
    if dataset_id:
        edits = _app_state.get_edits_for_dataset(dataset_id)
    else:
        edits = _app_state.pending_edits

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
            status_code=400, detail=f"Invalid trim range: {request.start_frame}-{request.end_frame} (length={episode_length})"
        )

    if request.start_frame >= request.end_frame:
        raise HTTPException(status_code=400, detail="Start frame must be less than end frame")

    # Remove existing trim for this episode
    _app_state.pending_edits = [
        e
        for e in _app_state.pending_edits
        if not (e.dataset_id == request.dataset_id and e.episode_index == request.episode_index and e.edit_type == "trim")
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
        logger.info(f"Set trim for episode {request.episode_index}: frames {request.start_frame}-{request.end_frame}")

    _save_edits_for_dataset(request.dataset_id)

    return {
        "status": "ok",
        "message": f"Episode {request.episode_index} will be trimmed to frames {request.start_frame}-{request.end_frame}",
    }


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

    count = len(_app_state.pending_edits if dataset_id is None else _app_state.get_edits_for_dataset(dataset_id))

    # Clear the edits file(s)
    if dataset_id:
        if dataset_id in _app_state.datasets:
            clear_edits_file(_app_state.datasets[dataset_id].root)
    else:
        # Clear all datasets' edit files
        for ds_id, dataset in _app_state.datasets.items():
            clear_edits_file(dataset.root)

    _app_state.clear_edits(dataset_id)

    logger.info(f"Discarded {count} pending edits")
    return {"status": "ok", "message": f"Discarded {count} pending edits"}


@router.post("/apply")
async def apply_edits(dataset_id: str):
    """Apply all pending edits for a dataset to disk."""
    from pathlib import Path

    from lerobot.datasets.dataset_tools import (
        delete_episodes_virtual,
        reaggregate_dataset_stats,
        trim_episode_virtual,
    )
    from lerobot.datasets.utils import load_episodes

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
        trim_episode_virtual,
    )
    from lerobot.datasets.utils import load_episodes

    edits = _app_state.get_edits_for_dataset(dataset_id)
    if not edits:
        return {"status": "ok", "message": "No edits to apply", "applied": 0}

    dataset = _app_state.datasets[dataset_id]
    original_root = Path(dataset.root)
    applied = 0
    errors = []

    # Sort edits: apply trims first (they modify in place), then deletes
    trim_edits = [e for e in edits if e.edit_type == "trim"]
    delete_edits = sorted([e for e in edits if e.edit_type == "delete"], key=lambda e: e.episode_index, reverse=True)

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
            logger.info(f"Applied trim to episode {edit.episode_index}: keeping frames {start_frame}-{end_frame - 1}")
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

    # Invalidate frame cache for this dataset
    logger.info(f"Invalidating frame cache for {dataset_id}...")
    num_invalidated = _app_state.frame_cache.invalidate_dataset(dataset_id)
    logger.info(f"Invalidated {num_invalidated} cached frames")

    # Clear video decoder cache to ensure fresh video data is loaded
    # The video decoder caches file handles which may become stale after edits
    try:
        from lerobot.datasets.video_utils import _default_decoder_cache

        cache_size = _default_decoder_cache.size()
        if cache_size > 0:
            _default_decoder_cache.clear()
            logger.info(f"Cleared video decoder cache ({cache_size} entries)")
    except Exception as e:
        logger.warning(f"Could not clear video decoder cache: {e}")

    # Reload dataset from disk to get updated metadata
    # We reload in-place to avoid LeRobotDataset constructor trying to access HuggingFace Hub
    try:
        from lerobot.datasets.utils import (
            load_info,
            load_stats,
            load_tasks,
            load_nested_dataset,
            get_hf_features_from_features,
            hf_transform_to_torch,
        )
        import datasets

        old_dataset = _app_state.datasets[dataset_id]
        root = old_dataset.root

        logger.info(f"Reloading dataset from {root}")

        # Reload all metadata from disk
        old_dataset.meta.info = load_info(root)
        old_dataset.meta.episodes = load_episodes(root)
        old_dataset.meta.stats = load_stats(root)
        old_dataset.meta.tasks = load_tasks(root)

        # Clean up any existing HuggingFace cache files for this dataset
        # This ensures we load fresh data, not cached Arrow files
        if old_dataset.hf_dataset is not None:
            try:
                num_cleaned = old_dataset.hf_dataset.cleanup_cache_files()
                if num_cleaned > 0:
                    logger.info(f"Cleaned up {num_cleaned} cache files")
            except Exception as e:
                logger.warning(f"Could not cleanup cache files: {e}")

        # Reload the HuggingFace dataset from parquet files
        # Use disable_caching to ensure fresh data is loaded
        datasets.disable_caching()
        try:
            features = get_hf_features_from_features(old_dataset.meta.features)
            old_dataset.hf_dataset = load_nested_dataset(root / "data", features=features)
            old_dataset.hf_dataset.set_transform(hf_transform_to_torch)
            # CRITICAL: Set _lazy_loading to False to prevent _ensure_hf_dataset_loaded
            # from reloading stale cached data and overwriting our fresh hf_dataset
            old_dataset._lazy_loading = False
        finally:
            datasets.enable_caching()

        logger.info(f"Reloaded dataset {dataset_id}: {old_dataset.meta.total_episodes} episodes, {old_dataset.meta.total_frames} frames")

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
