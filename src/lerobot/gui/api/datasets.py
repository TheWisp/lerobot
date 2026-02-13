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
"""Dataset API endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Will be set by server.py
_app_state: "AppState" = None  # type: ignore

# Track metadata file modification times for auto-reload
_dataset_info_mtime: dict[str, float] = {}


def _check_and_reload_metadata(dataset_id: str) -> bool:
    """Check if dataset metadata changed on disk and reload if needed.

    Returns True if metadata was reloaded.
    """
    if dataset_id not in _app_state.datasets:
        return False

    dataset = _app_state.datasets[dataset_id]
    info_file = Path(dataset.root) / "meta" / "info.json"

    if not info_file.exists():
        return False

    current_mtime = info_file.stat().st_mtime
    cached_mtime = _dataset_info_mtime.get(dataset_id)

    if cached_mtime is not None and current_mtime == cached_mtime:
        return False

    # Metadata changed - reload metadata AND HuggingFace dataset
    import datasets as hf_datasets

    from lerobot.datasets.utils import (
        get_hf_features_from_features,
        hf_transform_to_torch,
        load_episodes,
        load_info,
        load_nested_dataset,
        load_stats,
        load_tasks,
    )

    logger.info(f"Detected metadata change for {dataset_id}, reloading...")

    try:
        root = dataset.root

        # Reload all metadata
        dataset.meta.info = load_info(root)
        dataset.meta.episodes = load_episodes(root)
        dataset.meta.stats = load_stats(root)
        dataset.meta.tasks = load_tasks(root)
        _dataset_info_mtime[dataset_id] = current_mtime

        # CRITICAL: Also reload the HuggingFace dataset
        # Otherwise frame indices will be misaligned with the new metadata
        if dataset.hf_dataset is not None:
            try:
                num_cleaned = dataset.hf_dataset.cleanup_cache_files()
                if num_cleaned > 0:
                    logger.info(f"Cleaned up {num_cleaned} HF cache files")
            except Exception as e:
                logger.warning(f"Could not cleanup cache files: {e}")

        # Clear video decoder cache
        try:
            from lerobot.datasets.video_utils import _default_decoder_cache

            cache_size = _default_decoder_cache.size()
            if cache_size > 0:
                _default_decoder_cache.clear()
                logger.info(f"Cleared video decoder cache ({cache_size} entries)")
        except Exception as e:
            logger.warning(f"Could not clear video decoder cache: {e}")

        # Reload HF dataset with caching disabled
        hf_datasets.disable_caching()
        try:
            features = get_hf_features_from_features(dataset.meta.features)
            dataset.hf_dataset = load_nested_dataset(root / "data", features=features)
            dataset.hf_dataset.set_transform(hf_transform_to_torch)
            dataset._lazy_loading = False
        finally:
            hf_datasets.enable_caching()

        # Invalidate frame cache for this dataset
        num_invalidated = _app_state.frame_cache.invalidate_dataset(dataset_id)
        if num_invalidated > 0:
            logger.info(f"Invalidated {num_invalidated} cached frames")

        logger.info(
            f"Reloaded dataset: {dataset.meta.total_episodes} episodes, "
            f"{dataset.meta.total_frames} frames"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to reload dataset for {dataset_id}: {e}")
        return False


def set_app_state(state: "AppState") -> None:
    """Set the application state for API handlers."""
    global _app_state
    _app_state = state


class OpenDatasetRequest(BaseModel):
    """Request to open a dataset."""

    repo_id: str | None = None
    local_path: str | None = None


class DatasetInfo(BaseModel):
    """Summary info about a dataset."""

    id: str
    repo_id: str
    root: str
    total_episodes: int
    total_frames: int
    fps: int
    camera_keys: list[str]
    features: list[str]


class EpisodeInfo(BaseModel):
    """Summary info about an episode."""

    episode_index: int
    length: int
    duration_s: float
    task: str | None = None


@router.get("")
async def list_datasets() -> list[DatasetInfo]:
    """List all currently opened datasets."""
    result = []
    for dataset_id, ds in _app_state.datasets.items():
        result.append(
            DatasetInfo(
                id=dataset_id,
                repo_id=ds.repo_id,
                root=str(ds.root),
                total_episodes=ds.meta.total_episodes,
                total_frames=ds.meta.total_frames,
                fps=ds.fps,
                camera_keys=list(ds.meta.camera_keys),
                features=list(ds.meta.features.keys()),
            )
        )
    return result


@router.post("")
async def open_dataset(request: OpenDatasetRequest) -> DatasetInfo:
    """Open a dataset by repo_id or local path."""
    import datasets as hf_datasets

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import load_episodes

    try:
        if request.local_path:
            # Open local dataset
            local_path = Path(request.local_path)
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"Path not found: {request.local_path}")

            dataset_id = str(local_path)

            # Check if dataset is already open - return existing instance
            # This is important after edits: the existing instance has fresh data,
            # while creating a new instance might load stale cached Arrow files
            if dataset_id in _app_state.datasets:
                dataset = _app_state.datasets[dataset_id]
                logger.info(f"Returning existing dataset: {dataset_id} ({dataset.meta.total_episodes} episodes)")
                return DatasetInfo(
                    id=dataset_id,
                    repo_id=dataset.repo_id,
                    root=str(dataset.root),
                    total_episodes=dataset.meta.total_episodes,
                    total_frames=dataset.meta.total_frames,
                    fps=dataset.fps,
                    camera_keys=list(dataset.meta.camera_keys),
                    features=list(dataset.meta.features.keys()),
                )

            # Extract repo_id from path or use path name
            repo_id = request.repo_id or local_path.name

            # Disable HuggingFace caching to ensure fresh data is loaded
            # This is important for datasets that may have been edited
            hf_datasets.disable_caching()
            try:
                dataset = LeRobotDataset(repo_id, root=local_path)
            finally:
                hf_datasets.enable_caching()

        elif request.repo_id:
            dataset_id = request.repo_id

            # Check if dataset is already open
            if dataset_id in _app_state.datasets:
                dataset = _app_state.datasets[dataset_id]
                logger.info(f"Returning existing dataset: {dataset_id} ({dataset.meta.total_episodes} episodes)")
                return DatasetInfo(
                    id=dataset_id,
                    repo_id=dataset.repo_id,
                    root=str(dataset.root),
                    total_episodes=dataset.meta.total_episodes,
                    total_frames=dataset.meta.total_frames,
                    fps=dataset.fps,
                    camera_keys=list(dataset.meta.camera_keys),
                    features=list(dataset.meta.features.keys()),
                )

            # Open from HuggingFace Hub
            dataset = LeRobotDataset(request.repo_id)
        else:
            raise HTTPException(status_code=400, detail="Must provide either repo_id or local_path")

        # Ensure episodes are loaded
        if dataset.meta.episodes is None:
            dataset.meta.episodes = load_episodes(dataset.root)

        # Store in app state
        _app_state.datasets[dataset_id] = dataset

        # Track metadata mtime for auto-reload detection
        info_file = Path(dataset.root) / "meta" / "info.json"
        if info_file.exists():
            _dataset_info_mtime[dataset_id] = info_file.stat().st_mtime

        # Load any persisted pending edits from disk
        from lerobot.gui.state import load_edits_from_file

        persisted_edits = load_edits_from_file(dataset.root, dataset_id)
        for edit in persisted_edits:
            _app_state.add_edit(edit)
        if persisted_edits:
            logger.info(f"Restored {len(persisted_edits)} pending edits from disk")

        logger.info(f"Opened dataset: {dataset_id} ({dataset.meta.total_episodes} episodes)")

        return DatasetInfo(
            id=dataset_id,
            repo_id=dataset.repo_id,
            root=str(dataset.root),
            total_episodes=dataset.meta.total_episodes,
            total_frames=dataset.meta.total_frames,
            fps=dataset.fps,
            camera_keys=list(dataset.meta.camera_keys),
            features=list(dataset.meta.features.keys()),
        )

    except Exception as e:
        logger.exception(f"Failed to open dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{dataset_id:path}")
async def close_dataset(dataset_id: str) -> dict[str, str]:
    """Close a dataset."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    del _app_state.datasets[dataset_id]
    _dataset_info_mtime.pop(dataset_id, None)
    logger.info(f"Closed dataset: {dataset_id}")

    return {"status": "ok", "message": f"Closed dataset: {dataset_id}"}


@router.get("/{dataset_id:path}/episodes")
async def list_episodes(dataset_id: str) -> list[EpisodeInfo]:
    """List all episodes in a dataset."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # Check if metadata changed on disk (e.g., new episodes recorded)
    _check_and_reload_metadata(dataset_id)

    dataset = _app_state.datasets[dataset_id]
    episodes = dataset.meta.episodes

    if episodes is None:
        from lerobot.datasets.utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    result = []
    for i in range(dataset.meta.total_episodes):
        ep = episodes[i]
        length = ep["length"]
        duration_s = length / dataset.fps

        # Get task if available (ep["tasks"] contains task name strings directly)
        task = None
        if "tasks" in ep and ep["tasks"]:
            task = ep["tasks"][0] if len(ep["tasks"]) > 0 else None

        result.append(
            EpisodeInfo(
                episode_index=i,
                length=length,
                duration_s=duration_s,
                task=task,
            )
        )

    return result


@router.get("/{dataset_id:path}/episodes/{episode_idx}/frame/{frame_idx}")
async def get_frame(dataset_id: str, episode_idx: int, frame_idx: int, camera: str | None = None) -> Response:
    """Get a single frame as JPEG.

    Args:
        dataset_id: Dataset identifier
        episode_idx: Episode index
        frame_idx: Frame index within the episode
        camera: Camera key (optional, returns first camera if not specified)
    """
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]

    # Validate episode index
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    # Get episode metadata
    episodes = dataset.meta.episodes
    if episodes is None:
        from lerobot.datasets.utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    ep = episodes[episode_idx]
    ep_length = ep["length"]

    # Validate frame index
    if frame_idx < 0 or frame_idx >= ep_length:
        raise HTTPException(status_code=404, detail=f"Frame not found: {frame_idx} (episode has {ep_length} frames)")

    # Determine camera key
    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        raise HTTPException(status_code=400, detail="Dataset has no camera/image keys")

    if camera:
        if camera not in camera_keys:
            raise HTTPException(status_code=400, detail=f"Camera not found: {camera}. Available: {camera_keys}")
        camera_key = camera
    else:
        camera_key = camera_keys[0]

    # Calculate global index
    global_idx = ep["dataset_from_index"] + frame_idx

    # Check if this camera is already cached
    jpeg_bytes = _app_state.frame_cache.get(dataset_id, episode_idx, frame_idx, camera_key)

    if jpeg_bytes is None:
        # Not cached - decode ALL cameras at once (dataset[idx] decodes all anyway)
        # This avoids redundant decoding when multiple cameras are requested
        from lerobot.gui.frame_cache import encode_frame_to_jpeg

        item = dataset[global_idx]

        # Cache all cameras from this single decode
        for cam in camera_keys:
            if cam in item:
                cam_jpeg = encode_frame_to_jpeg(item[cam])
                _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, cam, cam_jpeg)
                if cam == camera_key:
                    jpeg_bytes = cam_jpeg

        # Fallback if camera wasn't in camera_keys list
        if jpeg_bytes is None:
            jpeg_bytes = encode_frame_to_jpeg(item[camera_key])
            _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, camera_key, jpeg_bytes)

    # Prevent browser caching - frames may change after edits
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/{dataset_id:path}/episodes/{episode_idx}/frames")
async def get_frames_batch(
    dataset_id: str,
    episode_idx: int,
    start: int = 0,
    count: int = 10,
    camera: str | None = None,
) -> dict[str, Any]:
    """Get multiple frames as base64-encoded JPEGs.

    Args:
        dataset_id: Dataset identifier
        episode_idx: Episode index
        start: Starting frame index
        count: Number of frames to return (max 100)
        camera: Camera key (optional)

    Returns:
        Dict with frame data and metadata
    """
    import base64

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    count = min(count, 100)  # Limit batch size

    # Get episode metadata
    episodes = dataset.meta.episodes
    if episodes is None:
        from lerobot.datasets.utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    ep = episodes[episode_idx]
    ep_length = ep["length"]

    # Determine camera key
    camera_keys = list(dataset.meta.camera_keys)
    if camera:
        if camera not in camera_keys:
            raise HTTPException(status_code=400, detail=f"Camera not found: {camera}")
        camera_key = camera
    else:
        camera_key = camera_keys[0] if camera_keys else None

    if not camera_key:
        raise HTTPException(status_code=400, detail="No camera available")

    # Collect frames
    from lerobot.gui.frame_cache import encode_frame_to_jpeg

    frames = []
    for i in range(start, min(start + count, ep_length)):
        global_idx = ep["dataset_from_index"] + i

        # Check cache first
        jpeg_bytes = _app_state.frame_cache.get(dataset_id, episode_idx, i, camera_key)

        if jpeg_bytes is None:
            # Decode all cameras at once and cache them
            item = dataset[global_idx]
            for cam in camera_keys:
                if cam in item:
                    cam_jpeg = encode_frame_to_jpeg(item[cam])
                    _app_state.frame_cache.put(dataset_id, episode_idx, i, cam, cam_jpeg)
                    if cam == camera_key:
                        jpeg_bytes = cam_jpeg

            # Fallback
            if jpeg_bytes is None:
                jpeg_bytes = encode_frame_to_jpeg(item[camera_key])
                _app_state.frame_cache.put(dataset_id, episode_idx, i, camera_key, jpeg_bytes)

        frames.append(
            {
                "frame_idx": i,
                "data": base64.b64encode(jpeg_bytes).decode("ascii"),
            }
        )

    return {
        "episode_idx": episode_idx,
        "camera": camera_key,
        "start": start,
        "count": len(frames),
        "total_frames": ep_length,
        "frames": frames,
    }


@router.get("/{dataset_id:path}/cache/stats")
async def get_cache_stats(dataset_id: str) -> dict[str, Any]:
    """Get frame cache statistics."""
    return _app_state.frame_cache.stats()


@router.post("/{dataset_id:path}/episodes/{episode_idx}/visualize")
async def visualize_episode(dataset_id: str, episode_idx: int) -> dict[str, str]:
    """Launch Rerun visualization for an episode.

    Starts lerobot-dataset-viz in the background for the specified episode.
    """
    import subprocess

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]

    # Validate episode index
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    # Build the command
    cmd = [
        "lerobot-dataset-viz",
        "--repo-id", dataset.repo_id,
        "--episode-index", str(episode_idx),
        "--root", str(dataset.root),
        "--display-compressed-images", "False",
    ]

    logger.info(f"Launching Rerun viz: {' '.join(cmd)}")

    # Launch in background (don't wait for it)
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch visualizer: {e}")

    return {"status": "ok", "message": f"Launched Rerun for episode {episode_idx}"}


