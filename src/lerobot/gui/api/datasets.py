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
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import load_episodes

    try:
        if request.local_path:
            # Open local dataset
            local_path = Path(request.local_path)
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"Path not found: {request.local_path}")

            # Extract repo_id from path or use path name
            repo_id = request.repo_id or local_path.name
            dataset = LeRobotDataset(repo_id, root=local_path)
            dataset_id = str(local_path)
        elif request.repo_id:
            # Open from HuggingFace Hub
            dataset = LeRobotDataset(request.repo_id)
            dataset_id = request.repo_id
        else:
            raise HTTPException(status_code=400, detail="Must provide either repo_id or local_path")

        # Ensure episodes are loaded
        if dataset.meta.episodes is None:
            dataset.meta.episodes = load_episodes(dataset.root)

        # Store in app state
        _app_state.datasets[dataset_id] = dataset

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
    logger.info(f"Closed dataset: {dataset_id}")

    return {"status": "ok", "message": f"Closed dataset: {dataset_id}"}


@router.get("/{dataset_id:path}/episodes")
async def list_episodes(dataset_id: str) -> list[EpisodeInfo]:
    """List all episodes in a dataset."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

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

    # Try to get from cache
    def decode_frame():
        item = dataset[global_idx]
        return item[camera_key]

    jpeg_bytes = _app_state.frame_cache.get_or_decode(
        dataset_id=dataset_id,
        episode_idx=episode_idx,
        frame_idx=frame_idx,
        camera_key=camera_key,
        decode_fn=decode_frame,
    )

    return Response(content=jpeg_bytes, media_type="image/jpeg")


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
    frames = []
    for i in range(start, min(start + count, ep_length)):
        global_idx = ep["dataset_from_index"] + i

        def decode_frame(idx=global_idx):
            item = dataset[idx]
            return item[camera_key]

        jpeg_bytes = _app_state.frame_cache.get_or_decode(
            dataset_id=dataset_id,
            episode_idx=episode_idx,
            frame_idx=i,
            camera_key=camera_key,
            decode_fn=decode_frame,
        )

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
