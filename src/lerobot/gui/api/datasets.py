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
import threading
from concurrent.futures import ThreadPoolExecutor
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

# Cache episode start indices (cumulative sum of lengths)
_episode_start_indices: dict[str, list[int]] = {}

# Background prefetch state
# Single worker: sequential frame access is optimal for video decoding.
# The prefetch thread uses its own VideoDecoderCache, so it does NOT
# contend with the main thread's decoder — no lock needed.
_prefetch_executor = ThreadPoolExecutor(max_workers=1)
_prefetch_generation: int = 0
_prefetch_current: tuple[str, int] | None = None  # (dataset_id, episode_idx)
_prefetch_last_frame: int = 0  # Last frame requested by _maybe_start_prefetch
_prefetch_lock = threading.Lock()

# Threshold for detecting seeks vs. normal sequential playback.
# If the frame delta between consecutive _maybe_start_prefetch calls
# exceeds this, we cancel the current prefetch and restart from the new position.
_PREFETCH_SEEK_THRESHOLD = 5


def _get_episode_start_index(dataset_id: str, episode_idx: int) -> int:
    """Get the global HuggingFace dataset index where an episode starts.

    The metadata's dataset_from_index is per-parquet-file, not global.
    This function computes the cumulative sum of episode lengths to get
    the correct global index.
    """
    if dataset_id not in _app_state.datasets:
        return 0

    # Check if we have cached start indices
    if dataset_id not in _episode_start_indices:
        dataset = _app_state.datasets[dataset_id]
        episodes = dataset.meta.episodes

        # Compute cumulative sum of episode lengths
        start_indices = [0]
        for i in range(len(episodes) - 1):
            start_indices.append(start_indices[-1] + episodes[i]["length"])

        _episode_start_indices[dataset_id] = start_indices
        logger.debug(f"Computed episode start indices for {dataset_id}: {len(start_indices)} episodes")

    indices = _episode_start_indices[dataset_id]
    if episode_idx < len(indices):
        return indices[episode_idx]
    return 0


def _invalidate_episode_start_indices(dataset_id: str) -> None:
    """Clear cached episode start indices when metadata changes."""
    _episode_start_indices.pop(dataset_id, None)


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

        # Check and repair episode metadata indices if needed
        from lerobot.datasets.dataset_tools import repair_episode_indices

        repaired = repair_episode_indices(root)
        if repaired > 0:
            logger.info(f"Repaired {repaired} episode indices with incorrect dataset_from_index")
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

        # Invalidate episode start index cache
        _invalidate_episode_start_indices(dataset_id)

        # Verify dataset integrity after reload
        from lerobot.datasets.dataset_tools import verify_dataset

        verification = verify_dataset(root, check_videos=False, verbose=False)
        if not verification.is_valid:
            for err in verification.errors:
                logger.warning(f"Post-reload verification: {err.message}")
        for warn in verification.warnings:
            logger.warning(f"Post-reload verification warning: {warn.message}")

        logger.info(
            f"Reloaded dataset: {dataset.meta.total_episodes} episodes, "
            f"{dataset.meta.total_frames} frames"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to reload dataset for {dataset_id}: {e}")
        return False


_PREFETCH_BATCH_SIZE = 30  # Frames per batch (~1 second at 30 fps)


def _prefetch_episode(
    dataset_id: str, episode_idx: int, ep_length: int, generation: int, start_frame: int = 0
) -> None:
    """Decode and cache all frames of an episode in a background thread.

    Starts from start_frame and wraps around to cover the entire episode.
    Stops early if _prefetch_generation changes (meaning a different episode was selected).

    Uses batch decoding (multiple timestamps per decode call) for efficiency —
    the video decoder can read sequential frames without re-seeking. Also uses
    its own VideoDecoderCache so it never contends with the main thread.
    """
    import time

    from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec
    from lerobot.gui.frame_cache import encode_frame_to_jpeg

    if dataset_id not in _app_state.datasets:
        return

    dataset = _app_state.datasets[dataset_id]
    video_keys = list(dataset.meta.video_keys)
    first_camera = list(dataset.meta.camera_keys)[0] if dataset.meta.camera_keys else None
    ep = dataset.meta.episodes[episode_idx]
    fps = dataset.fps
    tolerance_s = 1 / fps * 0.7

    # Own decoder cache — completely independent from the main thread's decoders
    prefetch_decoder_cache = VideoDecoderCache()

    cached_count = 0
    decoded_count = 0
    total_decode_ms = 0.0
    total_encode_ms = 0.0
    prefetch_start = time.perf_counter()

    # Build two contiguous ranges: [start_frame, ep_length) then [0, start_frame)
    # Keeping frame indices sequential within each range lets the decoder
    # read forward without seeking backward.
    contiguous_ranges = [range(start_frame, ep_length)]
    if start_frame > 0:
        contiguous_ranges.append(range(0, start_frame))

    try:
        for frame_range in contiguous_ranges:
            for batch_start in range(frame_range.start, frame_range.stop, _PREFETCH_BATCH_SIZE):
                # Check cancellation between batches
                if _prefetch_generation != generation:
                    logger.info(
                        f"Prefetch cancelled for episode {episode_idx} at frame {batch_start}/{ep_length} "
                        f"(decoded {decoded_count}, skipped {cached_count} cached)"
                    )
                    return

                batch_end = min(batch_start + _PREFETCH_BATCH_SIZE, frame_range.stop)

                # Filter out already-cached frames
                uncached_frames = []
                for fi in range(batch_start, batch_end):
                    if first_camera and _app_state.frame_cache.contains(
                        dataset_id, episode_idx, fi, first_camera
                    ):
                        cached_count += 1
                    else:
                        uncached_frames.append(fi)

                if not uncached_frames:
                    continue

                # Batch-decode all uncached frames for each camera
                try:
                    for vid_key in video_keys:
                        from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
                        timestamps = [from_timestamp + fi / fps for fi in uncached_frames]
                        video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, vid_key)

                        t1 = time.perf_counter()
                        frames = decode_video_frames_torchcodec(
                            video_path, timestamps, tolerance_s,
                            decoder_cache=prefetch_decoder_cache,
                        )
                        t2 = time.perf_counter()
                        total_decode_ms += (t2 - t1) * 1000

                        # JPEG-encode each frame and cache it
                        for k, fi in enumerate(uncached_frames):
                            cam_jpeg = encode_frame_to_jpeg(frames[k])
                            _app_state.frame_cache.put(dataset_id, episode_idx, fi, vid_key, cam_jpeg)

                        t3 = time.perf_counter()
                        total_encode_ms += (t3 - t2) * 1000

                    decoded_count += len(uncached_frames)
                except Exception:
                    logger.warning(
                        f"Prefetch failed for batch {batch_start}-{batch_end} of episode {episode_idx}",
                        exc_info=True,
                    )

        elapsed = (time.perf_counter() - prefetch_start) * 1000
        avg_decode = total_decode_ms / decoded_count if decoded_count else 0
        avg_encode = total_encode_ms / decoded_count if decoded_count else 0
        logger.info(
            f"Prefetch complete for episode {episode_idx}: "
            f"decoded {decoded_count}, skipped {cached_count} cached, {ep_length} total in {elapsed:.0f}ms "
            f"(avg decode={avg_decode:.1f}ms, encode={avg_encode:.1f}ms)"
        )
    finally:
        prefetch_decoder_cache.clear()


def _maybe_start_prefetch(dataset_id: str, episode_idx: int, ep_length: int, start_frame: int = 0) -> None:
    """Start background prefetching for an episode if not already in progress.

    Deduplicates by (dataset_id, episode_idx) for sequential playback.
    Detects seeks (frame jumps > _PREFETCH_SEEK_THRESHOLD) and restarts
    the prefetch from the new position.
    """
    global _prefetch_generation, _prefetch_current, _prefetch_last_frame

    with _prefetch_lock:
        if _prefetch_current == (dataset_id, episode_idx):
            # Same episode — only restart on significant seek
            frame_delta = abs(start_frame - _prefetch_last_frame)
            _prefetch_last_frame = start_frame
            if frame_delta <= _PREFETCH_SEEK_THRESHOLD:
                return  # Normal sequential advance, let current prefetch continue
            # Big jump detected — cancel old prefetch and restart from new position
            logger.info(
                f"Seek detected (delta={frame_delta}), restarting prefetch "
                f"for episode {episode_idx} from frame {start_frame}"
            )

        _prefetch_generation += 1
        generation = _prefetch_generation
        _prefetch_current = (dataset_id, episode_idx)
        _prefetch_last_frame = start_frame

    logger.info(f"Starting prefetch for episode {episode_idx} from frame {start_frame} ({ep_length} frames)")
    _prefetch_executor.submit(_prefetch_episode, dataset_id, episode_idx, ep_length, generation, start_frame)


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
    warnings: list[str] = []  # Any warnings (repairs, verification issues) from loading


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

        # Check and repair episode metadata indices if needed
        from lerobot.datasets.dataset_tools import repair_episode_indices, verify_dataset

        warnings: list[str] = []

        repaired = repair_episode_indices(dataset.root)
        if repaired > 0:
            logger.info(f"Repaired {repaired} episode indices with incorrect dataset_from_index")
            dataset.meta.episodes = load_episodes(dataset.root)
            warnings.append(f"Repaired {repaired} episode indices with incorrect metadata")

        # Verify dataset integrity
        verification = verify_dataset(dataset.root, check_videos=False, verbose=False)
        if not verification.is_valid:
            for err in verification.errors:
                logger.warning(f"Dataset verification: {err.message}")
                warnings.append(err.message)
        for warn in verification.warnings:
            logger.warning(f"Dataset verification warning: {warn.message}")
            warnings.append(warn.message)
        if verification.is_valid and not verification.warnings:
            logger.info("Dataset verification passed with no errors")

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
            warnings=warnings,
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

    # Calculate global index (cumulative sum of episode lengths, not per-file offset)
    episode_start = _get_episode_start_index(dataset_id, episode_idx)
    global_idx = episode_start + frame_idx

    import time

    # Check if this camera is already cached
    jpeg_bytes = _app_state.frame_cache.get(dataset_id, episode_idx, frame_idx, camera_key)

    if jpeg_bytes is None:
        # Not cached - decode ALL cameras at once (dataset[idx] decodes all anyway)
        # This avoids redundant decoding when multiple cameras are requested
        from lerobot.gui.frame_cache import encode_frame_to_jpeg

        t0 = time.perf_counter()
        item = dataset[global_idx]
        t1 = time.perf_counter()

        # Cache all cameras from this single decode
        for cam in camera_keys:
            if cam in item:
                cam_jpeg = encode_frame_to_jpeg(item[cam])
                _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, cam, cam_jpeg)
                if cam == camera_key:
                    jpeg_bytes = cam_jpeg
        t2 = time.perf_counter()

        # Fallback if camera wasn't in camera_keys list
        if jpeg_bytes is None:
            jpeg_bytes = encode_frame_to_jpeg(item[camera_key])
            _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, camera_key, jpeg_bytes)

        decode_ms = (t1 - t0) * 1000
        encode_ms = (t2 - t1) * 1000
        logger.info(
            f"get_frame ep={episode_idx} frame={frame_idx} cam={camera_key}: "
            f"decode={decode_ms:.1f}ms encode={encode_ms:.1f}ms"
        )
    else:
        logger.debug(f"get_frame ep={episode_idx} frame={frame_idx} cam={camera_key}: cache hit")

    # Trigger background prefetching for this episode, starting from the current frame
    _maybe_start_prefetch(dataset_id, episode_idx, ep_length, start_frame=frame_idx)

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

    # Calculate episode start index (cumulative sum, not per-file offset)
    episode_start = _get_episode_start_index(dataset_id, episode_idx)

    frames = []
    for i in range(start, min(start + count, ep_length)):
        global_idx = episode_start + i

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


