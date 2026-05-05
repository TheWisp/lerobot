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

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from lerobot.datasets.dataset_tools import check_episode_video_duration
from lerobot.datasets.utils import DEFAULT_DATA_PATH
from lerobot.utils.constants import HF_LEROBOT_HOME

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Will be set by server.py
_app_state: AppState = None  # type: ignore

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


def _check_local_dataset_complete(local_path: Path) -> tuple[bool, list[str]]:
    """Check whether a local dataset directory has all data + video files.

    Used to short-circuit ``LeRobotDataset.__init__``'s implicit Hub download
    when the user opens a local path: the editor is local-only and should not
    silently pull hundreds of MB from the Hub if local files are missing.

    Returns:
        ``(is_complete, problems)``. ``problems`` is a list of human-readable
        strings; empty when the cache is complete. Never raises.
    """
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    if not (local_path / "meta" / "info.json").exists():
        return False, ["meta/info.json is missing"]

    try:
        # Passing root= ensures the metadata loader reads from disk; for a local
        # dataset that has meta/, no Hub fetch is triggered.
        meta = LeRobotDatasetMetadata(repo_id=local_path.name, root=local_path)
    except Exception as e:
        return False, [f"failed to load metadata: {e}"]

    problems: list[str] = []

    missing_data: set[str] = set()
    for ep in range(meta.total_episodes):
        try:
            p = local_path / meta.get_data_file_path(ep)
        except Exception as e:
            problems.append(f"ep {ep}: cannot resolve data path ({e})")
            continue
        if not p.exists():
            missing_data.add(str(p))
    if missing_data:
        sample = sorted(missing_data)[0]
        problems.append(f"{len(missing_data)} data parquet file(s) missing (e.g. {sample})")

    missing_videos: set[str] = set()
    for vid_key in meta.video_keys:
        for ep in range(meta.total_episodes):
            try:
                p = local_path / meta.get_video_file_path(ep, vid_key)
            except Exception as e:
                problems.append(f"ep {ep} {vid_key}: cannot resolve video path ({e})")
                continue
            if not p.exists():
                missing_videos.add(str(p))
    if missing_videos:
        sample = sorted(missing_videos)[0]
        problems.append(f"{len(missing_videos)} video file(s) missing (e.g. {sample})")

    return (len(problems) == 0), problems


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
    _episode_action_stats.pop(dataset_id, None)
    _per_episode_features_cache.pop(dataset_id, None)


# Cache of per-episode action stats keyed by dataset_id. Surfaced via
# EpisodeInfo.action_stats so the GUI (and any other consumer) can apply
# generic quality heuristics — all-zero, static, saturated, jittery — on
# top of the same raw characteristics.
_episode_action_stats: dict[str, dict[int, EpisodeActionStats]] = {}


def _load_episode_action_stats(dataset_root: Path) -> dict[int, EpisodeActionStats]:
    """Load per-episode action stats from the dataset's episode metadata
    parquet files.

    LeRobot stores ``stats/action/{min,max,mean,std,count,q*}`` per episode
    at record time, so we don't need to rescan the data parquet — we just
    read the four summary keys from ``meta/episodes/*.parquet``.
    Falls back to an empty dict if the dataset has no action feature or
    the stats columns aren't present (older datasets, partially-built
    datasets); callers treat ``None`` action_stats as "unknown, render
    no badge".
    """
    out: dict[int, EpisodeActionStats] = {}
    episodes_dir = Path(dataset_root) / "meta" / "episodes"
    if not episodes_dir.exists():
        return out
    parquet_paths = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_paths:
        return out

    stat_cols = ["stats/action/min", "stats/action/max", "stats/action/mean", "stats/action/std"]
    for path in parquet_paths:
        try:
            df = pd.read_parquet(path, columns=["episode_index", *stat_cols])
        except (KeyError, ValueError):
            # `action` feature absent, or stats not pre-computed — bail
            # silently for the whole dataset rather than per-file, so we
            # don't half-populate the cache.
            return {}
        except Exception as e:
            logger.warning(f"action-stats: skipping unreadable {path.name}: {e}")
            continue
        for _, row in df.iterrows():
            try:
                out[int(row["episode_index"])] = EpisodeActionStats(
                    min=list(row["stats/action/min"]),
                    max=list(row["stats/action/max"]),
                    mean=list(row["stats/action/mean"]),
                    std=list(row["stats/action/std"]),
                )
            except (TypeError, ValueError):
                continue
    return out


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

    from lerobot.datasets.io_utils import (
        load_episodes,
        load_info,
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

        try:
            repaired = repair_episode_indices(root)
        except PermissionError as e:
            logger.warning(f"Skipping episode index repair on reload: {e}")
            repaired = 0
        if repaired > 0:
            logger.info(f"Repaired {repaired} episode indices with incorrect dataset_from_index")
            dataset.meta.episodes = load_episodes(root)

        dataset.meta.stats = load_stats(root)
        dataset.meta.tasks = load_tasks(root)
        _dataset_info_mtime[dataset_id] = current_mtime

        # CRITICAL: Also reload the HuggingFace dataset.
        # Post-refactor, hf_dataset is owned by DatasetReader. Direct assignment
        # to dataset.hf_dataset is rejected (read-only property), so route through
        # the reader's load_and_activate() public reload entry.
        if dataset.reader is not None and dataset.reader.hf_dataset is not None:
            try:
                num_cleaned = dataset.reader.hf_dataset.cleanup_cache_files()
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
            if dataset.reader is not None:
                dataset.reader.load_and_activate()
        finally:
            hf_datasets.enable_caching()

        from lerobot.gui.cache_invalidation import invalidate_caches

        invalidate_caches(
            _app_state, dataset_id, invalidate_episode_indices=_invalidate_episode_start_indices
        )

        # Verify dataset integrity after reload
        from lerobot.datasets.dataset_tools import verify_dataset

        verification = verify_dataset(root, check_videos=False, verbose=False)
        if not verification.is_valid:
            for err in verification.errors:
                logger.warning(f"Post-reload verification: {err.message}")
        for warn in verification.warnings:
            logger.warning(f"Post-reload verification warning: {warn.message}")

        logger.info(
            f"Reloaded dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to reload dataset for {dataset_id}: {e}")
        return False


_PREFETCH_BATCH_SIZE = 30  # Frames per batch (~1 second at 30 fps)

# After caching the current episode, keep prefetching subsequent episodes
# until at least this many frames ahead have been cached. This provides
# a comfortable buffer even at 2x playback speed with short episodes.
# 1000 frames ≈ 33s at 30fps, using ~100-240MB depending on resolution/cameras.
_PREFETCH_LOOKAHEAD_FRAMES = 1000


def _prefetch_episode(
    dataset_id: str, episode_idx: int, ep_length: int, generation: int, start_frame: int = 0
) -> None:
    """Decode and cache all frames of an episode in a background thread.

    Starts from start_frame and wraps around to cover the entire episode.
    Stops early if _prefetch_generation changes (meaning a different episode was selected).

    After completing the current episode, continues prefetching subsequent
    episodes until at least _PREFETCH_LOOKAHEAD_FRAMES have been cached
    ahead, or there are no more episodes.

    Uses batch decoding (multiple timestamps per decode call) for efficiency —
    the video decoder can read sequential frames without re-seeking. Also uses
    its own VideoDecoderCache so it never contends with the main thread.
    """
    from lerobot.datasets.video_utils import VideoDecoderCache

    if dataset_id not in _app_state.datasets:
        return

    dataset = _app_state.datasets[dataset_id]
    video_keys = list(dataset.meta.video_keys)
    first_camera = list(dataset.meta.camera_keys)[0] if dataset.meta.camera_keys else None
    fps = dataset.fps
    tolerance_s = 1 / fps * 0.7

    # Own decoder cache — completely independent from the main thread's decoders
    prefetch_decoder_cache = VideoDecoderCache()

    try:
        _prefetch_single_episode(
            dataset_id,
            dataset,
            episode_idx,
            ep_length,
            generation,
            start_frame,
            video_keys,
            first_camera,
            fps,
            tolerance_s,
            prefetch_decoder_cache,
        )

        # Keep prefetching subsequent episodes until we have enough lookahead
        lookahead_remaining = _PREFETCH_LOOKAHEAD_FRAMES
        next_idx = episode_idx + 1
        while next_idx < dataset.meta.total_episodes and lookahead_remaining > 0:
            if _prefetch_generation != generation:
                return
            next_ep = dataset.meta.episodes[next_idx]
            next_length = next_ep["length"]

            # Skip episodes already fully cached
            if first_camera and _app_state.frame_cache.is_episode_cached(
                dataset_id, next_idx, next_length, first_camera
            ):
                logger.debug(f"Lookahead: episode {next_idx} already cached, skipping")
                lookahead_remaining -= next_length
                next_idx += 1
                continue

            logger.info(
                f"Auto-prefetching episode {next_idx} ({next_length} frames, "
                f"{lookahead_remaining} lookahead remaining)"
            )
            # Clear decoder cache between episodes (different video files)
            prefetch_decoder_cache.clear()
            _prefetch_single_episode(
                dataset_id,
                dataset,
                next_idx,
                next_length,
                generation,
                0,
                video_keys,
                first_camera,
                fps,
                tolerance_s,
                prefetch_decoder_cache,
            )
            lookahead_remaining -= next_length
            next_idx += 1
    finally:
        prefetch_decoder_cache.clear()


def _prefetch_single_episode(
    dataset_id: str,
    dataset,
    episode_idx: int,
    ep_length: int,
    generation: int,
    start_frame: int,
    video_keys: list[str],
    first_camera: str | None,
    fps: float,
    tolerance_s: float,
    prefetch_decoder_cache,
) -> None:
    """Decode and cache all frames of a single episode."""
    import time

    from lerobot.datasets.video_utils import decode_video_frames_torchcodec
    from lerobot.gui.frame_cache import encode_frame_to_jpeg

    ep = dataset.meta.episodes[episode_idx]

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
                        video_path,
                        timestamps,
                        tolerance_s,
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
    msg = (
        f"Prefetch complete for episode {episode_idx}: "
        f"decoded {decoded_count}, skipped {cached_count} cached, {ep_length} total in {elapsed:.0f}ms "
        f"(avg decode={avg_decode:.1f}ms, encode={avg_encode:.1f}ms)"
    )
    # Use DEBUG for no-op prefetches (everything already cached) to reduce log noise
    if decoded_count == 0:
        logger.debug(msg)
    else:
        logger.info(msg)


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
            # Wrap-around detection: delta ≈ ep_length means playback looped
            if frame_delta >= ep_length - _PREFETCH_SEEK_THRESHOLD:
                return  # Loop wrap-around, not a real seek
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


def set_app_state(state: AppState) -> None:
    """Set the application state for API handlers."""
    global _app_state
    _app_state = state


# ---------------------------------------------------------------------------
# Dataset sources (folder browser)
# ---------------------------------------------------------------------------

SOURCES_FILE = Path.home() / ".config" / "lerobot" / "dataset_sources.json"
OPENED_FILE = Path.home() / ".config" / "lerobot" / "opened_datasets.json"


def _read_opened() -> list[dict]:
    """Read persisted list of opened datasets."""
    if not OPENED_FILE.exists():
        return []
    try:
        data = json.loads(OPENED_FILE.read_text())
        return data.get("datasets", [])
    except Exception:
        logger.warning("Failed to read opened datasets", exc_info=True)
        return []


def _write_opened(opened: list[dict]) -> None:
    """Persist the list of opened datasets."""
    OPENED_FILE.parent.mkdir(parents=True, exist_ok=True)
    OPENED_FILE.write_text(json.dumps({"version": 1, "datasets": opened}, indent=2))


def _save_opened_state() -> None:
    """Save the current set of opened datasets from app state."""
    entries = [{"root": str(ds.root)} for ds in _app_state.datasets.values()]
    _write_opened(entries)


def _read_sources() -> list[dict]:
    """Read source folders from config. Returns default source if file missing."""
    default_source = {
        "path": str(HF_LEROBOT_HOME),
        "removable": False,
        "expanded": True,
    }
    if not SOURCES_FILE.exists():
        return [default_source]
    try:
        data = json.loads(SOURCES_FILE.read_text())
        sources = data.get("sources", [])
        # Ensure default source is always present
        default_paths = {str(HF_LEROBOT_HOME)}
        has_default = any(s["path"] in default_paths for s in sources)
        if not has_default:
            sources.insert(0, default_source)
        return sources
    except Exception:
        logger.warning("Failed to read dataset sources, using defaults", exc_info=True)
        return [default_source]


def _write_sources(sources: list[dict]) -> None:
    """Persist source folders to config."""
    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "sources": sources}
    SOURCES_FILE.write_text(json.dumps(data, indent=2))


def _scan_source(source_path: str, max_depth: int = 3) -> list[dict]:
    """Scan a directory for datasets (subdirs containing meta/info.json).

    Returns lightweight metadata for each found dataset.
    """
    root = Path(source_path)
    if not root.is_dir():
        return []

    found = []
    _scan_recursive(root, root, found, max_depth, 0)
    # Sort by name
    found.sort(key=lambda d: d["name"])
    return found


def _scan_recursive(base: Path, current: Path, found: list[dict], max_depth: int, depth: int) -> None:
    """Recursively scan for datasets up to max_depth."""
    if depth > max_depth:
        return
    try:
        info_file = current / "meta" / "info.json"
        if info_file.is_file():
            # This directory is a dataset
            try:
                info = json.loads(info_file.read_text())
                rel = current.relative_to(base)
                found.append(
                    {
                        "name": str(rel),
                        "root": str(current),
                        "total_episodes": info.get("total_episodes", 0),
                        "total_frames": info.get("total_frames", 0),
                        "fps": info.get("fps", 0),
                        "robot_type": info.get("robot_type") or "",
                    }
                )
            except Exception:
                logger.debug(f"Failed to read info.json in {current}", exc_info=True)
            return  # Don't recurse into dataset subdirs

        # Not a dataset — recurse into subdirectories
        if depth < max_depth:
            try:
                for child in sorted(current.iterdir()):
                    if child.is_dir() and not child.name.startswith("."):
                        _scan_recursive(base, child, found, max_depth, depth + 1)
            except PermissionError:
                pass
    except Exception:  # nosec B110 - directory scan should never abort enumeration
        pass


class SourceRequest(BaseModel):
    path: str


class SourceInfo(BaseModel):
    path: str
    removable: bool
    expanded: bool


class SourceDatasetInfo(BaseModel):
    name: str
    root: str
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str = ""


@router.get("/previously-opened")
async def get_previously_opened() -> list[dict]:
    """Return the list of datasets that were open in the previous session."""
    return _read_opened()


@router.get("/sources")
async def list_sources() -> list[SourceInfo]:
    """List dataset source folders."""
    return [SourceInfo(**s) for s in _read_sources()]


@router.post("/sources")
async def add_source(req: SourceRequest) -> SourceInfo:
    """Add a dataset source folder."""
    path = str(Path(req.path).expanduser().resolve())
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {path}")

    sources = _read_sources()
    # Check if already exists
    if any(s["path"] == path for s in sources):
        raise HTTPException(status_code=409, detail="Source already exists")

    new_source = {"path": path, "removable": True, "expanded": True}
    sources.append(new_source)
    _write_sources(sources)
    logger.info(f"Added dataset source: {path}")
    return SourceInfo(**new_source)


@router.delete("/sources/{encoded_path:path}")
async def remove_source(encoded_path: str) -> dict[str, str]:
    """Remove a dataset source folder."""
    path = unquote(encoded_path)
    sources = _read_sources()
    source = next((s for s in sources if s["path"] == path), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")
    if not source.get("removable", True):
        raise HTTPException(status_code=400, detail="Cannot remove default source")

    sources = [s for s in sources if s["path"] != path]
    _write_sources(sources)
    logger.info(f"Removed dataset source: {path}")
    return {"status": "ok"}


@router.put("/sources/{encoded_path:path}/expanded")
async def set_source_expanded(encoded_path: str, expanded: bool = True) -> dict[str, str]:
    """Toggle source folder expansion state."""
    path = unquote(encoded_path)
    sources = _read_sources()
    source = next((s for s in sources if s["path"] == path), None)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")

    source["expanded"] = expanded
    _write_sources(sources)
    return {"status": "ok"}


@router.post("/open-in-files")
async def open_in_file_manager(body: dict) -> dict:
    """Open a directory in the system file manager."""
    import subprocess as _subprocess

    path = body.get("path", "")
    if not path or not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {path}")

    try:
        _subprocess.Popen(["xdg-open", path])  # nosec B607 - xdg-open is the standard Linux file-opener
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="xdg-open not found") from e

    return {"status": "ok"}


@router.get("/sources/{encoded_path:path}/datasets")
async def scan_source(encoded_path: str) -> list[SourceDatasetInfo]:
    """Scan a source folder for datasets."""
    import asyncio

    path = unquote(encoded_path)
    sources = _read_sources()
    if not any(s["path"] == path for s in sources):
        raise HTTPException(status_code=404, detail=f"Source not found: {path}")

    # Run scan in executor to avoid blocking
    loop = asyncio.get_event_loop()
    datasets = await loop.run_in_executor(None, _scan_source, path)
    return [SourceDatasetInfo(**d) for d in datasets]


class OpenDatasetRequest(BaseModel):
    """Request to open a dataset."""

    repo_id: str | None = None
    local_path: str | None = None
    # When opening a local path with an incomplete cache, the server returns 409
    # with a list of problems. Re-issue with confirm_hub_sync=True to authorize
    # the implicit Hub download (snapshot_download into the existing root).
    confirm_hub_sync: bool = False


class FeatureSchema(BaseModel):
    """Schema info for a single dataset feature.

    Mirrors the per-feature spec from ``info.json`` ``features`` dict —
    just the fields the GUI actually uses for rendering and validation.
    """

    dtype: str  # e.g. "float32", "int64", "bool", "string", "image", "video"
    shape: list[int]  # e.g. [1] for scalar, [14] for vector, [3, 480, 640] for image
    names: list[str] | None = None  # component names for vectors; None for scalars/strings
    is_per_episode: bool = False
    # True if every episode has uniform value for this feature — i.e. it's a logical
    # per-episode field broadcast across the per-frame column. Edits coerce to the
    # whole episode to preserve the broadcast invariant. Detected once per dataset
    # open via _detect_per_episode_features() and cached.
    observed_min: float | None = None
    observed_max: float | None = None
    # Dataset-wide observed extrema, sourced from ``meta/stats.json`` (aggregated
    # across episodes by ``compute_stats.py``). Populated only for scalar numeric
    # features (shape == [1] or empty). The GUI shows these next to the feature
    # name and uses them to scale the slider so the range is stable across
    # episodes. Distinct from any future *declared* min/max bound — these are
    # observed values, not enforced.


class DatasetInfo(BaseModel):
    """Summary info about a dataset."""

    id: str
    repo_id: str
    root: str
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str = ""
    camera_keys: list[str]
    features: list[str]  # feature names only — preserved for backwards-compat
    features_schema: dict[str, FeatureSchema] = {}
    # Full per-feature schema (dtype, shape, names) keyed by feature name.
    # Populated from ``ds.meta.features``. The GUI renderer registry
    # dispatches on (dtype, ndim) to pick row + Inspector widgets.
    errors: list[str] = []  # Verification errors (metadata mismatches — dataset may be corrupted)
    warnings: list[str] = []  # Non-critical warnings (stale stats, etc.)


_per_episode_features_cache: dict[str, set[str]] = {}


def _detect_per_episode_features(dataset_id: str, dataset) -> set[str]:
    """Identify features whose values are uniform within every episode.

    Pre: ``dataset`` is fully loaded.
    Post: returns a set of feature names. Only considers features that
    are non-image, non-video, non-DEFAULT_FEATURES, non-action,
    non-observation.* — the same gate the staging endpoint uses to
    decide editability. Result is cached per ``dataset_id``.

    Detection is by reading the data parquet shards once. For each
    feature, group rows by ``episode_index`` and check that every group
    has at most one unique value. We treat lists / numpy arrays as not
    per-episode (only scalars and strings can plausibly be broadcast).
    """
    if dataset_id in _per_episode_features_cache:
        return _per_episode_features_cache[dataset_id]

    skip_dtypes = {"image", "video"}
    default_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

    candidate_features: list[str] = []
    for name, ft in dataset.meta.features.items():
        if ft.get("dtype") in skip_dtypes:
            continue
        if name in default_features:
            continue
        if name == "action" or name.startswith("observation."):
            continue
        # Vectors and matrices can't reasonably be "uniform per episode" — skip.
        shape = ft.get("shape") or [1]
        if len(shape) > 1 or (len(shape) == 1 and shape[0] != 1):
            continue
        candidate_features.append(name)

    if not candidate_features:
        _per_episode_features_cache[dataset_id] = set()
        return _per_episode_features_cache[dataset_id]

    per_episode: set[str] = set()
    data_dir = Path(dataset.root) / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        _per_episode_features_cache[dataset_id] = set()
        return _per_episode_features_cache[dataset_id]

    cols_to_read = ["episode_index", *candidate_features]
    nunique_by_feature: dict[str, int] = dict.fromkeys(candidate_features, 0)

    try:
        for shard in parquet_files:
            df = pd.read_parquet(shard, columns=cols_to_read)
            for name in candidate_features:
                if nunique_by_feature[name] > 1:
                    continue
                # nunique() per episode_index group — max across groups tells us
                # whether ANY episode has variation. >1 disqualifies the feature.
                nunique_max = df.groupby("episode_index")[name].nunique(dropna=False).max()
                if nunique_max is None:
                    continue
                nunique_by_feature[name] = max(nunique_by_feature[name], int(nunique_max))
    except Exception as e:
        logger.warning(f"per-episode-feature detection failed for {dataset_id}: {e}")
        _per_episode_features_cache[dataset_id] = set()
        return _per_episode_features_cache[dataset_id]

    for name, nu in nunique_by_feature.items():
        if 0 < nu <= 1:
            per_episode.add(name)

    _per_episode_features_cache[dataset_id] = per_episode
    logger.info(f"detected per-episode features for {dataset_id}: {sorted(per_episode)}")
    return per_episode


def _invalidate_per_episode_features(dataset_id: str) -> None:
    """Drop the cached per-episode feature set when the dataset changes."""
    _per_episode_features_cache.pop(dataset_id, None)


# Hardcoded backend representation of the LeRobot 3.0 subtask format. The
# data layer stores ``subtask_index`` (int64[1]) plus ``meta/subtasks.parquet``
# (index → string lookup). The user always thinks in terms of strings, so
# whenever both pieces are present the schema endpoint synthesizes a
# ``subtask`` (string) feature in place of ``subtask_index``. Stage endpoint
# accepts either name and routes to the storage feature; PendingEdit stores
# the storage name so the apply pipeline doesn't need a special case.
SUBTASK_STORAGE_FEATURE = "subtask_index"
SUBTASK_DISPLAY_FEATURE = "subtask"


def _has_subtask_lookup(dataset) -> bool:
    """True if the dataset has a ``meta/subtasks.parquet`` lookup table."""
    return getattr(dataset.meta, "subtasks", None) is not None


def _scalar_observed_extrema(
    stats: dict | None, name: str, shape: list[int]
) -> tuple[float | None, float | None]:
    """Pull ``(min, max)`` for a scalar feature out of ``meta/stats.json``.

    ``stats`` is the dict loaded by :func:`load_stats`, structured as
    ``{feature_name: {"min": np.ndarray, "max": np.ndarray, ...}}``. Values are
    aggregated across all episodes by ``compute_stats.py``.

    Returns ``(None, None)`` if stats are missing, the feature isn't in stats,
    the values aren't numeric/finite, or the shape isn't scalar (we don't
    surface component-wise stats for vectors — too messy in the card header).
    """
    if stats is None or name not in stats:
        return None, None
    if shape and not (len(shape) == 1 and shape[0] == 1):
        return None, None  # vectors / matrices: skip
    entry = stats[name]
    if not isinstance(entry, dict):
        return None, None
    raw_min = entry.get("min")
    raw_max = entry.get("max")
    try:
        import numpy as _np

        mn = float(_np.asarray(raw_min).flatten()[0]) if raw_min is not None else None
        mx = float(_np.asarray(raw_max).flatten()[0]) if raw_max is not None else None
    except (TypeError, ValueError, IndexError):
        return None, None
    # Reject NaN / inf — they'd break JSON serialization and confuse the GUI.
    import math

    if mn is not None and not math.isfinite(mn):
        mn = None
    if mx is not None and not math.isfinite(mx):
        mx = None
    return mn, mx


def _build_features_schema(
    features: dict,
    per_episode: set[str] | None = None,
    *,
    subtask_synthesis: bool = False,
    stats: dict | None = None,
) -> dict[str, FeatureSchema]:
    """Convert ``ds.meta.features`` into the JSON-friendly FeatureSchema dict.

    Pre: ``features`` follows the LeRobot ``info.json`` shape — each value
    has ``dtype`` (str), ``shape`` (list/tuple of int), and optional
    ``names`` (list of str or None).
    Post: returned dict has the same keys; values are pydantic-validated
    FeatureSchema instances. Shapes are coerced to ``list[int]`` for
    JSON serialization stability.

    If ``subtask_synthesis=True`` and ``subtask_index`` is in ``features``,
    the storage entry is replaced by a synthetic ``subtask`` (string)
    entry that inherits the per-episode flag from ``subtask_index``. The
    caller passes ``subtask_synthesis=True`` only when the dataset also
    has ``meta/subtasks.parquet``.

    If ``stats`` is provided (the ``ds.meta.stats`` dict from
    ``meta/stats.json``), scalar numeric features get their dataset-wide
    observed ``(min, max)`` populated.
    """
    per_episode = per_episode or set()
    out: dict[str, FeatureSchema] = {}
    for name, ft in features.items():
        if subtask_synthesis and name == SUBTASK_STORAGE_FEATURE:
            # Skip the storage entry; it will be replaced by the synthetic
            # display entry below. We don't include both — the user thinks
            # in strings, so exposing both would just leak the storage name.
            continue
        shape = ft.get("shape", [])
        shape_list = [int(x) for x in shape] if shape is not None else []
        names = ft.get("names")
        if names is not None and not isinstance(names, list):
            if isinstance(names, dict):
                vals = next(iter(names.values()), None)
                names = list(vals) if isinstance(vals, list) else None
            else:
                names = None
        obs_min, obs_max = _scalar_observed_extrema(stats, name, shape_list)
        out[name] = FeatureSchema(
            dtype=str(ft.get("dtype", "")),
            shape=shape_list,
            names=names,
            is_per_episode=name in per_episode,
            observed_min=obs_min,
            observed_max=obs_max,
        )

    if subtask_synthesis and SUBTASK_STORAGE_FEATURE in features:
        # Per-episode flag transfers from the storage feature: if every episode
        # had uniform subtask_index, every episode also has a uniform subtask
        # string, so edits should still coerce to whole-episode.
        out[SUBTASK_DISPLAY_FEATURE] = FeatureSchema(
            dtype="string",
            shape=[1],
            names=None,
            is_per_episode=SUBTASK_STORAGE_FEATURE in per_episode,
        )
    return out


def _dataset_info_from(
    dataset_id: str,
    dataset,
    *,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> DatasetInfo:
    """Build a DatasetInfo response from an opened LeRobotDataset.

    Pre: ``dataset.meta`` is fully loaded (call ``ensure_episodes_loaded``
    first if opening). Post: returns a DatasetInfo with both the legacy
    ``features`` name list and the full ``features_schema`` mapping.
    """
    per_episode = _detect_per_episode_features(dataset_id, dataset)
    # Synthesize the user-facing "subtask" string feature only when the
    # dataset has BOTH the storage column AND the lookup table — an
    # incomplete dataset (one but not the other) would not let us decode.
    subtask_synthesis = SUBTASK_STORAGE_FEATURE in dataset.meta.features and _has_subtask_lookup(dataset)
    feature_names = list(dataset.meta.features.keys())
    if subtask_synthesis:
        # Mirror the schema synthesis in the legacy `features: list[str]` field
        # so frontends that read that list see "subtask" instead of "subtask_index".
        feature_names = [
            SUBTASK_DISPLAY_FEATURE if n == SUBTASK_STORAGE_FEATURE else n for n in feature_names
        ]
    return DatasetInfo(
        id=dataset_id,
        repo_id=dataset.repo_id,
        root=str(dataset.root),
        total_episodes=dataset.meta.total_episodes,
        total_frames=dataset.meta.total_frames,
        fps=dataset.fps,
        robot_type=getattr(dataset.meta, "robot_type", "") or "",
        camera_keys=list(dataset.meta.camera_keys),
        features=feature_names,
        features_schema=_build_features_schema(
            dataset.meta.features,
            per_episode=per_episode,
            subtask_synthesis=subtask_synthesis,
            stats=getattr(dataset.meta, "stats", None),
        ),
        errors=errors or [],
        warnings=warnings or [],
    )


class EpisodeActionStats(BaseModel):
    """Per-component summary statistics for an episode's recorded action.

    These mirror the per-episode stats LeRobot already stores in
    ``meta/episodes/*.parquet`` under ``stats/action/{min,max,mean,std}``,
    pre-computed at record time — we just expose them in the per-episode
    response so consumers (GUI, tests, future tooling) can apply quality
    heuristics generically rather than each one re-scanning the data
    parquet. Examples of derivable checks:

    * ``all-zero``  → ``max(|min|) == 0 AND max(|max|) == 0``
    * ``static``    → ``max(std) == 0`` (action never changed)
    * ``saturated`` → some component pinned to action-space bounds
    * ``jittery``   → ``mean(std)`` unusually high relative to peers

    None of those checks are baked into the API — the GUI decides which
    visual treatment to apply, based on these raw characteristics.
    """

    min: list[float]
    max: list[float]
    mean: list[float]
    std: list[float]


class EpisodeInfo(BaseModel):
    """Summary info about an episode."""

    episode_index: int
    length: int
    duration_s: float
    task: str | None = None
    video_extra_frames: int = 0  # Frame count difference (positive=extra, negative=missing)
    video_length: int = 0  # Total video frame count (0 = same as length)
    action_stats: EpisodeActionStats | None = None
    # Per-component action stats from dataset metadata (pre-computed at
    # record time). None when the dataset has no action feature, or stats
    # aren't present in the episode metadata (older / partially-built
    # datasets). Consumers derive quality flags from these — see
    # EpisodeActionStats docs.


@router.get("")
async def list_datasets() -> list[DatasetInfo]:
    """List all currently opened datasets."""
    return [_dataset_info_from(dataset_id, ds) for dataset_id, ds in _app_state.datasets.items()]


@router.post("")
async def open_dataset(request: OpenDatasetRequest) -> DatasetInfo:
    """Open a dataset by repo_id or local path."""
    import datasets as hf_datasets

    from lerobot.datasets.io_utils import load_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
                # Check for metadata changes (e.g. new episodes recorded externally)
                _check_and_reload_metadata(dataset_id)
                dataset = _app_state.datasets[dataset_id]
                logger.info(
                    f"Returning existing dataset: {dataset_id} ({dataset.meta.total_episodes} episodes)"
                )
                return _dataset_info_from(dataset_id, dataset)

            # Extract repo_id from path: if under HF_LEROBOT_HOME, use owner/name
            repo_id = request.repo_id
            if not repo_id:
                try:
                    rel = local_path.relative_to(HF_LEROBOT_HOME)
                    parts = rel.parts
                except ValueError:
                    # Not under HF_LEROBOT_HOME — best-effort: use the last two
                    # path components as owner/name. Matches the canonical
                    # "<root>/<owner>/<name>" layout that mirrors HF.
                    parts = local_path.parts[-2:]
                repo_id = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else local_path.name

            # Editor is local-only by default: surface incomplete on-disk
            # caches as a 409 so the frontend can ask the user to confirm.
            # On confirm (confirm_hub_sync=True), we skip the pre-check and let
            # LeRobotDataset.__init__ pull the missing files via snapshot_download
            # into local_path — same code path the existing /hub/download
            # endpoint uses.
            if not request.confirm_hub_sync:
                is_complete, problems = _check_local_dataset_complete(local_path)
                if not is_complete:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "incomplete_local_cache",
                            "message": "Local dataset cache is incomplete.",
                            "problems": problems,
                            "repo_id": repo_id,
                            "local_path": str(local_path),
                            "hub_sync_available": True,
                        },
                    )

            # Disable HuggingFace caching to ensure fresh data is loaded
            # This is important for datasets that may have been edited
            hf_datasets.disable_caching()
            try:
                dataset = LeRobotDataset(repo_id, root=local_path)
            except Exception as e:
                # LeRobotDataset.__init__ tries to download from Hub when
                # cached data doesn't match info.json (episode count mismatch).
                # For local datasets this is wrong — surface the real issue.
                err_msg = str(e)
                if "Repository Not Found" in err_msg or "doesn't contain all requested episodes" in err_msg:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Dataset metadata is inconsistent: info.json episode/frame counts "
                            "don't match the actual parquet data. This usually means episodes "
                            "were added or removed without updating info.json. "
                            "Run dataset verification to identify and repair the issue."
                        ),
                    ) from e
                raise
            finally:
                hf_datasets.enable_caching()

        elif request.repo_id:
            dataset_id = request.repo_id

            # Check if dataset is already open
            if dataset_id in _app_state.datasets:
                _check_and_reload_metadata(dataset_id)
                dataset = _app_state.datasets[dataset_id]
                logger.info(
                    f"Returning existing dataset: {dataset_id} ({dataset.meta.total_episodes} episodes)"
                )
                return _dataset_info_from(dataset_id, dataset)

            # Open from HuggingFace Hub
            dataset = LeRobotDataset(request.repo_id)
        else:
            raise HTTPException(status_code=400, detail="Must provide either repo_id or local_path")

        # Ensure episodes are loaded
        from lerobot.gui.dataset_reload import ensure_episodes_loaded

        ensure_episodes_loaded(dataset)

        # Check and repair episode metadata indices if needed
        from lerobot.datasets.dataset_tools import repair_episode_indices, verify_dataset

        errors: list[str] = []
        warnings: list[str] = []

        try:
            repaired = repair_episode_indices(dataset.root)
        except PermissionError as e:
            # Read-only dataset (e.g. backup directory). Skip repair, surface a
            # warning, and let the user open the dataset for inspection.
            logger.warning(f"Skipping episode index repair: {e}")
            warnings.append(f"Episode index repair skipped (dataset is read-only): {e}")
            repaired = 0
        if repaired > 0:
            logger.info(f"Repaired {repaired} episode indices with incorrect dataset_from_index")
            dataset.meta.episodes = load_episodes(dataset.root)
            warnings.append(f"Repaired {repaired} episode indices with incorrect metadata")

        # Verify dataset integrity
        verification = verify_dataset(dataset.root, check_videos=False, verbose=False)
        if not verification.is_valid:
            for err in verification.errors:
                logger.warning(f"Dataset verification ERROR: {err.message}")
                errors.append(err.message)
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
        _save_opened_state()

        return _dataset_info_from(dataset_id, dataset, errors=errors, warnings=warnings)

    except HTTPException:
        # Preserve intentional HTTP responses (e.g. 400 / 409) — don't wrap
        # them in a 500 with stringified detail.
        raise
    except Exception as e:
        logger.exception(f"Failed to open dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{dataset_id:path}")
async def close_dataset(dataset_id: str) -> dict[str, str]:
    """Close a dataset."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    if _app_state.is_locked(dataset_id):
        raise HTTPException(status_code=423, detail="Dataset is busy (operation in progress)")

    del _app_state.datasets[dataset_id]
    _dataset_info_mtime.pop(dataset_id, None)
    _save_opened_state()
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
        from lerobot.datasets.io_utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    # Per-episode action stats from dataset metadata (pre-computed at
    # record time). Cached until metadata mtime changes — the cache is
    # invalidated alongside _episode_start_indices by the metadata-reload
    # path. Returns an empty dict for datasets without an action feature
    # or pre-computed stats; consumers treat that as "unknown".
    if dataset_id not in _episode_action_stats:
        _episode_action_stats[dataset_id] = _load_episode_action_stats(Path(dataset.root))
    action_stats_by_ep = _episode_action_stats[dataset_id]

    result = []
    for i in range(dataset.meta.total_episodes):
        ep = episodes[i]
        length = ep["length"]
        duration_s = length / dataset.fps

        # Get task if available (ep["tasks"] contains task name strings directly)
        task = None
        if "tasks" in ep and ep["tasks"]:
            task = ep["tasks"][0] if len(ep["tasks"]) > 0 else None

        # Check for video-data duration mismatch (re-recording artifact or truncation)
        diff_per_cam = check_episode_video_duration(ep, dataset.fps)
        video_extra_frames = max(diff_per_cam.values(), key=abs) if diff_per_cam else 0

        # Total video frame count (matches length when no mismatch)
        video_length = length + max(0, video_extra_frames)

        result.append(
            EpisodeInfo(
                episode_index=i,
                length=length,
                duration_s=duration_s,
                task=task,
                video_extra_frames=video_extra_frames,
                video_length=video_length,
                action_stats=action_stats_by_ep.get(i),
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
        from lerobot.datasets.io_utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes

    ep = episodes[episode_idx]
    ep_length = ep["length"]

    # Compute video length for episodes with extra video frames
    diff_per_cam = check_episode_video_duration(ep, dataset.fps)
    video_extra = max(diff_per_cam.values(), key=abs) if diff_per_cam else 0
    video_length = ep_length + video_extra if video_extra > 0 else ep_length

    # Validate frame index (allow up to video_length for flagged episodes)
    if frame_idx < 0 or frame_idx >= video_length:
        raise HTTPException(
            status_code=404,
            detail=f"Frame not found: {frame_idx} (episode has {ep_length} data frames, {video_length} video frames)",
        )

    # Determine camera key
    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        raise HTTPException(status_code=400, detail="Dataset has no camera/image keys")

    if camera:
        if camera not in camera_keys:
            raise HTTPException(
                status_code=400, detail=f"Camera not found: {camera}. Available: {camera_keys}"
            )
        camera_key = camera
    else:
        camera_key = camera_keys[0]

    import time

    # Check if this camera is already cached
    jpeg_bytes = _app_state.frame_cache.get(dataset_id, episode_idx, frame_idx, camera_key)

    if jpeg_bytes is None:
        from lerobot.gui.frame_cache import encode_frame_to_jpeg

        if frame_idx < ep_length:
            # Normal frame — decode via dataset[global_idx] (gets all cameras at once)
            episode_start = _get_episode_start_index(dataset_id, episode_idx)
            global_idx = episode_start + frame_idx

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
        else:
            # Extra video frame beyond data length — decode directly from video file
            from lerobot.datasets.video_utils import decode_video_frames_torchcodec

            fps = dataset.fps
            from_ts = ep.get(f"videos/{camera_key}/from_timestamp", 0.0)
            timestamp = from_ts + frame_idx / fps
            tolerance_s = 1 / fps * 0.7

            video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, camera_key)

            t0 = time.perf_counter()
            frames = decode_video_frames_torchcodec(video_path, [timestamp], tolerance_s)
            t1 = time.perf_counter()

            jpeg_bytes = encode_frame_to_jpeg(frames[0])
            _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, camera_key, jpeg_bytes)
            t2 = time.perf_counter()

        decode_ms = (t1 - t0) * 1000
        encode_ms = (t2 - t1) * 1000
        logger.info(
            f"get_frame ep={episode_idx} frame={frame_idx} cam={camera_key}: "
            f"decode={decode_ms:.1f}ms encode={encode_ms:.1f}ms"
        )
    else:
        logger.debug(f"get_frame ep={episode_idx} frame={frame_idx} cam={camera_key}: cache hit")

    # Trigger background prefetching for this episode, starting from the current frame
    _maybe_start_prefetch(dataset_id, episode_idx, ep_length, start_frame=min(frame_idx, ep_length - 1))

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


def _coerce_feature_value_to_json(value: Any, dtype: str) -> Any:
    """Convert a single sample's feature value into JSON-serializable form.

    Pre: ``value`` comes from ``dataset[i][name]`` — typically a torch.Tensor,
    np.ndarray, str, bool, int, or float.
    Post: returns a JSON-compatible Python type (bool / int / float / str / list).
    Image / video tensors are NEVER passed in (caller must skip those features).
    """
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            scalar = value.item()
            return bool(scalar) if dtype == "bool" else scalar
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return list(value)
    # Pass-through for str / int / float / bool. Anything else falls back to str()
    # so the response stays JSON-encodable (better than failing the whole frame).
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


@router.get("/{dataset_id:path}/episodes/{episode_idx}/frame/{frame_idx}/features")
async def get_frame_features(dataset_id: str, episode_idx: int, frame_idx: int) -> dict[str, Any]:
    """Return all per-frame feature values at a single frame.

    Skips ``image`` / ``video`` features (those have a dedicated frame
    endpoint). Returns JSON-serializable values: scalars for shape ``[1]``,
    lists for vectors, strings for ``string`` features. ``subtask_index``
    and the decoded ``subtask`` string both appear when the dataset has a
    subtasks lookup table.
    """
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]

    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    ep = dataset.meta.episodes[episode_idx]
    ep_length = ep["length"]
    if frame_idx < 0 or frame_idx >= ep_length:
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frame_idx} out of range for episode {episode_idx} (length={ep_length})",
        )

    episode_start = _get_episode_start_index(dataset_id, episode_idx)
    global_idx = episode_start + frame_idx
    item = dataset[global_idx]

    # Filter out image/video features — those have their own frame endpoints.
    skip_dtypes = {"image", "video"}
    out: dict[str, Any] = {}
    for name, ft in dataset.meta.features.items():
        dtype = ft.get("dtype", "")
        if dtype in skip_dtypes:
            continue
        if name not in item:
            # Decoder can drop columns it doesn't know how to load; skip rather than error.
            continue
        try:
            out[name] = _coerce_feature_value_to_json(item[name], dtype)
        except Exception as e:
            logger.warning(f"Failed to coerce feature {name!r} at frame {frame_idx}: {e}")
            # Don't fail the whole response over one bad cell.
            out[name] = None

    # ``dataset[i]`` includes both ``task`` (decoded string) and ``task_index``
    # automatically; same for ``subtask`` / ``subtask_index`` when the lookup
    # table is present. They flow through the loop above.
    return {"frame_index": frame_idx, "episode_index": episode_idx, "values": out}


@router.get("/{dataset_id:path}/episodes/{episode_idx}/feature-series")
async def get_episode_feature_series(
    dataset_id: str,
    episode_idx: int,
    features: str = "",
) -> dict[str, Any]:
    """Return the per-frame trajectory of one or more features for an episode.

    The frontend uses this to render line / band / stripe rows under the timeline.
    Image / video features are rejected (they have their own video decode path).

    Pre: ``features`` is a comma-separated list of feature names. Empty / omitted
    means "all non-image, non-video features the dataset declares".
    Post: response shape is ``{episode_index, length, series: {name: [v0..v_{N-1}]}}``.
    For ``task`` / ``subtask``, the decoded string is returned per frame (matching
    what ``dataset[i]`` would yield) — backed by the dataset's lookup table.
    """
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    requested = [name.strip() for name in features.split(",") if name.strip()] if features else None

    # Resolve which raw columns we need to read from parquet, and which
    # decoded views (task / subtask) the caller wants on top of those.
    feature_dict = dataset.meta.features
    skip_dtypes = {"image", "video"}

    # Synthetic columns: not in features dict, but materialized by dataset[i].
    has_subtasks = getattr(dataset.meta, "subtasks", None) is not None and "subtask_index" in feature_dict
    synthetic_decoded = {"task": "task_index"}
    if has_subtasks:
        synthetic_decoded["subtask"] = "subtask_index"

    if requested is None:
        # Default: every per-frame feature except image/video, plus the synthetic
        # decoded views the dataset supports.
        raw_cols: list[str] = [
            name for name, ft in feature_dict.items() if ft.get("dtype") not in skip_dtypes
        ]
        decoded_cols: list[str] = [name for name in synthetic_decoded if synthetic_decoded[name] in raw_cols]
    else:
        raw_cols = []
        decoded_cols = []
        for name in requested:
            if name in synthetic_decoded:
                # caller asked for "task"/"subtask" — read the *_index column and decode.
                if synthetic_decoded[name] in feature_dict:
                    raw_cols.append(synthetic_decoded[name])
                    decoded_cols.append(name)
                continue
            if name not in feature_dict:
                raise HTTPException(status_code=400, detail=f"Unknown feature: {name!r}")
            if feature_dict[name].get("dtype") in skip_dtypes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature {name!r} is image/video — fetch via /frame/{{idx}} instead.",
                )
            raw_cols.append(name)

    ep = dataset.meta.episodes[episode_idx]
    chunk_idx = int(ep["data/chunk_index"])
    file_idx = int(ep["data/file_index"])
    parquet_path = Path(dataset.root) / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)

    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail=f"Data parquet missing: {parquet_path}")

    # Always pull episode_index too so we can slice in-memory. Avoids reading the whole
    # shard's worth of columns when we only need a slice — though for 1 episode we read
    # the matching rows only.
    cols_to_read = list(dict.fromkeys(["episode_index", *raw_cols]))
    df = pd.read_parquet(parquet_path, columns=cols_to_read)
    df = df[df["episode_index"] == episode_idx].reset_index(drop=True)

    series: dict[str, list[Any]] = {}
    for name in raw_cols:
        col = df[name].tolist() if name in df.columns else []
        # Pandas keeps numpy arrays for vector cells. Coerce per-row.
        series[name] = [_coerce_feature_value_to_json(v, feature_dict[name].get("dtype", "")) for v in col]

    # Decode synthetic columns ("task", "subtask") from the underlying *_index series.
    for decoded_name in decoded_cols:
        idx_col = synthetic_decoded[decoded_name]
        lookup = dataset.meta.tasks if decoded_name == "task" else dataset.meta.subtasks
        if lookup is None:
            continue
        idx_values = series.get(idx_col, [])
        try:
            series[decoded_name] = [lookup.iloc[int(i)].name for i in idx_values]
        except Exception as e:
            logger.warning(f"Failed to decode {decoded_name} via {idx_col}: {e}")
            series[decoded_name] = [None] * len(idx_values)

    return {
        "episode_index": episode_idx,
        "length": int(ep["length"]),
        "series": series,
    }


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
        from lerobot.datasets.io_utils import load_episodes

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


def _build_visualize_cmd(repo_id: str, episode_idx: int, root: str) -> list[str]:
    """Build the argv for ``lerobot-dataset-viz`` — extracted so tests can pin
    its shape against the target script's actual argparse (see
    ``tests/gui/test_visualize_episode.py``).

    Two things have bitten us here:

    1. PATH-based ``lerobot-dataset-viz`` resolves to whichever env's bin/
       comes first — typically the *base* conda env rather than the lerobot
       env running the GUI. That env can ship incompatible torch / torchcodec
       versions and the viewer crashes mid-decode with
       ``NotImplementedError: torchcodec_ns::_convert_to_tensor``. Use
       ``sys.executable -m`` so we always run under the same Python as the
       GUI.
    2. ``--display-compressed-images`` is a store_true flag (no value) since
       the upstream refactor; passing ``"False"`` as the next argv makes
       argparse treat ``"False"`` as an unknown positional and crash silently.
       Default (uncompressed) is what we want for the standalone viewer, so
       just omit the flag.
    """
    import sys

    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_dataset_viz",
        "--repo-id",
        repo_id,
        "--episode-index",
        str(episode_idx),
        "--root",
        str(root),
    ]


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

    cmd = _build_visualize_cmd(dataset.repo_id, episode_idx, str(dataset.root))

    logger.info(f"Launching Rerun viz: {' '.join(cmd)}")

    # Tee stdout/stderr to a per-launch log so silent crashes don't disappear.
    log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "gui" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime as _dt

    log_path = log_dir / f"rerun_viz_{_dt.now().strftime('%Y%m%d_%H%M%S')}_ep{episode_idx}.log"
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - subprocess owns the handle until exit

    try:
        subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as e:
        log_fh.close()
        raise HTTPException(status_code=500, detail=f"Failed to launch visualizer: {e}") from e
    logger.info(f"Rerun viz log: {log_path}")

    return {"status": "ok", "message": f"Launched Rerun for episode {episode_idx}"}


# ---------------------------------------------------------------------------
# HuggingFace Hub operations
# ---------------------------------------------------------------------------


@router.get("/hub/auth-status")
async def hub_auth_status():
    """Check if the user is logged in to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.whoami()
        return {"logged_in": True, "username": info.get("name", info.get("fullname", "unknown"))}
    except Exception:
        return {"logged_in": False, "username": None}


@router.get("/hub/repo-info")
async def hub_repo_info(repo_id: str):
    """Get info about a dataset repo on HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id, files_metadata=True)
        siblings = info.siblings or []
        total_size = sum(s.size for s in siblings if s.size)
        # Fetch episode/frame counts from remote info.json
        remote_episodes = None
        remote_frames = None
        remote_fps = None
        try:
            import json as _json

            from huggingface_hub import hf_hub_download

            info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
            remote_info = _json.loads(Path(info_path).read_text())
            remote_episodes = remote_info.get("total_episodes")
            remote_frames = remote_info.get("total_frames")
            remote_fps = remote_info.get("fps")
        except Exception:  # nosec B110 - remote info is best-effort metadata
            pass

        return {
            "exists": True,
            "repo_id": info.id,
            "private": info.private,
            "last_modified": str(info.last_modified) if info.last_modified else None,
            "downloads": info.downloads,
            "files": len(siblings),
            "total_size_mb": round(total_size / 1e6, 1),
            "sha": info.sha[:12] if info.sha else None,
            "total_episodes": remote_episodes,
            "total_frames": remote_frames,
            "fps": remote_fps,
        }
    except Exception:
        return {"exists": False, "repo_id": repo_id}


@router.get("/{dataset_id:path}/hub/diff")
async def hub_diff(dataset_id: str, repo_id: str | None = None):
    """Compare local dataset against HuggingFace Hub version by file size.

    Returns lists of modified, local-only, and remote-only files.
    Fast: no downloads, just file size comparison.
    """
    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    target_repo_id = repo_id or dataset.repo_id
    root = dataset.root

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(target_repo_id, files_metadata=True)
    except Exception:
        return {"status": "error", "message": f"Repo not found: {target_repo_id}"}

    remote_files = {}
    for s in info.siblings or []:
        remote_files[s.rfilename] = {
            "size": s.size,
            "sha": s.lfs.sha256 if s.lfs else s.blob_id,
        }

    local_only = []
    remote_only = []
    modified = []
    unchanged = 0

    for rname, rinfo in remote_files.items():
        local_path = root / rname
        if not local_path.exists():
            remote_only.append(rname)
            continue
        local_size = local_path.stat().st_size
        if rinfo["size"] and local_size != rinfo["size"]:
            modified.append({"file": rname, "local_size": local_size, "remote_size": rinfo["size"]})
        else:
            unchanged += 1

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if rel.startswith(".cache/") or rel.startswith(".lerobot"):
            continue
        if rel not in remote_files:
            local_only.append(rel)

    in_sync = len(modified) == 0 and len(local_only) == 0 and len(remote_only) == 0
    return {
        "status": "ok",
        "in_sync": in_sync,
        "unchanged": unchanged,
        "modified": modified,
        "local_only": local_only,
        "remote_only": remote_only,
    }


class HubUploadRequest(BaseModel):
    repo_id: str | None = None  # If None, uses dataset's repo_id


class HubDownloadRequest(BaseModel):
    repo_id: str | None = None  # If None, uses dataset's repo_id


@router.post("/{dataset_id:path}/hub/upload")
async def hub_upload(dataset_id: str, request: HubUploadRequest | None = None):
    """Push local dataset to HuggingFace Hub (overwrites remote)."""
    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        # Auto-open if path exists on disk (handles GUI restart with stale frontend)
        p = Path(dataset_id)
        if p.exists() and (p / "meta" / "info.json").exists():
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            _app_state.datasets[dataset_id] = LeRobotDataset(str(p), local_files_only=True)
            logger.info("Auto-opened dataset for upload: %s", dataset_id)
        else:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    lock = _app_state.get_lock(dataset_id)
    if lock.locked():
        raise HTTPException(status_code=423, detail="Dataset is busy")

    dataset = _app_state.datasets[dataset_id]
    repo_id = (request.repo_id if request and request.repo_id else None) or dataset.repo_id

    async with lock:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.whoami()
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail="Not logged in to HuggingFace Hub. Run `huggingface-cli login` in terminal.",
            ) from e

        logger.info(f"Uploading dataset to {repo_id} from {dataset.root}")
        try:
            from huggingface_hub import upload_folder

            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
            upload_folder(
                folder_path=str(dataset.root),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Update from LeRobot GUI",
            )
        except Exception as e:
            logger.exception(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {e}") from e

    logger.info(f"Upload complete: {repo_id}")
    return {
        "status": "ok",
        "message": f"Uploaded to {repo_id}",
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }


@router.post("/{dataset_id:path}/hub/download")
async def hub_download(dataset_id: str, request: HubDownloadRequest | None = None):
    """Pull dataset from HuggingFace Hub, overwriting local copy.

    TODO: clean up stale HF download cache/lock files before downloading
    to avoid deadlocks from interrupted previous downloads.
    """
    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    lock = _app_state.get_lock(dataset_id)
    if lock.locked():
        raise HTTPException(status_code=423, detail="Dataset is busy")

    dataset = _app_state.datasets[dataset_id]
    repo_id = (request.repo_id if request and request.repo_id else None) or dataset.repo_id
    root = dataset.root

    async with lock:
        logger.info(f"Downloading dataset {repo_id} to {root}")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(root),
            )
        except Exception as e:
            logger.exception(f"Download failed: {e}")
            raise HTTPException(status_code=500, detail=f"Download failed: {e}") from e

        # Reload dataset in-place (same pattern as merge_into / apply_edits)
        try:
            from lerobot.gui.dataset_reload import reload_dataset_from_disk

            reload_dataset_from_disk(dataset, root=root)

            from lerobot.gui.cache_invalidation import invalidate_caches

            invalidate_caches(
                _app_state, dataset_id, invalidate_episode_indices=_invalidate_episode_start_indices
            )

        except Exception as e:
            logger.exception(f"Reload after download failed: {e}")
            raise HTTPException(status_code=500, detail=f"Download succeeded but reload failed: {e}") from e

    logger.info(f"Download complete: {repo_id} ({dataset.meta.total_episodes} episodes)")
    return {
        "status": "ok",
        "message": f"Downloaded {repo_id} ({dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames)",
    }
