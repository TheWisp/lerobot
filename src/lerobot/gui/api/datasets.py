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

import asyncio
import contextlib
import gzip
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from lerobot.datasets.dataset_tools import check_episode_video_duration
from lerobot.datasets.utils import DEFAULT_DATA_PATH
from lerobot.gui.config_paths import gui_config_dir
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


# Dedicated single-worker executor for on-demand frame decode.
#
# Why a dedicated 1-worker pool rather than asyncio's default executor:
# the frame-fetch endpoint receives N parallel HTTP requests per frame
# change (the frontend's <img src=...> grid fires one per camera; for a
# 4-cam dataset that's 4 simultaneous requests targeting the same
# global_idx). Those would all land on different threads in the default
# multi-worker pool, and all of them would call dataset[global_idx]
# concurrently — which goes through the module-level
# ``video_utils._default_decoder_cache``. That cache returns the SAME
# torchcodec VideoDecoder instance to every caller, and libdav1d
# crashes (SIGSEGV at libdav1d.so.7.0.0+0x52c0d) on simultaneous use of
# one decoder. Confirmed in the wild on 2026-05-14 + reproduced
# locally with faulthandler.
#
# Two parts to the fix:
#   1. Use this single-worker executor so only one thread is ever
#      inside the decode work-fn at a time. Eliminates the libdav1d
#      thread-safety problem by construction.
#   2. Re-check the frame_cache inside the work-fn: by the time the
#      2nd, 3rd, ... requests reach the worker, the first one has
#      already decoded ALL cameras for that frame and populated the
#      JPEG cache. The N−1 followers find their cam in cache and skip
#      the redundant decode. Measured: 4 parallel same-frame requests
#      cost ~14 ms total (= 1 decode) instead of ~58 ms (= 4 decodes).
#
# Why not multi-worker with per-thread decoder caches? Benchmarked: the
# per-thread approach actually *loses* across the board because (a)
# libdav1d already uses internal multi-threading and saturates the
# physical cores, so user-level parallelism barely helps (~9 %), and
# (b) without singleflight it duplicates work (4 simultaneous same-frame
# requests do 4 decodes instead of 1). The single-worker + cache
# re-check pattern matches the access profile exactly: dedupe within
# a frame, queue across frames. Throughput ceiling is the same either
# way (limited by libdav1d's own per-decoder rate).
_decode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-decode")

# Whole-directory copies and deletes. Kept off the default executor, which is
# contended with frame decode and camera work: a dataset here runs to gigabytes,
# so one copy would hold a shared thread for seconds and stall playback.
# Single-threaded on purpose — these are disk-bound, and serialising them also
# means two copies cannot interleave onto the same target.
_fileops_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-fileops")


def shutdown_prefetch_executor() -> None:
    """Drop pending prefetch tasks and release the thread on server shutdown.

    The executor's single worker is a daemon thread so it dies on process
    exit anyway, but `shutdown(wait=False, cancel_futures=True)` also
    cancels any queued futures — without it, a long-running prefetch
    (a multi-second `_prefetch_episode` decode pass) would keep logging
    progress after uvicorn has already torn down logging handlers,
    producing the "I/O operation on closed file" stack traces.
    """
    _prefetch_executor.shutdown(wait=False, cancel_futures=True)


def shutdown_decode_executor() -> None:
    """Mirror of :func:`shutdown_prefetch_executor` for the decode pool."""
    _decode_executor.shutdown(wait=False, cancel_futures=True)


def _check_local_dataset_complete(local_path: Path) -> tuple[str, list[str]]:
    """Classify whether a local dataset directory can be opened from disk, and if not, how.

    Used to short-circuit ``LeRobotDataset.__init__``'s implicit Hub download
    when the user opens a local path: the editor is local-only and should not
    silently pull hundreds of MB from the Hub if local files are missing or the
    on-disk metadata is self-inconsistent.

    Two failure modes are distinguished so the caller can offer the right
    recovery (or none) and never show a misleading download prompt:

      * ``"missing_files"`` — the metadata is internally consistent but some
        referenced data/video files are absent on disk (e.g. a partial Hub
        download). Re-downloading from the Hub, when the repo exists, completes
        the cache.
      * ``"metadata_inconsistent"`` — the on-disk metadata contradicts itself:
        ``info.json``'s episode count disagrees with the per-episode metadata
        table, or an episode's path cannot be resolved. A download does not
        address this; the inconsistency is surfaced as-is.

    Returns:
        ``(kind, problems)`` where ``kind`` is ``"complete"``,
        ``"missing_files"``, or ``"metadata_inconsistent"``. ``problems``
        faithfully lists what's wrong; empty only when ``kind == "complete"``.
        Never raises.
    """
    from huggingface_hub.errors import HFValidationError

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    if not (local_path / "meta" / "info.json").exists():
        return "missing_files", ["meta/info.json is missing"]

    # If any meta-side file is missing, `LeRobotDatasetMetadata.__init__` falls
    # through to `_pull_from_repo` → `snapshot_download`, which validates the
    # `repo_id`. Folder names with spaces / non-alphanumeric characters (a
    # legitimate local scenario — users rename / duplicate cached datasets)
    # then surface as a confusing "Repo id must use alphanumeric chars…" error
    # instead of the actual diagnosis (meta files missing). Probe the
    # remaining required meta paths directly so we report the real problem.
    missing_meta: list[str] = []
    if not (local_path / "meta" / "tasks.parquet").exists():
        missing_meta.append("meta/tasks.parquet")
    if not (local_path / "meta" / "episodes").is_dir():
        missing_meta.append("meta/episodes/")
    if missing_meta:
        return "missing_files", [f"{p} is missing" for p in missing_meta]

    try:
        # Passing root= ensures the metadata loader reads from disk; for a local
        # dataset that has meta/, no Hub fetch is triggered.
        meta = LeRobotDatasetMetadata(repo_id=local_path.name, root=local_path)
    except HFValidationError:
        # The constructor only validates `repo_id` against the Hub when local
        # meta is incomplete enough to trigger a Hub fallback. Above we
        # already covered the common "info.json present, episodes missing"
        # case; reaching here means a subtler meta-side problem (a partial
        # episodes/ tree, version mismatch, etc.).
        return "metadata_inconsistent", ["meta/ contents are inconsistent (failed to load on disk)"]
    except Exception as e:
        return "metadata_inconsistent", [f"failed to load metadata: {e}"]

    problems: list[str] = []
    kind = "complete"

    # Metadata self-consistency: `info.json.total_episodes` is the authoritative
    # count, but the per-episode metadata table (`meta/episodes/`) is what
    # resolves data/video paths. When they disagree, path resolution for the
    # "extra" episodes raises IndexError — a self-inconsistency, not a missing
    # file, and not something a download addresses.
    n_info = meta.total_episodes
    n_table = len(meta.episodes) if meta.episodes is not None else 0
    if n_info != n_table:
        kind = "metadata_inconsistent"
        detail = (
            f" (episodes {n_table}–{n_info - 1} have data but no metadata row)" if n_info > n_table else ""
        )
        problems.append(
            f"info.json reports {n_info} episode(s) but the episode metadata table describes {n_table}{detail}"
        )

    # Only probe files for episodes the metadata table can actually resolve;
    # iterating past `n_table` would just re-report the count mismatch above as a
    # pile of "cannot resolve path" noise.
    resolvable = min(n_info, n_table)

    missing_data: set[str] = set()
    for ep in range(resolvable):
        try:
            p = local_path / meta.get_data_file_path(ep)
        except Exception as e:
            # A resolution failure within the resolvable range is a deeper
            # inconsistency, not a missing file.
            kind = "metadata_inconsistent"
            problems.append(f"ep {ep}: cannot resolve data path ({e})")
            continue
        if not p.exists():
            missing_data.add(str(p))
    if missing_data:
        sample = sorted(missing_data)[0]
        problems.append(f"{len(missing_data)} data parquet file(s) missing (e.g. {sample})")

    missing_videos: set[str] = set()
    for vid_key in meta.video_keys:
        for ep in range(resolvable):
            try:
                p = local_path / meta.get_video_file_path(ep, vid_key)
            except Exception as e:
                kind = "metadata_inconsistent"
                problems.append(f"ep {ep} {vid_key}: cannot resolve video path ({e})")
                continue
            if not p.exists():
                missing_videos.add(str(p))
    if missing_videos:
        sample = sorted(missing_videos)[0]
        problems.append(f"{len(missing_videos)} video file(s) missing (e.g. {sample})")

    # Missing files are only the diagnosis when no metadata contradiction was
    # found first.
    if problems and kind == "complete":
        kind = "missing_files"
    return kind, problems


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
    _per_episode_source_cache.pop(dataset_id, None)
    _per_episode_warnings_cache.pop(dataset_id, None)


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
    dataset_id: str,
    episode_idx: int,
    ep_length: int,
    generation: int,
    start_frame: int = 0,
    profile: str = "full",
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
            profile,
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
                profile,
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
    profile: str = "full",
) -> None:
    """Decode and cache all frames of a single episode.

    ``profile`` must match what the scrub endpoint will ask for — the cache is
    keyed by it, so warming the wrong one costs a full decode pass and serves
    nothing.
    """
    import time

    from lerobot.datasets.video_utils import decode_video_frames_torchcodec
    from lerobot.gui.frame_cache import encode_frame_for_profile

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
                        cam_jpeg = encode_frame_for_profile(frames[k], profile)
                        _app_state.frame_cache.put(dataset_id, episode_idx, fi, vid_key, cam_jpeg, profile)

                    t3 = time.perf_counter()
                    total_encode_ms += (t3 - t2) * 1000

                decoded_count += len(uncached_frames)
            except IndexError:
                # The episode vanished mid-prefetch — a concurrently-running
                # re-record discards the in-progress episode's metadata and
                # files, so get_video_file_path starts raising out-of-range
                # for every remaining batch. Expected when browsing a dataset
                # that is actively being recorded; abort quietly instead of
                # logging a traceback per batch.
                logger.info(
                    f"Prefetch aborted for episode {episode_idx}: episode no longer exists "
                    f"(re-recorded or deleted while prefetching)"
                )
                return
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


def _maybe_start_prefetch(
    dataset_id: str,
    episode_idx: int,
    ep_length: int,
    start_frame: int = 0,
    profile: str = "full",
) -> None:
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
    _prefetch_executor.submit(
        _prefetch_episode, dataset_id, episode_idx, ep_length, generation, start_frame, profile
    )


def set_app_state(state: AppState) -> None:
    """Set the application state for API handlers."""
    global _app_state
    _app_state = state


# ---------------------------------------------------------------------------
# Dataset sources (folder browser)
# ---------------------------------------------------------------------------

# Where the GUI keeps the state a user would notice losing: which folders are
# configured as dataset sources, and which datasets to restore on next launch.
#
# The directory is overridable because the GUI also runs as a subprocess — in
# tests, and in the e2e flows that launch it for real. A subprocess re-imports
# this module and cannot see a monkeypatched constant, so without an env
# channel those runs write the developer's actual config. That is not
# hypothetical: it left the GUI opening with a "Failed to open dataset" toast
# pointing at a deleted pytest directory.
_GUI_CONFIG_DIR = gui_config_dir()

SOURCES_FILE = _GUI_CONFIG_DIR / "dataset_sources.json"
OPENED_FILE = _GUI_CONFIG_DIR / "opened_datasets.json"


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
    """Persist the list of opened datasets (atomic write).

    Without staging, a crash between `write_text`'s truncate and the
    actual byte write leaves the file empty / partial; the next session
    would lose track of every dataset the user had open. Tiny file but
    not self-recovering — better to be safe.
    """
    import os

    OPENED_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OPENED_FILE.with_suffix(OPENED_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "datasets": opened}, indent=2))
    os.replace(tmp, OPENED_FILE)


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
    """Persist source folders to config (atomic write)."""
    import os

    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "sources": sources}
    tmp = SOURCES_FILE.with_suffix(SOURCES_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, SOURCES_FILE)


def _scan_source(source_path: str, max_depth: int = 3) -> list[dict]:
    """Scan a directory for datasets (subdirs containing meta/info.json).

    Returns lightweight metadata for each found dataset.
    """
    root = Path(source_path)
    if not root.is_dir():
        return []

    # A copy or a delete that was interrupted leaves a dot-prefixed remnant.
    # Invisible is the point — half-formed data must never list as a dataset —
    # but invisible also means nobody notices the disk it holds. A scan is when
    # we are already walking this tree, so it is the cheapest place to reclaim.
    from lerobot.gui.api._datasets_core import sweep_remnants

    sweep_remnants(root)

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
                        # Bit vocabularies, so callers can offer quality labels
                        # without opening the dataset. Read here because this
                        # function already has info.json parsed.
                        "flags": {
                            name: list(spec["flags"])
                            for name, spec in (info.get("features") or {}).items()
                            if isinstance(spec, dict) and spec.get("flags")
                        },
                        "cameras": [
                            name.removeprefix("observation.images.")
                            for name in (info.get("features") or {})
                            if name.startswith("observation.images.")
                        ],
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
    flags: dict[str, list[str]] = {}
    # Declared bit vocabularies keyed by feature name, straight from
    # info.json — see feature_utils.is_flags_feature. Empty for a dataset
    # that was never labelled, which is how callers tell "no labels" from
    # "not looked up yet".
    cameras: list[str] = []
    # Visual feature names with the observation.images. prefix removed, so a
    # caller can offer a camera choice without opening the dataset. Declared
    # here because this is a response model: a field the scan returns but the
    # model does not name is dropped silently on the way out.


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


class DuplicateDatasetRequest(BaseModel):
    """Request to copy the dataset at ``path`` to a sibling directory."""

    path: str
    new_name: str


# Typed errors from the shared core, mapped to the status codes the frontend
# already branches on. The same errors reach the MCP tools untranslated.
_DATASET_OP_STATUS = {
    "NotADatasetError": 404,
    "InvalidNameError": 400,
    "DatasetExistsError": 409,
    "DatasetBusyError": 423,
    # 500 either way, but the message is already a readable sentence rather
    # than shutil's per-file list, which ran to 228 KB when a source was
    # renamed mid-copy and reached the user as an alert() of that length.
    "CopyFailedError": 500,
    "DeleteFailedError": 500,
}


def _run_dataset_op(fn, *args):
    """Call a ``_datasets_core`` function, translating its typed errors.

    Blocking by design — it copies or removes a directory tree — so callers
    hand it to ``_fileops_executor`` rather than the default one, which is
    contended with frame decode and camera work.
    """
    try:
        return fn(*args)
    except Exception as e:
        status = _DATASET_OP_STATUS.get(type(e).__name__)
        if status is None:
            logger.error(f"Dataset operation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e
        raise HTTPException(status_code=status, detail=str(e)) from e


@router.post("/duplicate")
async def duplicate_dataset(request: DuplicateDatasetRequest) -> dict[str, Any]:
    """Copy a dataset to a sibling directory under a new name."""
    import asyncio

    from lerobot.gui.api import _datasets_core

    return await asyncio.get_running_loop().run_in_executor(
        _fileops_executor,
        _run_dataset_op,
        _datasets_core.duplicate_dataset,
        _app_state,
        request.path,
        request.new_name,
    )


@router.delete("/files")
async def delete_dataset_files(path: str) -> dict[str, Any]:
    """Delete a dataset from disk, closing it first if it is open.

    ``path`` travels as a query parameter rather than a path segment:
    ``{dataset_id:path}`` is greedy, so a suffix route under it would capture a
    close for any dataset whose folder carried the suffix name.
    """
    import asyncio

    from lerobot.gui.api import _datasets_core

    return await asyncio.get_running_loop().run_in_executor(
        _fileops_executor, _run_dataset_op, _datasets_core.delete_dataset, _app_state, path
    )


@router.post("/open-in-files")
async def open_in_file_manager(body: dict) -> dict:
    """Open a directory in the system file manager.

    Spawns the subprocess in the default executor so a slow fork/exec
    (heavy desktop session, many open FDs) cannot stall the FastAPI
    event loop.
    """
    import asyncio
    import subprocess as _subprocess

    path = body.get("path", "")
    if not path or not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {path}")

    def _spawn() -> None:
        _subprocess.Popen(["xdg-open", path])  # nosec B607 - xdg-open is the standard Linux file-opener

    try:
        await asyncio.get_event_loop().run_in_executor(None, _spawn)
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


class FlagImpact(BaseModel):
    """What excluding one label would cost, on its own."""

    label: str
    feature: str
    per_episode: bool
    frames: int
    episodes: int
    chunks_dropped: int = 0
    # Chunk starts this label alone would remove. The number that matters:
    # a training sample is a chunk, and one flagged frame disqualifies every
    # chunk containing it, so a thinly scattered label can cost many times
    # what its frame count suggests.
    # Episodes with at least one frame carrying the label. For a per-episode
    # column that is the whole episode either way; for a per-frame one it says
    # how widely the label is spread, which a frame count alone does not.


class FlagImpactResponse(BaseModel):
    total_frames: int
    total_episodes: int
    total_chunks: int = 0
    labels: list[FlagImpact] = []
    selected_chunks_kept: int | None = None
    selected_frames: int | None = None
    # Exact cost of the requested combination, not the sum of its parts:
    # labels overlap, and their chunk costs overlap more than their frames do.


_flags_impact_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-flags-impact")


def _chunk_starts_kept(episodes, flagged, chunk_size: int) -> int:
    """Starts whose whole window avoids ``flagged``. Mirrors the trainer's rule."""
    import numpy as np

    n = len(episodes)
    if not len(flagged):
        return n
    bad = np.zeros(n, dtype=bool)
    bad[flagged] = True
    boundaries = np.flatnonzero(np.diff(episodes)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [n]])
    ends = np.repeat(stops, stops - starts)
    cumulative = np.concatenate([[0], np.cumsum(bad)])
    idx = np.arange(n)
    window_end = np.minimum(idx + chunk_size, ends)
    return int(((cumulative[window_end] - cumulative[idx]) == 0).sum())


def _read_flags_impact(root: str, chunk_size: int = 50, selected: tuple = ()) -> dict:
    """Count, per declared label, the frames and episodes carrying it.

    Reads the parquet columns directly rather than opening a LeRobotDataset:
    this runs while the operator is filling in a form, and opening a dataset
    resolves against the Hub.

    Pre: ``root`` is a dataset directory containing ``meta/info.json``.
    """
    import json

    import numpy as np
    import pyarrow.parquet as pq

    base = Path(root)
    info = json.loads((base / "meta" / "info.json").read_text())
    vocab = {
        name: list(spec["flags"])
        for name, spec in (info.get("features") or {}).items()
        if isinstance(spec, dict) and spec.get("flags")
    }
    out: dict = {
        "total_frames": int(info.get("total_frames", 0)),
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_chunks": int(info.get("total_frames", 0)),
        "labels": [],
    }
    if not vocab:
        return out

    shards = sorted((base / "data").rglob("*.parquet"))
    if not shards:
        return out
    columns = ["episode_index", *vocab]
    table = pq.ParquetDataset([str(x) for x in shards]).read(columns=columns)
    episode = np.asarray(table.column("episode_index"), dtype=np.int64)
    out["total_frames"] = int(len(episode))

    out["total_chunks"] = _chunk_starts_kept(episode, np.array([], dtype=np.int64), chunk_size)
    selected_mask = np.zeros(len(episode), dtype=bool)
    for feature, labels in vocab.items():
        values = np.asarray(table.column(feature), dtype=np.int64).reshape(-1)
        per_episode = bool((info["features"][feature] or {}).get("per_episode"))
        for bit, label in enumerate(labels):
            hit = (values & (1 << bit)) != 0
            kept = _chunk_starts_kept(episode, np.flatnonzero(hit), chunk_size)
            out["labels"].append(
                {
                    "label": label,
                    "feature": feature,
                    "per_episode": per_episode,
                    "frames": int(hit.sum()),
                    "episodes": int(len(np.unique(episode[hit]))),
                    "chunks_dropped": out["total_chunks"] - kept,
                }
            )
            if label in selected:
                selected_mask |= hit

    if selected:
        out["selected_frames"] = int(selected_mask.sum())
        out["selected_chunks_kept"] = _chunk_starts_kept(episode, np.flatnonzero(selected_mask), chunk_size)
    return out


@router.get("/flags-impact", response_model=FlagImpactResponse)
async def flags_impact(root: str, chunk_size: int = 50, labels: str = "") -> FlagImpactResponse:
    """Per-label frame and episode counts for a dataset, by filesystem root.

    Keyed by root rather than by an opened dataset id so the training form can
    price labels for a dataset nobody has opened.
    """
    import asyncio

    if not (Path(root) / "meta" / "info.json").is_file():
        raise HTTPException(status_code=404, detail=f"No dataset at {root}")
    loop = asyncio.get_event_loop()
    try:
        picked = tuple(x.strip() for x in labels.split(",") if x.strip())
        data = await loop.run_in_executor(
            _flags_impact_executor, _read_flags_impact, root, chunk_size, picked
        )
    except Exception as e:
        logger.exception(f"flags-impact failed for {root}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return FlagImpactResponse(**data)


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
    flags: list[str] | None = None
    # Bit vocabulary for a bitset feature (feature_utils.is_flags_feature): bit
    # i means flags[i], so several labels can hold on one frame. Distinct from
    # names, whose value is an index and therefore exclusive. Carried through
    # because the timeline draws one sub-lane per flag and the row legend names
    # them; without it a bitset renders as a bare integer.
    derived: bool = False
    # True when the column is computed from other data (feature_utils
    # .is_derived_feature). The editor shows it and refuses to change it —
    # an edit to a derived value is discarded by the next recompute, so
    # offering the control at all is offering a no-op.
    is_per_episode: bool = False
    # True if every episode has uniform value for this feature — i.e. it's a logical
    # per-episode field broadcast across the per-frame column. Edits coerce to the
    # whole episode to preserve the broadcast invariant. Detected once per dataset
    # open via _detect_per_episode_features() and cached.
    per_episode_source: str | None = None
    # How was ``is_per_episode`` set? "declared" means the writer wrote
    # ``"per_episode": true`` in ``info.json`` — authoritative, persists across
    # saves. "detected" means the GUI scanned the parquet and inferred it from
    # uniform-within-episode values — empirical, can flip if a single frame
    # changes. ``None`` when ``is_per_episode`` is False. Power users can
    # surface this in tooltips to know which.
    observed_min: float | None = None
    observed_max: float | None = None
    # Dataset-wide observed extrema, sourced from ``meta/stats.json`` (aggregated
    # across episodes by ``compute_stats.py``). Populated only for scalar numeric
    # features (shape == [1] or empty). The GUI shows these next to the feature
    # name and uses them to scale the slider so the range is stable across
    # episodes. Distinct from declared bounds (below): observed, not enforced.
    declared_min: float | None = None
    declared_max: float | None = None
    # Optional declared bounds from the feature spec in ``info.json``. When
    # present, ``validate_feature_dtype_and_shape`` rejects values outside
    # ``[declared_min, declared_max]`` at add_frame and stage time. The GUI
    # uses them to scale the slider (preferred over observed_min/max) and may
    # show them in the card header. Categorical integer features use the
    # ``names`` field instead — the legal range is implicitly ``[0, len(names))``.


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
# Per-name source tag: "declared" (info.json said so — authoritative) vs
# "detected" (the GUI inferred it from uniform-within-episode data).
_per_episode_source_cache: dict[str, dict[str, str]] = {}
# Warnings surfaced when a feature was declared per_episode but the data
# isn't actually uniform within every episode. Surfaced via DatasetInfo
# warnings so the user knows their data violates their own declaration.
_per_episode_warnings_cache: dict[str, list[str]] = {}


def _per_episode_warnings_for(dataset_id: str) -> list[str]:
    """Return the (possibly empty) warning list cached by the last
    detector run for ``dataset_id``."""
    return list(_per_episode_warnings_cache.get(dataset_id, []))


def _detect_per_episode_features(dataset_id: str, dataset) -> set[str]:
    """Identify features that are per-episode — declared OR detected.

    Pre: ``dataset`` is fully loaded.
    Post: returns a set of feature names. The set is the union of:

    1. **Declared**: features whose info.json entry has ``per_episode: true``.
       Authoritative — survives saves, doesn't flip if the data drifts.
    2. **Detected**: features without the declaration whose values happen
       to be uniform within every episode (nunique-per-group ≤ 1).
       Heuristic — kept for backward compat with datasets pre-flag.

    Side effects (cached by ``dataset_id``):

    * ``_per_episode_source_cache[dataset_id]`` records the source
      ("declared" or "detected") per feature, so the schema endpoint can
      surface ``per_episode_source`` to the GUI.
    * ``_per_episode_warnings_cache[dataset_id]`` records any feature
      that was declared but whose data isn't actually uniform — the user's
      declaration is being trusted but their data violates it. The
      :class:`DatasetInfo` response surfaces these in ``warnings``.

    Only considers features that are non-image, non-video, non-DEFAULT_FEATURES,
    non-action, non-observation.* — the same gate the staging endpoint uses
    to decide editability.
    """
    if dataset_id in _per_episode_features_cache:
        return _per_episode_features_cache[dataset_id]

    skip_dtypes = {"image", "video"}
    default_features = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

    # Declared `per_episode` hint in the feature spec wins over inference,
    # in BOTH directions. Mirrors the same precedence used in
    # _build_features_schema. Without this, a freshly-added per-frame
    # column (e.g. reward) initialized to a constant fill would look
    # uniform-per-episode and be silently coerced to whole-episode by
    # the staging endpoint — even though the schema says is_per_episode=false.
    declared: set[str] = set()
    declared_not_per_episode: set[str] = set()
    candidate_features: list[str] = []
    for name, ft in dataset.meta.features.items():
        if ft.get("dtype") in skip_dtypes:
            continue
        if name in default_features:
            continue
        if name == "action" or name.startswith("observation."):
            continue
        declared_pe = ft.get("per_episode") if "per_episode" in ft else None
        if declared_pe is False:
            declared_not_per_episode.add(name)
            continue
        # Vectors and matrices can't reasonably be "uniform per episode" — skip.
        shape = ft.get("shape") or [1]
        if len(shape) > 1 or (len(shape) == 1 and shape[0] != 1):
            continue
        if declared_pe is True:
            declared.add(name)
            # Still scan declared features below — to verify the data
            # actually matches the declaration. If it doesn't, we trust
            # the declaration but emit a warning so the user can fix the data.
        candidate_features.append(name)

    source_map: dict[str, str] = dict.fromkeys(declared, "declared")
    warnings: list[str] = []

    if not candidate_features:
        # Still record any declared per-episode features even when there's
        # nothing to infer (e.g. all features are declared one way or the
        # other already).
        _per_episode_features_cache[dataset_id] = set(declared)
        _per_episode_source_cache[dataset_id] = source_map
        _per_episode_warnings_cache[dataset_id] = warnings
        return _per_episode_features_cache[dataset_id]

    data_dir = Path(dataset.root) / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        # No data to scan — trust declared, nothing to detect.
        _per_episode_features_cache[dataset_id] = declared
        _per_episode_source_cache[dataset_id] = source_map
        _per_episode_warnings_cache[dataset_id] = warnings
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
        # Detection scan failed — fall back to declared-only.
        _per_episode_features_cache[dataset_id] = declared
        _per_episode_source_cache[dataset_id] = source_map
        _per_episode_warnings_cache[dataset_id] = warnings
        return _per_episode_features_cache[dataset_id]

    per_episode: set[str] = set(declared)
    for name, nu in nunique_by_feature.items():
        is_uniform = 0 < nu <= 1
        if name in declared:
            # Verify the declaration. If the data isn't uniform, trust the
            # declaration (per the design — declared > detected) but warn so
            # the user knows their data has drifted.
            if not is_uniform:
                msg = (
                    f"Feature {name!r} is declared per_episode=true in info.json, "
                    f"but its data is not uniform within every episode "
                    f"(max nunique-per-episode = {nu}). The GUI is treating it "
                    f"as per-episode anyway — fix the data or remove the flag."
                )
                logger.warning(msg)
                warnings.append(msg)
        else:
            if is_uniform:
                per_episode.add(name)
                source_map[name] = "detected"

    _per_episode_features_cache[dataset_id] = per_episode
    _per_episode_source_cache[dataset_id] = source_map
    _per_episode_warnings_cache[dataset_id] = warnings
    logger.info(
        f"per-episode features for {dataset_id}: declared={sorted(declared)} "
        f"detected={sorted(per_episode - declared)} warnings={len(warnings)}"
    )
    return per_episode


# Hardcoded backend representation of the LeRobot 3.0 subtask format. The
# data layer stores ``subtask_index`` (int64[1]) plus ``meta/subtasks.parquet``
# (index → string lookup). The user always thinks in terms of strings, so
# whenever both pieces are present the schema endpoint synthesizes a
# ``subtask`` (string) feature in place of ``subtask_index``. Stage endpoint
# accepts either name and routes to the storage feature; PendingEdit stores
# the storage name so the apply pipeline doesn't need a special case.
SUBTASK_STORAGE_FEATURE = "subtask_index"
SUBTASK_DISPLAY_FEATURE = "subtask"

# The language instruction has the same storage/display split as subtasks:
# ``task_index`` (int64[1]) plus ``meta/tasks.parquet`` (index → string). Unlike
# subtasks it is present in every LeRobot dataset, and until now only the index
# reached the GUI — so the data view offered `task_index` reading `0` where the
# instruction should be, and the string the policy is conditioned on could not
# be seen while reviewing episodes.
TASK_STORAGE_FEATURE = "task_index"
TASK_DISPLAY_FEATURE = "task"


def _has_subtask_lookup(dataset) -> bool:
    """True if the dataset has a ``meta/subtasks.parquet`` lookup table."""
    return getattr(dataset.meta, "subtasks", None) is not None


def _has_task_lookup(dataset) -> bool:
    """True if the dataset has a ``meta/tasks.parquet`` lookup table."""
    return getattr(dataset.meta, "tasks", None) is not None


def _coerce_optional_float(v: Any) -> float | None:
    """Coerce a value from ``info.json`` to a JSON-friendly Python float.

    Tolerates ``None``, missing, non-numeric, or NaN/inf — returns ``None`` for
    anything we can't safely surface as a slider bound.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math

    if not math.isfinite(f):
        return None
    return f


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
    task_synthesis: bool = False,
    stats: dict | None = None,
    per_episode_source: dict[str, str] | None = None,
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
    per_episode_source = per_episode_source or {}
    out: dict[str, FeatureSchema] = {}
    for name, ft in features.items():
        if subtask_synthesis and name == SUBTASK_STORAGE_FEATURE:
            # Skip the storage entry; it will be replaced by the synthetic
            # display entry below. We don't include both — the user thinks
            # in strings, so exposing both would just leak the storage name.
            continue
        if task_synthesis and name == TASK_STORAGE_FEATURE:
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
        # Declared bounds: optional ``min`` / ``max`` keys in info.json — coerce
        # to JSON-friendly float and tolerate missing/non-numeric values silently.
        decl_min = _coerce_optional_float(ft.get("min"))
        decl_max = _coerce_optional_float(ft.get("max"))
        # Per-episode: declared hint in the feature spec wins over inference,
        # in BOTH directions (explicit false also overrides inferred-true).
        # This matters for freshly-added per-frame features whose constant
        # initial fill would otherwise look like uniform-per-episode and
        # mis-coerce range edits to the whole episode.
        is_per_ep = bool(ft["per_episode"]) if "per_episode" in ft else name in per_episode
        out[name] = FeatureSchema(
            dtype=str(ft.get("dtype", "")),
            shape=shape_list,
            names=names,
            flags=(ft.get("flags") if isinstance(ft, dict) else None),
            derived=bool(ft.get("derived")),
            is_per_episode=is_per_ep or name in per_episode,
            per_episode_source=per_episode_source.get(name),
            observed_min=obs_min,
            observed_max=obs_max,
            declared_min=decl_min,
            declared_max=decl_max,
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
            per_episode_source=per_episode_source.get(SUBTASK_STORAGE_FEATURE),
        )
    if task_synthesis and TASK_STORAGE_FEATURE in features:
        # Per-episode by construction, not by detection. `task_index` is stored
        # per frame, but upstream derives that column from `episode_index` —
        # `modify_tasks` maps episode → one task and rewrites every row, and
        # writes the episodes-table `tasks` array as a single element. So one
        # instruction per episode is the format's contract, not an observation
        # about a particular dataset.
        #
        # It cannot be inherited from `_detect_per_episode_features` either:
        # that detector skips DEFAULT_FEATURES, `task_index` among them, so the
        # lookup is always False and the row would render as a full-width
        # single-color band across the timeline.
        #
        # Intra-episode language is a different mechanism — the
        # `language_persistent` / `language_events` columns in datasets/language.py.
        out[TASK_DISPLAY_FEATURE] = FeatureSchema(
            dtype="string",
            shape=[1],
            names=None,
            is_per_episode=True,
            per_episode_source="declared",
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
    per_episode_source = dict(_per_episode_source_cache.get(dataset_id, {}))
    pe_warnings = _per_episode_warnings_for(dataset_id)
    # Merge declared-but-inconsistent warnings into the per-call ``warnings``
    # list so the GUI surfaces them alongside any other open-time issues.
    warnings_out = list(warnings or []) + pe_warnings
    # Synthesize the user-facing "subtask" string feature only when the
    # dataset has BOTH the storage column AND the lookup table — an
    # incomplete dataset (one but not the other) would not let us decode.
    subtask_synthesis = SUBTASK_STORAGE_FEATURE in dataset.meta.features and _has_subtask_lookup(dataset)
    task_synthesis = TASK_STORAGE_FEATURE in dataset.meta.features and _has_task_lookup(dataset)
    feature_names = list(dataset.meta.features.keys())
    if subtask_synthesis:
        # Mirror the schema synthesis in the legacy `features: list[str]` field
        # so frontends that read that list see "subtask" instead of "subtask_index".
        feature_names = [
            SUBTASK_DISPLAY_FEATURE if n == SUBTASK_STORAGE_FEATURE else n for n in feature_names
        ]
    if task_synthesis:
        feature_names = [TASK_DISPLAY_FEATURE if n == TASK_STORAGE_FEATURE else n for n in feature_names]
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
            task_synthesis=task_synthesis,
            stats=getattr(dataset.meta, "stats", None),
            per_episode_source=per_episode_source,
        ),
        errors=errors or [],
        warnings=warnings_out,
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


class VideoStreamInfo(BaseModel):
    """What a video file actually contains, as opposed to what info.json says."""

    codec: str
    width: int = 0
    height: int = 0
    pix_fmt: str = ""
    fps: float = 0.0
    bitrate_kbps: int = 0


_codec_cache: dict[str, VideoStreamInfo | None] = {}


def _probe_video(path: Path) -> VideoStreamInfo | None:
    """Stream properties for a video file, cached by path+mtime.

    ffprobe costs tens of milliseconds and a file does not change under us
    without its mtime moving, so the cache makes repeat listings free while
    staying correct across a re-encode.

    Post: returns None when the file is missing or unreadable; a probe failure
    must never break the panel that displays it.
    """
    import subprocess

    try:
        key = f"{path}:{path.stat().st_mtime_ns}"
    except OSError:
        return None
    if key in _codec_cache:
        return _codec_cache[key]
    info = None
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt,avg_frame_rate,bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        fields = dict(line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line)
        if fields.get("codec_name"):
            num, _, den = fields.get("avg_frame_rate", "0/1").partition("/")
            fps = float(num) / float(den) if den and float(den) else 0.0
            info = VideoStreamInfo(
                codec=fields["codec_name"],
                width=int(fields.get("width") or 0),
                height=int(fields.get("height") or 0),
                pix_fmt=fields.get("pix_fmt", "") or "",
                fps=round(fps, 3),
                bitrate_kbps=int(int(fields.get("bit_rate") or 0) / 1000),
            )
    except (subprocess.SubprocessError, OSError, ValueError):
        info = None
    if info is not None:
        _codec_cache[key] = info
    return info


def _codecs_by_episode(dataset, episode_indices) -> dict[int, dict[str, VideoStreamInfo]]:
    """Codec per camera for each episode, probing each video file once.

    Episodes share files, so the probe count is the number of distinct
    (camera, chunk, file) triples rather than the number of episodes.

    Pre: call from a worker thread — this runs ffprobe. Post: never raises;
    a dataset whose files cannot be probed simply reports nothing.
    """
    out: dict[int, dict[str, str]] = {}
    try:
        episodes = dataset.meta.episodes
        if episodes is None:
            return out
        cams = list(dataset.meta.camera_keys)
        for i in episode_indices:
            ep = episodes[i]
            per_cam = {}
            for cam in cams:
                chunk = ep.get(f"videos/{cam}/chunk_index")
                fidx = ep.get(f"videos/{cam}/file_index")
                if chunk is None or fidx is None:
                    continue
                path = (
                    Path(dataset.root)
                    / "videos"
                    / cam
                    / f"chunk-{int(chunk):03d}"
                    / f"file-{int(fidx):03d}.mp4"
                )
                stream = _probe_video(path)  # cached by path + mtime
                if stream is not None:
                    per_cam[cam] = stream
            if per_cam:
                out[i] = per_cam
    except Exception:  # noqa: BLE001 - a missing codec must never break the panel
        logger.debug("codec probe failed", exc_info=True)
    return out


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
    video_streams: dict[str, VideoStreamInfo] = {}
    # What this episode's own video files contain, per camera: codec,
    # resolution, pixel format, frame rate, bitrate. Probed rather than read
    # from info.json, which records what the writer intended and can only
    # describe one encoding — after a merge reconciles two, it describes
    # neither.


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
                kind, problems = _check_local_dataset_complete(local_path)
                if kind != "complete":
                    # Carry `kind` so the frontend can say what's wrong and only
                    # offer a Hub download when it's actually a missing-files
                    # case; a metadata inconsistency isn't a download problem.
                    hub_sync_available = kind == "missing_files"
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "incomplete_local_cache",
                            "kind": kind,
                            "message": (
                                "Local dataset cache is incomplete."
                                if hub_sync_available
                                else "Local dataset metadata is inconsistent."
                            ),
                            "problems": problems,
                            "repo_id": repo_id,
                            "local_path": str(local_path),
                            "hub_sync_available": hub_sync_available,
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

        # Crash-recovery: sweep any orphan .tmp files left from an
        # interrupted schema-add or value-edit save before serving the
        # dataset. Fast (one rglob over data/ + meta/) and almost always
        # a no-op on healthy datasets.
        try:
            from lerobot.datasets.dataset_tools import _sweep_orphan_tmp_shards

            removed = _sweep_orphan_tmp_shards(dataset.root)
            if removed:
                logger.info(f"Cleaned {removed} orphan .tmp file(s) for {dataset_id}")
        except Exception as e:
            logger.warning(f"Orphan .tmp sweep failed for {dataset_id}: {e}")

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


# ── Schema-add endpoints (in-place column add) ─────────────────────────


def _migrate_next_success_inplace(dataset) -> None:
    """Collapse per-frame ``next.success`` (bool) into per-episode ``success`` (int8 tri-state).

    ``lerobot-eval`` rollouts on ``main`` write ``next.success`` as a per-frame
    boolean column — typically all-False except the success-transition frame.
    The new ``success`` feature is per-episode int8 tri-state (-1/0/+1). The
    rename machinery only handles same-dtype renames, so this is a small
    bespoke migration.

    Mapping:
      * any frame with ``next.success == True`` → +1 (success)
      * all frames False                        → -1 (failure)

    Then ``next.success`` is dropped. info.json + per-episode stats are
    updated by the underlying ``add_features_inplace`` /
    ``remove_features_inplace`` primitives.

    Pre: ``next.success`` exists in ``dataset.meta.features`` with bool dtype,
    and ``success`` does NOT exist. Caller (``add_default_features``) gates on
    these. No-op for empty datasets — the underlying primitives raise.
    """
    import os

    import numpy as np

    from lerobot.datasets.dataset_tools import (
        _add_new_feature_stats_to_episodes,
        add_features_inplace,
        remove_features_inplace,
    )
    from lerobot.datasets.utils import DATA_DIR

    work_root = Path(dataset.root)
    data_dir = work_root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    # ── Pass 1: compute per-episode tri-state (any-True → +1, else -1) ──
    # next.success may be stored as scalar bool or shape-[1] list-of-bool.
    # Coerce element extraction so both layouts collapse the same way.
    def _scalarize(v: Any) -> bool:
        if hasattr(v, "__len__") and not isinstance(v, (bytes, str)):
            return bool(v[0]) if len(v) else False
        return bool(v)

    per_episode_value: dict[int, int] = {}
    for shard in parquet_files:
        df = pd.read_parquet(shard, columns=["episode_index", "next.success"])
        for ep_idx, group in df.groupby("episode_index"):
            ep_idx_int = int(ep_idx)
            if per_episode_value.get(ep_idx_int) == 1:
                continue  # already a success — multi-shard episode, no need to recheck
            any_true = any(_scalarize(v) for v in group["next.success"].tolist())
            per_episode_value[ep_idx_int] = 1 if any_true else -1

    # ── Pass 2: add `success` int8[1] column with fill=0 ────────────────
    # Reuses existing primitive so info.json + episode stats get the right
    # entries (per_episode=True hint is preserved into info.json).
    add_features_inplace(
        dataset,
        features={
            "success": (
                0,
                {"dtype": "int8", "shape": [1], "names": None, "per_episode": True},
            )
        },
        recompute_stats=False,  # we'll overwrite the column in pass 3 — stats
        # would be wrong if computed off the all-zeros fill
    )

    # ── Pass 3: rewrite each shard's `success` column from per_episode_value ──
    pending_renames: list[tuple[Path, Path]] = []
    try:
        for shard in parquet_files:
            df = pd.read_parquet(shard)
            mapped = df["episode_index"].map(lambda i: per_episode_value.get(int(i), 0))
            df["success"] = pd.Series(mapped.to_numpy(), dtype=np.dtype("int8"))
            tmp = shard.with_suffix(shard.suffix + ".tmp")
            df.to_parquet(tmp, compression="snappy", index=False)
            pending_renames.append((tmp, shard))
    except Exception:
        for tmp, _ in pending_renames:
            if tmp.exists():
                # safe-destruct: our own .tmp file we just wrote in this function
                tmp.unlink()
        raise
    for tmp, final in pending_renames:
        os.replace(tmp, final)

    # ── Pass 4: recompute stats for `success` from the rewritten column ──
    # add_features_inplace was called with recompute_stats=False, so stats
    # are absent. Computing here (after pass 3 wrote correct values) avoids
    # stale stats from the all-zeros initial fill. The helper overwrites
    # existing stats columns; we wrote none, so it just adds them.
    _add_new_feature_stats_to_episodes(work_root, {"success": dataset.meta.features["success"]})

    # ── Pass 5: drop next.success (drops shard column + stats + info.json) ──
    remove_features_inplace(dataset, ["next.success"])


# Default features the GUI offers to add via the banner. Any keys the user's
# dataset is missing get appended in a single ``add_features_inplace`` call;
# this is the convenience layer on top of the generic POST .../features.
_DEFAULT_FEATURE_SPECS = {
    "reward": {
        "fill_value": 0.0,
        # per_episode=false declared explicitly so that the constant 0.0
        # initial fill doesn't get mis-inferred as a per-episode broadcast
        # column. Otherwise drag-select range edits would coerce to the
        # full episode for reward, which is the opposite of what users want.
        "info": {"dtype": "float32", "shape": [1], "names": None, "per_episode": False},
        # Existing column names that should be preserved (renamed, not
        # discarded) when present in the dataset. Renaming requires the
        # source column to be type-compatible with the destination spec
        # (same dtype + shape after spec_overrides applied).
        "rename_from": ["next.reward"],
    },
    "success": {
        "fill_value": 0,
        "info": {"dtype": "int8", "shape": [1], "names": None, "per_episode": True},
        # next.success (bool, per-frame, written by lerobot-eval rollouts)
        # can't be a plain rename target — dtype + cardinality both differ
        # (per-frame bool → per-episode int8 tri-state). Migration is
        # handled separately by _migrate_next_success_inplace, invoked from
        # add_default_features before the rename/add planning loop.
        "rename_from": [],
    },
}


class AddFeatureRequest(BaseModel):
    """Body for ``POST /api/datasets/{id}/features``.

    The ``fill_value`` is auto-typed by ``add_features_inplace`` based on
    ``dtype``; bool/int/float strings sent from JS get coerced. Only V1
    dtypes are currently editable in the GUI: bool, int8, int64, float32,
    string. Vector / image / video features can be added but won't have
    an editable widget yet.
    """

    name: str
    dtype: str
    shape: list[int] = [1]
    per_episode: bool = False
    fill_value: Any = 0
    flags: list[str] | None = None
    # Label vocabulary for a bitset column (feature_utils.is_flags_feature).
    # When present the column is int64[1] and bit i means flags[i], so several
    # labels can hold on one frame — which is the point, since labels from
    # different passes overlap.


class AddFeatureResponse(BaseModel):
    added: list[str]
    info: DatasetInfo
    # Optional: renames performed (when /features/defaults reuses an
    # existing column under a different name instead of adding a duplicate).
    # Each entry is "<old_name>→<new_name>". Empty for the generic
    # POST /features endpoint.
    renamed: list[str] = []


def _refresh_dataset_after_schema_change(dataset_id: str) -> None:
    """Drop schema-bound caches after a schema mutation.

    Keeps in-memory state in sync with the new ``info.json`` / parquet
    contents. ``_dataset_info_mtime`` is popped so the next dataset-open
    call re-detects the schema; ``_per_episode_features_cache`` and
    ``_episode_start_indices`` are cleared via the existing helper.
    ``add_features_inplace`` already replaces ``dataset.meta`` in place,
    so future ``_dataset_info_from`` calls will see the new schema.
    """
    from lerobot.gui.cache_invalidation import invalidate_caches

    _dataset_info_mtime.pop(dataset_id, None)
    invalidate_caches(_app_state, dataset_id, invalidate_episode_indices=_invalidate_episode_start_indices)


@router.post("/{dataset_id:path}/features", response_model=AddFeatureResponse)
async def add_dataset_feature(dataset_id: str, body: AddFeatureRequest) -> AddFeatureResponse:
    """Add one new feature column to the dataset in place.

    Refuses (409) if any pending ``feature_set`` edits exist for the
    dataset — the user must Save or Discard them first to avoid mixing
    schema and value mutations on the same parquet shards.
    """
    from lerobot.datasets.dataset_tools import add_features_inplace
    from lerobot.utils.constants import DEFAULT_FEATURES

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    if body.name in DEFAULT_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"'{body.name}' is a reserved DEFAULT_FEATURE",
        )
    if body.name in _DEFAULT_FEATURE_SPECS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{body.name}' is a default feature — use POST /features/defaults "
                "(banner) instead of the generic dialog."
            ),
        )

    pending = _app_state.pending_feature_set_edits_for_dataset(dataset_id)
    if pending:
        raise HTTPException(
            status_code=409,
            detail=f"{len(pending)} pending feature edits — Save or Discard them first",
        )

    async with _app_state.get_lock(dataset_id):
        dataset = _app_state.datasets[dataset_id]
        if body.name in dataset.meta.features:
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{body.name}' already exists in dataset",
            )

        # Always write the declared per_episode value (including False) so
        # the FeatureSchema construction picks it up over the inference
        # fallback. A constant initial fill would otherwise be mis-inferred
        # as per-episode-uniform and silently coerce range edits to whole
        # episodes for non-per-episode columns.
        info = {
            "dtype": body.dtype,
            "shape": list(body.shape),
            "names": None,
            "per_episode": bool(body.per_episode),
        }
        if body.flags is not None:
            from lerobot.datasets.feature_utils import MAX_FLAGS

            labels = [f.strip() for f in body.flags if f and f.strip()]
            if not labels:
                raise HTTPException(status_code=400, detail="A label column needs at least one label")
            if len(labels) != len(set(labels)):
                dupes = sorted({x for x in labels if labels.count(x) > 1})
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate label(s) {dupes}: a bit index must name one label",
                )
            if len(labels) > MAX_FLAGS:
                raise HTTPException(
                    status_code=400,
                    detail=f"{len(labels)} labels exceeds the {MAX_FLAGS} bits an int64 holds",
                )
            # The storage shape is dictated by the contract, not chosen: a
            # bitset is one integer per frame.
            info.update({"dtype": "int64", "shape": [1], "flags": labels})

        try:
            add_features_inplace(dataset, features={body.name: (body.fill_value, info)})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception(f"add_features_inplace failed for {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        _refresh_dataset_after_schema_change(dataset_id)
        return AddFeatureResponse(added=[body.name], info=_dataset_info_from(dataset_id, dataset))


def _compatible_for_rename(existing_spec: dict, target_spec: dict) -> bool:
    """True if `existing_spec` is type-compatible with the target default spec.

    Renaming preserves the on-disk values, so the source column's dtype
    and shape must already match what the target spec declares; otherwise
    we'd silently misadvertise the data type. Optional spec keys (names,
    per_episode) are intentionally ignored — those can be overridden via
    spec_overrides during the rename.
    """
    if existing_spec.get("dtype") != target_spec.get("dtype"):
        return False
    es = list(existing_spec.get("shape") or [])
    ts = list(target_spec.get("shape") or [])
    return es == ts


class FlagLabelRequest(BaseModel):
    """Body for appending to / renaming within a flags vocabulary."""

    label: str


def _flags_spec_for_edit(dataset_id: str, feature_name: str) -> tuple[Any, dict]:
    """The dataset and feature spec, or the reason this vocabulary is off limits.

    Pre: caller holds no dataset lock. Post: returns ``(dataset, spec)`` for a
    non-derived flags feature.

    Raises:
        HTTPException: dataset or feature missing, not a bitset, or derived —
            a derived vocabulary belongs to the code that computes the values.
    """
    from lerobot.datasets.feature_utils import is_derived_feature, is_flags_feature

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = _app_state.datasets[dataset_id]
    spec = dataset.meta.features.get(feature_name)
    if not isinstance(spec, dict) or not is_flags_feature(spec):
        raise HTTPException(status_code=400, detail=f"'{feature_name}' is not a label column")
    if is_derived_feature(spec):
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{feature_name}' is computed from the recorded data — its labels are "
                "defined by whatever produces them. Add hand labels to a column of your own."
            ),
        )
    return dataset, spec


def _write_flags_vocabulary(dataset, dataset_id: str, feature_name: str, labels: list[str]) -> None:
    """Persist a new vocabulary for ``feature_name``.

    Only ``meta/info.json`` is rewritten: a vocabulary change that keeps every
    existing bit at its existing index does not change a single stored value,
    which is what makes appending and renaming cheap enough to do mid-session.
    """
    import json

    info_path = Path(dataset.root) / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][feature_name]["flags"] = labels
    info_path.write_text(json.dumps(info, indent=4))
    dataset.meta.features[feature_name]["flags"] = labels
    _refresh_dataset_after_schema_change(dataset_id)


@router.post("/{dataset_id:path}/features/{feature_name}/flags", response_model=AddFeatureResponse)
async def append_flag_label(dataset_id: str, feature_name: str, body: FlagLabelRequest) -> AddFeatureResponse:
    """Append a label to a flags column, taking the next unused bit."""
    from lerobot.datasets.feature_utils import MAX_FLAGS

    dataset, spec = _flags_spec_for_edit(dataset_id, feature_name)
    label = body.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label cannot be empty")
    labels = list(spec["flags"])
    if label in labels:
        raise HTTPException(status_code=400, detail=f"'{label}' is already bit {labels.index(label)}")
    if len(labels) >= MAX_FLAGS:
        raise HTTPException(
            status_code=400, detail=f"{feature_name} already uses all {MAX_FLAGS} bits of an int64"
        )
    labels.append(label)
    async with _app_state.get_lock(dataset_id):
        _write_flags_vocabulary(dataset, dataset_id, feature_name, labels)
    logger.info(f"Appended flag {label!r} as bit {len(labels) - 1} of {feature_name}")
    return AddFeatureResponse(added=[label], info=_dataset_info_from(dataset_id, dataset))


@router.patch("/{dataset_id:path}/features/{feature_name}/flags/{bit}", response_model=AddFeatureResponse)
async def rename_flag_label(
    dataset_id: str, feature_name: str, bit: int, body: FlagLabelRequest
) -> AddFeatureResponse:
    """Rename the label at ``bit``, leaving which bit it is alone.

    Stored values are untouched, so this is safe to do at any point — it is the
    intended fix for a mistyped label, since deleting one is not offered.
    """
    dataset, spec = _flags_spec_for_edit(dataset_id, feature_name)
    label = body.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label cannot be empty")
    labels = list(spec["flags"])
    if not 0 <= bit < len(labels):
        raise HTTPException(
            status_code=400, detail=f"{feature_name} declares bits 0…{len(labels) - 1}, not {bit}"
        )
    if label in labels and labels.index(label) != bit:
        raise HTTPException(status_code=400, detail=f"'{label}' is already bit {labels.index(label)}")
    previous = labels[bit]
    labels[bit] = label
    async with _app_state.get_lock(dataset_id):
        _write_flags_vocabulary(dataset, dataset_id, feature_name, labels)
    logger.info(f"Renamed bit {bit} of {feature_name}: {previous!r} -> {label!r}")
    return AddFeatureResponse(
        added=[label], renamed=[f"{previous}→{label}"], info=_dataset_info_from(dataset_id, dataset)
    )


@router.post("/{dataset_id:path}/features/defaults", response_model=AddFeatureResponse)
async def add_default_features(dataset_id: str) -> AddFeatureResponse:
    """Reconcile the dataset's schema against the default features.

    For each missing default:
      * if a known alternate column exists with a compatible dtype/shape
        (e.g. ``next.reward`` for ``reward``), rename it in place — the
        existing recorded values are preserved instead of being shadowed
        by a fresh all-zeros column.
      * otherwise, add a new column with the default fill value.

    Idempotent: returns ``added=[], renamed=[]`` when nothing's needed.
    Used by the banner shown on dataset open.
    """
    from lerobot.datasets.dataset_tools import add_features_inplace, rename_features_inplace

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    pending = _app_state.pending_feature_set_edits_for_dataset(dataset_id)
    if pending:
        raise HTTPException(
            status_code=409,
            detail=f"{len(pending)} pending feature edits — Save or Discard them first",
        )

    async with _app_state.get_lock(dataset_id):
        dataset = _app_state.datasets[dataset_id]

        # Special-case migration: next.success (bool, per-frame) → success
        # (int8 tri-state, per-episode). Rename machinery rejects dtype
        # changes, so this is a bespoke transform run before the planner.
        # Gated on next.success being bool to avoid stomping unexpected dtypes.
        migrated_next_success = False
        if (
            "next.success" in dataset.meta.features
            and "success" not in dataset.meta.features
            and dataset.meta.features["next.success"].get("dtype") == "bool"
        ):
            try:
                _migrate_next_success_inplace(dataset)
                migrated_next_success = True
            except Exception as e:
                logger.exception(f"_migrate_next_success_inplace failed for {dataset_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e

        # Plan: split missing defaults into renames (alternate exists +
        # compatible) vs adds (everything else).
        renames: dict[str, str] = {}
        rename_overrides: dict[str, dict] = {}
        to_add: dict[str, tuple] = {}

        for name, spec in _DEFAULT_FEATURE_SPECS.items():
            if name in dataset.meta.features:
                continue
            target_info = dict(spec["info"])
            picked_alternate = None
            for alt in spec.get("rename_from", []):
                if alt in dataset.meta.features and _compatible_for_rename(
                    dataset.meta.features[alt], target_info
                ):
                    picked_alternate = alt
                    break
            if picked_alternate is not None:
                renames[picked_alternate] = name
                # The alternate may not declare per_episode (e.g. legacy
                # next.reward); merge in the default's hint so the
                # renamed column gets the right is_per_episode behavior.
                rename_overrides[name] = {
                    k: v for k, v in target_info.items() if k in ("per_episode", "names")
                }
            else:
                to_add[name] = (spec["fill_value"], target_info)

        if not renames and not to_add and not migrated_next_success:
            return AddFeatureResponse(added=[], renamed=[], info=_dataset_info_from(dataset_id, dataset))

        try:
            # Renames first so the new names are present before any
            # subsequent add validates against an updated schema.
            if renames:
                rename_features_inplace(dataset, renames=renames, spec_overrides=rename_overrides)
            if to_add:
                add_features_inplace(dataset, features=to_add)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception(f"add_default_features failed for {dataset_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        _refresh_dataset_after_schema_change(dataset_id)
        renamed_pairs = [f"{old}→{new}" for old, new in renames.items()]
        if migrated_next_success:
            renamed_pairs.append("next.success→success")
        return AddFeatureResponse(
            added=sorted(to_add.keys()),
            renamed=sorted(renamed_pairs),
            info=_dataset_info_from(dataset_id, dataset),
        )


class RemoveFeatureResponse(BaseModel):
    removed: list[str]
    info: DatasetInfo


@router.delete(
    "/{dataset_id:path}/features/{feature_name}",
    response_model=RemoveFeatureResponse,
)
async def remove_dataset_feature(dataset_id: str, feature_name: str) -> RemoveFeatureResponse:
    """Drop one feature column from the dataset in place.

    Refuses (400) for ``DEFAULT_FEATURES`` and image/video features
    (their on-disk filenames also encode the feature name — would need
    the forked ``remove_feature`` and a video re-encode).
    Refuses (409) when pending ``feature_set`` edits exist on the
    dataset, same as the schema-add path.
    """
    from lerobot.datasets.dataset_tools import remove_features_inplace
    from lerobot.utils.constants import DEFAULT_FEATURES

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    if feature_name in DEFAULT_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"'{feature_name}' is a reserved DEFAULT_FEATURE",
        )

    pending = _app_state.pending_feature_set_edits_for_dataset(dataset_id)
    if pending:
        raise HTTPException(
            status_code=409,
            detail=f"{len(pending)} pending feature edits — Save or Discard them first",
        )

    async with _app_state.get_lock(dataset_id):
        dataset = _app_state.datasets[dataset_id]
        if feature_name not in dataset.meta.features:
            raise HTTPException(
                status_code=404,
                detail=f"Feature '{feature_name}' not found in dataset",
            )
        try:
            remove_features_inplace(dataset, names=feature_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception(f"remove_features_inplace failed for {dataset_id}/{feature_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

        _refresh_dataset_after_schema_change(dataset_id)
        return RemoveFeatureResponse(removed=[feature_name], info=_dataset_info_from(dataset_id, dataset))


def forget_dataset(dataset_id: str) -> None:
    """Drop ``dataset_id`` from the registry and every cache keyed on it.

    Pre: ``dataset_id`` is present in ``_app_state.datasets``.
    Post: no registry entry, cached episode index, per-dataset lock, staged
    edit, or frame-cache entry references it, and the opened-state file on disk
    no longer lists it.

    Shared by close and delete-from-disk. Deleting the files while any of this
    survived would leave the registry pointing at a directory that is gone.
    """
    del _app_state.datasets[dataset_id]
    _dataset_info_mtime.pop(dataset_id, None)
    # Drop every dataset-scoped cache + per-dataset lock. Without this, each
    # close/re-open cycle left behind cached episode indices, per-episode
    # action stats, the asyncio.Lock, and any frame-cache entries — a slow
    # leak in long-running sessions and a correctness hazard if the same
    # dataset_id is later re-opened against different on-disk content.
    _invalidate_episode_start_indices(dataset_id)
    from lerobot.gui.cache_invalidation import invalidate_caches

    invalidate_caches(_app_state, dataset_id)
    _app_state.clear_edits(dataset_id)
    _app_state.discard_lock(dataset_id)
    _save_opened_state()


@router.delete("/{dataset_id:path}")
async def close_dataset(dataset_id: str) -> dict[str, str]:
    """Close a dataset. The files on disk are untouched."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    if _app_state.is_locked(dataset_id):
        raise HTTPException(status_code=423, detail="Dataset is busy (operation in progress)")

    forget_dataset(dataset_id)
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

    # Probing runs ffprobe, so it goes to the bounded pool rather than the
    # loop. Distinct video files only, cached by path+mtime, so this costs
    # something on the first listing of a dataset and nothing afterwards.
    import asyncio

    codecs_by_ep = await asyncio.get_event_loop().run_in_executor(
        _flags_impact_executor,
        _codecs_by_episode,
        dataset,
        range(dataset.meta.total_episodes),
    )

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
                video_streams=codecs_by_ep.get(i, {}),
            )
        )

    return result


@router.get("/{dataset_id:path}/episodes/{episode_idx}/frame/{frame_idx}")
async def get_frame(
    dataset_id: str,
    episode_idx: int,
    frame_idx: int,
    camera: str | None = None,
    profile: str = "full",
) -> Response:
    """Get a single frame as JPEG.

    Args:
        dataset_id: Dataset identifier
        episode_idx: Episode index
        frame_idx: Frame index within the episode
        camera: Camera key (optional, returns first camera if not specified)
        profile: Still-frame profile (frame_cache.STILL_PROFILES) — "low" and
            "medium" downscale before encoding. Defaults to "full" (source
            resolution) so callers that predate the option are unaffected.
    """
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # A misspelled profile falls back to the source resolution otherwise —
    # the most expensive variant — for a caller that asked for a cheap one.
    from lerobot.gui.frame_cache import STILL_PROFILES

    if profile not in STILL_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile {profile!r}; expected one of {sorted(STILL_PROFILES)}",
        )

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

    import asyncio
    import time

    # Check if this camera is already cached (cheap lock-protected dict lookup).
    jpeg_bytes = _app_state.frame_cache.get(dataset_id, episode_idx, frame_idx, camera_key, profile)

    if jpeg_bytes is None:
        # Cache miss: do the heavy decode+encode work off the event loop.
        # Otherwise every scrub on a long video stalls FastAPI's loop and
        # cascades into stuck SSE keepalives + delayed concurrent requests.
        from lerobot.gui.frame_cache import encode_frame_for_profile

        def _decode_and_cache() -> bytes:
            # Re-check the JPEG cache inside the worker. Multiple browser
            # requests for the same frame (one per camera) all hit cache-
            # miss outside, all submit to this single-worker executor, and
            # the first one to run decodes ALL cameras and caches them.
            # The 2nd .. Nth submissions wake up, find the cache populated,
            # and return immediately without redundant decode work.
            cached = _app_state.frame_cache.get(dataset_id, episode_idx, frame_idx, camera_key, profile)
            if cached is not None:
                return cached

            if frame_idx < ep_length:
                # Normal frame — decode via dataset[global_idx] (all cameras at once)
                episode_start = _get_episode_start_index(dataset_id, episode_idx)
                global_idx = episode_start + frame_idx

                t0 = time.perf_counter()
                item = dataset[global_idx]
                t1 = time.perf_counter()

                # Cache all cameras from this single decode
                primary: bytes | None = None
                for cam in camera_keys:
                    if cam in item:
                        cam_jpeg = encode_frame_for_profile(item[cam], profile)
                        _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, cam, cam_jpeg, profile)
                        if cam == camera_key:
                            primary = cam_jpeg
                t2 = time.perf_counter()

                if primary is None:
                    # Fallback when the requested camera isn't in camera_keys.
                    primary = encode_frame_for_profile(item[camera_key], profile)
                    _app_state.frame_cache.put(
                        dataset_id, episode_idx, frame_idx, camera_key, primary, profile
                    )
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

                primary = encode_frame_for_profile(frames[0], profile)
                _app_state.frame_cache.put(dataset_id, episode_idx, frame_idx, camera_key, primary, profile)
                t2 = time.perf_counter()

            decode_ms = (t1 - t0) * 1000
            encode_ms = (t2 - t1) * 1000
            logger.info(
                f"get_frame ep={episode_idx} frame={frame_idx} cam={camera_key}: "
                f"decode={decode_ms:.1f}ms encode={encode_ms:.1f}ms"
            )
            return primary

        jpeg_bytes = await asyncio.get_event_loop().run_in_executor(_decode_executor, _decode_and_cache)
    else:
        logger.debug(f"get_frame ep={episode_idx} frame={frame_idx} cam={camera_key}: cache hit")

    # Trigger background prefetching for this episode, starting from the current frame
    _maybe_start_prefetch(
        dataset_id, episode_idx, ep_length, start_frame=min(frame_idx, ep_length - 1), profile=profile
    )

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


def _resolve_dataset_urdf_spec(dataset_id: str, episode_idx: int):
    """Validate args + resolve the URDF viz spec from this dataset's schema.

    Returns ``(dataset, episode_row, state_names, action_names_or_None, spec)``
    or ``None`` when no description matches the dataset's motor set.
    Raises 404 ``HTTPException`` for unknown dataset / episode (everything
    else returns ``None`` to surface as ``{"available": false}``).
    """
    from lerobot.gui.urdf_viz import resolve_robot

    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = _app_state.datasets[dataset_id]
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")
    ep = dataset.meta.episodes[episode_idx]

    feature_dict = dataset.meta.features
    state_feat = feature_dict.get("observation.state")
    if state_feat is None or "names" not in state_feat:
        return None
    state_names: list[str] = list(state_feat["names"])
    spec = resolve_robot(state_names)
    if spec is None:
        return None
    action_feat = feature_dict.get("action")
    action_names = list(action_feat["names"]) if action_feat and "names" in action_feat else None
    return dataset, ep, state_names, action_names, spec


@router.get("/{dataset_id:path}/episodes/{episode_idx}/urdf-viz/meta")
async def get_urdf_viz_dataset_meta(dataset_id: str, episode_idx: int) -> dict:
    """One-shot identity + advertised sources for a dataset's URDF viewer.

    Sibling of the live ``/api/run/urdf-viz/meta``. ``sources`` lists what
    the dataset's schema can serve — ``"state"`` whenever ``observation.state``
    is present and matches a vendored description; ``"action"`` when an
    ``action`` feature is also there. The frontend fetches this once when
    the viewer mounts, then polls per-source via the sibling endpoint.
    """
    resolved = _resolve_dataset_urdf_spec(dataset_id, episode_idx)
    if resolved is None:
        return {"available": False}
    _, _, _, action_names, spec = resolved
    sources = ["state"]
    if action_names:
        sources.append("action")
    return {
        "available": True,
        "name": spec.name,
        "urdf": f"/urdf-assets/{spec.urdf_url_path}",
        # Mirrored-arm robots (e.g. OpenArm) ship a separate right-arm URDF;
        # None means both arms load ``urdf``.
        "urdf_right": f"/urdf-assets/{spec.urdf_url_path_right}" if spec.urdf_url_path_right else None,
        # Per-arm base offsets (URDF world frame) from the description;
        # None lets the frontend use its default side-by-side spacing.
        "base_offsets": spec.base_offsets,
        "bimanual": len(spec.arms) == 2,
        "sources": sources,
        # ee_link is None for descriptions that didn't declare one; the
        # frontend skips polyline rendering in that case.
        "ee_link": spec.ee_link,
    }


@router.get("/{dataset_id:path}/episodes/{episode_idx}/urdf-viz")
async def get_urdf_viz_dataset_source(
    dataset_id: str,
    episode_idx: int,
    frame: int = 0,
    source: str = "state",
    horizon: int = 1,
) -> dict:
    """Per-arm URDF joint angles for one named source at one frame.

    Mirrors the live endpoint's shape (``arms[*].frames[]``). When
    ``horizon`` is 1 (the default) ``frames`` has length 1 — the single
    recorded pose at ``frame``. With ``horizon`` > 1 the endpoint returns
    up to ``horizon`` consecutive frames starting at ``frame`` (clipped at
    the end of the episode), enabling the frontend to draw an EE-trajectory
    polyline through the future poses. ``source`` is one of the names from
    ``/urdf-viz/meta``.
    """
    from lerobot.gui.urdf_viz import compute_joint_angles

    if horizon < 1:
        raise HTTPException(status_code=400, detail=f"horizon must be >= 1 (got {horizon})")

    resolved = _resolve_dataset_urdf_spec(dataset_id, episode_idx)
    if resolved is None:
        return {"available": False}
    dataset, ep, state_names, action_names, spec = resolved

    ep_length = int(ep["length"])
    if frame < 0 or frame >= ep_length:
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frame} out of range for episode {episode_idx} (length={ep_length})",
        )

    if source == "state":
        col, names = "observation.state", state_names
    elif source == "action":
        if not action_names:
            return {"available": False}
        col, names = "action", action_names
    else:
        raise HTTPException(status_code=400, detail=f"unknown source: {source!r}")

    # Pull only the column we need, scoped to this episode. Mirrors the
    # feature-series endpoint: avoids decoding any video features the full
    # ``dataset[i]`` accessor would pull in.
    chunk_idx = int(ep["data/chunk_index"])
    file_idx = int(ep["data/file_index"])
    parquet_path = Path(dataset.root) / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail=f"Data parquet missing: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["episode_index", col])
    df = df[df["episode_index"] == episode_idx].reset_index(drop=True)
    if frame >= len(df):
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frame} not present in parquet (have {len(df)} rows)",
        )

    # Clip the horizon at the end of the available rows. The renderer is
    # fine with shorter-than-requested trajectories (just a shorter line).
    end = min(frame + horizon, len(df))
    arms_payload: list[dict] = [{"prefix": a.obs_prefix, "frames": []} for a in spec.arms]
    for i in range(frame, end):
        vec = df[col].iloc[i]
        sample = {n: float(vec[j]) for j, n in enumerate(names) if j < len(vec)}
        angles = compute_joint_angles(spec, sample)
        for k, a in enumerate(spec.arms):
            arms_payload[k]["frames"].append({"joints": angles.get(a.obs_prefix, {})})
    return {"available": True, "arms": arms_payload}


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


# --------------------------------------------------------------------------
# Segmentation masks
#
# Whole episode in ONE response, deliberately. The live overlay path costs a
# publish POST plus a pull GET per displayed frame, so at the ~240 ms RTT an
# operator on the other side of the world actually has, it tops out near two
# frames per second no matter how fast the segmenter is. Masks stored as a
# feature are already computed, so the client can take the episode in a single
# round trip and scrub locally at full speed.
#
# The payload is gzipped here rather than by middleware: RLE is repetitive
# ASCII and compresses several-fold, and this is the one response whose size is
# dominated by that.
# --------------------------------------------------------------------------


def _mask_features(dataset) -> dict[str, dict]:
    """Every declared mask column, by feature key."""
    return {
        name: ft
        for name, ft in dataset.meta.features.items()
        if ft.get("mask_encoding") == "coco_rle"
    }


@router.get("/{dataset_id:path}/episodes/{episode_idx}/masks")
async def get_episode_masks(dataset_id: str, episode_idx: int, camera: str = "") -> Response:
    """Return every stored mask for one episode, gzipped, in one response.

    Pre: the dataset declares at least one column with ``mask_encoding`` set;
    ``camera`` optionally narrows to a single mask feature key or its short name.

    Post: ``{episode_index, length, from_index, cameras: {key: {labels, size,
    encoding, frames}}}`` where ``frames`` has one entry per episode frame, in
    order, each a list of ``[label_id, rle]`` pairs. An empty list means the
    frame was segmented and nothing was found — distinct from a missing column.
    """
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = _app_state.datasets[dataset_id]
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    wanted = _mask_features(dataset)
    if camera:
        short = camera.rsplit(".", 1)[-1]
        wanted = {k: v for k, v in wanted.items() if k == camera or k.rsplit(".", 1)[-1] == short}
    if not wanted:
        raise HTTPException(status_code=404, detail="No mask features on this dataset")

    start = int(dataset.meta.episodes["dataset_from_index"][episode_idx])
    length = int(dataset.meta.episodes["length"][episode_idx])

    def _build() -> bytes:
        cameras: dict[str, Any] = {}
        for key, ft in wanted.items():
            column = dataset.hf_dataset[key][start : start + length]
            frames = []
            for cell in column:
                raw = cell[0] if isinstance(cell, (list, tuple)) else cell
                frames.append(json.loads(raw) if raw else [])
            cameras[key] = {
                "labels": ft.get("mask_labels", []),
                "size": ft.get("mask_size"),
                "encoding": ft.get("mask_encoding"),
                "frames": frames,
            }
        body = {
            "episode_index": episode_idx,
            "length": length,
            "from_index": start,
            "cameras": cameras,
        }
        return gzip.compress(json.dumps(body, separators=(",", ":")).encode(), compresslevel=6)

    # Reading a whole episode's column and gzipping it is real CPU work; keep it
    # off the event loop, on the pool that already serves dataset decodes.
    payload = await asyncio.get_event_loop().run_in_executor(_decode_executor, _build)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"},
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
    import asyncio

    from lerobot.gui.frame_cache import encode_frame_to_jpeg

    # Calculate episode start index (cumulative sum, not per-file offset)
    episode_start = _get_episode_start_index(dataset_id, episode_idx)

    def _decode_one_frame(i: int) -> bytes:
        """Decode + encode one frame on cache miss. Caches every camera in
        the decoded item to amortise the multi-camera read cost.

        Runs on the dedicated ``_decode_executor`` (single worker), so
        only one thread is ever inside the underlying torchcodec /
        libdav1d decoder at a time — see the module-level comment on
        ``_decode_executor`` for the libdav1d thread-safety background.
        """
        # Re-check the JPEG cache: a sibling request for the same frame
        # (different camera) may have already populated all cameras via
        # the side-effect cache writes below. If it did, skip the redundant
        # decode entirely.
        cached = _app_state.frame_cache.get(dataset_id, episode_idx, i, camera_key)
        if cached is not None:
            return cached

        global_idx = episode_start + i
        item = dataset[global_idx]
        primary: bytes | None = None
        for cam in camera_keys:
            if cam in item:
                cam_jpeg = encode_frame_to_jpeg(item[cam])
                _app_state.frame_cache.put(dataset_id, episode_idx, i, cam, cam_jpeg)
                if cam == camera_key:
                    primary = cam_jpeg
        if primary is None:
            primary = encode_frame_to_jpeg(item[camera_key])
            _app_state.frame_cache.put(dataset_id, episode_idx, i, camera_key, primary)
        return primary

    loop = asyncio.get_event_loop()
    frames = []
    for i in range(start, min(start + count, ep_length)):
        # Check cache first (cheap)
        jpeg_bytes = _app_state.frame_cache.get(dataset_id, episode_idx, i, camera_key)
        if jpeg_bytes is None:
            # Decode-encode work is multi-ms per miss; push off the event loop.
            jpeg_bytes = await loop.run_in_executor(_decode_executor, _decode_one_frame, i)
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
    del dataset_id  # URL-scoped for symmetry with sibling routes; cache is global.
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
    # Open the log file, hand it to the subprocess as stdout/stderr, then
    # close the parent's copy. The subprocess inherits its own FD via fork,
    # so it can still write to the file. Without the close(), every viz
    # launch leaks one FD in the GUI server process until garbage
    # collection runs — and the server is long-lived.
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - handed to subprocess; closed below
    try:
        try:
            subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to launch visualizer: {e}") from e
    finally:
        log_fh.close()
    logger.info(f"Rerun viz log: {log_path}")

    return {"status": "ok", "message": f"Launched Rerun for episode {episode_idx}"}


# ---------------------------------------------------------------------------
# HuggingFace Hub operations
# ---------------------------------------------------------------------------


@router.get("/hub/auth-status")
async def hub_auth_status():
    """Check if the user is logged in to HuggingFace Hub.

    Runs in a worker thread: ``whoami()`` is a synchronous network call,
    and when the Hub is unreachable (DNS poisoning, firewall blackhole)
    the TCP connect hangs until kernel timeouts — which would freeze the
    whole event loop (static files, websockets, everything) if done inline.
    """
    import asyncio

    from lerobot.gui.api._hub_core import get_auth_status

    return await asyncio.to_thread(get_auth_status)


@router.post("/hub/open-job-folder")
async def hub_open_job_folder() -> dict:
    """Open the per-job IPC directory in the GUI host's file manager.

    Same pattern as ``/open-in-files``, but the path is fixed to the
    hub-jobs directory so the frontend doesn't have to know (or be
    trusted with) the absolute path. The directory contains one trio of
    files per job (``<job_id>.json``, ``.log``, ``.pid``); from there
    the user can open the ``.log`` to read the raw HF output that drove
    the milestone string they saw in the Transfers tray.

    Caveat: this opens on the *server*, not the frontend machine — same
    constraint as the existing Open-data-directory affordance. Useful
    when the GUI is running locally; degrades gracefully (xdg-open fails
    with a clean 500) on a headless server.
    """
    import asyncio
    import subprocess as _subprocess

    from lerobot.gui.hub_jobs import JOBS_DIR

    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def _spawn() -> None:
        _subprocess.Popen(["xdg-open", str(JOBS_DIR)])  # nosec B607 - standard Linux file-opener

    try:
        await asyncio.get_event_loop().run_in_executor(None, _spawn)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="xdg-open not found") from e

    return {"status": "ok", "path": str(JOBS_DIR)}


@router.get("/hub/repo-info")
async def hub_repo_info(repo_id: str, repo_type: str = "dataset"):
    """Get info about a repo on HuggingFace Hub.

    ``repo_type`` selects the namespace — models and datasets are separate ID
    spaces, so a model looked up as a dataset reports "not found" for a repo
    that exists.

    Threaded for the same reason as ``/hub/auth-status`` — the sync HF
    call must not block the event loop when the network stalls.
    """
    import asyncio

    from lerobot.gui.api._hub_core import get_repo_info

    return await asyncio.to_thread(get_repo_info, repo_id, repo_type)


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
        import asyncio

        from huggingface_hub import HfApi

        api = HfApi()
        # Threaded: sync network call; a stalled Hub connection must not
        # freeze the event loop.
        info = await asyncio.to_thread(api.dataset_info, target_repo_id, files_metadata=True)
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
    # If True, bypasses the completeness check that warns about uploading
    # from a local copy missing files present on the remote. Used by the
    # frontend's "Upload anyway" follow-up after the user sees the warning.
    confirm_force: bool = False
    # Route this transfer through classic LFS instead of Xet. Per-job rather
    # than a process-wide env var because it is a property of the network
    # path, not of the installation: on a link where the Xet CAS endpoints
    # stall, a 200 MB upload that never completed via Xet finished in 405 s
    # over LFS at full link speed. The cost is losing Xet's chunk-level
    # dedup, so a re-upload of an edited dataset resends whole changed files.
    disable_xet: bool = False


class HubDownloadRequest(BaseModel):
    repo_id: str | None = None  # If None, uses dataset's repo_id


def _verify_hub_auth() -> None:
    """Raise HTTPException 401 if not logged in to the Hub.

    Cheap (single GET to whoami); we check before kicking off a background
    job so the frontend can surface "not logged in" synchronously rather
    than via a delayed job-failed status in the Transfers tray.
    """
    try:
        from huggingface_hub import HfApi

        HfApi().whoami()
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Not logged in to HuggingFace Hub. Run `huggingface-cli login` in terminal.",
        ) from e


# Async lock keyed by dataset_id to serialise the active-job check and
# the job-registration step for the same dataset. Two rapid Upload clicks
# do NOT both run: the first registers a job and spawns a worker; the
# second acquires the lock, finds the just-registered job via
# active_hub_job_for(), and is rejected with 409 carrying the first's
# job_id (so both tabs/clicks end up watching the same Transfers card).
# Lock is held only across check + registration; the actual transfer
# runs in a subprocess outside the lock so the GUI server stays responsive.
_hub_spawn_locks: dict[str, Any] = {}  # values are asyncio.Lock


def _hub_spawn_lock_for(dataset_id: str):
    """Get-or-create the spawn lock for a dataset."""
    import asyncio

    lock = _hub_spawn_locks.get(dataset_id)
    if lock is None:
        lock = asyncio.Lock()
        _hub_spawn_locks[dataset_id] = lock
    return lock


def _refresh_progress_from_file(job) -> None:
    """Merge the worker's progress JSON into the in-memory HubJobState.

    Called on every /hub/jobs and /hub/progress poll. If the worker's
    progress file is missing or unreadable, leaves the in-memory state
    as-is — the server-side fallback (PID liveness check + sweep) handles
    the "worker is dead" case separately.
    """
    import json

    from lerobot.gui.hub_jobs import JOBS_DIR, JobPaths

    paths = JobPaths.for_job(job.job_id, JOBS_DIR)
    try:
        snap = json.loads(paths.progress.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    job.merge_progress(snap)


def _send_signal_with_identity_check(job, sig, *, fail_if_absent: bool = True) -> bool:
    """Send ``sig`` to ``job``'s worker, verifying (pid, start_time) first.

    Returns True if the signal was sent; False if the worker isn't alive
    or the PID has been recycled (in which case we mark the job failed).
    """
    import os

    from lerobot.gui.hub_jobs import JOBS_DIR, JobPaths, is_worker_alive, read_pid_file

    paths = JobPaths.for_job(job.job_id, JOBS_DIR)
    payload = read_pid_file(paths.pid)
    if payload is None or not is_worker_alive(payload):
        # Worker is already gone — synthesize a failure state and drop the
        # stale PID file so we don't keep re-checking it on every cancel
        # attempt (the startup sweep would catch it eventually, but only on
        # next server restart).
        #
        # `fail_if_absent=False` is for the cancel path, where "no PID file"
        # is ambiguous: the worker may simply not have written it yet. It is
        # written at worker startup, so a Cancel clicked in the first moments
        # of a transfer lands in that window. Declaring failure there is
        # wrong twice over — the job ends terminal while its worker is very
        # much alive and still uploading, and being terminal stops the poll
        # loop escalating and stops it blocking a second transfer on the
        # same dataset. Staying `cancelling` lets the escalation finish the
        # job properly once the grace period expires.
        if fail_if_absent and job.status not in ("complete", "failed", "cancelled"):
            job.status = "failed"
            job.error = "The transfer ended without reporting a result."
            job.error_class = "other"
            job.finished_at = time.time()
            # Removed only once we've concluded the worker is gone: on the
            # cancel path the file may simply not exist yet, and deleting it
            # after the worker writes it would lose our handle on it.
            # safe-destruct: stale PID file from a dead worker we owned
            paths.pid.unlink(missing_ok=True)
        return False
    try:
        os.kill(payload["pid"], sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _request_cancel(job) -> None:
    """Move ``job`` into ``cancelling`` and SIGTERM its worker. Idempotent.

    Sets the server-side status *before* signalling so the very next poll
    renders "Cancelling…" no matter what the worker's progress file still
    says. The old behaviour — signal only, status untouched — meant the
    next poll re-rendered a plain "running" card, which is what made a
    cancel look like it had done nothing at all.
    """
    import signal as _signal

    if job.cancel_requested_at is None:
        job.cancel_requested_at = time.time()
    from lerobot.gui.hub_jobs import JOBS_DIR, JobPaths

    job.status = "cancelling"
    job.milestone = "Cancelling…"
    if _send_signal_with_identity_check(job, _signal.SIGTERM, fail_if_absent=False):
        return

    # Couldn't signal. Two very different reasons, and only one is knowable:
    # a PID file that exists but names a dead process is proof the worker is
    # gone, so finish now rather than making the user watch "Cancelling…" for
    # the whole grace period. No PID file is ambiguous — the worker may still
    # be starting — so leave it to the escalation, which ends the job either
    # way once the grace expires.
    if JobPaths.for_job(job.job_id, JOBS_DIR).pid.exists():
        job.status = "cancelled"
        job.error = "Cancelled by user"
        job.error_class = "cancelled"
        job.milestone = "Cancelled"
        job.finished_at = time.time()
        _record_terminal_outcome(job)


def _escalate_cancel_if_overdue(job, *, now: float | None = None) -> bool:
    """SIGKILL a worker that outlived the cancel grace period.

    Called on every poll for jobs in ``cancelling``. The worker normally
    force-exits itself (see ``_force_cancel_exit``), so reaching here means
    it is genuinely wedged — inside an uninterruptible HF call, or stopped.
    SIGKILL cannot be caught, so this terminates the transfer for real.

    Returns True if the job was escalated and finalised on this call.
    """
    import signal as _signal

    from lerobot.gui.hub_jobs import CANCEL_GRACE_S

    if job.status != "cancelling":
        return False
    requested_at = job.cancel_requested_at or job.started_at
    if (time.time() if now is None else now) - requested_at < CANCEL_GRACE_S:
        return False

    # False means the worker is already gone; either way the job is over.
    _send_signal_with_identity_check(job, _signal.SIGKILL)
    job.status = "cancelled"
    job.error = "Cancelled by user"
    job.error_class = "cancelled"
    job.milestone = "Cancelled"
    job.finished_at = time.time()
    logger.warning(
        "Hub job %s ignored SIGTERM for %.0fs; escalated to SIGKILL",
        job.job_id,
        CANCEL_GRACE_S,
    )
    _record_terminal_outcome(job)
    return True


def _fail_if_heartbeat_dead(job, *, now: float | None = None) -> bool:
    """Fail a job whose worker is alive but has stopped reporting.

    The worker rewrites its progress file ~2 Hz regardless of transfer
    activity, so an mtime older than ``HEARTBEAT_FAULT_S`` means the
    reporting path itself is broken — not that the transfer is slow. The
    server previously had no check for this: its only health signal was
    process liveness, which stays true while a worker's writer thread is
    dead, so a transfer that had gone dark still rendered as healthy and
    running indefinitely.

    The worker is killed rather than left running. We have no visibility
    into it and no way to cancel it through the normal path, and leaving
    it alive would let a subsequent Retry spawn a second worker against
    the same upload cache and draft PR. Retry is cheap by design (Xet
    dedupe + PR resume), so ending it is the conservative choice.

    Returns True if the job was faulted on this call.
    """
    import signal as _signal

    from lerobot.gui.hub_jobs import HEARTBEAT_FAULT_S, JOBS_DIR, JobPaths

    if job.status not in ("running", "cancelling"):
        # `pending` has no worker yet, so it has no heartbeat to miss.
        return False

    paths = JobPaths.for_job(job.job_id, JOBS_DIR)
    try:
        last_write = paths.progress.stat().st_mtime
    except OSError:
        # No progress file yet; the spawn path stubs one, so treat the
        # job's own start as the floor rather than faulting immediately.
        last_write = job.started_at
    reference = max(last_write, job.started_at)
    if (time.time() if now is None else now) - reference < HEARTBEAT_FAULT_S:
        return False

    _send_signal_with_identity_check(job, _signal.SIGKILL)
    job.status = "failed"
    job.error = (
        f"The transfer stopped responding for over {HEARTBEAT_FAULT_S:.0f}s and was ended. "
        "Some data may already have been uploaded; Retry continues from where it stopped."
    )
    job.error_class = "unresponsive"
    job.milestone = "Worker unresponsive"
    job.finished_at = time.time()
    logger.error(
        "Hub job %s heartbeat dead (no progress write for >%.0fs); terminated worker",
        job.job_id,
        HEARTBEAT_FAULT_S,
    )
    _record_terminal_outcome(job)
    return True


def _record_terminal_outcome(job) -> None:
    """Append a server-decided ending to the durable transfer history.

    The worker records its own ending, but not when the server ends it *for*
    it: a SIGKILLed worker writes nothing, and those — a cancel that had to be
    forced, a worker that stopped reporting — are the endings a user is most
    likely to come back asking about. Both sides append; the reader keeps the
    last line per job, and the server writes later in every case where both do.
    """
    from lerobot.gui.hub_history import _record_from_job, append_outcome

    # `append_outcome` cannot raise, but `_record_from_job` can — it reads a
    # dozen attributes off the job. Unguarded, that puts an AttributeError on
    # the cancel path, which is the path this whole feature exists because it
    # failed. The worker's own recorder suppresses; the server's must too.
    with contextlib.suppress(Exception):
        append_outcome(_record_from_job(job))


def _sweep_orphan_temp_files(*, min_age_s: float = 300.0) -> int:
    """Delete stale ``*.tmp`` staging files left in the jobs dir.

    ``atomic_write_json`` unlinks its own temp on a failed write, but it
    cannot clean up after a hard kill: SIGKILL, a power loss, or the
    worker's own ``os._exit`` paths can land between the write and the
    rename. Because temp names are unique per (pid, thread) — required so
    concurrent writers don't destroy each other's staging file — such
    orphans accumulate rather than being overwritten by the next writer.

    ``min_age_s`` keeps us well clear of temps belonging to a write in
    flight right now; a single write takes microseconds, so anything this
    old is certainly abandoned.

    Called on server startup alongside the PID sweep. Returns the number
    of files removed.
    """
    from lerobot.gui.hub_jobs import JOBS_DIR

    if not JOBS_DIR.exists():
        return 0
    now = time.time()
    removed = 0
    for tmp_path in JOBS_DIR.glob("*.tmp"):
        try:
            if now - tmp_path.stat().st_mtime < min_age_s:
                continue
            # safe-destruct: abandoned staging file we wrote ourselves
            tmp_path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    if removed:
        logger.info("Removed %d orphan Hub temp file(s) on startup", removed)
    return removed


def _sweep_orphan_pid_files() -> int:
    """Inspect every <job_id>.pid in the jobs dir; reap dead workers.

    For each PID file:
      * If the worker is alive AND we have a registry entry → adopt
        (no-op; the entry's progress JSON catches up on next poll).
      * If the worker is alive AND we have no registry entry → leave
        the file alone; future restart-sweep iterations will pick it up
        once the registry knows about the job. (This case is rare —
        the registry is the spawn-time source of truth.)
      * If the worker is dead → mark any matching registry entry
        ``failed`` and delete the PID file.

    Called once on server startup. Returns the number of orphan files
    that were reaped.
    """
    from lerobot.gui.hub_jobs import JOBS_DIR, is_worker_alive, read_pid_file

    if not JOBS_DIR.exists():
        return 0

    reaped = 0
    for pid_path in JOBS_DIR.glob("*.pid"):
        job_id = pid_path.stem
        payload = read_pid_file(pid_path)
        if payload is None:
            # safe-destruct: malformed PID file we wrote; cleaning up our own debris
            pid_path.unlink(missing_ok=True)
            reaped += 1
            continue
        if is_worker_alive(payload):
            continue
        # Worker is dead.
        entry = _app_state.hub_jobs.get(job_id)
        if entry is not None and entry.status not in ("complete", "failed", "cancelled"):
            entry.status = "failed"
            entry.error = "Worker exited without finalizing (detected on server startup)"
            entry.error_class = "other"
            entry.finished_at = time.time()
        # safe-destruct: stale PID file from a dead worker we owned
        pid_path.unlink(missing_ok=True)
        reaped += 1
    if reaped:
        logger.info("Reaped %d orphan Hub worker PID file(s) on startup", reaped)
    return reaped


def _find_existing_pr_for_retry(dataset_id: str, repo_id: str, repo_type: str = "dataset") -> int | None:
    """For an upload retry, find a draft PR we can reuse.

    Looks back through ``_app_state.hub_jobs`` for the most recent
    terminal (failed/cancelled) entry matching this (dataset, repo). If
    that entry has a ``pr_num`` AND the PR is still in draft state on
    HF, return it for the new worker to resume into.

    Returns ``None`` if no resumable PR exists; the new worker will
    create a fresh one.

    Side effect: when a usable PR is found, **transfers ownership of
    that pr_num to the new caller** by clearing ``pr_num`` on every
    terminal source-job entry that pointed at the same PR. Callers
    that subsequently invoke ``hub_progress_dismiss`` on those source
    entries will then skip the ``change_discussion_status(closed)``
    branch — without this transfer, a Retry click would close the very
    PR the new worker is trying to resume into.

    ``repo_type`` flows through so the lookup works for ``"model"`` repos
    when the future Model Tab adds model-upload endpoints — the
    underlying mechanism is the same, only the repo namespace differs.
    """
    candidates = sorted(
        (
            j
            for j in _app_state.hub_jobs.values()
            if j.dataset_id == dataset_id
            and j.repo_id == repo_id
            and j.repo_type == repo_type
            and j.status in ("failed", "cancelled")
            and j.pr_num is not None
        ),
        key=lambda j: j.started_at,
        reverse=True,
    )
    if not candidates:
        # Nothing in the registry — but the registry is not the only record,
        # and it is routinely emptied: clearing a card with ✕ drops the entry
        # while deliberately leaving the draft PR open on HF, and a server
        # restart drops every entry. Without this fallback the PR is orphaned
        # — unreachable from the tray and invisible to Retry — so the next
        # upload opens a fresh one and re-sends everything, which is exactly
        # what the ✕ tooltip promises does not happen.
        #
        # The durable outcome record carries pr_num, so it can answer this.
        # The draft-status check below is what makes reading a possibly stale
        # record safe: a PR that was merged, closed, or already consumed by
        # another retry is rejected there.
        return _pr_from_history(dataset_id, repo_id, repo_type)
    pr_num = candidates[0].pr_num
    try:
        from huggingface_hub import HfApi

        details = HfApi().get_discussion_details(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=pr_num,
        )
        if details.status == "draft":
            # Transfer ownership: clear pr_num on every source-entry pointing
            # at this PR so a follow-up dismiss does not close it.
            for src in candidates:
                if src.pr_num == pr_num:
                    src.pr_num = None
            return pr_num
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not check PR #%d on %s (%s) for resume: %s. Will create a fresh PR.",
            pr_num,
            repo_id,
            repo_type,
            e,
        )
    return None


def _pr_from_history(dataset_id: str, repo_id: str, repo_type: str) -> int | None:
    """Most recent draft PR for this (dataset, repo) from the durable history.

    Fallback for :func:`_find_existing_pr_for_retry` when the in-memory
    registry has no entry. Returns the PR number only if HF still reports it
    as a draft, so a merged or already-consumed PR is never handed out.
    """
    from lerobot.gui.hub_history import read_recent

    for rec in read_recent(limit=100):
        if (
            rec.get("dataset_id") == dataset_id
            and rec.get("repo_id") == repo_id
            and rec.get("repo_type", "dataset") == repo_type
            and rec.get("status") in ("failed", "cancelled")
            and rec.get("pr_num") is not None
        ):
            pr_num = rec["pr_num"]
            try:
                from huggingface_hub import HfApi

                details = HfApi().get_discussion_details(
                    repo_id=repo_id, repo_type=repo_type, discussion_num=pr_num
                )
                if details.status == "draft":
                    logger.info("Reusing draft PR #%d for %s from transfer history", pr_num, repo_id)
                    return pr_num
            except Exception as e:  # noqa: BLE001 — a stale record must not break the upload
                logger.warning("Could not check PR #%d from history: %s", pr_num, e)
            return None
    return None


def _spawn_hub_worker(
    *,
    job,
    local_path: Path,
    reuse_pr_num: int | None = None,
    private: bool = True,
    commit_message: str | None = None,
) -> None:
    """Spawn the worker subprocess for ``job``.

    Pre: ``job`` already in ``_app_state.hub_jobs``. Caller holds the
    per-dataset spawn lock.
    Post: ``job.pid`` is set; the worker is running detached
    (``start_new_session=True``) so killing the GUI server doesn't
    immediately reap the worker until PR_SET_PDEATHSIG kicks in inside
    the worker.
    """
    import subprocess
    import sys

    from lerobot.gui.hub_jobs import (
        DEFAULT_UPLOAD_IGNORES,
        JOBS_DIR,
        JobConfig,
        JobPaths,
        atomic_write_json,
    )

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    paths = JobPaths.for_job(job.job_id, JOBS_DIR)

    # Build the config. For uploads we forward the default ignore set;
    # downloads pass them through too (snapshot_download will skip these
    # patterns from the remote sibling list if they exist).
    cfg = JobConfig(
        job_id=job.job_id,
        dataset_id=job.dataset_id,
        direction=job.direction,
        repo_id=job.repo_id,
        repo_type=job.repo_type,
        local_path=str(local_path),
        jobs_dir=str(JOBS_DIR),
        private=private,
        commit_message=commit_message,
        allow_patterns=None,
        ignore_patterns=DEFAULT_UPLOAD_IGNORES if job.direction == "upload" else None,
        reuse_pr_num=reuse_pr_num,
    )

    # Stub the progress file so a /hub/jobs poll right after spawn has
    # something to read. The worker rewrites it on its own schedule.
    atomic_write_json(
        paths.progress,
        {
            "job_id": job.job_id,
            "status": "pending",
            "milestone": f"Starting {job.direction}",
            "milestone_at": job.started_at,
        },
    )

    env = os.environ.copy()
    env["LEROBOT_HUB_WORKER_CONFIG"] = cfg.to_json()
    # The selector is authoritative in both directions. Injected here rather
    # than inside the worker because huggingface_hub reads this into a module
    # constant at import time (constants.HF_HUB_DISABLE_XET); setting it after
    # any part of the library has been imported would silently do nothing.
    #
    # Clearing it matters as much as setting it: the worker inherits our
    # environment, so a server started with HF_HUB_DISABLE_XET=1 already
    # exported — a plausible workaround for a stalling link, and one this
    # feature exists to replace — would otherwise leave every transfer on
    # LFS while the modal claimed Xet was selected.
    if job.disable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    else:
        env.pop("HF_HUB_DISABLE_XET", None)

    proc = subprocess.Popen(  # noqa: S603 — args are well-controlled
        [sys.executable, "-m", "lerobot.gui.hub_worker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    job.pid = proc.pid
    logger.info(
        "Spawned Hub worker pid=%d job=%s dir=%s repo=%s reuse_pr=%s",
        proc.pid,
        job.job_id,
        job.direction,
        job.repo_id,
        reuse_pr_num,
    )


@router.post("/{dataset_id:path}/hub/upload")
async def hub_upload(dataset_id: str, request: HubUploadRequest | None = None):
    """Start a Hub upload. Returns ``{job_id}`` immediately.

    Spawns a subprocess worker that runs the full PR pipeline
    (create_pull_request → upload_large_folder → super_squash_history →
    merge_pull_request). The Transfers tray polls /hub/jobs for progress.

    On retry of a failed/cancelled job for the same (dataset, repo), the
    worker is told to resume into the previous PR branch rather than
    creating a new one.
    """
    from lerobot.gui.hub_jobs import check_upload_completeness, make_job

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

    dataset = _app_state.datasets[dataset_id]
    repo_id = (request.repo_id if request and request.repo_id else None) or dataset.repo_id

    # Spawn lock: serialises concurrent spawn attempts for the same dataset.
    async with _hub_spawn_lock_for(dataset_id):
        active = _app_state.active_hub_job_for(dataset_id)
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail={"message": "A Hub transfer is already in progress", "job_id": active.job_id},
            )

        import asyncio

        # Threaded: whoami() is a sync network call that can hang for
        # minutes when the Hub is unreachable; never block the event loop.
        await asyncio.to_thread(_verify_hub_auth)

        # Upload-time completeness check: defends against download-fail-then-upload
        # corruption. If local is missing files that exist on the remote, warn the
        # caller. The frontend can re-issue with confirm_force=true to override.
        confirm_force = request.confirm_force if request else False
        if not confirm_force:
            try:
                missing = await asyncio.to_thread(check_upload_completeness, Path(dataset.root), repo_id)
            except Exception as e:  # noqa: BLE001 — completeness check is best-effort
                logger.warning("Completeness check failed for %s vs %s: %s", dataset_id, repo_id, e)
                missing = {"missing_locally": [], "incomplete_locally": []}
            if missing["missing_locally"] or missing["incomplete_locally"]:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "incomplete_local_state",
                        "message": (
                            "Local copy is missing files that exist on the remote. "
                            "Re-download first, or confirm to upload anyway."
                        ),
                        "missing_locally": missing["missing_locally"][:20],
                        "incomplete_locally": missing["incomplete_locally"][:20],
                    },
                )

        reuse_pr = _find_existing_pr_for_retry(dataset_id, repo_id, repo_type="dataset")
        job = make_job(dataset_id=dataset_id, direction="upload", repo_id=repo_id)
        job.disable_xet = bool(request and request.disable_xet)
        _app_state.hub_jobs[job.job_id] = job
        if reuse_pr is not None:
            job.pr_num = reuse_pr

        logger.info(
            "Hub upload start: dataset=%s repo=%s job=%s reuse_pr=%s xet=%s",
            dataset_id,
            repo_id,
            job.job_id,
            reuse_pr,
            "off" if job.disable_xet else "on",
        )
        _spawn_hub_worker(
            job=job,
            local_path=Path(dataset.root),
            reuse_pr_num=reuse_pr,
            private=True,
        )

    return {"job_id": job.job_id, "status": "started"}


@router.post("/{dataset_id:path}/hub/download")
async def hub_download(dataset_id: str, request: HubDownloadRequest | None = None):
    """Start a Hub download. Returns ``{job_id}`` immediately.

    snapshot_download writes directly into dataset.root; HF's etag-skip
    and .incomplete-resume primitives handle resumability without a temp
    staging directory.
    """
    from lerobot.gui.hub_jobs import make_job

    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    dataset = _app_state.datasets[dataset_id]
    repo_id = (request.repo_id if request and request.repo_id else None) or dataset.repo_id

    async with _hub_spawn_lock_for(dataset_id):
        active = _app_state.active_hub_job_for(dataset_id)
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail={"message": "A Hub transfer is already in progress", "job_id": active.job_id},
            )

        import asyncio

        # Threaded: whoami() is a sync network call that can hang for
        # minutes when the Hub is unreachable; never block the event loop.
        await asyncio.to_thread(_verify_hub_auth)

        job = make_job(dataset_id=dataset_id, direction="download", repo_id=repo_id)
        _app_state.hub_jobs[job.job_id] = job
        logger.info("Hub download start: dataset=%s repo=%s job=%s", dataset_id, repo_id, job.job_id)
        _spawn_hub_worker(job=job, local_path=Path(dataset.root))

    return {"job_id": job.job_id, "status": "started"}


@router.get("/hub/jobs")
async def hub_jobs():
    """Return all Hub transfers, sorted newest-first.

    Each entry is the merged view of the server's in-memory HubJobState
    + the worker's latest progress JSON. Calling this opportunistically
    garbage-collects terminal jobs older than 30 minutes.
    """
    from lerobot.gui.api._hub_core import list_hub_jobs

    result = list_hub_jobs(_app_state)
    # Preserve the legacy FastAPI response shape (the GUI's Transfers
    # tray only reads the "jobs" key).
    return {"jobs": result["jobs"]}


@router.get("/hub/history")
async def hub_history(limit: int = 20):
    """Past transfers and how they ended, newest first.

    Survives both the 30-minute GC of finished jobs and a server restart,
    neither of which the live ``/hub/jobs`` list does — so this is what
    answers "did my upload actually land?" hours after the fact.
    """
    from lerobot.gui.api._hub_core import list_hub_history

    return list_hub_history(limit=max(1, min(limit, 200)))


@router.get("/hub/progress/{job_id}")
async def hub_progress(job_id: str):
    """Single-job snapshot for clients that want to attach to one specific job."""
    from lerobot.gui.api._hub_core import HubJobNotFoundError, get_job_progress

    try:
        return get_job_progress(_app_state, job_id)
    except HubJobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/hub/progress/{job_id}/cancel")
async def hub_progress_cancel(job_id: str):
    """Cancel an active transfer. Idempotent; a repeat call force-kills.

    First call moves the job to ``cancelling`` and SIGTERMs the worker.
    Calling again on an already-cancelling job escalates to SIGKILL
    immediately rather than waiting out the grace period — that second
    click is the user telling us the polite path isn't working.
    """
    from lerobot.gui.hub_jobs import CANCEL_GRACE_S

    job = _app_state.hub_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status == "cancelling":
        _escalate_cancel_if_overdue(job, now=time.time() + CANCEL_GRACE_S)
    elif job.status in ("pending", "running"):
        _request_cancel(job)
    return {"status": "cancel_requested", "job_id": job_id, "job_status": job.status}


@router.post("/hub/progress/{job_id}/dismiss")
async def hub_progress_dismiss(job_id: str, close_pr: bool = True):
    """Remove a terminal job from the registry + clean up its IPC files.

    For cancelled/failed uploads whose ``pr_num`` is still set, also close
    the draft PR on HF to prevent it from cluttering the repo's discussion
    list. PR-ownership transfer happens earlier in the retry path: when a
    new worker inherits a source job's PR via ``_find_existing_pr_for_retry``,
    that function clears the source's ``pr_num`` so a subsequent dismiss
    on the source skips the close branch. A Retry-then-Discard sequence
    therefore does not close the PR the retry is resuming into.
    """
    from lerobot.gui.hub_jobs import ACTIVE_STATUSES, JOBS_DIR, JobPaths

    job = _app_state.hub_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="Job is still running; cancel it before dismissing.",
        )

    # Close the draft PR if we created one and it's still open. Only on
    # cancelled/failed paths — a completed upload's PR was already merged
    # (and HF auto-cleans it).
    # `close_pr=false` separates clearing the list from destroying the
    # artifact — the rule browser download managers follow: clearing a
    # download from the panel never deletes the file, and deleting it is its
    # own explicit action. Without this, tidying a failed transfer out of the
    # tray was only possible by closing the draft PR it could have resumed
    # from, so the list and the remote state could not be managed separately.
    if close_pr and job.status in ("cancelled", "failed") and job.pr_num is not None:
        try:
            from huggingface_hub import HfApi

            details = HfApi().get_discussion_details(
                repo_id=job.repo_id,
                repo_type=job.repo_type,
                discussion_num=job.pr_num,
            )
            if details.status == "draft":
                HfApi().change_discussion_status(
                    repo_id=job.repo_id,
                    repo_type=job.repo_type,
                    discussion_num=job.pr_num,
                    new_status="closed",
                    comment="Discarded from LeRobot GUI.",
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not close PR #%d for discarded job %s: %s", job.pr_num, job_id, e)

    # Clean up the per-job IPC files.
    paths = JobPaths.for_job(job_id, JOBS_DIR)
    import contextlib as _contextlib

    # Includes any abandoned `<name>.<pid>.<tid>.tmp` staging files, which
    # a hard-killed writer can leave behind next to the real ones.
    strays = list(JOBS_DIR.glob(f"{job_id}.*.tmp"))
    for p in (paths.progress, paths.log, paths.pid, *strays):
        with _contextlib.suppress(OSError):
            # safe-destruct: per-job IPC files we created, user-confirmed dismiss
            p.unlink(missing_ok=True)

    del _app_state.hub_jobs[job_id]
    return {"status": "dismissed", "job_id": job_id}


# --------------------------------------------------------------------------
# Frame quality labelling
#
# Same parquet the HVLA trainer reads (frame_quality.parquet in the dataset
# root, schema index/episode_index/frame_index/exclude), so a label taken here
# is usable by the next run with no export step. Stored inside the dataset
# rather than beside it, matching how RABC ships per-frame values.
# --------------------------------------------------------------------------

QUALITY_FILENAME = "frame_quality.parquet"


class QualityRangeRequest(BaseModel):
    episode_idx: int
    start_frame: int
    end_frame: int  # exclusive
    exclude: bool = True


def _quality_path(dataset) -> Path:
    return Path(dataset.root) / QUALITY_FILENAME


def _load_quality_frame(dataset):
    """Full-length table, created on first use."""
    import numpy as np
    import pandas as pd

    path = _quality_path(dataset)
    if path.is_file():
        return pd.read_parquet(path)
    total = dataset.meta.total_frames
    return pd.DataFrame(
        {
            "index": np.arange(total, dtype=np.int64),
            "episode_index": np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64),
            "frame_index": np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64),
            "exclude": np.zeros(total, dtype=bool),
        }
    )


@router.get("/{dataset_id:path}/quality")
async def get_quality(dataset_id: str) -> dict:
    """Excluded frame counts, overall and for each episode that has any."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = _app_state.datasets[dataset_id]
    path = _quality_path(dataset)
    if not path.is_file():
        return {"total_excluded": 0, "per_episode": {}, "path": str(path), "exists": False}
    import pandas as pd

    table = pd.read_parquet(path)
    bad = table[table["exclude"].astype(bool)]
    per_ep = {int(k): int(v) for k, v in bad.groupby("episode_index").size().items()}
    return {
        "total_excluded": int(len(bad)),
        "total_frames": int(len(table)),
        "per_episode": per_ep,
        "path": str(path),
        "exists": True,
    }


@router.post("/{dataset_id:path}/quality")
async def set_quality(dataset_id: str, body: QualityRangeRequest) -> dict:
    """Mark (or clear) a frame range within one episode as low quality."""
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = _app_state.datasets[dataset_id]
    if body.end_frame <= body.start_frame:
        raise HTTPException(400, "end_frame must be greater than start_frame")

    episode_start = _get_episode_start_index(dataset_id, body.episode_idx)
    lo = episode_start + body.start_frame
    hi = episode_start + body.end_frame  # exclusive

    table = _load_quality_frame(dataset)
    mask = (table["index"] >= lo) & (table["index"] < hi)
    n = int(mask.sum())
    if n == 0:
        raise HTTPException(400, f"range covers no frames (global {lo}..{hi})")
    table.loc[mask, "exclude"] = bool(body.exclude)

    path = _quality_path(dataset)
    table.to_parquet(path, index=False)
    total = int(table["exclude"].astype(bool).sum())
    logger.info(
        "Frame quality: %s %d frames of episode %d (global %d..%d); %d excluded overall",
        "marked" if body.exclude else "cleared",
        n,
        body.episode_idx,
        lo,
        hi,
        total,
    )
    return {"changed": n, "total_excluded": total, "path": str(path)}


# --------------------------------------------------------------------------
# Episode playback as video
#
# The per-frame JPEG endpoint above addresses frames exactly, which is what
# scrubbing and the feature plots need. It is the wrong shape for *playing*:
# every frame is compressed independently and re-encoded from footage that is
# already H.264 on disk. This transcodes an episode's slice once and lets the
# browser do the rest, the same trade the Run tab makes for live cameras.
# --------------------------------------------------------------------------

PLAYBACK_PROFILES = {
    # name: (longest edge, video bitrate)
    "low": (640, "500k"),
    "medium": (1280, "1500k"),
    "full": (0, "6000k"),  # 0 = keep source resolution
}
_playback_locks: dict[str, asyncio.Lock] = {}


def _playback_cache_dir() -> Path:
    d = Path.home() / ".cache" / "lerobot" / "playback_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _transcode_episode(src: Path, out: Path, start_s: float, duration_s: float, profile: str) -> None:
    """Cut one episode out of its shard and re-encode it for the browser."""
    import shutil
    import subprocess

    max_edge, bitrate = PLAYBACK_PROFILES[profile]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise HTTPException(500, "ffmpeg not found on the server")

    encoder = os.environ.get("LEROBOT_PREVIEW_ENCODER", "libx264")
    if encoder not in {"libx264", "h264_nvenc"}:
        encoder = "libx264"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        # -ss before -i seeks by keyframe and is far faster on a long shard.
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.3f}",
        "-an",
    ]
    if max_edge:
        # Even width/height (-2) or H.264 rejects the frame size.
        cmd += ["-vf", f"scale='if(gt(iw,ih),{max_edge},-2)':'if(gt(iw,ih),-2,{max_edge})'"]
    if profile == "full":
        # Nothing to scale, so re-encoding would only lose quality and cost
        # time. Copy the source stream: exact frames, no decode at all.
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", encoder, "-b:v", bitrate]
        cmd += ["-preset", "p4", "-rc", "vbr"] if encoder == "h264_nvenc" else ["-preset", "veryfast"]
    # Frequent keyframes: seeking lands near the requested frame instead of
    # rewinding to the previous GOP boundary.
    if profile != "full":
        # Keyframes twice a second: seeking lands near the requested frame
        # instead of rewinding to the previous GOP boundary.
        cmd += ["-g", "15", "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", str(out)]

    tmp = out.with_suffix(".partial.mp4")
    cmd[-1] = str(tmp)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)  # safe-destruct: this transcode's own partial output
        raise HTTPException(500, f"transcode failed: {result.stderr[-300:]}")
    tmp.replace(out)


@router.get("/{dataset_id:path}/episodes/{episode_idx}/video")
async def get_episode_video(
    dataset_id: str, episode_idx: int, camera: str | None = None, profile: str = "low"
):
    """Stream one episode of one camera as H.264, transcoding on first request."""
    from fastapi.responses import FileResponse

    if profile not in PLAYBACK_PROFILES:
        raise HTTPException(400, f"profile must be one of {sorted(PLAYBACK_PROFILES)}")
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = _app_state.datasets[dataset_id]
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")

    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        raise HTTPException(400, "Dataset has no camera/image keys")
    camera_key = camera or camera_keys[0]
    if camera_key not in camera_keys:
        raise HTTPException(400, f"Camera not found: {camera_key}. Available: {camera_keys}")

    episodes = dataset.meta.episodes
    if episodes is None:
        from lerobot.datasets.io_utils import load_episodes

        episodes = load_episodes(dataset.root)
        dataset.meta.episodes = episodes
    ep = episodes[episode_idx]

    safe = dataset_id.replace("/", "_")
    out = _playback_cache_dir() / f"{safe}__ep{episode_idx}__{camera_key.split('.')[-1]}__{profile}.mp4"

    if not out.is_file():
        key = str(out)
        lock = _playback_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if not out.is_file():  # another request may have finished while we waited
                src = dataset.root / dataset.meta.get_video_file_path(episode_idx, camera_key)
                if not src.is_file():
                    raise HTTPException(404, f"video shard missing: {src}")
                start_s = float(ep.get(f"videos/{camera_key}/from_timestamp", 0.0) or 0.0)
                duration_s = float(ep["length"]) / float(dataset.fps)
                logger.info(
                    "Transcoding episode %d %s (%.2fs from %.2fs) profile=%s",
                    episode_idx,
                    camera_key,
                    duration_s,
                    start_s,
                    profile,
                )
                await asyncio.get_event_loop().run_in_executor(
                    _decode_executor, _transcode_episode, src, out, start_s, duration_s, profile
                )
                logger.info("Transcode done: %s (%d bytes)", out.name, out.stat().st_size)

    # FileResponse handles HTTP range requests, which is what lets the browser
    # seek without downloading the whole clip.
    return FileResponse(out, media_type="video/mp4", filename=out.name)
