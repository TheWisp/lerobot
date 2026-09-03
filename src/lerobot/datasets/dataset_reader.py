#!/usr/bin/env python

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
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
import numpy as np
import torch

from lerobot.configs import (
    DEFAULT_DEPTH_UNIT,
    DEPTH_METER_UNIT,
    DepthEncoderConfig,
)
from lerobot.utils.feature_utils import resolve_flag_masks

from .dataset_metadata import LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .video_utils import decode_video_frames


def _int_column(hf_dataset: datasets.Dataset, name: str) -> np.ndarray:
    """An integer column as a flat numpy array, bypassing the torch transform.

    ``hf_dataset[name]`` applies ``hf_transform_to_torch`` and materialises a
    tensor per row, which is orders of magnitude slower than reading the Arrow
    column for values that are only ever compared or masked as integers.

    Pre: ``name`` is a scalar or length-1-list integer column.
    Post: a 1-D int64 array with one entry per row, in row order.
    """
    column = hf_dataset.data.column(name)
    try:
        values = column.to_numpy(zero_copy_only=False)
    except TypeError:  # older pyarrow without the kwarg on this column type
        values = np.asarray(column.to_pylist())
    values = np.asarray(values)
    if values.dtype == object:  # list<int64>: a one-element list per row
        values = np.concatenate([np.asarray(v).reshape(-1) for v in values])
    return values.reshape(-1).astype(np.int64, copy=False)


class DatasetReader:
    """Encapsulates read-side state and methods for LeRobotDataset.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        return_uint8: bool = False,
        record_images: bool = True,
        decode_videos: bool = True,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        exclude_flags: Sequence[str] | None = None,
        frame_compositor=None,
    ):
        """Initialize the reader with metadata, filtering, and transform config.

        The HF dataset is not loaded here — call :meth:`try_load` or
        :meth:`load_and_activate` afterward.

        Args:
            meta: Dataset metadata instance.
            root: Local dataset root directory.
            episodes: Optional list of episode indices to select. ``None``
                means all episodes.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            video_backend: Video decoding backend identifier.
            delta_timestamps: Optional dict mapping feature keys to lists of
                relative timestamp offsets for temporal context windows.
            image_transforms: Optional torchvision v2 transform applied to
                visual features.
            decode_videos: When ``False``, video frames are not decoded and the
                returned item carries no camera keys. For callers that decode
                the batch themselves, by index, somewhere else.
            record_images: When ``False``, the cache-sufficiency check skips
                the per-episode video-file existence check (used by fast-eval
                workflows that don't write images/video).
            return_uint8: If True, return RGB video frames as raw uint8 tensors
                instead of normalized float32.
            depth_output_unit: Physical unit depth maps are dequantized to
                (``"m"`` or ``"mm"``). Defaults to ``"mm"``.
            exclude_flags: Flags whose frames must not be learned. Each
                such frame acts as an episode end for any window reaching it --
                see :meth:`_get_query_indices`. ``None`` excludes nothing, so a
                dataset that was never annotated reads exactly as before.
        """
        self._meta = meta
        self.root = root
        self.episodes = episodes
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms
        #: Optional SavedMaskCompositor: reproduces the stored mask recipe on
        #: decoded frames, BEFORE image_transforms (augmentation must see the
        #: composited frame, the same order the GUI playback path uses).
        self._frame_compositor = frame_compositor
        #: Mask columns are load-time inputs (RLE strings consumed by the
        #: compositor), not model features: they are dropped from items so
        #: batches stay collate-clean whether or not compositing is on. Raw
        #: rows remain reachable via hf_dataset / get_raw_item.
        self._mask_columns = {k for k, ft in meta.features.items() if ft.get("mask_encoding")}
        self._return_uint8 = return_uint8
        self._record_images = record_images
        self._decode_videos = decode_videos
        self._depth_output_unit = depth_output_unit
        self._exclude_flags = list(exclude_flags) if exclude_flags else []
        # Resolved eagerly so an unknown label fails at construction, next to the
        # dataset that does not declare it, rather than on the first __getitem__
        # somewhere inside a dataloader worker.
        self._flag_masks = resolve_flag_masks(meta.features, self._exclude_flags)

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None
        # Absolute indices of flagged frames, sorted. Only the flagged frames are
        # stored, not a mask over every frame: annotations are sparse, and one
        # searchsorted per sample answers "where does this window stop".
        self._flagged_indices: np.ndarray | None = None

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

        self._depth_encoder_configs: dict[str, DepthEncoderConfig] = {
            vid_key: DepthEncoderConfig.from_video_info(self._meta.features[vid_key].get("info"))
            for vid_key in self._meta.depth_keys
        }

        # Get the input unit of each depth feature stored as raw images.
        self._image_depth_units: dict[str, str | None] = {
            key: (self._meta.features[key].get("info") or {}).get("depth_unit")
            for key in self._meta.depth_keys
            if key in self._meta.image_keys
        }

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform applied to visual observations."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """Remove the transform applied to visual observations."""
        self._image_transforms = None

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        self._build_flag_boundaries()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()
        self._build_flag_boundaries()

    def _build_flag_boundaries(self) -> None:
        """Collect the absolute indices of frames carrying an excluded label.

        Reads the Arrow column directly rather than through ``hf_dataset[key]``,
        which applies the torch transform and materialises a tensor per row --
        seconds on a large dataset, against milliseconds here, for values that
        are only ever compared as integers.

        Post: ``_flagged_indices`` is sorted ascending, or None when nothing is
        excluded.
        """
        self._flagged_indices = None
        if not self._flag_masks or self.hf_dataset is None:
            return

        absolute = _int_column(self.hf_dataset, "index")
        selected = np.zeros(len(absolute), dtype=bool)
        for key, mask in self._flag_masks.items():
            selected |= (_int_column(self.hf_dataset, key) & mask) != 0
        flagged = absolute[selected]
        if flagged.size:
            self._flagged_indices = np.sort(flagged)

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            indices = self.hf_dataset.data.column("index").to_numpy()
            self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(len(indices)), strict=True))

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self._meta.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0 and self._record_images:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes.

        Used to build the ``allow_patterns`` list for ``snapshot_download``.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps."""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        # A flagged frame ends the window exactly as the episode does: positions
        # from it onward clamp to the last good action and are marked padding.
        # Truncating rather than punching a hole is what keeps the supervised
        # actions contiguous -- a masked gap with real targets on both sides
        # would train the model to jump across data we decided not to trust,
        # and the model still emits those positions at inference.
        #
        # Only the forward direction is bounded. A negative delta (observation
        # history) still reads across a flagged frame; no policy consumes an
        # observation-side pad mask today, so marking one would change nothing.
        if self._flagged_indices is not None:
            position = int(np.searchsorted(self._flagged_indices, abs_idx, side="left"))
            if position < self._flagged_indices.size:
                ep_end = min(ep_end, int(self._flagged_indices[position]))
        # `max(ep_end - 1, abs_idx)` rather than `ep_end - 1`: when abs_idx is
        # itself excluded the boundary above sets ep_end == abs_idx, and the
        # upper clamp alone then resolves *every* delta to abs_idx - 1 --
        # including delta 0, so this frame's row would be paired with the
        # previous frame's images. Training never reaches it (the sampler does
        # not draw such a start), but `dataset[i]` does: the eval loop and the
        # viewers index directly. The padding mask below is unaffected, so these
        # positions stay marked padding either way; this only decides which row
        # the value is read from.
        floor = max(ep_end - 1, abs_idx)
        query_indices = {
            key: [max(ep_start, min(floor, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys."""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault.
        """
        ep = self._meta.episodes[ep_idx]

        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
                is_depth=vid_key in self._meta.depth_keys,
            )
            if vid_key in self._meta.depth_keys:
                depth_encoder = self._depth_encoder_configs[vid_key]
                frames = dequantize_depth(
                    frames,
                    depth_min=depth_encoder.depth_min,
                    depth_max=depth_encoder.depth_max,
                    shift=depth_encoder.shift,
                    use_log=depth_encoder.use_log,
                    output_unit=self._depth_output_unit,
                )
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())

        # Single camera: no threading overhead
        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        # Multi-camera: decode in parallel (video decoding releases the GIL)
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
            return dict(f.result() for f in futures)

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded.

        ``idx`` is a *relative* index into the (possibly episode-filtered)
        HF dataset, **not** the absolute frame index stored in the ``index``
        column.  The absolute index is retrieved from the row itself.
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # A caller that decodes elsewhere -- the GPU data path fetches frames by
        # index on device -- must not pay for a decode here as well. The item
        # still carries `index`, which is what such a caller fetches against.
        if len(self._meta.video_keys) > 0 and self._decode_videos:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        # Both of these belong to the path that DECODES here. When this reader is
        # not decoding, the GPU data path owns the frames and does the
        # compositing itself -- and it needs the RLE rows in the batch to do it.
        # Dropping them unconditionally is why that path raised
        # `KeyError: 'masks.<camera>'` on its first real training run: it asked
        # for a column the reader had already removed. Compositing here would be
        # a no-op in that state anyway, since no camera frame was decoded.
        if self._decode_videos:
            if self._frame_compositor is not None:
                item = self._frame_compositor.apply(item, ep_idx)
            for key in self._mask_columns:
                item.pop(key, None)

        if self._image_transforms is not None:
            for cam in self._meta.camera_keys:
                if cam in self._meta.depth_keys:
                    continue
                item[cam] = self._image_transforms(item[cam])

        # Convert depth features to the output unit.
        for key, stored_unit in self._image_depth_units.items():
            if key in item and stored_unit is not None and stored_unit != self._depth_output_unit:
                item[key] = (
                    item[key] * MM_PER_METRE if stored_unit == DEPTH_METER_UNIT else item[key] / MM_PER_METRE
                )

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        return item
