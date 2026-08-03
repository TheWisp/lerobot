# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prepare an HVLA-ready derivative of a LeRobot dataset.

Every RGB video is decoded, resized to 224x224 with the exact torchvision
call HVLA training uses (bilinear, antialias=True), and re-encoded as
lossless H.264. The result is a plain standard LeRobot Dataset: existing
TRAIN loads it unchanged, and its ``--resize-images 224x224`` resize becomes
a same-size no-op (torchvision returns the input early when sizes match).

This module is a single public function on purpose: no profile registry, no
plugin system, no job framework. CLI and GUI both call
:func:`prepare_hvla_dataset` directly.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
import torchvision.transforms.functional as TF

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_video_info
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

HVLA_IMAGE_SIZE = (224, 224)

# Lossless H.264 (crf=0) in 4:4:4 chroma keeps the re-encode step from adding
# any error beyond uint8 quantization; at 224x224 the files stay small.
# g=2 keeps timestamp-based seeking cheap and accurate. H.264 is also much
# cheaper to decode than the AV1 sources this was built for.
HVLA_VIDEO_ENCODER = RGBEncoderConfig(
    vcodec="h264",
    pix_fmt="yuv444p",
    crf=0,
    g=2,
    preset="fast",
)

# progress(done_files, total_files, current_relative_video_path)
ProgressCallback = Callable[[int, int, str], None]

_VIDEO_DIR = "videos"


def _resize_frame_to_uint8(frame_rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 -> resized HWC uint8, using HVLA TRAIN's exact resize call."""
    img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div_(255.0)
    img = TF.resize(
        img,
        list(HVLA_IMAGE_SIZE),
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


def _convert_video(input_path: Path, output_path: Path, fps: int) -> tuple[int, int]:
    """Decode -> resize -> H.264 encode one video file.

    Returns (input_frame_count, input_fps). Frame PTS follow the dataset FPS
    contract explicitly so lerobot timestamp seeking behaves identically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    container_in = av.open(str(input_path))
    try:
        stream_in = container_in.streams.video[0]
        input_fps = int(stream_in.base_rate)
        container_out = av.open(str(output_path), "w")
        try:
            stream_out = container_out.add_stream(
                HVLA_VIDEO_ENCODER.vcodec,
                fps,
                options=HVLA_VIDEO_ENCODER.get_codec_options(as_strings=True),
            )
            stream_out.pix_fmt = HVLA_VIDEO_ENCODER.pix_fmt
            stream_out.width, stream_out.height = HVLA_IMAGE_SIZE[1], HVLA_IMAGE_SIZE[0]
            stream_out.time_base = Fraction(1, fps)

            frame_count = 0
            for frame in container_in.decode(stream_in):
                resized = _resize_frame_to_uint8(frame.to_ndarray(format="rgb24"))
                video_frame = av.VideoFrame.from_ndarray(resized, format="rgb24")
                video_frame.pts = frame_count
                video_frame.time_base = Fraction(1, fps)
                for packet in stream_out.encode(video_frame):
                    container_out.mux(packet)
                frame_count += 1
            # Flush the encoder.
            for packet in stream_out.encode():
                container_out.mux(packet)
        finally:
            container_out.close()
    finally:
        container_in.close()
    return frame_count, input_fps


def _probe_frame_count(video_path: Path) -> int:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.frames > 0:
            return stream.frames
        if stream.duration is not None:
            return round(float(stream.duration * stream.time_base) * float(stream.base_rate))
        return round(container.duration / av.time_base * float(stream.base_rate))


def _validate_source(dataset: LeRobotDataset) -> None:
    meta = dataset.meta
    if not meta.video_keys:
        raise ValueError("Dataset has no video features; nothing to prepare.")
    if getattr(meta, "depth_keys", None):
        raise ValueError(f"Depth video is not supported (found {meta.depth_keys}).")
    for key in meta.video_keys:
        feature = meta.features[key]
        if feature.get("dtype") != "video":
            raise ValueError(f"Feature {key!r} is {feature.get('dtype')!r}, expected 'video'.")
        if feature["info"].get("is_depth_map"):
            raise ValueError(f"Feature {key!r} is a depth map; not supported.")
        shape = feature.get("shape")
        if not shape or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Feature {key!r} shape {shape} is not HWC RGB.")
    if not isinstance(meta.fps, int) or meta.fps <= 0:
        raise ValueError(f"Dataset fps must be a positive integer, got {meta.fps!r}.")


def _validate_output(
    staging_root: Path,
    output_repo_id: str,
    source: LeRobotDataset,
    input_frame_counts: dict[str, int],
    input_fps: dict[str, int],
) -> None:
    """Cheap mechanical checks on the staged derivative before publishing."""
    dataset = LeRobotDataset(output_repo_id, root=staging_root)
    meta = dataset.meta
    if meta.total_episodes != source.meta.total_episodes:
        raise RuntimeError(
            f"Episode count mismatch: {meta.total_episodes} != {source.meta.total_episodes}"
        )
    if meta.total_frames != source.meta.total_frames:
        raise RuntimeError(f"Frame count mismatch: {meta.total_frames} != {source.meta.total_frames}")
    if sorted(meta.video_keys) != sorted(source.meta.video_keys):
        raise RuntimeError(f"Camera keys mismatch: {meta.video_keys} != {source.meta.video_keys}")
    for key in meta.video_keys:
        if list(meta.features[key]["shape"]) != [HVLA_IMAGE_SIZE[0], HVLA_IMAGE_SIZE[1], 3]:
            raise RuntimeError(f"Feature {key!r} shape was not updated to 224x224.")
    for rel_path, in_count in input_frame_counts.items():
        out_path = staging_root / rel_path
        info = get_video_info(out_path)
        if (info["video.width"], info["video.height"]) != (HVLA_IMAGE_SIZE[1], HVLA_IMAGE_SIZE[0]):
            raise RuntimeError(f"{rel_path}: probed size is not 224x224.")
        if info["video.fps"] != input_fps[rel_path]:
            raise RuntimeError(f"{rel_path}: fps changed ({info['video.fps']} != {input_fps[rel_path]}).")
        out_count = _probe_frame_count(out_path)
        if out_count != in_count:
            raise RuntimeError(f"{rel_path}: frame count changed ({out_count} != {in_count}).")
    for index in (0, len(dataset) - 1):
        sample = dataset[index]
        for key in meta.video_keys:
            if tuple(sample[key].shape) != (3, HVLA_IMAGE_SIZE[0], HVLA_IMAGE_SIZE[1]):
                raise RuntimeError(
                    f"Sample {index} feature {key!r} tensor is {tuple(sample[key].shape)}, expected CHW 224x224."
                )


def prepare_hvla_dataset(
    *,
    source_repo_id: str,
    source_root: Path | str | None,
    output_repo_id: str,
    output_root: Path | str,
    progress: ProgressCallback | None = None,
) -> Path:
    """Create a 224x224 lossless-H.264 derivative dataset for HVLA training.

    Returns the completed output root. Refuses to overwrite the source or an
    existing output. On any failure the staging directory it created is
    removed and the source dataset is left untouched.
    """
    src_root = Path(source_root) if source_root is not None else HF_LEROBOT_HOME / source_repo_id
    out_root = Path(output_root)
    if src_root.resolve() == out_root.resolve():
        raise ValueError("output_root must differ from the source root.")
    if out_root.exists():
        raise FileExistsError(f"Output already exists, refusing to overwrite: {out_root}")

    source = LeRobotDataset(source_repo_id, root=src_root)
    _validate_source(source)
    meta = source.meta

    video_files: list[tuple[str, Path]] = []
    for key in meta.video_keys:
        key_dir = src_root / _VIDEO_DIR / key
        if not key_dir.is_dir():
            raise ValueError(f"Expected video directory missing: {key_dir}")
        for path in sorted(key_dir.rglob("*.mp4")):
            video_files.append((key, path))
    if not video_files:
        raise ValueError(f"No .mp4 files found under {src_root / _VIDEO_DIR}.")
    total = len(video_files)

    staging_root = out_root.parent / f".{out_root.name}.staging-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(f"Leftover staging directory in the way: {staging_root}")

    input_frame_counts: dict[str, int] = {}
    input_fps: dict[str, int] = {}
    try:
        staging_root.mkdir(parents=True)
        # Copy everything except the videos; data/meta are small, plain copies
        # are safer than hardlinks (a hardlinked edit can never touch the source).
        for entry in src_root.iterdir():
            if entry.name in (_VIDEO_DIR, ".cache") or entry.name.startswith("tmp_"):
                continue
            dest = staging_root / entry.name
            if entry.is_dir():
                shutil.copytree(entry, dest)
            else:
                shutil.copy2(entry, dest)

        for done, (_key, src_video) in enumerate(video_files):
            rel_path = src_video.relative_to(src_root)
            if progress is not None:
                progress(done, total, str(rel_path))
            logger.info("Converting %s (%d/%d)", rel_path, done + 1, total)
            count, fps_in = _convert_video(src_video, staging_root / rel_path, meta.fps)
            input_frame_counts[str(rel_path)] = count
            input_fps[str(rel_path)] = fps_in
        if progress is not None:
            progress(total, total, "")

        # Update only what must change in meta/info.json: camera shapes and
        # per-feature video info probed from the actual output files.
        # NOTE: image stats in meta/stats.json are kept from the source.
        # HVLA Flow S1 does not consume image stats; do not generalize this
        # to policies that normalize images from dataset stats.
        staging_meta = LeRobotDatasetMetadata(output_repo_id, staging_root)
        for key in staging_meta.video_keys:
            feature = staging_meta.info.features[key]
            feature["shape"] = [HVLA_IMAGE_SIZE[0], HVLA_IMAGE_SIZE[1], 3]
            first_video = staging_root / _VIDEO_DIR / key / "chunk-000" / "file-000.mp4"
            if not first_video.is_file():
                first_video = sorted((staging_root / _VIDEO_DIR / key).rglob("*.mp4"))[0]
            feature["info"].update(get_video_info(first_video, video_encoder=HVLA_VIDEO_ENCODER))
        write_info(staging_meta.info, staging_root)

        _validate_output(staging_root, output_repo_id, source, input_frame_counts, input_fps)
        os.rename(staging_root, out_root)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    logger.info("Prepared HVLA dataset at %s", out_root)
    return out_root
