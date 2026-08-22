# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GPU data path: decode, composite and resize a training batch on-device.

The CPU path decodes video, expands masks, composites and resizes inside
DataLoader workers; measured on the training host that is ~90% of a masked
720p step (data_s 2.11 s of a 2.34 s step). This module does the image half of
a batch on the GPU instead: NVDEC decodes straight into device memory (only
compressed bytes cross PCIe), masks expand from interval endpoints with one
scatter+cumsum, the composite runs as the equivalence-pinned
:class:`GpuMaskComposite`, and the resize is torchvision's, on-device.

Selected by the trainer's ``--data-path gpu``; the CPU path stays the default
and the fallback for recipes or hardware the GPU path does not support. Both
paths produce the same tensors: per selected camera, float32 in [0, 1], CHW at
the training resolution, within the composite-equivalence contract (≤2 levels,
>1 rarer than 1e-4 — tests/datasets/test_gpu_composite_equivalence.py).

CUDA decode needs a torchcodec built with the CUDA interface plus NVDEC
prerequisites; :func:`probe_cuda_decode` checks by doing, and its error
carries the working recipe (established empirically in-container: the
``+cu128`` torchcodec wheel, ``nvidia-npp-cu12`` preloaded, ffmpeg ≥ 5 shared
libraries, and the container's NVIDIA_DRIVER_CAPABILITIES including
``video``).
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import logging
import time
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def preload_nvidia_codec_libs() -> None:
    """Load pip-installed NPP libraries before torchcodec's CUDA core.

    The ``+cu128`` torchcodec links libnppicc but pip's nvidia wheels are not
    on the loader path; importing torch preloads only torch's own set.
    Idempotent, harmless where the libs are absent or already resolvable.
    """
    for site in __import__("site").getsitepackages():
        for p in sorted(glob.glob(f"{site}/nvidia/*npp*/lib/*.so*")):
            with contextlib.suppress(OSError):  # best effort by design
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)


def probe_cuda_decode(video_path: str) -> None:
    """Fail fast, with the install recipe, if CUDA decode cannot work here."""
    from torchcodec.decoders import VideoDecoder

    preload_nvidia_codec_libs()
    try:
        VideoDecoder(video_path, device="cuda").get_frames_at(indices=[0])
    except Exception as e:
        raise RuntimeError(
            "--data-path gpu needs CUDA video decode, which this environment "
            f"cannot do ({type(e).__name__}: {e}). The working recipe: install "
            "torchcodec==<torch-minor>+cu128 from https://download.pytorch.org/whl/cu128, "
            "install nvidia-npp-cu12, provide ffmpeg>=5 shared libraries on the "
            "loader path, and run the container with NVIDIA_DRIVER_CAPABILITIES "
            "including 'video'. Or use --data-path cpu."
        ) from e


class GpuFrameSource:
    """Random-access frames for one camera, batched, decoded on ``device``.

    Owns one decoder per video file (datasets shard each camera into a handful
    of files, so all stay open). The frame index arrays are built once from
    the dataset's episode metadata: global dataset index -> (file ordinal,
    frame index within that file).
    """

    def __init__(self, dataset, camera: str, device: str = "cuda"):
        from torchcodec.decoders import VideoDecoder

        self._VideoDecoder = VideoDecoder
        self.camera = camera
        self.device = device
        meta = dataset.meta
        fps = float(meta.fps)
        eps = meta.episodes
        n = dataset.num_frames
        self.file_of = np.empty(n, dtype=np.int32)
        self.local_of = np.empty(n, dtype=np.int64)
        paths: dict[tuple[int, int], int] = {}
        self.files: list[str] = []
        for e in range(meta.total_episodes):
            start = int(eps["dataset_from_index"][e])
            length = int(eps["length"][e])
            key = (int(eps[f"videos/{camera}/chunk_index"][e]), int(eps[f"videos/{camera}/file_index"][e]))
            if key not in paths:
                paths[key] = len(self.files)
                self.files.append(
                    str(
                        dataset.root
                        / meta.video_path.format(video_key=camera, chunk_index=key[0], file_index=key[1])
                    )
                )
            base = round(float(eps[f"videos/{camera}/from_timestamp"][e]) * fps)
            self.file_of[start : start + length] = paths[key]
            self.local_of[start : start + length] = base + np.arange(length)
        self._decoders: dict[int, Any] = {}

    def _decoder(self, file_ord: int):
        d = self._decoders.get(file_ord)
        if d is None:
            d = self._decoders[file_ord] = self._VideoDecoder(self.files[file_ord], device=self.device)
        return d

    def fetch(self, indices: np.ndarray) -> torch.Tensor:
        """uint8 (B,3,H,W) on ``device``, in the order of ``indices``."""
        out: torch.Tensor | None = None
        files = self.file_of[indices]
        locals_ = self.local_of[indices]
        for f in np.unique(files):
            sel = np.flatnonzero(files == f)
            order = sel[np.argsort(locals_[sel])]  # decoders like sorted access
            frames = self._decoder(int(f)).get_frames_at(indices=locals_[order].tolist()).data
            if out is None:
                out = torch.empty((len(indices), *frames.shape[1:]), dtype=frames.dtype, device=frames.device)
            out[torch.from_numpy(order)] = frames
        assert out is not None, "empty index batch"
        return out


class GpuImagePipeline:
    """Batch images for the selected cameras, prepared on the GPU.

    ``prepare(batch)`` consumes the collated batch (which carries the global
    ``index`` tensor and, for masked cameras, the RLE string rows) and returns
    ``{camera: float32 (B,3,resize,resize) in [0,1] on device}`` — the same
    tensors the CPU path produces, produced on-device.

    Telemetry: phase wall-times are accumulated on ``timed`` calls (the caller
    samples every N steps; CUDA syncs between phases cost real time, so every
    step would slow the thing being measured) and read via :meth:`report`.
    """

    PHASES = ("decode", "rle_parse", "mask_expand", "composite", "resize")

    def __init__(
        self,
        dataset,
        cameras: list[str],
        resize_to: tuple[int, int] | None,
        device: str = "cuda",
    ):
        self.device = device
        self.cameras = list(cameras)
        self.resize_to = resize_to
        self.sources = {cam: GpuFrameSource(dataset, cam, device) for cam in self.cameras}
        # Imported where it is used, not at module scope: a dataset without
        # saved masks composites nothing, and the pipeline must load on an
        # installation that has no mask compositing at all. Both dicts stay
        # empty in that case and every use below is already a lookup.
        self.composites: dict[str, GpuMaskComposite] = {}  # noqa: F821
        self.mask_key: dict[str, str] = {}
        for cam in self.cameras:
            key = cam.replace(".images.", ".masks.")
            spec = dataset.meta.features.get(key)
            if spec is not None and spec.get("mask_encoding") == "coco_rle":
                from lerobot.datasets.gpu_mask_composite import GpuMaskComposite

                self.composites[cam] = GpuMaskComposite(spec, device=device)
                self.mask_key[cam] = key
        self._totals = dict.fromkeys(self.PHASES, 0.0)
        self._samples = 0
        logger.info(
            "GPU data path: %d cameras (%s), masked: %s, resize %s",
            len(self.cameras),
            ", ".join(c.rsplit(".", 1)[-1] for c in self.cameras),
            ", ".join(c.rsplit(".", 1)[-1] for c in self.composites) or "none",
            self.resize_to,
        )

    def _tick(self, timed: bool, t0: float, phase: str) -> float:
        if not timed:
            return t0
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self._totals[phase] += t1 - t0
        return t1

    def prepare(self, batch: dict[str, Any], timed: bool = False) -> dict[str, torch.Tensor]:
        from torchvision.transforms import functional as tvf

        indices = batch["index"].cpu().numpy().astype(np.int64)
        out: dict[str, torch.Tensor] = {}
        if timed:
            torch.cuda.synchronize()
        for cam in self.cameras:
            t0 = time.perf_counter()
            frames = self.sources[cam].fetch(indices)
            t0 = self._tick(timed, t0, "decode")

            comp = self.composites.get(cam)
            if comp is not None:
                rows = batch[self.mask_key[cam]]
                rows = [r[0] if isinstance(r, (list, tuple)) else r for r in rows]
                starts, ends = comp.union_intervals(rows)
                t0 = self._tick(timed, t0, "rle_parse")
                union = comp.union_from_intervals(starts, ends, len(rows))
                t0 = self._tick(timed, t0, "mask_expand")
                frames = comp(frames, union)
                t0 = self._tick(timed, t0, "composite")

            image = frames.to(torch.float32) / 255.0
            if self.resize_to is not None:
                image = tvf.resize(
                    image, list(self.resize_to), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True
                )
            self._tick(timed, t0, "resize")
            out[cam] = image
        if timed:
            self._samples += 1
        return out

    def report(self) -> dict[str, float]:
        """Mean ms per timed step, per phase — for the training log line."""
        if not self._samples:
            return {}
        return {f"gpu_prep_{k}_ms": 1e3 * v / self._samples for k, v in self._totals.items()} | {
            "gpu_prep_steps_timed": float(self._samples)
        }
