# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GPU data path: decode, composite and resize a training batch on-device.

The CPU path decodes video, expands masks, composites and resizes inside
DataLoader workers; measured on the training host that is most of a masked
720p step (data_s 1.49 s of a 1.72 s step at 16 workers). This module does the
image half of a batch on the GPU instead: NVDEC decodes straight into device
memory (only compressed bytes cross PCIe), masks expand from interval
endpoints with one scatter+cumsum, the composite runs as the
equivalence-pinned :class:`GpuMaskComposite`, and the resize is torchvision's,
on-device.

Decoding goes through NVIDIA's own NVDEC binding (PyNvVideoCodec) rather than
torchcodec's CUDA device. That is not a preference: on this host
(Blackwell/sm_120, driver 580, torch 2.11+cu128) torchcodec 0.11.1+cu128
returns a constant frame for every file it decodes on CUDA -- every codec
tried, every ffmpeg major it supports (7 and 8), every seek mode and API
entry point, while the NVDEC engine reports 21-26% busy and no error is
raised. ffmpeg's own NVDEC path on the same machine is correct to 0.84 levels,
so the hardware, driver and container capabilities are sound and the fault is
in that integration. PyNvVideoCodec on the same files agrees with the CPU
decoder to a mean of 0.91 levels (max 3, over 40 random frames), which is the
4:2:0 chroma round-trip and nothing more.

Both paths therefore produce the same tensors: per selected camera, float32 in
[0, 1], CHW at the training resolution. The composite is pinned to <=2 levels
(tests/datasets/test_gpu_composite_equivalence.py) and the decode is verified
against the CPU decoder at startup for the dataset actually being trained on,
by :func:`calibrate_decode` -- which also picks the YUV->RGB conversion by
measurement instead of assuming one, because a wrong colour matrix is a
plausible-looking image that is quietly wrong by ~11 levels everywhere.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Luma coefficients (kr, kb) per colour standard.
_YUV_STANDARDS = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722)}

# Hardware decode is not bit-identical to software decode: 4:2:0 chroma is
# upsampled, and the two implementations site the samples slightly
# differently. The measured agreement for the CORRECT conversion is mean 0.91
# / max 3, and for a WRONG colour matrix it is mean 11.3 -- an order of
# magnitude apart, so these bounds separate them with room to spare.
DECODE_TOLERANCE_MEAN = 2.0
DECODE_TOLERANCE_MAX = 8.0


@dataclass(frozen=True)
class NV12Conversion:
    """Which YUV->RGB conversion reproduces this file's CPU-decoded pixels."""

    standard: str
    limited_range: bool

    def __str__(self) -> str:
        return f"{self.standard.upper()} {'limited' if self.limited_range else 'full'} range"


def nv12_to_rgb(plane: torch.Tensor, height: int, width: int, conv: NV12Conversion) -> torch.Tensor:
    """NV12 (H*3/2, W) uint8 -> RGB (3, H, W) uint8, on the plane's device.

    Chroma is replicated 2x2 (nearest), which is what matched the CPU decoder;
    bilinear upsampling measured worse (max 21 levels against 3).
    """
    y = plane[:height].float()
    uv = plane[height:].reshape(height // 2, width // 2, 2).float()
    u = uv[..., 0].repeat_interleave(2, 0).repeat_interleave(2, 1) - 128.0
    v = uv[..., 1].repeat_interleave(2, 0).repeat_interleave(2, 1) - 128.0
    if conv.limited_range:
        y = (y - 16.0) * (255.0 / 219.0)
        u = u * (255.0 / 224.0)
        v = v * (255.0 / 224.0)
    kr, kb = _YUV_STANDARDS[conv.standard]
    kg = 1.0 - kr - kb
    r = y + 2 * (1 - kr) * v
    b = y + 2 * (1 - kb) * u
    g = y - (kb * 2 * (1 - kb) / kg) * u - (kr * 2 * (1 - kr) / kg) * v
    return torch.stack([r, g, b]).clamp_(0, 255).to(torch.uint8)


def select_conversion(
    reference: torch.Tensor, plane: torch.Tensor, height: int, width: int
) -> tuple[float, float, NV12Conversion]:
    """Best (mean, max, conversion) reproducing ``reference`` from an NV12 plane.

    Pure and device-agnostic so the gate itself is testable without a GPU.
    """
    best: tuple[float, float, NV12Conversion] | None = None
    for standard in _YUV_STANDARDS:
        for limited in (True, False):
            conv = NV12Conversion(standard, limited)
            diff = (reference - nv12_to_rgb(plane, height, width, conv).float().to(reference.device)).abs()
            if best is None or diff.mean().item() < best[0]:
                best = (diff.mean().item(), diff.max().item(), conv)
    assert best is not None, "no conversion candidates"
    return best


def calibrate_decode(video_path: str, frame_index: int = 5) -> tuple[NV12Conversion, tuple[int, int, int]]:
    """Pick and verify the GPU decode for this file. Returns (conversion, CHW).

    Preconditions: ``video_path`` exists and has more than ``frame_index``
    frames. Postcondition: on return, GPU decode of this file with the
    returned conversion matches the CPU decoder within DECODE_TOLERANCE_*;
    otherwise RuntimeError and the caller must use the CPU path.

    Checking that a decode returns without raising is NOT sufficient, and that
    is not hypothetical: torchcodec's CUDA decoder on this host returns a flat
    frame for every file, raising nothing, and a training run consumed it for
    800 steps with a loss curve indistinguishable from the correct run's
    (0.2128 against 0.2056 at step 800) because state and action structure
    dominate the loss at that horizon. Neither an exception nor the loss would
    have caught it. Only a pixel comparison does, so this is a pixel
    comparison -- against the CPU decoder, on the file being trained on.

    The colour conversion is chosen the same way. Stream metadata here reports
    an unknown colour space, and picking the wrong matrix yields a
    normal-looking image that is wrong by ~11 levels everywhere, so the
    candidates are tried and the measured best one is verified rather than
    assumed.
    """
    import PyNvVideoCodec
    from torchcodec.decoders import VideoDecoder

    reference = VideoDecoder(video_path, device="cpu").get_frames_at(indices=[frame_index]).data[0].float()
    channels, height, width = (int(x) for x in reference.shape)
    plane = torch.from_dlpack(PyNvVideoCodec.SimpleDecoder(video_path, use_device_memory=True)[frame_index])

    mean, worst, conv = select_conversion(reference, plane, height, width)
    if mean > DECODE_TOLERANCE_MEAN or worst > DECODE_TOLERANCE_MAX:
        raise RuntimeError(
            f"GPU decode of {video_path} does not reproduce the CPU decoder's pixels "
            f"(best conversion {conv}: mean {mean:.2f} levels, max {worst:.0f}; allowed "
            f"{DECODE_TOLERANCE_MEAN} / {DECODE_TOLERANCE_MAX}). Training on these frames "
            "would not raise and would not show in the loss, so the GPU path is refused."
        )
    logger.info("GPU decode calibrated: %s, agreeing with the CPU decoder to %.2f levels", conv, mean)
    return conv, (channels, height, width)


class GpuFrameSource:
    """Random-access frames for one camera, batched, decoded on the GPU.

    Owns one decoder per video file (datasets shard each camera into a handful
    of files, so all stay open). The frame index arrays are built once from
    the dataset's episode metadata: global dataset index -> (file ordinal,
    frame index within that file). An off-by-one here is silent -- training
    would run on temporally shifted frames and nothing would error -- so
    tests/datasets/test_gpu_data_pipeline.py compares every fetched frame
    against what LeRobotDataset returns for the same index.
    """

    def __init__(self, dataset, camera: str, device: str = "cuda"):
        # Production decodes with NVDEC. A CPU device selects torchcodec's CPU
        # decoder instead, which exists so the index mapping, compositing and
        # resize logic stay testable on machines without a GPU -- it is not a
        # fallback for training, which the resolver handles by choosing the
        # CPU data path outright.
        self._cuda = str(device).startswith("cuda")
        if self._cuda:
            import PyNvVideoCodec

            self._nvc = PyNvVideoCodec
        else:
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
        if self._cuda:
            self.conversion, self.shape = calibrate_decode(self.files[0])
        else:
            self.conversion = None
            probe = self._decoder(0).get_frames_at(indices=[0]).data[0]
            self.shape = tuple(int(x) for x in probe.shape)

    def _decoder(self, file_ord: int):
        d = self._decoders.get(file_ord)
        if d is None:
            d = self._decoders[file_ord] = (
                self._nvc.SimpleDecoder(self.files[file_ord], use_device_memory=True)
                if self._cuda
                else self._VideoDecoder(self.files[file_ord], device="cpu")
            )
        return d

    def fetch(self, indices: np.ndarray) -> torch.Tensor:
        """uint8 (B, 3, H, W) on ``device``, in the order of ``indices``."""
        _, height, width = self.shape
        out = torch.empty((len(indices), *self.shape), dtype=torch.uint8, device=self.device)
        files = self.file_of[indices]
        locals_ = self.local_of[indices]
        # Sorted access within a file measured ~2x the random-access rate.
        for f in np.unique(files):
            sel = np.flatnonzero(files == f)
            decoder = self._decoder(int(f))
            order = sel[np.argsort(locals_[sel])]
            if not self._cuda:
                out[torch.from_numpy(order)] = decoder.get_frames_at(indices=locals_[order].tolist()).data.to(
                    out.dtype
                )
                continue
            for j in order:
                plane = torch.from_dlpack(decoder[int(locals_[j])])
                # Converting here also copies: the decoder recycles its surface
                # pool, so the NV12 view must not outlive the next decode.
                out[j] = nv12_to_rgb(plane, height, width, self.conversion)
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
        self._peak_bytes = 0
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
            torch.cuda.reset_peak_memory_stats()
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
            self._peak_bytes = max(self._peak_bytes, torch.cuda.max_memory_allocated())
        return out

    def report(self) -> dict[str, float]:
        """Mean ms per timed step, per phase — for the training log line."""
        if not self._samples:
            return {}
        return {f"gpu_prep_{k}_ms": 1e3 * v / self._samples for k, v in self._totals.items()} | {
            "gpu_prep_steps_timed": float(self._samples),
            # Peak includes the model's own allocations (the counter is
            # process-wide), so it is an upper bound on what the pipeline
            # costs, which is the safe direction for a memory headroom check.
            "gpu_prep_peak_mb": self._peak_bytes / (1 << 20),
        }
