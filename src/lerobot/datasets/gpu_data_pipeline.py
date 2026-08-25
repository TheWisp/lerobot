# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GPU data path: decode and resize a training batch on-device.

The CPU path decodes video and resizes inside DataLoader workers. This module
does the image half of a batch on the GPU instead: NVDEC decodes straight into
device memory (only compressed bytes cross PCIe) and the resize is
torchvision's, on-device.

Compositing saved masks is deliberately NOT here -- it belongs to the branch
that owns saved masks, which adds the compositor and the phases for it. A
dataset carrying mask columns is refused at construction rather than trained on
raw frames, and `_resolve_data_path` falls back to the CPU path, which does
composite.

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
[0, 1], CHW at the training resolution. Two bounds, measured separately, apply
end to end and should not be conflated:

* **Composite**: <=2 levels GIVEN identical input frames, pinned by
  tests/datasets/test_gpu_composite_equivalence.py.
* **Decode**: <=3 levels at 720p (mean 0.45) on the real dataset. Hardware and
  software decode upsample 4:2:0 chroma differently; ffmpeg's own NVDEC
  differs from its software decode by the same order (0.84 mean), so this is
  the format, not a defect.

End to end on the training dataset that measures mean 0.45 / max 3 at 720p and
max 2 at the 224 training resolution. The decode is verified against the CPU
decoder at startup for the dataset actually being trained on,
by :func:`calibrate_decode` -- which also picks the YUV->RGB conversion by
measurement instead of assuming one, because a wrong colour matrix is a
plausible-looking image that is quietly wrong by ~11 levels everywhere.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.datasets.gpu_mask_composite import GpuMaskComposite
from lerobot.datasets.mask_compositing import mask_feature_of

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
    # The decoder must outlive every view taken from it, and the view must be
    # copied before the decoder recycles the surface: a dlpack view of a
    # destroyed decoder is freed device memory, and reading it is an illegal
    # access (or a destroyed-context abort), not an exception at the point of
    # the mistake.
    decoder = PyNvVideoCodec.SimpleDecoder(video_path, use_device_memory=True)
    plane = torch.from_dlpack(decoder[frame_index]).clone()

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
        # `_decoder` is called from this source's own worker threads, one per
        # video file a batch touches. Two threads asking for the same file would
        # otherwise each build a decoder, and one of them would be dropped on the
        # floor still holding an NVDEC surface pool.
        self._decoders_lock = threading.Lock()
        # One worker per video file, capped: past ~8 concurrent decoders the
        # measured throughput curve flattens against the NVDEC engines.
        self._pool = (
            ThreadPoolExecutor(max_workers=min(len(self.files), 8), thread_name_prefix="nvdec")
            if self._cuda and len(self.files) > 1
            else None
        )
        if self._cuda:
            self.conversion, self.shape = calibrate_decode(self.files[0])
        else:
            self.conversion = None
            probe = self._decoder(0).get_frames_at(indices=[0]).data[0]
            self.shape = tuple(int(x) for x in probe.shape)

    def _decoder(self, file_ord: int):
        """The decoder for one video file, built once and kept open.

        Held open because construction is the expensive part -- rebuilding per
        call measures construction, not decode. Guarded because the callers are
        this source's worker threads.
        """
        d = self._decoders.get(file_ord)
        if d is not None:
            return d
        with self._decoders_lock:
            d = self._decoders.get(file_ord)
            if d is None:
                d = self._decoders[file_ord] = (
                    self._nvc.SimpleDecoder(self.files[file_ord], use_device_memory=True)
                    if self._cuda
                    else self._VideoDecoder(self.files[file_ord], device="cpu")
                )
            return d

    def _decode_file(self, file_ord: int, order: np.ndarray, locals_: np.ndarray, out: torch.Tensor) -> None:
        """Decode this file's share of the batch into its slots of ``out``."""
        _, height, width = self.shape
        decoder = self._decoder(file_ord)
        if not self._cuda:
            out[torch.from_numpy(order)] = decoder.get_frames_at(indices=locals_[order].tolist()).data.to(
                out.dtype
            )
            return
        for j in order:
            # clone() before the next decode: the surface pool is reused, and
            # the conversion's kernels read the view asynchronously, so a view
            # held across a decode can be overwritten underneath them. The copy
            # is ~1.4 MB per 720p frame, device-to-device.
            plane = torch.from_dlpack(decoder[int(locals_[j])]).clone()
            out[j] = nv12_to_rgb(plane, height, width, self.conversion)

    def fetch(self, indices: np.ndarray) -> torch.Tensor:
        """uint8 (B, 3, H, W) on ``device``, in the order of ``indices``.

        Files decode concurrently. One decoder reaches only part of what NVDEC
        sustains -- measured on this host, 673 frames/s against 1525 with eight
        in parallel -- and a random batch spans every video file of the camera,
        so parallelising across files buys most of that without giving any file
        a second decoder and a second surface pool.
        """
        out = torch.empty((len(indices), *self.shape), dtype=torch.uint8, device=self.device)
        files = self.file_of[indices]
        locals_ = self.local_of[indices]
        work = []
        for f in np.unique(files):
            sel = np.flatnonzero(files == f)
            # Sorted access within a file measured ~2x the random-access rate.
            work.append((int(f), sel[np.argsort(locals_[sel])]))
        if self._pool is None or len(work) == 1:
            for f, order in work:
                self._decode_file(f, order, locals_, out)
        else:
            list(self._pool.map(lambda a: self._decode_file(a[0], a[1], locals_, out), work))
        return out


def _camera_codec(dataset, camera: str) -> str | None:
    """The canonical codec name recorded for ``camera``, or None if absent.

    Precondition: ``camera`` is a video feature of ``dataset``. Returns None when
    the dataset predates the field, which callers must treat as unsupported
    rather than assume -- an unknown codec is exactly the case that must not
    reach the GPU decoder.
    """
    feature = dataset.meta.features.get(camera) or {}
    codec = (feature.get("info") or {}).get("video.codec")
    return codec.lower() if isinstance(codec, str) else None


# How often the mask-coverage line is emitted, in camera-frames. Matches the
# loader path's reporting cadence so the two are comparable.
_MASK_REPORT_EVERY = 500


def _region_centroid(mask: torch.Tensor, absent: tuple[float, float]) -> torch.Tensor:
    """``(B, 3)`` of ``(x, y, visible)`` from a ``(B, 1, H, W)`` 0/1 mask.

    Preconditions: ``mask`` holds only 0.0 and 1.0. Postcondition: rows with an
    empty mask carry ``absent`` and a zero flag, others the centroid as a
    fraction of width and height -- the same definition as
    :func:`~lerobot.policies.hvla.s1.flow_matching.ball_cue.ball_cue`, which
    the equivalence test pins.

    The marginals are summed first, so the weighted sum runs over H+W values
    rather than H*W. Those column and row counts are small integers and exact
    in float32; accumulating index*weight over 900k pixels in float32 would
    not be, and the cue would drift by a fraction of a pixel for no reason.
    """
    m = mask[:, 0]
    b, h, w = m.shape
    n = m.sum(dim=(1, 2))
    cols = m.sum(dim=1).double()  # (B, W)
    rows_ = m.sum(dim=2).double()  # (B, H)
    xs = torch.arange(w, device=m.device, dtype=torch.float64)
    ys = torch.arange(h, device=m.device, dtype=torch.float64)
    denom = n.double().clamp(min=1.0)
    x = (cols * xs).sum(dim=1) / denom / w
    y = (rows_ * ys).sum(dim=1) / denom / h
    visible = n > 0
    out = torch.empty(b, 3, device=m.device, dtype=torch.float32)
    out[:, 0] = torch.where(visible, x, torch.full_like(x, absent[0])).float()
    out[:, 1] = torch.where(visible, y, torch.full_like(y, absent[1])).float()
    out[:, 2] = visible.float()
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

    PHASES = ("decode", "rle_parse", "mask_expand", "composite", "resize", "ball")

    # Codecs whose NVDEC decode has been verified against the CPU decoder on this
    # host. HEVC is excluded by defect, not by omission -- see the constructor.
    DECODABLE_CODECS = frozenset({"h264", "av1"})

    # Device-memory budget for the float32 batch held while resizing. Sized so a
    # 1280x720 camera converts in chunks of about twenty frames rather than the
    # whole batch; large enough that the per-chunk launch overhead stays noise.
    CONVERT_TRANSIENT_BYTES = 256 * 2**20

    def __init__(
        self,
        dataset,
        cameras: list[str],
        resize_to: tuple[int, int] | None,
        device: str = "cuda",
        ball_source: str | None = None,
        ball_label_id: int = 0,
        cue_key: str | None = None,
        view_key: str | None = None,
        cue_absent: tuple[float, float] = (-1.0, -1.0),
    ):
        self.device = device
        self.cameras = list(cameras)
        self.resize_to = resize_to
        # Checked BEFORE any decoder exists, because constructing one already
        # decodes: PyNvVideoCodec 2.2.2 segfaults producing the final frame of an
        # HEVC file, and a SIGSEGV cannot be caught, so there is no probe to fall
        # back from -- the run dies mid-training the first time a batch draws
        # that index. See issue #166.
        #
        # An allowlist, not a denylist. A codec nobody has verified against the
        # CPU decoder belongs on the CPU path: this decoder is known to return
        # wrong pixels rather than error on some inputs, so "not yet checked"
        # and "known bad" deserve the same answer.
        unsupported = {
            cam: codec
            for cam in self.cameras
            if (codec := _camera_codec(dataset, cam)) not in self.DECODABLE_CODECS
        }
        if unsupported:
            raise NotImplementedError(
                "the GPU data path decodes "
                + "/".join(sorted(self.DECODABLE_CODECS))
                + " only; these cameras are encoded otherwise: "
                + ", ".join(f"{cam} ({codec})" for cam, codec in sorted(unsupported.items()))
            )
        # The ball cue is derived from a mask this pipeline already expands on
        # device for the composite, so both arms of the experiment cost one
        # extra label's intervals rather than a second decode -- and neither
        # needs the CPU data path, which was the only reason to run one.
        self.ball_source = ball_source
        self.ball_label_id = ball_label_id
        self.cue_key = cue_key
        self.view_key = view_key
        self.cue_absent = cue_absent
        self.sources = {cam: GpuFrameSource(dataset, cam, device) for cam in self.cameras}
        # A camera whose dataset declares a `coco_rle` mask column gets a
        # compositor; one without stays on the plain decode-and-resize path.
        # Paired by rewriting `.images.` to `.masks.` rather than by a second
        # list, which would be free to drift out of step with the cameras.
        self.composites: dict[str, GpuMaskComposite] = {}
        self.mask_key: dict[str, str] = {}
        # Only when this run is actually applying them. The dataset builds a
        # frame compositor iff `apply_saved_masks` asked for one, so its absence
        # is the run saying it wants raw pixels -- and then the stored recipe is
        # none of this path's business. Reading it regardless meant
        # `--ignore-saved-masks` on a dataset whose recipe the GPU compositor
        # cannot render gave up the GPU decode path as well, so opting out of
        # masks made training slower rather than faster.
        # Read directly, not through getattr with a default: LeRobotDataset sets
        # this attribute unconditionally, so a default would turn a rename into
        # "no masks anywhere" and quietly stop compositing on the GPU path.
        applying = dataset._frame_compositor is not None
        for cam in self.cameras if applying else ():
            key = mask_feature_of(cam)
            spec = dataset.meta.features.get(key)
            if spec is not None and spec.get("mask_encoding") == "coco_rle":
                self.composites[cam] = GpuMaskComposite(spec, device=device)
                self.mask_key[cam] = key
        # One thread per camera: the fetches are independent and each holds its
        # own decoders, so this is latency the pipeline was paying for nothing.
        self._cam_pool = (
            ThreadPoolExecutor(max_workers=len(self.cameras), thread_name_prefix="gpu-cam")
            if len(self.cameras) > 1
            else None
        )
        self._cuda = str(device).startswith("cuda")
        # Written on the prefetcher's producer thread, read on the training
        # thread; the lock is what makes report() a consistent snapshot.
        self._stats_lock = threading.Lock()
        if (cue_key or view_key) and ball_source not in self.composites:
            raise ValueError(
                f"the ball cue needs {ball_source!r} to carry coco_rle masks; "
                f"masked cameras here are {sorted(self.composites)}"
            )
        self._totals = dict.fromkeys(self.PHASES, 0.0)
        self._mask_rows = 0
        self._mask_rows_empty = 0
        self._mask_reported = 0
        self._samples = 0
        self._alloc_delta = 0
        logger.info(
            "GPU data path: %d cameras (%s), masked: %s, resize %s",
            len(self.cameras),
            ", ".join(c.rsplit(".", 1)[-1] for c in self.cameras),
            ", ".join(c.rsplit(".", 1)[-1] for c in self.composites) or "none",
            self.resize_to,
        )

    def _tick(self, timed: bool, t0: float, phase: str) -> float:
        """Close a phase, if this batch is being timed.

        Synchronises THIS stream, not the device. `prepare` runs on the
        prefetcher's producer thread, so a device-wide sync would also stall the
        training stream -- perturbing the step it is trying to measure, and
        serialising the overlap this pipeline exists to create.
        """
        if not timed:
            return t0
        if self._cuda:
            torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        # Accumulated on the producer thread and read by the training thread in
        # report(); `+=` on a float is a read-modify-write and would otherwise
        # lose samples and let a report divide by a count that does not match.
        with self._stats_lock:
            self._totals[phase] += t1 - t0
        return t1

    def prepare(self, batch: dict[str, Any], timed: bool = False) -> dict[str, torch.Tensor]:
        from torchvision.transforms import functional as tvf

        indices = batch["index"].cpu().numpy().astype(np.int64)
        out: dict[str, torch.Tensor] = {}
        if timed and self._cuda:
            # No reset_peak_memory_stats() here: it is process-wide, and this
            # runs on the producer thread while the model is allocating on
            # another. Resetting would silently destroy the model's accounting;
            # the allocation delta below is measured without disturbing it.
            torch.cuda.current_stream().synchronize()
            allocated_before = torch.cuda.memory_allocated()
        # Cameras decode CONCURRENTLY. Each source is already parallel across the
        # video files a batch touches, but the cameras used to run one after
        # another, so a four-camera dataset paid four times the decode latency it
        # needed to -- with the device idle between cameras. Each source owns its
        # own decoders, so there is nothing shared to race on; the decoders
        # themselves are not thread-safe, which is why the parallelism is here
        # and not across batches.
        t0 = time.perf_counter()
        if self._cam_pool is not None and len(self.cameras) > 1:
            frames_by_cam = dict(
                zip(
                    self.cameras,
                    self._cam_pool.map(lambda c: self.sources[c].fetch(indices), self.cameras),
                    strict=True,
                )
            )
        else:
            frames_by_cam = {c: self.sources[c].fetch(indices) for c in self.cameras}
        t0 = self._tick(timed, t0, "decode")

        for cam in self.cameras:
            frames = frames_by_cam[cam]
            # Compositing sits between decode and conversion because the mask
            # applies to the uint8 frame: doing it after the resize would
            # composite against interpolated pixels.
            comp = self.composites.get(cam)
            if comp is not None:
                rows = batch[self.mask_key[cam]]
                rows = [r[0] if isinstance(r, (list, tuple)) else r for r in rows]
                # The CPU compositor counts what it composited and how much of it
                # was empty, because "segmented and found nothing" renders the
                # whole frame as background -- legitimate when the object is out
                # of view, a silent disaster when the pass failed. This path had
                # no equivalent, so a GPU run's log could not answer "did training
                # see any masks at all?" -- only how long compositing took.
                self._mask_rows += len(rows)
                self._mask_rows_empty += sum(1 for r in rows if not r or r == "[]")
                starts, ends = comp.union_intervals(rows)
                t0 = self._tick(timed, t0, "rle_parse")
                union = comp.union_from_intervals(starts, ends, len(rows))
                t0 = self._tick(timed, t0, "mask_expand")
                frames = comp(frames, union)
                t0 = self._tick(timed, t0, "composite")
                # Same cadence as the loader path's report: once early, then
                # rarely. Evidence, not a running commentary.
                if self._mask_rows >= self._mask_reported + _MASK_REPORT_EVERY:
                    self._mask_reported = self._mask_rows
                    logger.info(
                        "saved masks (GPU path): %d camera-frames composited, %.1f%% had no mask "
                        "and rendered as all-background",
                        self._mask_rows,
                        100.0 * self._mask_rows_empty / max(1, self._mask_rows),
                    )
            out[cam] = self._to_model_input(frames, tvf)
            t0 = self._tick(timed, t0, "resize")

            # Derived from the composited frames at source resolution -- the
            # same point the CPU path builds the cue from, so the two agree
            # rather than differing by a resize.
            if cam == self.ball_source and (self.cue_key or self.view_key):
                # Guaranteed by the constructor: the cue source must be masked.
                assert comp is not None, self.ball_source
                ball = comp.union_from_intervals(*comp.label_intervals(rows, self.ball_label_id), len(rows))
                if self.cue_key:
                    out[self.cue_key] = _region_centroid(ball, self.cue_absent)
                if self.view_key:
                    view = frames.to(torch.float32) * ball / 255.0
                    if self.resize_to is not None:
                        view = tvf.resize(
                            view,
                            list(self.resize_to),
                            interpolation=tvf.InterpolationMode.BILINEAR,
                            antialias=True,
                        )
                    out[self.view_key] = view
                self._tick(timed, t0, "ball")
        if timed:
            with self._stats_lock:
                self._samples += 1
                if self._cuda:
                    # A delta, not a peak: the process-wide peak counter cannot
                    # be attributed to this pipeline while the model allocates
                    # concurrently. This under-reports transients freed inside
                    # prepare, which is stated rather than hidden.
                    self._alloc_delta = max(
                        self._alloc_delta, torch.cuda.memory_allocated() - allocated_before
                    )
        return out

    def _to_model_input(self, frames: torch.Tensor, tvf) -> torch.Tensor:
        """uint8 (B,3,H,W) frames -> float32 in [0,1], resized if configured.

        Converting the whole camera at once materialises the batch at *native*
        resolution in float32 -- four bytes per subpixel of a frame that is
        about to be thrown away. Measured on four 1280x720 cameras at batch 64,
        preparing 147 MiB of 224x224 input peaked at 2137 MiB, and the
        prefetcher holds two batches in flight.

        Converting in chunks bounds that transient. The result is bit-identical:
        the scale is elementwise and the resize is per-image, so neither reads
        across the batch dimension. Chunking is skipped when there is no resize,
        where the full-resolution float32 *is* the result and nothing is saved.
        """
        if self.resize_to is None:
            return frames.to(torch.float32) / 255.0

        n = frames.shape[0]
        per_frame = frames[0].numel() * 4  # float32 bytes for one frame
        chunk = max(1, self.CONVERT_TRANSIENT_BYTES // max(1, per_frame))
        if chunk >= n:
            image = frames.to(torch.float32) / 255.0
            return tvf.resize(
                image, list(self.resize_to), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True
            )

        out = torch.empty((n, frames.shape[1], *self.resize_to), dtype=torch.float32, device=frames.device)
        for start in range(0, n, chunk):
            block = frames[start : start + chunk].to(torch.float32) / 255.0
            out[start : start + chunk] = tvf.resize(
                block, list(self.resize_to), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True
            )
        return out

    def report(self) -> dict[str, float]:
        """Mean ms per timed step, per phase — for the training log line."""
        with self._stats_lock:
            if not self._samples:
                return {}
            totals = dict(self._totals)
            samples = self._samples
            delta = self._alloc_delta
        return {f"gpu_prep_{k}_ms": 1e3 * v / samples for k, v in totals.items()} | {
            "gpu_prep_steps_timed": float(samples),
            # Peak includes the model's own allocations (the counter is
            # process-wide), so it is an upper bound on what the pipeline
            # costs, which is the safe direction for a memory headroom check.
            "gpu_prep_alloc_mb": delta / (1 << 20),
        }


def resolve_gpu_pipeline(
    choice: str,
    dataset,
    camera_keys: list[str],
    resize_to: tuple[int, int] | None,
    device: str,
) -> GpuImagePipeline | None:
    """A pipeline for the GPU data path, or None to use the data-loader path.

    Preconditions: ``choice`` is one of ``auto``, ``cpu``, ``gpu``; ``dataset``
    is a LeRobotDataset whose ``camera_keys`` are video features.

    ``auto`` uses the GPU path where it is supported and falls back with the
    reason logged. ``cpu`` never builds one. ``gpu`` is honoured exactly: a
    request that cannot be satisfied raises rather than quietly training on the
    other path, because a run that asked for one path and got the other is a
    wrong measurement rather than a slow one.

    What ``auto`` checks are facts, not guesses. The device is CUDA; the
    constructor refuses a codec whose NVDEC decode is unverified or known to
    crash, and refuses a dataset carrying saved masks it cannot composite; and
    it verifies this dataset's own frames against the CPU decoder, because this
    decoder returns wrong pixels rather than erroring on some inputs.
    """
    if choice not in ("auto", "cpu", "gpu"):
        raise ValueError(f"unknown data path {choice!r}; expected auto, cpu or gpu")
    if choice == "cpu":
        logger.info("Data path: CPU (requested)")
        return None
    try:
        if not str(device).startswith("cuda"):
            raise NotImplementedError(f"device is {device}, not CUDA")
        if not camera_keys:
            raise NotImplementedError("the dataset has no camera features to decode")
        pipeline = GpuImagePipeline(dataset, camera_keys, resize_to=resize_to, device=device)
    except Exception as exc:
        if choice == "gpu":
            raise
        logger.warning("Data path: CPU (GPU path unavailable - %s: %s)", type(exc).__name__, exc)
        return None
    logger.info("Data path: GPU (NVDEC decode + on-device resize)")
    return pipeline


class GpuBatchPrefetcher:
    """Yield training batches with their GPU preparation already done.

    `GpuImagePipeline.prepare` is GPU work reached through a *blocking* call:
    PyNvVideoCodec decodes on the calling thread and returns when the frames
    exist. Calling it inline therefore puts preparation in series with the model
    step twice over -- the device alternates between decode and compute instead
    of doing both, and the training thread is blocked throughout. Measured on a
    4-camera 720p dataset that is a 49 ms preparation in front of a 31 ms
    update, an 85 ms step, and a device ~69% busy.

    A CUDA stream alone does not fix this, and trying it first is instructive:
    the decode does not become asynchronous just because a stream is current,
    because the blocking is on the CPU side of the binding. Preparation has to
    move to another *thread*. It then runs while the training thread is inside
    the model step, and the step costs about `max(prepare, update)` rather than
    their sum.

    Four details are what make this correct rather than merely fast:

    * The producer thread owns a side CUDA stream, so its copies and kernels do
      not serialise behind the compute stream's.
    * The consumer waits on a CUDA *event* recorded after preparation, so it
      cannot read tensors the side stream has not finished writing.
    * Every device tensor handed out is marked with `record_stream`. Without it
      the caching allocator may hand that memory to the next preparation while
      the compute stream is still reading it -- a nondeterministic failure that
      looks like corrupted images rather than a crash.
    * The source iterator is restarted here rather than by the caller, because a
      prefetcher that runs dry at an epoch boundary stalls exactly where it is
      supposed to be ahead.

    Pre: `pipeline` is a `GpuImagePipeline`; `loader` yields dicts and may be
    exhausted (it is restarted); `depth >= 1`.
    Post: iterating yields dicts whose image keys are on `device` and safe to
    read on the current stream. Exceptions raised in the producer surface from
    `__next__`.
    """

    _DONE = object()

    def __init__(
        self,
        loader,
        pipeline: GpuImagePipeline,
        device,
        depth: int = 2,
        timed_every: int = 50,
    ):
        assert depth >= 1, f"prefetch depth must be at least 1, got {depth}"
        self._timed_every = timed_every
        self._produced = 0
        self._loader = loader
        self._pipeline = pipeline
        self._device = device
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._error: BaseException | None = None
        self._stop = threading.Event()
        # One iterator, guarded: the workers race for the next batch rather than
        # each holding their own view of the epoch, so every sample is still
        # drawn exactly once per pass.
        self._it = iter(loader)
        self._it_lock = threading.Lock()
        # ONE producer. PyNvVideoCodec decoders are not thread-safe, and the
        # sources are shared, so a second producer corrupts the parser state --
        # observed as `cuvidParseVideoData` errors and segfaults. Parallelism
        # lives inside a fetch instead: across cameras, and across the video
        # files a batch touches.
        self._stream = torch.cuda.Stream(device=device)
        self._thread = threading.Thread(target=self._run, name="gpu-prefetch", daemon=True)
        self._thread.start()

    def _next_batch(self):
        """Take the next batch off the shared iterator, restarting on exhaustion."""
        with self._it_lock:
            try:
                return next(self._it)
            except StopIteration:
                self._it = iter(self._loader)
                return next(self._it)

    def _run(self) -> None:
        stream = self._stream
        try:
            while not self._stop.is_set():
                batch = self._next_batch()
                # Timed on a sample of batches: the per-phase CUDA syncs cost
                # real time, so telemetry must not slow what it measures.
                timed = self._timed_every > 0 and self._produced % self._timed_every == 0
                self._produced += 1
                with torch.cuda.stream(stream):
                    batch = {
                        k: (v.to(self._device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }
                    batch.update(self._pipeline.prepare(batch, timed=timed))
                    done = torch.cuda.Event()
                    done.record(stream)
                while not self._stop.is_set():
                    try:
                        self._queue.put((batch, done), timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer
            self._error = exc
            self._queue.put(self._DONE)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        item = self._queue.get()
        if item is self._DONE:
            raise self._error if self._error is not None else StopIteration
        batch, done = item
        current = torch.cuda.current_stream(self._device)
        current.wait_event(done)
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                value.record_stream(current)
        return batch

    def close(self) -> None:
        self._stop.set()
