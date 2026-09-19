# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Batched GPU reproduction of the saved-mask composite.

This is the GPU data path's counterpart to :func:`composite_from_store`: the
same recipe, applied to a whole batch of frames on the device. It exists for
training throughput — measured on real 720p rows, the CPU composite is
4.6–7.3 ms per frame while the batched GPU form is ~0.2 — and it is only
allowed to exist because its output is pinned against the CPU implementation
by test (tests/datasets/test_gpu_composite_equivalence.py), on real rows, with
the measured difference stated there rather than assumed.

Scope, deliberately narrow (v1): every object treatment ``none`` and a
background of ``blur``, ``solid`` or ``none`` — the recipe class the masked
datasets in use actually carry. Anything else raises, loudly, and the caller
falls back to the CPU path. ``random`` backgrounds need the per-episode
texture draw and are not implemented here yet.

Pre: RLE rows follow mask_codec's format; frames are uint8 (B,3,H,W) CUDA
tensors at the camera's ``mask_size`` resolution. Post: uint8 (B,3,H,W) CUDA
tensors, composited. All-empty rows render the whole frame as background,
matching the CPU path.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch.nn import functional as tnf

from lerobot.datasets.mask_codec import _string_to_counts

#: Feather geometry — must mirror effects.feathered_alpha (dilate with an
#: 11x11 ellipse, then an 11-tap Gaussian at cv2's default sigma for that
#: kernel size). Changing either changes composited pixels on both paths.
_FEATHER = 5
_FEATHER_K = 2 * _FEATHER + 1


def _cv2_default_sigma(ksize: int) -> float:
    """cv2's sigma when GaussianBlur is called with sigma=0."""
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


def _gaussian_taps(ksize: int, sigma: float) -> torch.Tensor:
    """cv2.getGaussianKernel's taps (float64), so both paths share the blur."""
    x = torch.arange(ksize, dtype=torch.float64) - (ksize - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def _toeplitz_reflect101(n: int, taps: torch.Tensor) -> torch.Tensor:
    """Banded matrix computing 1-D convolution with cv2's REFLECT_101 border.

    A k-tap single-channel conv2d is cuDNN's worst case (measured 508 ms per
    256-frame step against 51 for the same math as GEMMs), so the blur runs as
    x @ T. cv2 reflects at borders (edge pixel not repeated); zero padding
    instead produced differences up to 173/255 at frame edges against the CPU
    path, so the reflection is folded into the matrix: out-of-range source
    indices mirror back inside.
    """
    t = torch.zeros(n, n, dtype=torch.float64)
    r = len(taps) // 2
    for i in range(n):
        for k, v in enumerate(taps.tolist()):
            j = i + k - r
            if j < 0:
                j = -j
            elif j >= n:
                j = 2 * n - 2 - j
            t[i, j] += float(v)
    return t


def _ellipse_row_extents(k: int) -> list[int]:
    """Per-row half-widths of cv2's MORPH_ELLIPSE structuring element.

    cv2.getStructuringElement(MORPH_ELLIPSE, (k, k)) is what feathered_alpha
    dilates with; a square max-pool over-dilates its corners. The ellipse
    dilation decomposes exactly into one horizontal max-pool per row offset,
    each with that row's width — same result, GPU-friendly shape.
    """
    # Read the extents off cv2's own element rather than reproducing its
    # inclusion rule: a hand-derived formula disagreed with the real element
    # on the first and middle rows, and parity by construction cannot drift.
    import cv2

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return [(int(row.sum()) - 1) // 2 if row.any() else -1 for row in el]


def _ellipse_dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
    """cv2.dilate with MORPH_ELLIPSE (k,k), batched. mask: (B,1,H,W) float."""
    r = k // 2
    out = torch.zeros_like(mask)
    padded = tnf.pad(mask, (0, 0, r, r))  # pad H for the row shifts
    h = mask.shape[-2]
    for i, ext in enumerate(_ellipse_row_extents(k)):
        if ext < 0:
            continue  # an all-zero element row contributes nothing
        row = padded[:, :, i : i + h, :]
        if ext > 0:
            row = tnf.max_pool2d(row, kernel_size=(1, 2 * ext + 1), stride=1, padding=(0, ext))
        out = torch.maximum(out, row)
    return out


class GpuMaskComposite:
    """Per-camera batched compositor. Build once per (camera spec, device)."""

    SUPPORTED_BACKGROUNDS = ("blur", "solid", "none")

    def __init__(self, spec: dict[str, Any], device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.labels: list[str] = list(spec.get("mask_labels", []))
        self.h, self.w = (int(x) for x in spec["mask_size"])

        treatments = spec.get("mask_treatments") or {}
        non_none = {k: v for k, v in treatments.items() if (v or {}).get("key") not in (None, "", "none")}
        bg = spec.get("mask_background") or {"key": "none"}
        self.bg_key = bg.get("key") or "none"
        self.bg_params = bg.get("params") or {}
        if non_none or self.bg_key not in self.SUPPORTED_BACKGROUNDS:
            raise NotImplementedError(
                f"GPU composite v1 supports all-'none' objects with a "
                f"{self.SUPPORTED_BACKGROUNDS} background; this recipe has "
                f"objects={non_none or 'all none'} background={self.bg_key!r}. "
                "Use the CPU data path for it."
            )

        # The effect blur, exactly as effects._treat builds it.
        sigma = max(1.0, float(self.bg_params.get("strength", 12)))
        k = int(sigma * 4) | 1
        # float32, not the compositor's working dtype: fp16 tap quantization
        # (~1e-3 relative) times a steep edge produced errors of tens of
        # levels on a worst-case test image. fp32 matmul costs more and is
        # paid knowingly; the equivalence test pins the result, not the speed.
        # The W-side matrix is applied as x @ T, which computes with T's
        # TRANSPOSE: out[w'] = sum_w x[w] * T[w, w']. Interior rows are
        # symmetric so this is invisible there, but the reflect fold is not —
        # applying the untransposed matrix halved corner weights and put every
        # >1 difference in the border bands. Stored pre-transposed.
        self._blur_w = (
            _toeplitz_reflect101(self.w, _gaussian_taps(k, sigma)).T.contiguous().to(device, torch.float32)
        )
        self._blur_h = _toeplitz_reflect101(self.h, _gaussian_taps(k, sigma)).to(device, torch.float32)
        # The feather blur, exactly as feathered_alpha builds it (sigma=0 in
        # cv2 terms -> the ksize-derived default).
        ftaps = _gaussian_taps(_FEATHER_K, _cv2_default_sigma(_FEATHER_K))
        self._feather_w = _toeplitz_reflect101(self.w, ftaps).T.contiguous().to(device, torch.float32)
        self._feather_h = _toeplitz_reflect101(self.h, ftaps).to(device, torch.float32)
        if self.bg_key == "solid":
            color = self.bg_params.get("color") or [0, 0, 0]
            self._solid = torch.tensor(color, dtype=self.dtype, device=device).view(1, 3, 1, 1)

    # -- masks ---------------------------------------------------------------
    def union_intervals(self, rows: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """CPU half: parse each row's RLE into flat 'on' interval endpoints.

        Cheap (measured 55 ms serial for 256 six-label rows) and the only part
        of the composite that stays on CPU; callers may thread it or overlap
        it with decode.
        """
        hw = self.h * self.w
        starts, ends = [], []
        for j, row in enumerate(rows):
            off = j * hw
            for _label_id, counts in json.loads(row or "[]"):
                b = np.concatenate(([0], np.cumsum(np.asarray(_string_to_counts(counts), dtype=np.int64))))
                s = b[1:-1:2]
                e = b[2::2][: len(s)]
                starts.append(s + off)
                ends.append(e + off)
        if not starts:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        return np.concatenate(starts), np.concatenate(ends)

    def union_from_intervals(self, starts: np.ndarray, ends: np.ndarray, batch: int) -> torch.Tensor:
        """GPU half: expand endpoint intervals into a (B,1,H,W) float mask.

        Intervals close within each frame, so the running sum returns to zero
        at frame boundaries and one global cumsum over the whole batch is
        exact. Measured 3.0 ms for 256 frames against 144.8 for the threaded
        CPU expansion this replaces.
        """
        hw = self.h * self.w
        buf = torch.zeros(batch * hw + 1, dtype=torch.int32, device=self.device)
        if len(starts):
            s = torch.from_numpy(starts).to(self.device, non_blocking=True)
            e = torch.from_numpy(ends).to(self.device, non_blocking=True)
            ones = torch.ones(len(starts), dtype=torch.int32, device=self.device)
            buf.index_add_(0, s, ones)
            buf.index_add_(0, e, ones, alpha=-1)
        union = torch.cumsum(buf[:-1], 0, dtype=torch.int32) > 0
        # RLE flat order is column-major: view (W, H) then transpose.
        return union.view(batch, self.w, self.h).transpose(1, 2).unsqueeze(1).to(torch.float32)

    # -- the composite -------------------------------------------------------
    def __call__(self, frames: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
        """Composite the recipe onto uint8 (B,3,H,W) CUDA frames."""
        # These three say what they mean, because a bare tuple at 3am does not.
        # The last one is the consequential one: a union whose batch does not
        # match the frames composites frame f with frame g's mask, which is not
        # an error anywhere -- it is a quietly wrong training set.
        assert frames.dtype == torch.uint8 and frames.shape[1] == 3, (
            f"expected uint8 (B,3,H,W) frames, got {tuple(frames.shape)} of {frames.dtype}"
        )
        assert frames.shape[-2:] == (self.h, self.w), (
            f"frames are {tuple(frames.shape[-2:])} but these masks were segmented at "
            f"{(self.h, self.w)}; the GPU path composites before any resize, so they must match"
        )
        assert union.shape[0] == frames.shape[0], (
            f"{union.shape[0]} mask rows for {frames.shape[0]} frames; each frame must be "
            "composited with its OWN row, and a mismatch here silently pairs them wrongly"
        )
        if self.bg_key == "none":
            return frames

        x = frames.to(self.dtype)
        # Feather mirrors feathered_alpha: ellipse dilate, then the small
        # Gaussian, in float32 to keep the alpha's rounding independent of the
        # frame dtype. Objects are all 'none', so the only alpha is the
        # background's 1 - feathered(union).
        fg = _ellipse_dilate(union, _FEATHER_K)
        fg = (self._feather_h @ fg @ self._feather_w).clamp(0.0, 1.0)
        # feathered_alpha blurs a uint8 mask, so its alpha lives on the 1/255
        # grid; snapping to it keeps the two paths' rounding aligned.
        fg = torch.round(fg * 255.0) / 255.0
        a_bg = (1.0 - fg).to(self.dtype)

        if self.bg_key == "blur":
            xf = frames.to(torch.float32)
            treated = torch.matmul(torch.matmul(self._blur_h, xf), self._blur_w).to(self.dtype)
        else:  # solid
            treated = self._solid.expand_as(x)
        out = x + a_bg * (treated - x)
        return out.round().clamp(0, 255).to(torch.uint8)
