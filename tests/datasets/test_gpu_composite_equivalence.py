# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GPU composite must reproduce the CPU composite, and the bound is stated.

GpuMaskComposite exists only for speed; composite_from_store remains the
definition of the rendered pixels. Measured on real 720p rows and frames, the
two differ by at most 2 levels on ~2 pixels per 22M (cv2's uint8 fixed-point
blur rounds intermediates; the GPU blur is continuous float), and are
identical on ~97% of pixels. That is the contract pinned here: max 2, and
more-than-1 rarer than 1e-4.

Getting to that bound consumed four real defects, each of which this file now
guards piecewise, because each was invisible in an end-to-end diff number
alone: a hand-derived ellipse that disagreed with cv2's element (row extents
are now read off the element itself), zero-padded blur against cv2's
REFLECT_101 (errors up to 173/255 in the border bands), the W-side operator
applied transposed (invisible in the symmetric interior, halved corner
weights), and an fp16 blur whose tap quantization was withdrawn for fp32.

The masks here are synthetic but real-shaped: thresholded smooth noise gives
organic boundaries with thousands of RLE runs per label — the regime that
exposed decode_mask's Python loop — where rectangles (~2 runs) hide
everything.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from lerobot.datasets.gpu_mask_composite import (  # noqa: E402
    GpuMaskComposite,
    _ellipse_dilate,
    _ellipse_row_extents,
    _gaussian_taps,
    _toeplitz_reflect101,
)
from lerobot.datasets.mask_codec import encode_frame  # noqa: E402
from lerobot.datasets.mask_compositing import composite_from_store  # noqa: E402

H, W = 720, 1280
LABELS = ["a", "b", "c", "d"]

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU composite needs CUDA")


def _organic_masks(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for i, name in enumerate(LABELS):
        noise = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 6 + 2 * i)
        out[name] = noise > np.quantile(noise, 0.7)
    return out


def _spec(background: dict) -> dict:
    return {
        "mask_encoding": "coco_rle",
        "mask_labels": LABELS,
        "mask_size": [H, W],
        "mask_treatments": {n: {"key": "none", "params": {}} for n in LABELS},
        "mask_background": background,
    }


# ---- the pieces, each exact against its cv2 counterpart ---------------------


def test_ellipse_extents_come_from_cv2s_element():
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    assert _ellipse_row_extents(11) == [(int(r.sum()) - 1) // 2 if r.any() else -1 for r in el]


@cuda
def test_ellipse_dilate_matches_cv2_exactly():
    mask = next(iter(_organic_masks(1).values())).astype(np.uint8)
    ref = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    got = _ellipse_dilate(torch.from_numpy(mask)[None, None].float().cuda(), 11)
    assert (got[0, 0].cpu().numpy().astype(np.uint8) == ref).all()


def test_reflect_toeplitz_matches_cv2s_border():
    """One row through the matrix == cv2's separable pass along that axis."""
    taps = _gaussian_taps(49, 12.0)
    t = _toeplitz_reflect101(W, taps)
    row = np.zeros((1, W), np.float32)
    row[0, :60] = np.linspace(0, 255, 60)  # activity at the border, where it matters
    ref = cv2.sepFilter2D(row, -1, taps.numpy().astype(np.float32), np.array([1.0], np.float32))
    got = (torch.from_numpy(row).double() @ t.T).float().numpy()
    assert np.abs(ref - got).max() < 1e-3, np.abs(ref - got).max()


# ---- the whole composite, against the CPU implementation --------------------


@cuda
@pytest.mark.parametrize(
    "background", [{"key": "blur", "params": {}}, {"key": "solid", "params": {"color": [0, 255, 0]}}]
)
def test_composite_matches_cpu_within_the_stated_bound(background):
    spec = _spec(background)
    rows = [encode_frame(_organic_masks(s), LABELS) for s in range(4)] + [""]  # incl. an empty row
    y, x = np.mgrid[0:H, 0:W]
    frame = np.ascontiguousarray(
        np.stack([(x * 2) % 256, (y * 3) % 256, ((x + y) * 5) % 256], -1).astype(np.uint8)
    )

    cpu = np.stack([composite_from_store(frame, r, spec, episode=0, cache={}) for r in rows])
    g = GpuMaskComposite(spec)
    s_, e_ = g.union_intervals(rows)
    union = g.union_from_intervals(s_, e_, len(rows))
    frames = torch.from_numpy(
        np.ascontiguousarray(np.transpose(np.repeat(frame[None], len(rows), 0), (0, 3, 1, 2)))
    ).cuda()
    gpu = g(frames, union).cpu().numpy().transpose(0, 2, 3, 1)

    d = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
    per_px = d.max(axis=3)
    assert d.max() <= 2, f"composite diverged: max {d.max()}"
    assert (per_px > 1).mean() <= 1e-4, f">1-level pixels too common: {(per_px > 1).mean():.2e}"


@cuda
def test_an_empty_row_renders_the_whole_frame_as_background():
    """No detection means everything is background — same rule as the CPU path."""
    spec = _spec({"key": "solid", "params": {"color": [10, 20, 30]}})
    g = GpuMaskComposite(spec)
    s_, e_ = g.union_intervals([""])
    union = g.union_from_intervals(s_, e_, 1)
    frame = torch.randint(0, 255, (1, 3, H, W), dtype=torch.uint8, device="cuda")
    out = g(frame, union)
    # Deep interior is pure background colour (the feather only softens edges
    # of detections, of which there are none).
    assert (out[0, :, H // 2, W // 2].cpu() == torch.tensor([10, 20, 30], dtype=torch.uint8)).all()


def test_unsupported_recipes_refuse_loudly():
    spec = _spec({"key": "blur", "params": {}})
    spec["mask_treatments"]["a"] = {"key": "tint", "params": {"color": [255, 0, 0]}}
    with pytest.raises(NotImplementedError, match="CPU data path"):
        GpuMaskComposite(spec, device="cpu")
    spec2 = _spec({"key": "random", "params": {}})
    with pytest.raises(NotImplementedError, match="CPU data path"):
        GpuMaskComposite(spec2, device="cpu")
