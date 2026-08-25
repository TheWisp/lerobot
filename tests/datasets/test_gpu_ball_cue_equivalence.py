# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GPU data path must build the same ball cue the CPU path does.

The experiment compares two ways of presenting a cue to the policy. If the two
data paths disagreed about the cue itself, the comparison would be measuring
the data path instead, so the definition lives in ball_cue.py and this file
pins the on-device reimplementation against it.

Two things are checked separately because they fail differently: the
coordinate (a reduction, exact up to float rounding) and the rendered view (a
masked frame plus a resize, where an off-by-one in the RLE's column-major
order shows up as a transposed ball rather than a wrong number).

The masks are synthetic but real-shaped -- thresholded smooth noise, thousands
of RLE runs -- because a rectangle has ~2 runs and hides interval-order bugs.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from lerobot.datasets.gpu_data_pipeline import _region_centroid  # noqa: E402
from lerobot.datasets.gpu_mask_composite import GpuMaskComposite  # noqa: E402
from lerobot.datasets.mask_codec import encode_frame  # noqa: E402
from lerobot.policies.hvla.s1.flow_matching.ball_cue import (  # noqa: E402
    NOT_VISIBLE,
    ball_cue,
    render_ball_view,
)

H, W = 720, 1280
LABELS = ["yellow ball", "holder"]
RESIZE = (224, 224)


def _spec() -> dict:
    return {
        "mask_encoding": "coco_rle",
        "mask_labels": LABELS,
        "mask_size": [H, W],
        "mask_treatments": {n: {"key": "none", "params": {}} for n in LABELS},
        "mask_background": {"key": "none", "params": {}},
    }


def _blob(seed: int, quantile: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 9)
    return noise > np.quantile(noise, quantile)


def _rows(seeds) -> tuple[list[str], list[np.ndarray]]:
    """One RLE row per frame, plus the ball masks the CPU path would decode."""
    rows, balls = [], []
    for s in seeds:
        if s is None:  # a frame where nothing was detected
            rows.append("")
            balls.append(None)
            continue
        ball = _blob(s, 0.995)
        # A second, larger label the cue must ignore -- the composite unions
        # every label, so a cue that reused the union would track this instead.
        rows.append(encode_frame({"yellow ball": ball, "holder": _blob(s + 100, 0.90)}, LABELS))
        balls.append(ball)
    return rows, balls


def _device_masks(rows: list[str], device: str) -> torch.Tensor:
    comp = GpuMaskComposite(_spec(), device=device)
    starts, ends = comp.label_intervals(rows, 0)
    return comp.union_from_intervals(starts, ends, len(rows))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_the_on_device_mask_is_the_label_the_cpu_path_decodes(device):
    """Everything downstream is only as right as this."""
    rows, balls = _rows([1, 2, None, 3])
    got = _device_masks(rows, device).cpu().numpy()[:, 0] > 0.5
    for i, ball in enumerate(balls):
        expected = np.zeros((H, W), bool) if ball is None else ball
        assert (got[i] == expected).all(), f"frame {i}: on-device mask is not the ball label"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_the_coordinate_matches_ball_cue(device):
    rows, balls = _rows([1, 2, None, 3])
    got = _region_centroid(_device_masks(rows, device), NOT_VISIBLE[:2]).cpu().numpy()
    for i, ball in enumerate(balls):
        want = np.array(ball_cue(ball), dtype=np.float32)
        # float32 output of a float64 reduction: the tolerance is the storage,
        # not the arithmetic.
        assert np.allclose(got[i], want, atol=1e-6), f"frame {i}: {got[i]} != {want}"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_the_rendered_view_matches_render_ball_view(device):
    from torchvision.transforms import functional as tvf

    rows, balls = _rows([1, 2, None, 3])
    rng = np.random.default_rng(0)
    frames_np = rng.integers(0, 256, (len(rows), H, W, 3), dtype=np.uint8)

    mask = _device_masks(rows, device)
    frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous().to(device)
    got = frames.to(torch.float32) * mask / 255.0
    got = tvf.resize(got, list(RESIZE), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)

    for i, ball in enumerate(balls):
        view = render_ball_view(frames_np[i], ball)
        want = torch.from_numpy(view).permute(2, 0, 1).float() / 255.0
        want = tvf.resize(
            want[None], list(RESIZE), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True
        )[0]
        diff = (got[i].cpu() - want).abs().max()
        assert diff < 1e-5, f"frame {i}: views differ by {diff}"


def test_a_miss_renders_black_and_reports_not_visible():
    """The sentinel is what the policy branches on; it must survive the port."""
    mask = _device_masks(["", ""], "cpu")
    cue = _region_centroid(mask, NOT_VISIBLE[:2]).numpy()
    assert (cue == np.array(NOT_VISIBLE, dtype=np.float32)).all(), cue
    assert mask.sum() == 0


def test_the_label_index_is_validated():
    comp = GpuMaskComposite(_spec(), device="cpu")
    with pytest.raises(ValueError, match="outside the declared vocabulary"):
        comp.label_intervals([""], 7)
