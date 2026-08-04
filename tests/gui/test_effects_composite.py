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
"""Per-region composite — the unified WYSIWYG transform shared by the live
preview and the offline pass. Pure numpy/cv2; masks are hard-edged (feather=0) so
pixel assertions are exact."""

from __future__ import annotations

import numpy as np

from lerobot.overlays.effects import composite_regions, feathered_alpha, sample_treatment


def test_all_none_is_identity():
    rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    full = np.ones((8, 8), dtype=np.float32)
    out = composite_regions(rgb, [(full, {"key": "none"})], [{}])
    assert np.array_equal(out, rgb)  # a None region keeps every pixel


def test_background_treatment_keeps_object_pixels():
    h = w = 8
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    objm = np.zeros((h, w), dtype=np.uint8)
    objm[:, :4] = 1  # object = left half
    obj_alpha = feathered_alpha([objm], h, w, feather=0)
    bg_alpha = 1.0 - obj_alpha
    # Background solid colour; object kept as-is → GreenAug shape (deterministic here).
    regions = [(bg_alpha, {"key": "solid", "params": {"color": [10, 20, 30]}}), (obj_alpha, {"key": "none"})]
    out = composite_regions(rgb, regions, [{}, {}])
    assert np.array_equal(out[0, 0], [100, 100, 100])  # object pixel untouched
    assert np.array_equal(out[0, 7], [10, 20, 30])  # background replaced


def test_tint_blends_toward_colour():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)  # black
    full = np.ones((4, 4), dtype=np.float32)
    out = composite_regions(
        rgb, [(full, {"key": "tint", "params": {"color": [200, 0, 0], "strength": 0.5}})], [{}]
    )
    assert np.array_equal(out[0, 0], [100, 0, 0])  # black blended 0.5 toward red


def test_object_treatment_composites_over_background():
    h = w = 8
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    objm = np.zeros((h, w), dtype=np.uint8)
    objm[:, :4] = 1
    obj_alpha = feathered_alpha([objm], h, w, feather=0)
    bg_alpha = 1.0 - obj_alpha
    # Background solid black; object tinted fully red.
    regions = [
        (bg_alpha, {"key": "solid", "params": {"color": [0, 0, 0]}}),
        (obj_alpha, {"key": "tint", "params": {"color": [255, 0, 0], "strength": 1.0}}),
    ]
    out = composite_regions(rgb, regions, [{}, {}])
    assert np.array_equal(out[0, 0], [255, 0, 0])  # object → red
    assert np.array_equal(out[0, 7], [0, 0, 0])  # background → black


def test_sample_treatment_randomness():
    rng = np.random.default_rng(0)
    assert "bg" in sample_treatment("random", {}, 4, 4, rng)  # random = a texture, not a flat colour
    assert sample_treatment("none", {}, 4, 4, rng) == {}
    assert sample_treatment("tint", {}, 4, 4, rng) == {}


def _feathered_alpha_reference(masks, h, w, feather=5):
    """The pre-ROI full-frame implementation, kept verbatim as the equality oracle."""
    import cv2

    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m is not None and m.shape == (h, w):
            union |= m.astype(np.uint8)
    if not union.any():
        return np.zeros((h, w), dtype=np.float32)
    if feather > 0:
        ksz = feather * 2 + 1
        union = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
        soft = cv2.GaussianBlur(union.astype(np.float32) * 255.0, (ksz, ksz), 0) / 255.0
        return np.clip(soft, 0.0, 1.0)
    return union.astype(np.float32)


def test_feathered_alpha_roi_matches_full_frame():
    # The ROI-bounded feather must match the full-frame version to float32 rounding —
    # it is a WYSIWYG-committed value, so this is an output contract. Exact bit
    # equality is not achievable: cv2's SIMD accumulation order shifts with the
    # crop's row alignment (measured divergence ~1e-9). The 1e-6 bound is still
    # ~250x below one uint8 pixel step, so committed pixels cannot visibly differ.
    h, w = 120, 200
    rng = np.random.default_rng(7)
    cases = []
    m = np.zeros((h, w), dtype=bool)
    m[40:60, 80:120] = True
    cases.append([m])  # interior blob
    e = np.zeros((h, w), dtype=bool)
    e[0:15, 0:25] = True
    e[h - 8 :, w - 30 :] = True
    cases.append([e])  # frame-edge blobs (ROI clipped to the border)
    r1 = rng.random((h, w)) > 0.995
    cases.append([r1, m])  # sparse speckle + blob, multi-mask union
    cases.append([np.zeros((h, w), dtype=bool)])  # empty -> all-zero alpha
    for masks in cases:
        for feather in (0, 3, 5):
            got = feathered_alpha(masks, h, w, feather=feather)
            ref = _feathered_alpha_reference(masks, h, w, feather=feather)
            assert got.shape == ref.shape == (h, w)
            assert np.allclose(got, ref, atol=1e-6, rtol=0), (
                f"ROI feather diverged (feather={feather}, maxdiff={np.abs(got - ref).max()})"
            )


def test_overlapping_masks_contested_pixels_go_to_the_smaller_object():
    """Overlap policy: the most specific (smallest) object wins contested pixels,
    regardless of listing order. The bug this pins (measured on pick_ball): the arm
    mask swallowed the whole ball on grasp frames, and dict-order painting tinted
    the ball with the ARM's colour whenever the arm was listed later."""
    from lerobot.overlays.effects import build_and_sample_regions

    h = w = 32
    big = np.zeros((h, w), dtype=bool)
    big[4:28, 4:28] = True  # "robot arm"
    small = np.zeros((h, w), dtype=bool)
    small[12:18, 12:18] = True  # "ball", entirely inside the arm mask
    arm_t = {"key": "tint", "params": {"color": [255, 0, 0], "strength": 1.0}}
    ball_t = {"key": "tint", "params": {"color": [0, 0, 255], "strength": 1.0}}
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)

    outs = []
    for order in ({"ball": small, "robot arm": big}, {"robot arm": big, "ball": small}):
        regions, sampled = build_and_sample_regions(
            order, {"robot arm": arm_t, "ball": ball_t}, {"key": "none"}, h, w, rng, {}, feather=0
        )
        outs.append(composite_regions(frame, regions, sampled))
    # The ball's centre is tinted BLUE in both orders — never the arm's red.
    for out in outs:
        center = out[15, 15]
        assert center[2] > 200 and center[0] < 60, f"ball centre got the wrong tint: {center}"
    np.testing.assert_array_equal(outs[0], outs[1])  # order-independent


def test_kept_region_survives_bit_exact_under_feathering():
    """A region with treatment "none" must come back byte-identical.

    The bug: at a large object's centre the feathered alpha is 0.9999998 (float32
    epsilon, 1.0 for every practical purpose), so the blend lands at 219.99996 —
    and the final `astype(uint8)` TRUNCATED it to 219. Every treated pixel in every
    written dataset was biased down by up to one level, and "keep this region"
    quietly wasn't exact. Found by an end-to-end synthetic job run.
    """
    from lerobot.overlays.effects import build_and_sample_regions

    h, w = 48, 64
    mask = np.zeros((h, w), dtype=bool)
    mask[12:36, 4:28] = True  # 24x24 — comfortably larger than the 5 px feather
    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    frame[mask] = (220, 30, 30)

    regions, sampled = build_and_sample_regions(
        {"obj": mask},
        {"obj": {"key": "none"}},
        {"key": "tint", "params": {"color": [0, 0, 0], "strength": 1.0}},  # black background
        h,
        w,
        np.random.default_rng(0),
        {},
    )
    out = composite_regions(frame, regions, sampled)
    assert out[24, 16].tolist() == [220, 30, 30], "kept region must not be darkened by the cast"
