# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The template bank's reason to exist, demonstrated with a deliberately
rotation-VARIANT extractor: matching across a large in-plane rotation fails with a
single template and succeeds — with the right composed angle — once the examine
phase stores rotated copies. This mirrors the real situation (ViT patch features
are not rotation-equivariant) without any model download."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lerobot.fewshot.registration import Sim2  # noqa: E402
from lerobot.fewshot.templates import TemplateBank  # noqa: E402


class WindowExtractor:
    """Features = raw pixel windows on a grid inside the mask — axis-aligned, so
    inherently rotation-variant, like the real features but with zero deps."""

    def __init__(self, win: int = 11, stride: int = 4):
        self.win, self.stride = win, stride

    def extract(self, image: np.ndarray, mask: np.ndarray):
        h, w = mask.shape
        r = self.win // 2
        coords, feats = [], []
        ys, xs = np.nonzero(mask)
        for y in range(ys.min() + r, ys.max() - r, self.stride):
            for x in range(xs.min() + r, xs.max() - r, self.stride):
                if not mask[y, x]:
                    continue
                f = image[y - r : y + r + 1, x - r : x + r + 1].astype(np.float64).ravel()
                n = np.linalg.norm(f)
                if n < 1e-9:
                    continue
                coords.append([x, y])
                feats.append(f / n)
        assert feats, "mask too small for the window grid"
        return np.asarray(coords, dtype=np.float64), np.asarray(feats)


def _scene(seed=0):
    """A textured blob on black: smooth noise so nearby windows correlate.

    Sized so a ~10 deg residual rotation is RESOLVABLE: rotation is observable only
    where the arc displacement at the object's radius exceeds the patch spacing.
    At radius 45 px, 10 deg sweeps ~8 px against a 4 px grid — measurable. (The
    first version used radius 28 / stride 6, where 10 deg moves ~4 px: matching
    then locks to identical grid offsets and rotation quantises to the template
    step. That is a real property of patch-grid registration, worth remembering
    when choosing examine resolution for small objects.)"""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 255, size=(110, 110)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    canvas = np.zeros((320, 320, 3), np.uint8)
    canvas[105:215, 105:215] = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask = np.zeros((320, 320), bool)
    yy, xx = np.ogrid[:320, :320]
    mask[((yy - 160) ** 2 + (xx - 160) ** 2) < 45**2] = True
    return canvas, mask


def _rotated_view(image, mask, deg, t=(18.0, -9.0)):
    """Ground-truth live view: the scene rotated about the object centre + moved."""
    ys, xs = np.nonzero(mask)
    centre = (float(xs.mean()), float(ys.mean()))
    warp = cv2.getRotationMatrix2D(centre, deg, 1.0)
    warp[:, 2] += t
    img = cv2.warpAffine(image, warp, (image.shape[1], image.shape[0]))
    m = cv2.warpAffine(mask.astype(np.uint8), warp, (image.shape[1], image.shape[0])) > 0
    th = -np.deg2rad(deg)
    c = np.asarray(centre)
    rmat = Sim2.from_angle(th).R
    true = Sim2(1.0, rmat, c - rmat @ c + np.asarray(t))
    return img, m, true


def test_single_template_fails_across_a_large_rotation():
    image, mask = _scene()
    live_img, live_mask, _ = _rotated_view(image, mask, 40.0)
    bank = TemplateBank(WindowExtractor())
    bank.add_view("v0", image, mask, rotations_deg=(0.0,))
    m = bank.match(live_img, live_mask, allow_scale=False, inlier_dist=3.0)
    # Axis-aligned windows decorrelate at 40 deg: either no registration at all or
    # one too weak to trust. This failing case is WHY examine stores rotations.
    assert m is None or m.result.n_inliers < 8


@pytest.mark.parametrize("deg", [40.0, 90.0, 160.0])
def test_rotated_templates_recover_the_true_transform(deg):
    image, mask = _scene()
    live_img, live_mask, true = _rotated_view(image, mask, deg)
    bank = TemplateBank(WindowExtractor())
    bank.add_view("v0", image, mask, rotations_deg=tuple(float(d) for d in range(0, 360, 30)))
    m = bank.match(live_img, live_mask, allow_scale=False, inlier_dist=3.0)
    assert m is not None and m.result.n_inliers >= 8
    # The composed source->live transform must agree with the ground-truth warp.
    err_deg = np.rad2deg(abs((m.source_to_live.theta - true.theta + np.pi) % (2 * np.pi) - np.pi))
    assert err_deg < 6.0, f"angle error {err_deg:.1f} deg"
    ys, xs = np.nonzero(mask)
    centre = np.array([xs.mean(), ys.mean()])[None]
    np.testing.assert_allclose(
        m.source_to_live.apply(centre), true.apply(centre), atol=3.0
    )  # the object centre lands within a few px


def test_bank_requires_examination_first():
    bank = TemplateBank(WindowExtractor())
    with pytest.raises(AssertionError, match="examine first"):
        bank.match(np.zeros((32, 32, 3), np.uint8), np.ones((32, 32), bool))


def test_symmetric_texture_flags_rotation_ambiguous():
    """A texture with EXACT 180-deg symmetry: the 0- and 180-deg templates explain a
    live view equally well at composed angles 180 deg apart. The bank must say so —
    measured on real pixels, this failure mode produced a confidently wrong
    168.6-deg error with a single template, which no per-registration check caught."""
    image, mask = _scene()
    sym = np.maximum(image, cv2.rotate(image, cv2.ROTATE_180))  # f(x) == f(rot180(x))
    # Soften so the symmetry survives warpAffine resampling: the rotated template is
    # bilinearly resampled, and sharp noise decorrelates pixel-window features enough
    # to hide the (real) 180-deg alternate hypothesis behind interpolation loss.
    sym = cv2.GaussianBlur(sym, (0, 0), 1.5)
    bank = TemplateBank(WindowExtractor())
    bank.add_view("v0", sym, mask, rotations_deg=tuple(float(d) for d in range(0, 360, 30)))
    live_img, live_mask, _ = _rotated_view(sym, mask, 0.0)
    m = bank.match(live_img, live_mask, allow_scale=False, inlier_dist=3.0)
    assert m is not None
    assert m.rotation_ambiguous, "180-deg-symmetric object must not offer a trusted angle"
    assert not m.trust_rotation


def test_textured_object_is_unambiguous():
    image, mask = _scene()
    bank = TemplateBank(WindowExtractor())
    bank.add_view("v0", image, mask, rotations_deg=tuple(float(d) for d in range(0, 360, 30)))
    live_img, live_mask, _ = _rotated_view(image, mask, 40.0)
    m = bank.match(live_img, live_mask, allow_scale=False, inlier_dist=3.0)
    assert m is not None and not m.rotation_ambiguous and m.trust_rotation
