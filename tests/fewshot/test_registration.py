# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The registration core must be exact on clean data, robust to outliers, and honest
about what it cannot know (rotation of a featureless/symmetric object)."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.fewshot.registration import Sim2, fit_similarity, mutual_matches, ransac_register


def _random_sim2(rng) -> Sim2:
    return Sim2.from_angle(
        rng.uniform(-np.pi, np.pi), t=rng.uniform(-50, 50, size=2), s=rng.uniform(0.7, 1.4)
    )


def test_sim2_compose_and_inverse_are_consistent():
    rng = np.random.default_rng(0)
    a, b = _random_sim2(rng), _random_sim2(rng)
    pts = rng.uniform(-10, 10, size=(7, 2))
    np.testing.assert_allclose(a.compose(b).apply(pts), a.apply(b.apply(pts)), atol=1e-9)
    np.testing.assert_allclose(a.inverse().apply(a.apply(pts)), pts, atol=1e-9)


@pytest.mark.parametrize("allow_scale", [True, False])
def test_fit_similarity_recovers_exactly(allow_scale):
    rng = np.random.default_rng(1)
    true = Sim2.from_angle(0.7, t=(12.0, -3.0), s=1.0 if not allow_scale else 1.21)
    src = rng.uniform(0, 100, size=(20, 2))
    fit = fit_similarity(src, true.apply(src), allow_scale=allow_scale)
    np.testing.assert_allclose(fit.apply(src), true.apply(src), atol=1e-9)
    assert abs(fit.theta - true.theta) < 1e-9


def test_fit_similarity_never_produces_a_reflection():
    # A reflection can fit mirrored data better; Umeyama's sign correction must
    # refuse it — a robot trajectory must never be mirrored.
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dst = src * np.array([-1.0, 1.0])  # mirrored
    fit = fit_similarity(src, dst)
    assert float(np.linalg.det(fit.R)) > 0


def test_mutual_matches_finds_permuted_identity():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(30, 16))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    perm = rng.permutation(30)
    ia, ib = mutual_matches(f, f[perm])
    assert len(ia) == 30
    np.testing.assert_array_equal(perm[ib], ia)


def _featured_points(rng, n=60, d=32):
    pts = rng.uniform(0, 200, size=(n, 2))
    feats = rng.normal(size=(n, d))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return pts, feats


def test_ransac_recovers_under_outliers():
    rng = np.random.default_rng(3)
    pts, feats = _featured_points(rng)
    true = Sim2.from_angle(-1.1, t=(30.0, 8.0), s=1.0)
    dst = true.apply(pts)
    # 30% of the features on the B side are swapped between points: wrong matches
    # with plausible geometry on one end.
    feats_b = feats.copy()
    swap = rng.choice(len(pts), size=18, replace=False)
    feats_b[swap] = feats_b[rng.permutation(swap)]
    res = ransac_register(pts, dst, feats, feats_b, allow_scale=False, inlier_dist=2.0)
    assert res.ok
    assert abs(res.sim2.theta - true.theta) < 1e-6
    np.testing.assert_allclose(res.sim2.t, true.t, atol=1e-6)
    assert res.rotation_needed and res.rotation_trustworthy


def test_pure_translation_reports_rotation_not_needed():
    rng = np.random.default_rng(4)
    pts, feats = _featured_points(rng)
    res = ransac_register(pts, pts + np.array([40.0, -12.0]), feats, feats, allow_scale=False)
    assert res.ok and not res.rotation_needed
    assert not res.rotation_trustworthy  # angle must not be transferred
    np.testing.assert_allclose(res.sim2.t, [40.0, -12.0], atol=1e-6)


def test_featureless_object_is_reported_untrustworthy_not_wrong():
    """A symmetric/textureless object: every patch looks like every other. The
    danger is a confidently wrong angle; the contract is low confidence instead."""
    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 100, size=(40, 2))
    feats = np.tile(rng.normal(size=(1, 16)), (40, 1))  # identical features
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    res = ransac_register(pts, Sim2.from_angle(0.9).apply(pts), feats, feats, inlier_dist=2.0)
    assert (not res.ok) or (not res.rotation_trustworthy)


def test_too_few_matches_fails_closed():
    res = ransac_register(
        np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 8)) / np.sqrt(8), np.ones((1, 8)) / np.sqrt(8)
    )
    assert not res.ok
