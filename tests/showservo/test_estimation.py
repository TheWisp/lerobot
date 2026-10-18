# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The layer that says which tracker output is a measurement (src/lerobot/showservo/estimation.py)."""

from __future__ import annotations

import numpy as np

from lerobot.showservo.estimation import Candidate, SourceVerdict


def test_a_sources_verdict_takes_that_sources_candidate_when_it_is_confident_and_says_why_not():
    a, b = np.eye(4), np.eye(4) * 2.0
    out = SourceVerdict("second").estimate(
        1.0,
        {
            "block": [Candidate("first", a, ok=True), Candidate("second", b, ok=True)],
            "ring": [Candidate("first", a, ok=True), Candidate("second", b, ok=False, why="lost")],
            "cup": [Candidate("first", a, ok=True)],
        },
    )
    assert out["block"].motion is b and out["block"].why == "second: confident"
    assert out["ring"].motion is None and out["ring"].why == "second: lost"
    assert out["cup"].motion is None and out["cup"].why == "no second candidate"


# ── seen through: a pose whose surface the camera sees past ──

FX = FY = 600.0
CX, CY, H, W = 424.0, 240.0, 480, 848  # the rig camera's size; the camera is the base frame here
TABLE_M, TOP_M = 0.45, 0.44  # a 100 x 20 mm block's top, 10 mm over the table


def _block():
    x, y = np.meshgrid(np.arange(-0.05, 0.0501, 0.002), np.arange(-0.01, 0.0101, 0.002))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, TOP_M)]


def _depth(rng, occluder=None):
    """The table with the block on it, 1 mm of noise; ``occluder`` a pixel box (u0, u1, v0, v1) at 0.30 m."""
    v, u = np.mgrid[0:H, 0:W]
    x, y = (u - CX) / FX * TOP_M, (v - CY) / FY * TOP_M
    depth = np.where((np.abs(x) <= 0.05) & (np.abs(y) <= 0.01), TOP_M, TABLE_M)
    depth = depth + rng.normal(0.0, 0.001, depth.shape)
    if occluder is not None:
        u0, u1, v0, v1 = occluder
        depth[v0:v1, u0:u1] = 0.30
    return depth


def _shifted(dx, dz):
    m = np.eye(4)
    m[0, 3], m[2, 3] = dx, dz
    return m


def test_a_pose_the_camera_sees_through_is_refused_and_one_it_cannot_see_is_not():
    from lerobot.showservo.estimation import POINT2POSE, Frame, SeenThrough

    rng = np.random.default_rng(0)
    est = SeenThrough(POINT2POSE, {"block": _block()})
    frame = lambda occluder=None: Frame(_depth(rng, occluder), FX, FY, CX, CY, np.eye(4))  # noqa: E731
    view = lambda m: {"block": [Candidate(POINT2POSE, m, ok=True)]}  # noqa: E731
    for k in range(
        12
    ):  # its first second of views, at rest where it is: they pass while its tolerances are measured
        out = est.estimate(k / 10, view(np.eye(4)), frame())["block"]
        assert out.motion is not None
    assert "block" in est.limits
    assert est.estimate(1.3, view(np.eye(4)), frame())["block"].motion is not None
    # 80 mm to the side and 30 mm nearer the camera: where the camera sees the table, 40 mm farther than that surface
    phantom = _shifted(0.08, -0.03)
    refused = est.estimate(1.4, view(phantom), frame())["block"]
    assert refused.motion is None and "sees through" in refused.why
    # the same pose behind something nearer than it: hidden, not contradicted
    u = (np.array([0.03, 0.13]) / (TOP_M - 0.03) * FX + CX).astype(int)
    hidden = est.estimate(1.5, view(phantom), frame((u[0] - 20, u[1] + 20, 200, 280)))["block"]
    assert hidden.motion is not None
