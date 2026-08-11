# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A binder that always answers turns a mis-bind into a confident wrong motion. These
tests care as much about the abstentions as about the successes."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from lerobot.showservo.binder import BindGate, BindResult, SiftBinder, sift_keypoints  # noqa: E402
from lerobot.showservo.card import GoalRelation, Keypoint, Stage, Termination  # noqa: E402

CANVAS = 360
VIEW = 240


def _canvas(seed: int = 0) -> np.ndarray:
    """Blurred noise, not tidy shapes.

    A canvas of similar circles and squares produces near-duplicate SIFT descriptors,
    the ratio test correctly kills almost all of them, and the binder then abstains on
    3 inliers — which is the binder behaving well against an unrealistic fixture.
    Blurred noise gives the distinctive multi-scale structure a real textured object
    has, so the gate is tested rather than the fixture's poverty.
    """
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(CANVAS, CANVAS), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def _view(canvas: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    x0, y0 = 60 + dx, 60 + dy
    return np.ascontiguousarray(canvas[y0 : y0 + VIEW, x0 : x0 + VIEW])


def _stage_from(frame: np.ndarray, n: int = 40) -> Stage:
    """Compile a stage the way the offline compiler will: detect, describe, store."""
    pts, desc = sift_keypoints(frame, max_points=n)
    assert len(pts) >= 6
    return Stage(
        name="synthetic",
        camera="top",
        teams={"target": [Keypoint(uv=p, descriptor=d) for p, d in zip(pts, desc, strict=True)]},
        goal_relation=GoalRelation(held_uv=np.array([[30.0, 30.0]])),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("contact"),
    )


def test_descriptors_come_back_normalised_and_point_aligned():
    frame = _view(_canvas())
    pts, desc = sift_keypoints(frame, max_points=20)
    assert len(pts) == len(desc)
    assert desc.shape[1] == 128
    np.testing.assert_allclose(np.linalg.norm(desc, axis=1), 1.0, atol=1e-5)


def test_extraction_can_be_confined_to_the_designated_region():
    # §4 extracts keypoints on the SAM3 target mask; anything outside it is scene, and
    # scene in a card is exactly what invariant 3 forbids.
    frame = _view(_canvas(11))
    mask = np.zeros(frame.shape, dtype=bool)
    mask[40:140, 40:140] = True
    pts, _ = sift_keypoints(frame, mask)
    assert len(pts) > 0
    assert ((pts[:, 0] >= 38) & (pts[:, 0] <= 142) & (pts[:, 1] >= 38) & (pts[:, 1] <= 142)).all()


def test_binding_recovers_a_pure_translation():
    canvas = _canvas(1)
    stage = _stage_from(_view(canvas))
    result = SiftBinder().bind(_view(canvas, dx=18, dy=-11), stage)

    assert result.ok, result.reason
    np.testing.assert_allclose(result.sim2.t, [-18.0, 11.0], atol=2.0)
    assert abs(result.sim2.theta) < np.deg2rad(2.0)
    assert result.n_inliers >= 6


def test_binding_abstains_on_a_scene_it_was_never_taught():
    stage = _stage_from(_view(_canvas(2)))
    result = SiftBinder().bind(_view(_canvas(99)), stage)
    assert not result.ok
    assert result.reason


def test_a_blank_frame_abstains_rather_than_erroring():
    stage = _stage_from(_view(_canvas(3)))
    result = SiftBinder().bind(np.full((VIEW, VIEW), 128, dtype=np.uint8), stage)
    assert not result.ok
    assert result.reason


def test_a_card_without_descriptors_reports_why_it_cannot_be_bound():
    stage = Stage(
        camera="top",
        teams={"target": [Keypoint(uv=[10.0, 10.0]), Keypoint(uv=[40.0, 20.0])]},
        goal_relation=GoalRelation(held_uv=[[1.0, 1.0]]),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("contact"),
    )
    result = SiftBinder().bind(_view(_canvas(4)), stage)
    assert not result.ok
    assert "descriptors" in result.reason


def test_an_absent_team_is_reported_not_crashed(d1_stage):
    result = SiftBinder().bind(_view(_canvas(5)), d1_stage, team="held")
    assert not result.ok
    assert "no held team" in result.reason


def test_a_strict_gate_abstains_where_a_loose_one_would_commit():
    canvas = _canvas(6)
    stage = _stage_from(_view(canvas))
    frame = _view(canvas, dx=14)
    assert SiftBinder().bind(frame, stage).ok
    strict = SiftBinder(gate=BindGate(min_inliers=500, min_inlier_ratio=0.99, max_rms_px=0.001))
    assert not strict.bind(frame, stage).ok


def test_seeding_fills_unmatched_taught_points_through_the_transform():
    canvas = _canvas(7)
    stage = _stage_from(_view(canvas))
    result = SiftBinder().bind(_view(canvas, dx=12, dy=6), stage)
    assert result.ok

    taught = stage.team_uv("target")
    uv, measured = result.seed_points(taught)
    assert uv.shape == taught.shape and measured.shape == (len(taught),)
    assert measured.any()
    np.testing.assert_allclose(uv[measured], result.live_uv, atol=1e-9)
    # Points the matcher missed are still placed, via the consensus transform.
    if (~measured).any():
        np.testing.assert_allclose(uv[~measured], result.sim2.apply(taught[~measured]), atol=1e-9)


def test_seeding_from_a_failed_bind_is_refused():
    with pytest.raises(AssertionError, match="fabricate correspondence"):
        BindResult(ok=False).seed_points(np.array([[1.0, 2.0]]))
