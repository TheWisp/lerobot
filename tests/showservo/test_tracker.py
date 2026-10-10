# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""KLT's own status flag stays 1 through an occlusion, sliding the point onto whatever
texture arrives. The forward-backward gate is the only thing standing between that and
a fit built on a lie, so it is what these tests are really about."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from lerobot.showservo.tracker import KLTTracker, shi_tomasi_points  # noqa: E402

CANVAS = 320
VIEW = 200


def _canvas(seed: int = 0) -> np.ndarray:
    """High-contrast blobs: texture KLT can actually lock onto."""
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((CANVAS, CANVAS), 30, dtype=np.uint8)
    for _ in range(70):
        c = (int(rng.integers(0, CANVAS)), int(rng.integers(0, CANVAS)))
        cv2.circle(img, c, int(rng.integers(3, 9)), int(rng.integers(120, 255)), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _view(canvas: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    x0, y0 = 60 + dx, 60 + dy
    return np.ascontiguousarray(canvas[y0 : y0 + VIEW, x0 : x0 + VIEW])


def test_points_follow_a_translating_scene():
    canvas = _canvas()
    frame0 = _view(canvas)
    pts = shi_tomasi_points(frame0, max_points=12, min_distance=12.0)
    assert len(pts) >= 6, "fixture must offer enough corners to be a real test"

    tracker = KLTTracker()
    tracker.init(frame0, pts)
    for step in range(1, 6):
        state = tracker.step(_view(canvas, dx=2 * step, dy=step))

    assert state.n_valid >= len(pts) - 1
    expected = pts[state.valid] - np.array([10.0, 5.0])
    np.testing.assert_allclose(state.uv[state.valid], expected, atol=1.0)


def test_a_point_swallowed_by_a_blank_occluder_is_reported_lost():
    import cv2

    canvas = _canvas(1)
    frame0 = _view(canvas)
    pts = shi_tomasi_points(frame0, max_points=10, min_distance=15.0)
    tracker = KLTTracker()
    tracker.init(frame0, pts)

    victim = int(np.argmin(np.linalg.norm(pts - np.array([VIEW / 2, VIEW / 2]), axis=1)))
    for _ in range(4):
        occluded = _view(canvas).copy()
        x, y = pts[victim]
        cv2.rectangle(occluded, (int(x) - 22, int(y) - 22), (int(x) + 22, int(y) + 22), 90, -1)
        state = tracker.step(occluded)

    assert not state.valid[victim], "a point over a blank occluder must not be believed"


def test_a_lost_point_stays_lost():
    canvas = _canvas(2)
    frame0 = _view(canvas)
    pts = shi_tomasi_points(frame0, max_points=8, min_distance=15.0)
    tracker = KLTTracker()
    tracker.init(frame0, pts)
    tracker.drop([0])

    for step in range(1, 4):
        state = tracker.step(_view(canvas, dx=step))
    assert not state.valid[0], "resurrection would silently re-admit an evicted point"


def test_indices_are_stable_when_points_are_added_mid_track():
    canvas = _canvas(3)
    frame0 = _view(canvas)
    pts = shi_tomasi_points(frame0, max_points=6, min_distance=15.0)
    tracker = KLTTracker()
    tracker.init(frame0, pts)
    tracker.step(_view(canvas, dx=1))

    extra = shi_tomasi_points(_view(canvas, dx=1), max_points=3, min_distance=15.0, exclude=pts)
    idx = tracker.add(extra)
    assert list(idx) == list(range(len(pts), len(pts) + len(extra)))
    assert len(tracker.state.uv) == len(pts) + len(extra)


def test_replenishment_can_be_confined_to_a_region():
    # Replenishing from the whole frame would rent background into the team and break
    # invariant 3; the mask is what keeps new points on the object.
    frame = _view(_canvas(4))
    mask = np.zeros(frame.shape, dtype=bool)
    mask[20:80, 20:80] = True
    pts = shi_tomasi_points(frame, mask, max_points=20, min_distance=6.0)
    assert len(pts) > 0
    assert ((pts[:, 0] >= 20) & (pts[:, 0] < 80) & (pts[:, 1] >= 20) & (pts[:, 1] < 80)).all()


def test_replenishment_does_not_duplicate_points_already_tracked():
    frame = _view(_canvas(5))
    existing = shi_tomasi_points(frame, max_points=10, min_distance=10.0)
    fresh = shi_tomasi_points(frame, max_points=10, min_distance=4.0, exclude=existing, exclude_radius=10.0)
    if len(fresh):
        d = np.linalg.norm(fresh[:, None, :] - existing[None, :, :], axis=2)
        assert d.min() > 10.0


def test_seeding_outside_the_frame_is_refused():
    frame = _view(_canvas(6))
    tracker = KLTTracker()
    with pytest.raises(AssertionError, match="inside the frame"):
        tracker.init(frame, np.array([[VIEW + 10.0, 5.0]]))


def test_stepping_before_init_is_refused():
    with pytest.raises(AssertionError, match="before init"):
        KLTTracker().step(_view(_canvas(7)))


def test_an_rgb_frame_is_accepted_as_readily_as_grayscale():
    canvas = _canvas(8)
    rgb0 = np.repeat(_view(canvas)[:, :, None], 3, axis=2)
    pts = shi_tomasi_points(rgb0, max_points=8, min_distance=12.0)
    tracker = KLTTracker()
    tracker.init(rgb0, pts)
    state = tracker.step(np.repeat(_view(canvas, dx=3)[:, :, None], 3, axis=2))
    assert state.n_valid >= len(pts) - 1
