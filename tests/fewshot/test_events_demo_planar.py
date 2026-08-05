# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Event detection, demo extraction, homography, and the end-to-end transfer
invariant: move the object by a known transform and the replayed gripper must land
on the transformed grasp point exactly."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.fewshot.events import detect_interaction
from lerobot.fewshot.planar import PlanarDemo, apply_homography, fit_homography
from lerobot.fewshot.registration import Sim2

# ---- events ---------------------------------------------------------------


def _track(n=60, event=30, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    c = np.zeros((n, 2)) + rng.normal(scale=drift, size=(n, 2))
    c[event:] += np.cumsum(np.full((n - event, 2), 2.5), axis=0)  # steady motion
    a = np.full(n, 1000.0)
    return c, a


def test_event_found_at_motion_onset():
    c, a = _track()
    idx = detect_interaction(c, a)
    assert idx is not None and 30 <= idx <= 34


def test_still_object_never_fires():
    c, a = _track(event=60)  # never moves
    assert detect_interaction(c, a, move_px=4.0) is None


def test_jitter_below_threshold_never_fires():
    c, a = _track(event=60, drift=0.8, seed=1)
    assert detect_interaction(c, a, move_px=6.0) is None


def test_occlusion_dip_is_not_motion():
    """The gripper crossing in front shrinks the mask and yanks the visible
    centroid — the classic false trigger this detector exists to reject."""
    c, a = _track(event=60)
    a[20:26] = 100.0  # area collapses to 10%
    c[20:26] = [80.0, 80.0]  # visible-fragment centroid jumps far away
    assert detect_interaction(c, a) is None


def test_motion_during_partial_occlusion_still_detected_after():
    c, a = _track(event=25)
    a[25:35] = 100.0  # occluded exactly while motion starts
    idx = detect_interaction(c, a)
    assert idx is not None and idx >= 35  # detected once observable again


# ---- homography ------------------------------------------------------------


def _camera_h():
    # A plausible oblique camera: rotate + perspective terms, mapping px -> metres.
    return np.array([[0.0016, -0.0003, -0.4], [0.0002, -0.0019, 0.9], [1e-5, 3e-5, 1.0]])


def test_homography_roundtrip():
    h_mat = _camera_h()
    rng = np.random.default_rng(6)
    px = rng.uniform(0, 1280, size=(12, 2)) * [1, 0.5625]
    xy = apply_homography(h_mat, px)
    h_fit = fit_homography(px, xy)
    np.testing.assert_allclose(apply_homography(h_fit, px), xy, atol=1e-9)


def test_homography_rejects_collinear_points():
    px = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    with pytest.raises(AssertionError):
        fit_homography(px, px * 2.0)


# ---- demo extraction + transfer -------------------------------------------


def _demo(grasp_xy=(0.30, 0.10), yaw=0.5):
    """Synthetic demo: approach from above the grasp point, descend, close."""
    n = 40
    poses = np.zeros((n, 4))
    poses[:, 0] = np.linspace(grasp_xy[0] - 0.10, grasp_xy[0], n)
    poses[:, 1] = np.linspace(grasp_xy[1] + 0.06, grasp_xy[1], n)
    poses[:, 2] = np.linspace(0.12, 0.02, n)
    poses[:, 3] = yaw
    grip = np.concatenate([np.ones(35), np.zeros(5)])  # close at the end
    return poses, grip


def test_extract_starts_at_bottleneck_with_zero_offset():
    poses, grip = _demo()
    d = PlanarDemo.extract(poses, grip, event_idx=35, lead_frames=10)
    np.testing.assert_allclose(d.bottleneck_pose, poses[25], atol=1e-12)
    assert len(d.rel_traj) == 15
    np.testing.assert_allclose(d.rel_traj[0, :4], 0.0, atol=1e-12)


def test_identity_transfer_replays_the_demo_exactly():
    poses, grip = _demo()
    d = PlanarDemo.extract(poses, grip, event_idx=35, lead_frames=10)
    out = d.transfer(Sim2.identity(), np.array([0.3, 0.1]), trust_rotation=True)
    np.testing.assert_allclose(out[:, 0], poses[25:, 0], atol=1e-9)
    np.testing.assert_allclose(out[:, 1], poses[25:, 1], atol=1e-9)
    np.testing.assert_allclose(out[:, 3], poses[25:, 3], atol=1e-9)


def test_transfer_lands_on_the_transformed_grasp_point():
    """THE invariant: object moved by (R, t) -> final gripper XY is exactly
    (R, t) of the demo's final gripper XY, and approach yaw rotated with it."""
    poses, grip = _demo()
    d = PlanarDemo.extract(poses, grip, event_idx=35, lead_frames=10)
    motion = Sim2.from_angle(0.8, t=(0.05, -0.12))
    out = d.transfer(motion, np.array([0.30, 0.10]), trust_rotation=True)
    np.testing.assert_allclose(out[-1, :2], motion.apply(poses[-1, :2][None])[0], atol=1e-9)
    assert abs(out[-1, 3] - (poses[-1, 3] + 0.8)) < 1e-9
    np.testing.assert_allclose(out[:, 2], poses[25:, 2], atol=1e-12)  # z untouched
    np.testing.assert_allclose(out[:, 4], grip[25:], atol=1e-12)  # gripper timing kept


def test_untrusted_rotation_translates_by_the_object_centre():
    """Symmetric object: keep the demo approach direction, move to where the
    object went. The final grasp offset from the object centre is preserved in
    the DEMO's orientation, not swung around by an arbitrary angle."""
    poses, grip = _demo()
    d = PlanarDemo.extract(poses, grip, event_idx=35, lead_frames=10)
    centre = np.array([0.30, 0.10])
    motion = Sim2.from_angle(2.0, t=(0.05, -0.12))  # angle is arbitrary/garbage
    out = d.transfer(motion, centre, trust_rotation=False)
    d_centre = motion.apply(centre[None])[0] - centre
    np.testing.assert_allclose(out[-1, :2], poses[-1, :2] + d_centre, atol=1e-9)
    np.testing.assert_allclose(out[:, 3], poses[25:, 3], atol=1e-9)  # yaw unchanged


def test_transfer_refuses_a_scaled_object():
    poses, grip = _demo()
    d = PlanarDemo.extract(poses, grip, event_idx=35, lead_frames=10)
    with pytest.raises(AssertionError, match="refusing to scale"):
        d.transfer(Sim2.from_angle(0.0, s=1.3), np.zeros(2), trust_rotation=True)
