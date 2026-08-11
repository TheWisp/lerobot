# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""WHEN is transporting the goal through the target's 2D fit actually correct?

The servo transports the taught held-end position into the live frame using the
transform fitted to the TARGET team. That is exact under two conditions, and these
tests establish both the exactness and the size of the error when each is broken —
because a wrong goal is NOT something the closed loop repairs. The loop drives the
error it is given to zero; if the setpoint is wrong, it converges confidently to the
wrong place.

    1. The camera's optical axis is perpendicular to the plane the object moves in.
    2. The held team's features are COPLANAR with the target team's features.

Condition 2 is the one that bites in practice, and its mitigation is cheap: track the
FINGERTIPS, which at the grasp instant sit at the object's own surface.

The metric error from breaking condition 2, at the held plane, is

    |error| = |d| * (z_target - z_held) / (H - z_target)

for a camera at height H and an object displacement d — derived below and asserted
against a full pinhole projection, not against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.showservo.grouping import fit_team

F_PX = 600.0  # focal length in pixels
H_CAM = 0.60  # camera height above the table, metres
Z_TABLE_OBJ = 0.04  # the block's top face, where its trackable features live

TARGET_XY = np.array([[-0.03, -0.03], [0.03, -0.03], [0.03, 0.03], [-0.03, 0.03]])
HELD_XY = np.array([[-0.012, 0.0], [0.012, 0.0], [0.0, 0.015]])


def _camera(tilt_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Camera at ``H_CAM`` looking down, optionally tilted about the world x-axis.

    Post: ``(centre (3,), R (3,3))`` whose columns are the camera axes in world frame.
    """
    a = np.deg2rad(tilt_deg)
    ca, sa = np.cos(a), np.sin(a)
    x_cam = np.array([1.0, 0.0, 0.0])
    y_cam = np.array([0.0, -ca, sa])
    z_cam = np.array([0.0, sa, ca]) * -1.0  # looks down (-z world) when tilt is 0
    return np.array([0.0, 0.0, H_CAM]), np.column_stack([x_cam, y_cam, z_cam])


def _project(points_xyz: np.ndarray, centre: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Pinhole projection. Pre: every point is in front of the camera."""
    q = (np.asarray(points_xyz, float) - centre) @ rot
    assert (q[:, 2] > 1e-6).all(), "point behind the camera"
    return F_PX * q[:, :2] / q[:, 2:3]


def _at_height(xy: np.ndarray, z: float) -> np.ndarray:
    return np.hstack([xy, np.full((len(xy), 1), z)])


def _move(points_xyz: np.ndarray, yaw_deg: float, d_xy) -> np.ndarray:
    """Rigid motion in the table plane: yaw about world z, then translate."""
    a = np.deg2rad(yaw_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = np.asarray(points_xyz, float).copy()
    out[:, :2] = out[:, :2] @ rot.T + np.asarray(d_xy, float)
    return out


def _transport(
    z_held: float,
    *,
    tilt_deg: float = 0.0,
    yaw_deg: float = 30.0,
    d_xy=(0.10, 0.05),
    inlier_px: float = 2.0,
):
    """Fit on the target team, transport the held team, and return the residual.

    Post: ``(ok, error (N,2) px)`` — ``ok`` is the target fit's certificate, and the
    error is the gap between where the servo thinks the held end must go and where it
    must ACTUALLY go. When ``ok`` is False the servo abstains and the error is moot.
    """
    centre, rot = _camera(tilt_deg)
    target0 = _at_height(TARGET_XY, Z_TABLE_OBJ)
    held0 = _at_height(HELD_XY, z_held)

    taught_target = _project(target0, centre, rot)
    taught_held = _project(held0, centre, rot)
    live_target = _project(_move(target0, yaw_deg, d_xy), centre, rot)
    live_held = _project(_move(held0, yaw_deg, d_xy), centre, rot)

    fit = fit_team(taught_target, live_target, inlier_px=inlier_px)
    if not fit.ok:
        return False, None
    return True, fit.sim2.apply(taught_held) - live_held


def _transport_error_px(z_held: float, **kw):
    ok, err = _transport(z_held, **kw)
    assert ok, "the target team must fit, or the test is measuring the wrong thing"
    return err, None


def _metric_mm(err_px: np.ndarray, z_held: float) -> float:
    return float(np.linalg.norm(err_px, axis=1).mean() * (H_CAM - z_held) / F_PX * 1000.0)


def test_transport_is_exact_with_a_perpendicular_camera_and_coplanar_teams():
    # Both conditions met: the fitted transform is the true one for BOTH teams, so the
    # goal lands exactly where it must. This is the case the v0 servo assumes.
    err, _ = _transport_error_px(z_held=Z_TABLE_OBJ, tilt_deg=0.0)
    assert np.abs(err).max() < 1e-6


@pytest.mark.parametrize("yaw_deg", [0.0, 15.0, 30.0, 90.0, 180.0])
def test_in_plane_rotation_transfers_exactly_at_any_angle(yaw_deg):
    # A perpendicular camera turns a yaw about the table normal into a pure image
    # rotation of the same magnitude, at any angle — so rotation is NOT the fragile
    # part of the transport, which is worth knowing before adding machinery to fix it.
    err, _ = _transport_error_px(z_held=Z_TABLE_OBJ, tilt_deg=0.0, yaw_deg=yaw_deg)
    assert np.abs(err).max() < 1e-6


@pytest.mark.parametrize("z_held", [0.08, 0.12, 0.20])
def test_a_held_feature_at_the_wrong_height_produces_a_predictable_metric_error(z_held):
    """The failure the servo cannot see, quantified against the closed form."""
    d_xy = (0.10, 0.05)
    err_px, _ = _transport_error_px(z_held=z_held, tilt_deg=0.0, d_xy=d_xy)

    # Image error -> metric error at the held plane.
    metric = np.linalg.norm(err_px, axis=1).mean() * (H_CAM - z_held) / F_PX
    predicted = np.linalg.norm(d_xy) * abs(Z_TABLE_OBJ - z_held) / (H_CAM - Z_TABLE_OBJ)
    assert metric == pytest.approx(predicted, rel=1e-6)

    # The error is a pure translation: rotation still transferred correctly.
    spread = np.abs(err_px - err_px.mean(axis=0)).max()
    assert spread < 1e-6


def test_the_headline_number_a_gripper_feature_8cm_too_high_fails_the_grasp():
    # 11 cm of object displacement, gripper features 8 cm above the block's top face:
    # 16 mm of goal error. That is a missed grasp, produced by a loop that reports
    # perfect convergence — which is why the fingertips are the features to track.
    err_px, _ = _transport_error_px(z_held=0.12, tilt_deg=0.0, d_xy=(0.10, 0.05))
    metric_mm = np.linalg.norm(err_px, axis=1).mean() * (H_CAM - 0.12) / F_PX * 1000.0
    assert 15.0 < metric_mm < 17.0, f"{metric_mm:.1f} mm"


@pytest.mark.parametrize("tilt_deg", [10.0, 25.0, 40.0])
def test_an_oblique_camera_is_caught_by_the_fit_certificate(tilt_deg):
    # Off the perpendicular, a planar rigid motion induces a HOMOGRAPHY, which a Sim2
    # cannot represent. Under a tight consensus band the team fit cannot assemble
    # enough inliers and ABSTAINS — invariant 5 catching a modelling error, not just a
    # perception one. The servo stops rather than converging to a wrong goal.
    ok, _ = _transport(z_held=Z_TABLE_OBJ, tilt_deg=tilt_deg, inlier_px=2.0)
    assert not ok, f"a {tilt_deg} deg tilt slipped through a 2 px consensus band"


def test_modest_obliquity_is_a_sub_millimetre_effect_when_teams_are_coplanar():
    # Widening the band to the binder's default lets a 10 deg tilt through. The bias it
    # carries is ~0.3 mm — because coplanar teams share ONE homography, and a Sim2
    # approximates it well over a small patch. Obliquity is a real error source but a
    # minor one; it is not what breaks the transport.
    ok, err = _transport(z_held=Z_TABLE_OBJ, tilt_deg=10.0, inlier_px=8.0)
    assert ok
    assert _metric_mm(err, Z_TABLE_OBJ) < 1.0


def test_coplanarity_matters_far_more_than_perpendicularity():
    """The design conclusion: mount the camera roughly overhead, but put the tracked
    gripper features ON the object's plane — the second is worth ~50x the first."""
    _, err_tilt = _transport(z_held=Z_TABLE_OBJ, tilt_deg=10.0, inlier_px=8.0)
    _, err_height = _transport(z_held=0.12, tilt_deg=0.0)

    from_tilt = _metric_mm(err_tilt, Z_TABLE_OBJ)
    from_height = _metric_mm(err_height, 0.12)
    assert from_height > 20 * from_tilt, f"height {from_height:.2f} mm vs tilt {from_tilt:.2f} mm"


def test_obliquity_error_grows_with_tilt():
    errors = [
        _metric_mm(_transport(z_held=Z_TABLE_OBJ, tilt_deg=t, inlier_px=40.0)[1], Z_TABLE_OBJ)
        for t in (5.0, 20.0, 35.0)
    ]
    assert errors[0] < errors[1] < errors[2]
