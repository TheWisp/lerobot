"""Analytical controls plus real surface-session consumption of a temporal EKF."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    PoseHint,
)
from lerobot.policies.flash.surface_kalman import MeasuredGroupMotion, PoseKalmanFilter, rotate_covariance
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig
from tests.policies.flash.test_experimental_surface_session import points, prepared_frames, transform


@pytest.mark.parametrize("contradicted", [False, True])
@pytest.mark.parametrize("distance", [0.001, 0.1])
@pytest.mark.parametrize("information_scale", [0.0, 1e4, 1e6])
def test_saved_no_reset_setting_matches_the_tested_filter(contradicted, distance, information_scale):
    from benchmarks.flash_kalman_recovery_control import StrictPoseKalmanFilter

    old = StrictPoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config())
    new = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config(reset_on_geometry_conflict=False))
    for time_s in (0.1, 0.2):
        for f in (old, new):
            f.predict(time_s, np.eye(4), np.eye(4))
        measured = transform(z=distance)
        a = old.correct(measured, np.eye(6) * information_scale, measured[:3, 3], contradicted=contradicted)
        b = new.correct(measured, np.eye(6) * information_scale, measured[:3, 3], contradicted=contradicted)
        assert a == b
        np.testing.assert_array_equal(old.pose, new.pose)
        np.testing.assert_array_equal(old.covariance, new.covariance)


def test_no_reset_setting_rejects_the_jump_even_when_old_prediction_fails_geometry():
    f = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config(reset_on_geometry_conflict=False))
    before = f.snapshot()
    rejected = f.correct(transform(z=0.1), np.eye(6) * 1e6, np.array([0, 0, 0.1]), contradicted=True)
    assert not rejected["accepted"] and rejected["filter_reason"] == "innovation_conflict"
    np.testing.assert_array_equal(f.pose, before["pose"])
    np.testing.assert_array_equal(f.covariance, before["covariance"])
    # This configuration must still allow ordinary measured corrections.
    accepted = f.correct(transform(z=0.001), np.eye(6) * 1e6, np.array([0, 0, 0.001]), contradicted=True)
    assert accepted["accepted"] and f.pose[2, 3] > 0
    with pytest.raises(ValueError, match="booleans"):
        replace(config(), reset_on_geometry_conflict="false")


def config(**kwargs):
    return PoseUncertaintyConfig(enabled=True, mode="kalman", **kwargs)


@pytest.mark.parametrize("prior_std,measurement_std,expected_mm", [(0.002, 0.004, 0.8), (0.004, 0.002, 3.2)])
def test_both_covariances_change_the_output_as_analytic_scalar_kalman_gain(
    prior_std, measurement_std, expected_mm
):
    f = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config())
    f.covariance = np.eye(6) * prior_std**2
    info = np.zeros((6, 6))
    info[5, 5] = 1 / measurement_std**2
    result = f.correct(transform(z=0.004), info, np.array([0.0, 0.0, 0.004]))
    assert result["accepted"]
    assert f.pose[2, 3] * 1000 == pytest.approx(expected_mm)
    assert f.covariance[5, 5] == pytest.approx(0.0000032)
    # Unmeasured x/y directions retain their prior uncertainty.
    assert f.covariance[3, 3] == prior_std**2


def test_visible_plane_corrects_depth_without_inventing_in_plane_measurements():
    f = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config(initial_position_std_m=0.01))
    info = np.zeros((6, 6))
    info[5, 5] = 1 / 0.01**2
    result = f.correct(transform(x=0.2, z=0.004), info, np.array([0.2, 0.0, 0.004]))
    assert result["accepted"]
    assert f.pose[2, 3] == pytest.approx(0.002)
    assert f.pose[0, 3] == pytest.approx(0.0)


def test_rotation_correction_uses_fixed_object_center_not_camera_origin():
    center = np.array([0.2, -0.1, 0.6])
    f = PoseKalmanFilter(np.eye(4), center, 0.0, config(initial_rotation_std_deg=10))
    measurement = np.eye(4)
    measurement[:3, :3] = Rotation.from_euler("y", 20, degrees=True).as_matrix()
    measurement[:3, 3] = center - measurement[:3, :3] @ center
    info = np.zeros((6, 6))
    info[1, 1] = 1 / np.deg2rad(10) ** 2
    result = f.correct(measurement, info, center)
    assert result["accepted"]
    np.testing.assert_allclose(f.pose[:3, :3], Rotation.from_euler("y", 10, degrees=True).as_matrix())
    np.testing.assert_allclose(f.center(), center)
    assert f.covariance[1, 1] == pytest.approx(np.deg2rad(10) ** 2 / 2)


def test_sequential_updates_carry_posterior_covariance_instead_of_resetting():
    f = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config(initial_position_std_m=0.002))
    info = np.zeros((6, 6))
    info[5, 5] = 1 / 0.002**2
    f.correct(transform(z=0.002), info, np.array([0.0, 0.0, 0.002]))
    assert f.pose[2, 3] == pytest.approx(0.001)
    f.correct(transform(z=0.002), info, np.array([0.0, 0.0, 0.002]))
    assert f.pose[2, 3] == pytest.approx(0.004 / 3)
    assert f.covariance[5, 5] == pytest.approx(0.002**2 / 3)


def test_no_information_cannot_confirm_or_reset_a_pose():
    f = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config())
    before = f.covariance.copy()
    result = f.correct(transform(x=0.2), np.zeros((6, 6)), np.zeros(3), contradicted=True)
    assert not result["accepted"]
    np.testing.assert_array_equal(f.pose, np.eye(4))
    np.testing.assert_array_equal(f.covariance, before)


def test_changing_measurement_pivot_does_not_change_the_filter_update():
    # Same linearized measurement, written about two different origins.
    a = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config())
    b = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config())
    measured = transform(z=0.004)
    center = np.array([0.0, 0.0, 0.004])
    pivot = center + [0.1, 0.0, 0.0]
    information = np.diag([100.0, 200.0, 300.0, 1e5, 2e5, 3e5])
    # Translation at the shifted point gains omega cross [0.1, 0, 0].
    from lerobot.policies.flash.surface_kalman import skew

    center_to_pivot = np.eye(6)
    center_to_pivot[3:, :3] = -skew(pivot - center)
    inverse = np.linalg.inv(center_to_pivot)
    a.correct(measured, information, center)
    b.correct(measured, inverse.T @ information @ inverse, pivot)
    np.testing.assert_allclose(a.pose, b.pose, atol=1e-12)
    np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-12)


def test_hidden_group_moves_corrected_pose_and_increases_uncertainty():
    p = points()
    f = PoseKalmanFilter(transform(x=0.003), p.mean(0), 0.0, config())
    before = f.covariance.copy()
    motion = MeasuredGroupMotion(p, p + [0.01, 0.0, 0.0], 0.0, 0.1, "surroundings")
    f.predict(0.1, np.eye(4), np.eye(4), motion)
    assert f.pose[0, 3] == pytest.approx(0.013)
    assert np.linalg.eigvalsh(f.covariance - before).min() > 0
    assert f.last_prediction["motion_covariance"] is not None
    pose, covariance = f.pose.copy(), f.covariance.copy()
    with pytest.raises(ValueError, match="consecutive"):
        f.predict(0.2, transform(x=0.1), np.eye(4), motion)
    np.testing.assert_array_equal(f.pose, pose)
    np.testing.assert_array_equal(f.covariance, covariance)


def test_camera_rotation_transports_pose_and_covariance_without_world_motion():
    f = PoseKalmanFilter(transform(x=0.1), np.zeros(3), 0.0, config())
    f.covariance = np.diag([0.01, 0.02, 0.03, 0.001, 0.002, 0.003])
    original = f.covariance.copy()
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    camera[:3, 3] = [0.02, 0.03, 0.0]
    f.predict(0.1, camera, np.linalg.inv(camera))
    np.testing.assert_allclose(f.pose, camera @ transform(x=0.1), atol=1e-12)
    expected = rotate_covariance(original, camera[:3, :3])
    np.testing.assert_allclose(f.covariance - np.asarray(f.last_prediction["process_covariance"]), expected)


def test_process_noise_depends_on_elapsed_time_not_frame_count():
    a = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config())
    b = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config())
    a.predict(1.0, np.eye(4), np.eye(4))
    for t in np.linspace(0.1, 1.0, 10):
        b.predict(t, np.eye(4), np.eye(4))
    np.testing.assert_allclose(a.covariance, b.covariance)


def test_incompatible_measurement_is_not_fused_but_proven_motion_can_reset():
    f = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config())
    info = np.eye(6) * 1e6
    before = f.covariance.copy()
    measurement = transform(z=0.1)
    rejected = f.correct(measurement, info, np.array([0.0, 0.0, 0.1]))
    assert not rejected["accepted"]
    np.testing.assert_array_equal(f.pose, np.eye(4))
    np.testing.assert_array_equal(f.covariance, before)
    accepted = f.correct(measurement, info, np.array([0.0, 0.0, 0.1]), contradicted=True)
    assert accepted["accepted"]
    np.testing.assert_allclose(f.pose, measurement)


def test_actual_session_hidden_motion_camera_change_and_reappearance(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    session = GrowingSurfaceSession(
        prepare.shape_config, GrowingSurfaceConfig(uncertainty=config(), retain_unobserved_pose=True)
    )
    session.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    original = session.store.points.copy()
    prior = session.kalman.covariance.copy()
    p = points()
    hint = PoseHint(
        transform(z=0.5),
        "group",
        0.1,
        0.0,
        0,
        motion=MeasuredGroupMotion(p, p + [0.0, 0.0, 0.02], 0.0, 0.1, "group"),
    )
    result = session.update(prepare.prepare(rgb, depth, mask & False, k, 1), 0.1, hint=hint)
    assert result["status"] == "predicted"
    np.testing.assert_allclose(np.asarray(result["camera_from_reference"])[2, 3], 0.02, atol=1e-9)
    np.testing.assert_array_equal(session.store.points, original)
    assert np.linalg.eigvalsh(session.kalman.covariance - prior).min() > 0
    # Current depth is actually 30mm farther than initialization. The stale
    # group position must not prevent this real surface fit from correcting it.
    result = session.update(prepare.prepare(rgb, depth + 0.03, mask, k, 2), 0.2)
    assert result["status"] == "observed_candidate"
    assert abs(np.asarray(result["camera_from_reference"])[2, 3] - 0.03) < 0.005
    assert result["diagnostics"]["kalman"]["prior_rejected_by_depth_mask"]
    assert result["diagnostics"]["kalman"]["gain"] is not None


def test_group_rotation_is_measured_from_visible_points():
    p = points()
    pivot = p.mean(0)
    motion_pose = np.eye(4)
    motion_pose[:3, :3] = Rotation.from_euler("y", 20, degrees=True).as_matrix()
    motion_pose[:3, 3] = pivot - motion_pose[:3, :3] @ pivot
    f = PoseKalmanFilter(np.eye(4), pivot, 0.0, config())
    f.predict(0.1, np.eye(4), np.eye(4), MeasuredGroupMotion(p, move(p, motion_pose), 0.0, 0.1, "support"))
    np.testing.assert_allclose(f.pose, motion_pose, atol=1e-12)
    assert np.linalg.eigvalsh(f.covariance).min() > 0
