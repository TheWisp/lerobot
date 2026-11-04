"""Independent kinematic controls for the experimental motion predictor."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_velocity_filter import VelocityConfig, VelocityPoseFilter
from lerobot.policies.flash.surface_kalman import MeasuredGroupMotion
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig


def make_filter(**kwargs):
    return VelocityPoseFilter(
        np.eye(4),
        np.array([0.1, -0.05, 0.7]),
        0.0,
        PoseUncertaintyConfig(),
        velocity_config=VelocityConfig(**kwargs),
    )


def test_translation_covariance_matches_continuous_acceleration_integral():
    f = make_filter()
    prior = f.joint_covariance.copy()
    t = 0.2
    f.predict(t, np.eye(4), np.eye(4))
    q = f.velocity_config.linear_acceleration_std**2
    walk = f.config.position_process_std_m_sqrt_s**2
    # Position/velocity covariance has a closed-form scalar solution, independent
    # of the implementation's matrix exponential and of image fitting.
    assert f.joint_covariance[3, 3] == pytest.approx(
        prior[3, 3] + t * t * prior[9, 9] + q * t**3 / 3 + walk * t
    )
    assert f.joint_covariance[3, 9] == pytest.approx(t * prior[9, 9] + q * t * t / 2)
    assert f.joint_covariance[9, 9] == pytest.approx(prior[9, 9] + q * t)


def test_pose_measurement_updates_velocity_like_scalar_kalman_filter():
    f = make_filter()
    f.predict(0.1, np.eye(4), np.eye(4))
    prior = f.joint_covariance.copy()
    measurement = f.pose.copy()
    measurement[0, 3] += 0.004
    info = np.zeros((6, 6))
    info[3, 3] = 1 / 0.002**2
    gain = prior[:, 3] / (prior[3, 3] + 0.002**2)
    expected_cov = prior - np.outer(prior[:, 3], prior[3, :]) / (prior[3, 3] + 0.002**2)
    result = f.correct(measurement, info, f.center(measurement))
    assert result["accepted"]
    assert f.pose[0, 3] == pytest.approx(gain[3] * 0.004)
    assert f.velocity[3] == pytest.approx(gain[9] * 0.004)
    assert f.velocity[3] > 0
    np.testing.assert_allclose(f.joint_covariance, expected_cov, atol=1e-12)


def test_turning_prediction_is_about_fixed_anchor_and_independent_of_frame_rate():
    a, b = make_filter(), make_filter()
    for f in (a, b):
        f.velocity = np.array([0.0, 0.8, 0.0, 0.02, -0.01, 0.0])
    center = a.center().copy()
    a.predict(0.4, np.eye(4), np.eye(4))
    for t in np.linspace(0.04, 0.4, 10):
        b.predict(t, np.eye(4), np.eye(4))
    np.testing.assert_allclose(a.pose[:3, :3], Rotation.from_rotvec([0, 0.32, 0]).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(a.center(), center + [0.008, -0.004, 0], atol=1e-12)
    np.testing.assert_allclose(a.pose, b.pose, atol=1e-12)
    np.testing.assert_allclose(a.joint_covariance, b.joint_covariance, atol=1e-12)


def test_camera_motion_rotates_all_state_components_without_becoming_object_velocity():
    a, b = make_filter(), make_filter()
    a.velocity = b.velocity = np.array([0.0, 0.5, 0.0, 0.04, 0.0, 0.0])
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    camera[:3, 3] = [0.04, 0.01, 0.0]
    a.predict(0.1, np.eye(4), np.eye(4))
    b.predict(0.1, camera, np.linalg.inv(camera))
    np.testing.assert_allclose(b.pose, camera @ a.pose, atol=1e-12)
    axes = np.kron(np.eye(4), camera[:3, :3])
    np.testing.assert_allclose(b.joint_covariance, axes @ a.joint_covariance @ axes.T, atol=1e-12)
    np.testing.assert_allclose(b.velocity, np.kron(np.eye(2), camera[:3, :3]) @ a.velocity, atol=1e-12)


def test_measured_group_displacement_is_not_added_twice():
    f = make_filter()
    f.velocity[3] = 0.1
    p = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.05, 0.05) for z in (0.6, 0.8)])
    motion = MeasuredGroupMotion(p, p + [0.01, 0, 0], 0.0, 0.1, "visible surroundings")
    f.predict(0.1, np.eye(4), np.eye(4), motion)
    assert f.pose[0, 3] == pytest.approx(0.01)
    assert f.velocity[3] == pytest.approx(0.1)
    assert np.linalg.eigvalsh(f.joint_covariance).min() > 0


def test_rejected_measurement_and_session_rollback_preserve_all_filter_state():
    f = make_filter()
    f.predict(0.1, np.eye(4), np.eye(4))
    before = f.snapshot()
    bad = f.pose.copy()
    bad[0, 3] = 1.0
    result = f.correct(bad, np.eye(6) * 1e6, f.center(bad), contradicted=True)
    assert not result["accepted"]  # A failed old geometry check cannot bypass the gate.
    np.testing.assert_array_equal(f.joint_covariance, before["joint_covariance"])
    good = f.pose.copy()
    good[0, 3] = 0.003
    assert f.correct(good, np.eye(6) * 1e5, f.center(good))["accepted"]
    assert np.linalg.norm(f.velocity) > 0
    f.restore(before)
    np.testing.assert_array_equal(f.velocity, before["velocity"])
    np.testing.assert_array_equal(f.joint_covariance, before["joint_covariance"])
    np.testing.assert_array_equal(f.pose, before["pose"])


def test_unobserved_direction_cannot_create_velocity():
    f = make_filter()
    f.predict(0.1, np.eye(4), np.eye(4))
    measured = f.pose.copy()
    measured[0, 3] = 0.2
    measured[2, 3] = 0.003
    info = np.zeros((6, 6))
    info[5, 5] = 1e5
    assert f.correct(measured, info, f.center(measured))["accepted"]
    assert f.pose[0, 3] == 0
    assert f.velocity[3] == 0
    assert f.velocity[5] > 0


def test_group_update_matches_augmented_two_time_linear_filter():
    f = make_filter()
    p = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.05, 0.05) for z in (0.6, 0.8)])
    # Put the anchor at the support centroid: this isolates a genuinely linear
    # translation update without a correlated angular correction/retraction.
    f.center_object = p.mean(0)
    prior = f.joint_covariance.copy()
    dt = 0.1
    # Independent textbook augmented-state update for a translation. The
    # measurement is x_now - x_previous, not an absolute position measurement.
    transition = np.eye(12)
    transition[:6, 6:] = np.eye(6) * dt
    _, noise = f.discretize(dt)
    predicted = transition @ prior @ transition.T + noise
    cross = prior @ transition.T
    augmented = np.block([[prior, cross], [cross.T, predicted]])
    motion = MeasuredGroupMotion(p, p + [0.004, 0, 0], 0.0, dt, "support")
    f.predict(dt, np.eye(4), np.eye(4), motion)
    h = np.zeros((6, 24))
    h[:, :6] = -np.eye(6)
    h[:, 12:18] = np.eye(6)
    measurement_noise = np.array(f.last_prediction["motion_covariance"])
    gain = np.linalg.solve(h @ augmented @ h.T + measurement_noise, h @ augmented).T
    mean = gain @ np.array([0, 0, 0, 0.004, 0, 0])
    expected = augmented - gain @ h @ augmented
    np.testing.assert_allclose(f.joint_covariance, expected[12:, 12:], atol=1e-12)
    np.testing.assert_allclose(f.velocity, mean[18:], atol=1e-12)
    assert f.pose[0, 3] == pytest.approx(mean[15])


@pytest.mark.parametrize("partial_surface_matching", [False, True])
def test_session_geometry_rejection_rolls_back_velocity_and_map(monkeypatch, partial_surface_matching):
    from lerobot.policies.flash import experimental_surface_session as module
    from tests.policies.flash.test_experimental_surface_session import prepared_frames

    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    monkeypatch.setattr(module, "PoseKalmanFilter", VelocityPoseFilter)
    session = module.GrowingSurfaceSession(
        prepare.shape_config,
        module.GrowingSurfaceConfig(partial_surface_matching=partial_surface_matching),
    )
    session.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    initial_map = session.store.points.copy()
    initial_patches = len(session.store.patches)
    filter = session.kalman
    original_correct = filter.correct
    states = {}

    def record_correct(*args, **kwargs):
        states["before"] = filter.snapshot()
        result = original_correct(*args, **kwargs)
        states["after"] = filter.snapshot()
        assert result["accepted"]
        assert np.linalg.norm(filter.velocity) > 0
        return result

    monkeypatch.setattr(filter, "correct", record_correct)
    # Force the PUBLIC downstream group-consistency check to reject a real
    # fitted, Kalman-corrected observation. The fitter and filter are untouched.
    hint = module.PoseHint(np.eye(4), "test support", 0.1, 0.0, 0)
    result = session.update(
        prepare.prepare(rgb, depth + 0.003, mask, k, 1),
        0.1,
        hint=hint,
        candidate_check=lambda *_: (False, {"verdict": "rejected"}),
    )
    assert result["diagnostics"]["reason"] == "surface_group_disagreement"
    for key in ("pose", "covariance", "velocity", "joint_covariance"):
        np.testing.assert_array_equal(filter.snapshot()[key], states["before"][key])
    np.testing.assert_array_equal(session.store.points, initial_map)
    assert len(session.store.patches) == initial_patches


def test_sudden_stop_measurements_correct_learned_velocity():
    f = make_filter()
    info = np.diag([1e6] * 3 + [1e8] * 3)
    for i in range(1, 21):
        f.predict(i / 30, np.eye(4), np.eye(4))
        pose = np.eye(4)
        pose[0, 3] = min(i, 10) * 0.002
        assert f.correct(pose, info, f.center(pose))["accepted"]
    assert abs(f.velocity[3]) < 0.001
    assert abs(f.pose[0, 3] - 0.02) < 0.001


def test_hidden_session_has_bounded_memory(monkeypatch):
    from lerobot.policies.flash import experimental_surface_session as module
    from tests.policies.flash.test_recovery_acceptance_integration import (
        test_long_hidden_sequence_keeps_memory_and_work_bounded,
    )

    monkeypatch.setattr(module, "PoseKalmanFilter", VelocityPoseFilter)
    test_long_hidden_sequence_keeps_memory_and_work_bounded(3)


def test_rotation_transition_jacobian_matches_finite_difference_of_rigid_motion():
    from lerobot.policies.flash.surface_motion import pose_with_delta

    f = make_filter()
    f.velocity = np.array([0.7, -0.3, 0.2, 0.02, -0.04, 0.01])
    dt, epsilon = 0.13, 1e-7
    predicted = pose_with_delta(f.pose, f.center(), f.velocity * dt)
    nominal_center = f.center(predicted)
    numerical = np.empty((12, 12))
    for axis in range(12):
        perturbation = np.zeros(12)
        perturbation[axis] = epsilon
        start = pose_with_delta(f.pose, f.center(), perturbation[:6])
        velocity = f.velocity + perturbation[6:]
        end = pose_with_delta(start, f.center(start), velocity * dt)
        numerical[:, axis] = (
            np.r_[
                Rotation.from_matrix(end[:3, :3] @ predicted[:3, :3].T).as_rotvec(),
                f.center(end) - nominal_center,
                velocity - f.velocity,
            ]
            / epsilon
        )
    transition, _ = f.discretize(dt)
    np.testing.assert_allclose(transition, numerical, atol=2e-8)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_velocity_uncertainty_is_rejected(value):
    with pytest.raises(ValueError):
        make_filter(linear_acceleration_std=value)


@pytest.mark.xfail(
    strict=True,
    reason="Velocity inferred across an occluded move overshoots after the object stops; subsequent fitting loses it",
)
def test_recovery_after_hidden_motion_stays_accurate_when_the_object_stops():
    from benchmarks.flash_kalman_recovery_control import AFTER, run_scene, scene

    rows = run_scene(scene(), "current_reset", filter_factory=VelocityPoseFilter)["rows"]
    for row in rows[-AFTER:]:
        assert row["available"], (row["index"], row["reason"])
        assert row["error"]["median_point_error_mm"] < 5
        assert row["error"]["rotation_error_deg"] < 1


def test_recorded_bad_mug_recovery_remains_rejected_without_surface_changes():
    from benchmarks.flash_recovery_fixture import restore
    from benchmarks.flash_recovery_matrix import WorkProbe

    session, prepared, metadata, _ = restore("strict", filter_factory=VelocityPoseFilter)
    before = session.store.points.copy()
    with WorkProbe() as probe:
        result = session.update(prepared, metadata["time_s"])
    assert result["camera_from_reference"] is None
    assert not result["diagnostics"]["kalman"]["accepted"]
    assert result["diagnostics"]["kalman"]["filter_reason"] == "innovation_conflict"
    np.testing.assert_array_equal(session.store.points, before)
    assert not result["diagnostics"]["map_events"]
    work = probe.result()
    assert work["calls"]["kalman_correct"] == 1
    assert work["calls"]["fit_refinement"] <= 6


def test_reset_ablation_preserves_legacy_reset_but_discards_old_velocity():
    from benchmarks.flash_velocity_reset import VelocityResetPoseFilter
    from lerobot.policies.flash.surface_kalman import PoseKalmanFilter

    config = PoseUncertaintyConfig()
    a = VelocityResetPoseFilter(np.eye(4), np.zeros(3), 0.0, config)
    b = PoseKalmanFilter(np.eye(4), np.zeros(3), 0.0, config)
    a.velocity[:] = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03]
    measured = np.eye(4)
    measured[0, 3] = 0.1
    info = np.diag([1e3] * 3 + [1e6] * 3)
    assert a.correct(measured, info, measured[:3, 3], contradicted=True)["accepted"]
    assert b.correct(measured, info, measured[:3, 3], contradicted=True)["accepted"]
    np.testing.assert_array_equal(a.pose, b.pose)
    np.testing.assert_array_equal(a.covariance, b.covariance)
    np.testing.assert_array_equal(a.velocity, np.zeros(6))
    np.testing.assert_array_equal(a.joint_covariance[:6, 6:], np.zeros((6, 6)))
