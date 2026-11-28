"""Numerical controls for the optional experiment, not real-tracker accuracy."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_pose_covariance import CovarianceConfig, fuse_local, plane_information
from lerobot.policies.flash.experimental_surface_session import PointEvidence
from lerobot.policies.flash.surface_observation import move


def plane(width=0.08):
    x, y = np.meshgrid(np.linspace(-0.04, 0.04, 16), np.linspace(-width / 2, width / 2, 8))
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    return points, normals


def test_plane_has_three_unobservable_directions_and_narrow_strip_loses_tilt_information():
    cfg = CovarianceConfig()
    full, normals = plane()
    strip, _ = plane(0.006)
    a = plane_information(full, normals, np.zeros(3), cfg)
    b = plane_information(strip, normals, np.zeros(3), cfg)
    # Spin about normal and translation within the plane cannot be measured.
    np.testing.assert_allclose(a[[2, 3, 4]], 0, atol=1e-12)
    assert b[0, 0] < a[0, 0] / 4


def test_repeated_correlated_measurements_do_not_duplicate_information():
    points, normals = plane()
    cfg = CovarianceConfig(independent_error_m=1e-5, correlated_error_m=0.002)
    a = plane_information(points, normals, np.zeros(3), cfg)
    b = plane_information(np.repeat(points, 2, axis=0), np.repeat(normals, 2, axis=0), np.zeros(3), cfg)
    assert b[5, 5] / a[5, 5] < 1.03
    # Mutation control: the independent-noise assumption doubles the confidence.
    independent = CovarianceConfig(correlated_error_m=0)
    c = plane_information(points, normals, np.zeros(3), independent)
    d = plane_information(
        np.repeat(points, 2, axis=0), np.repeat(normals, 2, axis=0), np.zeros(3), independent
    )
    assert d[5, 5] / c[5, 5] == pytest.approx(2)


def test_information_is_invariant_to_normal_signs_and_coordinate_origin():
    points, normals = plane()
    original = plane_information(points, normals, np.zeros(3), CovarianceConfig())
    normals[::2] *= -1
    offset = np.array([2.0, -5.0, 8.0])
    transformed = plane_information(points + offset, normals, offset, CovarianceConfig())
    np.testing.assert_allclose(original, transformed, rtol=1e-9, atol=1e-6)


def test_clear_geometry_can_correct_wrong_supportive_prediction():
    points, normals = plane()
    reference = np.vstack([points, points[:, [2, 0, 1]], points[:, [1, 2, 0]]])
    normals = np.vstack([normals, normals[:, [2, 0, 1]], normals[:, [1, 2, 0]]])
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("x", 3, degrees=True).as_matrix()
    truth[:3, 3] = [0.003, -0.002, 0.004]
    evidence = [PointEvidence(reference, reference, "wrong_prediction", "predicted", 0.005)]
    result, details = fuse_local(
        truth,
        np.eye(4),
        reference,
        move(reference, truth),
        normals @ truth[:3, :3].T,
        evidence,
        CovarianceConfig(),
        0.03,
    )
    error = np.linalg.norm(move(reference, result) - move(reference, truth), axis=1)
    assert details["applied"]
    assert error.max() < 0.001
    covariance = np.array(details["posterior_covariance"])
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert np.linalg.norm(result[:3, 3]) > 0.004  # A frozen prior must fail.


def test_unobservable_slide_does_not_override_support_and_no_evidence_does_not_invent_one():
    points, normals = plane()
    geometry = np.eye(4)
    geometry[0, 3] = 0.02
    evidence = [PointEvidence(points, points, "support", "predicted", 0.005)]
    result, _ = fuse_local(
        geometry, np.eye(4), points, move(points, geometry), normals, evidence, CovarianceConfig(), 0.03
    )
    np.testing.assert_allclose(result, np.eye(4), atol=1e-10)
    unchanged, details = fuse_local(
        geometry, np.eye(4), points, move(points, geometry), normals, [], CovarianceConfig(), 0.03
    )
    np.testing.assert_array_equal(unchanged, geometry)
    assert details["reason"] == "no_supportive_evidence"


def test_consistency_check_rejects_contradicted_support_without_freezing_real_motion():
    points, normals = plane()
    moved = np.eye(4)
    moved[2, 3] = 0.06  # Motion perpendicular to a plane is directly observed.
    evidence = [PointEvidence(points, points, "stale_support", "predicted", 0.005)]
    cfg = CovarianceConfig(correlated_error_m=0.005, compatibility_probability=0.99)
    result, details = fuse_local(moved, np.eye(4), points, move(points, moved), normals, evidence, cfg, 0.03)
    assert details["consistency"]["support_rejected"]
    np.testing.assert_allclose(result, moved)
    # A large in-plane disagreement is still unobservable. Size alone cannot
    # decide that the support is wrong.
    moved[2, 3], moved[0, 3] = 0, 0.06
    result, details = fuse_local(moved, np.eye(4), points, move(points, moved), normals, evidence, cfg, 0.03)
    assert not details["consistency"]["support_rejected"]
    np.testing.assert_allclose(result, np.eye(4), atol=1e-10)
