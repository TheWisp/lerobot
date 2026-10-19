"""Metric object-frame registration across demonstration and live views."""

import numpy as np
import pytest

from lerobot.policies.flash.registration import (
    DegenerateRegistrationError,
    TrackerFrameBinding,
    register_object,
    rigid_fit,
    transform_points,
    unproject_pixels,
)


def pose(x=0.0):
    value = np.eye(4)
    value[0, 3] = x
    return value


def test_correspondence_fit_rejects_outlier_and_binds_tracker_frame():
    source = np.array([[0, 0, 0], [0.04, 0, 0], [0, 0.03, 0], [0, 0, 0.02], [0.03, 0.02, 0.02]], dtype=float)
    target = transform_points(pose(0.2), source)
    target[-1] += [0.1, 0, 0]
    result = register_object(source, target, threshold_m=0.001, min_inliers=4, trials=100, seed=7)
    assert len(result.inlier_indices) == 4
    np.testing.assert_allclose(result.base_from_demo_object, pose(0.2), atol=1e-9)
    binding = TrackerFrameBinding.at_initialization(pose(1.0), result, "object-v1")
    np.testing.assert_allclose(binding.demo_object_pose(pose(1.1)), pose(0.3), atol=1e-9)


def test_collinear_correspondences_do_not_claim_an_orientation():
    source = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], dtype=float)
    with pytest.raises(DegenerateRegistrationError, match="collinear"):
        rigid_fit(source, source + [0, 0.1, 0])


def test_aligned_depth_unprojection_uses_metric_scale():
    depth = np.full((3, 3), 1000, dtype=np.uint16)
    intrinsics = np.array([[100, 0, 1], [0, 100, 1], [0, 0, 1]], dtype=float)
    points = unproject_pixels(depth, intrinsics, np.array([[1, 1], [2, 1]]), 0.001)
    np.testing.assert_allclose(points, [[0, 0, 1], [0.01, 0, 1]])
