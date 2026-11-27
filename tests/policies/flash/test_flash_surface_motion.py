"""Geometry observability and motion ownership, with known physical transforms."""

import numpy as np

from lerobot.policies.flash.surface_motion import (
    measured_surface_information,
)


def planar_cloud():
    x, y = np.meshgrid(np.linspace(-0.06, 0.06, 41), np.linspace(-0.04, 0.04, 31))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.6)]


def test_precomputed_depth_normals_report_partial_motion_without_an_axis_label():
    points = planar_cloud()
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    full = measured_surface_information(points, np.eye(4), points, normals, correspondence_distance_m=0.005)
    patch = np.linalg.norm(points[:, :2], axis=1) < 0.01
    small = measured_surface_information(
        points, np.eye(4), points[patch], normals[patch], correspondence_distance_m=0.005
    )
    assert full.matched_points == full.reference_points == len(points)
    assert small.matched_points < full.matched_points
    assert full.strength.shape == small.strength.shape == (6,)
    # Translation within the plane and rotation around its normal do not move
    # a depth plane, while losing most of the surface weakens the other axes.
    assert np.count_nonzero(full.strength < 1e-12) == 3
    assert small.strength[-1] < full.strength[-1]
    np.testing.assert_allclose(full.basis[[2, 3, 4], -3:], 0, atol=1e-8)


def test_conditional_information_rejects_a_pose_with_no_surface_correspondence():
    points = planar_cloud()
    pose = np.eye(4)
    pose[2, 3] = 0.05
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    support = measured_surface_information(points, pose, points, normals, correspondence_distance_m=0.005)
    assert support.matched_points == 0
    assert support.basis.shape == (6, 0)
    np.testing.assert_array_equal(support.strength, np.zeros(6))
