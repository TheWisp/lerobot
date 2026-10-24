"""Checks for benchmark geometry and two-sided measured-depth gating."""

import cv2
import numpy as np
from compare import measured_pairs, move, robust_fit
from scipy.spatial.transform import Rotation


def test_robust_fit_recovers_known_rigid_motion_with_outliers():
    cv2.setRNGSeed(0)
    generator = np.random.default_rng(12)
    source = generator.normal(0, 0.015, (100, 3)) + [0, 0, 0.35]
    rotation = Rotation.from_euler("xyz", [0.11, -0.06, 0.09]).as_matrix()
    target = source @ rotation.T + [0.008, -0.011, 0.003]
    target[:20] += generator.normal(0, 0.04, (20, 3))
    estimated, inliers, _ = robust_fit(source, target)
    assert estimated is not None
    assert inliers >= 80
    np.testing.assert_allclose(move(source[20:], estimated), target[20:], atol=1e-6)


def test_correspondence_requires_valid_source_and_target_depth():
    depth = np.full((4, 4), 0.5)
    frame_a = {"depth": depth, "usable": np.zeros((4, 4), bool)}
    frame_b = {"depth": depth, "usable": np.zeros((4, 4), bool)}
    frame_a["usable"][1, 1] = frame_a["usable"][2, 1] = True
    frame_b["usable"][1, 2] = True
    k = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]])
    source, target = measured_pairs(
        frame_a, frame_b, np.array([[1, 1], [1, 2]]), np.array([[2, 1], [2, 2]]), k
    )
    assert source.shape == target.shape == (1, 3)
    np.testing.assert_allclose(source[0], [0.005, 0.005, 0.5])
    np.testing.assert_allclose(target[0], [0.01, 0.005, 0.5])
