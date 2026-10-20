"""Causal point motion must use current RGB-D rather than recycled pose time."""

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.causal_points import CausalRigidPointTracker
from lerobot.policies.flash.perception import RGBDFrame


def frame(rgb, source_time_s, depth=None):
    height, width = rgb.shape[:2]
    intrinsics = np.array([[100.0, 0, width / 2], [0, 100.0, height / 2], [0, 0, 1]])
    return RGBDFrame(
        int(source_time_s * 1000),
        rgb,
        np.full((height, width), 1000, dtype=np.uint16) if depth is None else depth,
        intrinsics,
        0.001,
        np.eye(4),
        source_time_s,
        source_time_s,
        "camera-v1",
    )


@pytest.mark.parametrize("image_scale", [1.0, 0.5])
def test_causal_points_recover_metric_motion_and_abstain_without_depth(image_scale):
    gray = np.random.default_rng(1).integers(0, 255, (96, 96), dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mask = np.zeros(gray.shape, dtype=bool)
    mask[20:76, 20:76] = True
    tracker = CausalRigidPointTracker(min_inliers=12, image_scale=image_scale)
    assert tracker.seed(frame(rgb, 0), mask) >= 12

    shifted = cv2.warpAffine(rgb, np.float32([[1, 0, 2], [0, 1, 0]]), (96, 96))
    motion = tracker.update(frame(shifted, 0.033))
    assert motion.base_motion_from_seed is not None
    assert motion.inliers >= 12
    center = motion.base_motion_from_seed @ np.array([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(center[:3], [0.02, 0.0, 1.0], atol=0.005)

    no_depth = tracker.update(frame(shifted, 0.066, np.zeros(gray.shape, dtype=np.uint16)))
    assert no_depth.base_motion_from_seed is None
    assert no_depth.depth_points == 0


def test_causal_points_reject_unobserved_frame_gap():
    gray = np.random.default_rng(2).integers(0, 255, (96, 96), dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    tracker = CausalRigidPointTracker(max_frame_gap_s=0.05)
    tracker.seed(frame(rgb, 0), np.ones(gray.shape, dtype=bool))
    with pytest.raises(ValueError, match="lost a camera-frame interval"):
        tracker.update(frame(rgb, 0.2))
