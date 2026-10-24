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
    assert no_depth.tracked_points >= 12

    recovered = tracker.update(frame(shifted, 0.099))
    assert recovered.base_motion_from_seed is not None
    assert recovered.inliers >= 12


def test_causal_points_reject_unobserved_frame_gap():
    gray = np.random.default_rng(2).integers(0, 255, (96, 96), dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    tracker = CausalRigidPointTracker(max_frame_gap_s=0.05)
    tracker.seed(frame(rgb, 0), np.ones(gray.shape, dtype=bool))
    with pytest.raises(ValueError, match="lost a camera-frame interval"):
        tracker.update(frame(rgb, 0.2))


def test_hole_adjacent_to_rgb_track_withholds_only_its_metric_sample():
    gray = np.random.default_rng(5).integers(0, 255, (96, 96), dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mask = np.zeros(gray.shape, dtype=bool)
    mask[20:76, 20:76] = True
    direct = CausalRigidPointTracker(min_inliers=12, depth_support_radius_px=0)
    guarded = CausalRigidPointTracker(min_inliers=12, depth_support_radius_px=1)
    assert direct.seed(frame(rgb, 0), mask) == guarded.seed(frame(rgb, 0), mask)
    x, y = guarded.current_pixels(gray.shape)[0]
    depth = np.full(gray.shape, 1000, dtype=np.uint16)
    depth[y, x + 1] = 0

    direct_motion = direct.update(frame(rgb, 0.033, depth))
    guarded_motion = guarded.update(frame(rgb, 0.033, depth))
    assert guarded_motion.tracked_points == direct_motion.tracked_points
    assert guarded_motion.depth_points < direct_motion.depth_points


def test_half_scale_tracking_keeps_features_on_small_object():
    gray = np.zeros((128, 128), dtype=np.uint8)
    gray[32:64, 32:64] = np.random.default_rng(4).integers(0, 255, (32, 32), dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mask = np.zeros(gray.shape, dtype=bool)
    mask[32:64, 32:64] = True
    tracker = CausalRigidPointTracker(min_inliers=12, image_scale=0.5)

    assert tracker.seed(frame(rgb, 0), mask) >= 12
    shifted = cv2.warpAffine(rgb, np.float32([[1, 0, 2], [0, 1, 0]]), (128, 128))
    motion = tracker.update(frame(shifted, 0.033))
    assert motion.base_motion_from_seed is not None
    assert motion.inliers >= 12
    np.testing.assert_allclose(motion.base_motion_from_seed[:3, 3], [0.02, 0, 0], atol=0.005)


def test_current_mask_rejects_wrong_object_before_fitting_and_allows_real_motion():
    gray = np.random.default_rng(8).integers(0, 255, (96, 96), dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mask = np.zeros(gray.shape, bool)
    mask[20:76, 20:76] = True
    tracker = CausalRigidPointTracker()
    tracker.seed(frame(rgb, 0), mask)
    shifted = cv2.warpAffine(rgb, np.float32([[1, 0, 2], [0, 1, 0]]), (96, 96))
    excluded = tracker.update(frame(shifted, 0.033), mask=np.zeros_like(mask))
    assert excluded.base_motion_from_seed is None
    assert excluded.depth_points == 0
    # RGB identities survive a mask gap, but no depth/mask observation is invented.
    recovered = tracker.update(frame(shifted, 0.066), mask=mask)
    assert recovered.base_motion_from_seed is not None
    np.testing.assert_allclose(recovered.base_motion_from_seed[:3, 3], [0.02, 0, 0], atol=0.005)


def test_release_handoff_tracks_object_motion_without_following_departing_gripper():
    from lerobot.policies.flash.surface_motion import AcceptedPosePointBridge, RigidMotionState

    gray = np.random.default_rng(9).integers(0, 255, (96, 96), dtype=np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mask = np.zeros(gray.shape, bool)
    mask[20:76, 20:76] = True
    state = RigidMotionState()
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 1.0]
    state.predict(0, "grasp", np.eye(4))
    state.accept(pose, measured=True, carrier=np.eye(4))
    bridge = AcceptedPosePointBridge()
    assert bridge.seed(frame(rgb, 0), mask, state.pose)
    shifted = cv2.warpAffine(rgb, np.float32([[1, 0, 2], [0, 1, 0]]), (96, 96))
    motion = bridge.advance(frame(shifted, 0.033), mask)
    result = state.predict(
        0.033, "object-points", motion.base_motion_from_seed, anchor_pose=bridge.anchor_pose
    )
    np.testing.assert_allclose(result[:3, 3], [0.12, 0.2, 1.0], atol=0.005)
    assert state.last_measurement_s == 0  # no new absolute-pose evidence
    missing = bridge.advance(frame(shifted, 0.066, np.zeros(gray.shape, np.uint16)), mask)
    assert missing.base_motion_from_seed is None
    assert state.predict(0.066, "object-points", None, anchor_pose=bridge.anchor_pose) is None
