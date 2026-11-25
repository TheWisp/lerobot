"""Geometry and RGB-D observation contracts."""

import numpy as np
import pytest

from lerobot.policies.flash.geometry import held_object_goal
from lerobot.policies.flash.observations import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.pose import Evidence, PoseEstimate


def pose(x=0.0, y=0.0, z=0.0):
    value = np.eye(4)
    value[:3, 3] = x, y, z
    return value


def test_live_held_object_follows_live_target_with_changed_grasp_attachment():
    demo_held = pose(0.2, 0, 0.1)
    demo_target = pose(0.2, 0, 0)
    live_target = pose(0.4, 0.1, 0)
    live_attachment = pose(0.01, 0.02, 0)
    ee_goal = held_object_goal(demo_held, demo_target, live_target, live_attachment)
    np.testing.assert_allclose((ee_goal @ live_attachment)[:3, 3], [0.4, 0.1, 0.1])


def test_partial_pose_is_represented_but_cannot_satisfy_strict_six_dof_goal():
    estimate = PoseEstimate(
        "held", pose(), 0, 0, Evidence.OBSERVED, "v1", (True, True, True), (True, False, False)
    )
    assert estimate.pose is not None
    assert not estimate.usable(0.1, 1.0, allow_prediction=False)
    assert estimate.usable(
        0.1,
        1.0,
        allow_prediction=False,
        translation=(True, True, True),
        rotation=(True, False, False),
    )


def test_rgbd_alignment_and_pose_handoff_preserve_source_time():
    frame = RGBDFrame(
        1,
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.full((2, 3), 1000, dtype=np.uint16),
        np.eye(3),
        0.001,
        pose(),
        1.0,
        1.05,
        "cal1",
    )
    np.testing.assert_allclose(frame.metric_depth(), 1.0)
    with pytest.raises(ValueError, match="aligned"):
        RGBDFrame(2, frame.rgb, np.zeros((3, 2), dtype=np.uint16), np.eye(3), 0.001, pose(), 1.0, 1.1, "cal1")
    store = LatestPoseStore()
    fresh = PoseEstimate(
        "target", pose(0.2), 1.0, 1.1, Evidence.OBSERVED, "ref1", (True, True, True), (True, True, True)
    )
    stale = PoseEstimate(
        "target", pose(0.1), 0.9, 1.2, Evidence.OBSERVED, "ref1", (True, True, True), (True, True, True)
    )
    assert store.publish((fresh,)) == 1
    assert store.publish((stale,)) == 1
    assert store.snapshot()[1]["target"].source_time_s == 1.0
