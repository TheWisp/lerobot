"""Demo evidence compares a visible held object with the recorded grasp assumption."""

import numpy as np
import pytest

from lerobot.policies.flash.held_pose_diagnostics import compare_demo_held_pose
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def _estimate(pose: np.ndarray, time_s: float, evidence: Evidence = Evidence.OBSERVED) -> PoseEstimate:
    return PoseEstimate("held", pose, time_s, time_s, evidence, "object-v1", (True,) * 3, (True,) * 3)


def test_compares_held_views_at_camera_time_and_separates_observed_runs():
    motor_times = np.array([0.0, 2.0, 4.0])
    motor_poses = np.stack([_pose(0.0), _pose(0.2), _pose(0.4)])
    rows = [
        {},
        {"held": _estimate(_pose(0.12), 1.0)},
        {"held": _estimate(_pose(0.17), 1.5)},
        {"held": _estimate(_pose(0.22), 2.0)},
        {"held": _estimate(_pose(0.27), 2.5, Evidence.PREDICTED)},
        {"held": _estimate(_pose(0.32), 3.0)},
    ]

    comparison = compare_demo_held_pose(
        role="held",
        estimates=rows,
        arm_times_s=motor_times,
        arm_poses=motor_poses,
        grasp_time_s=0.5,
        release_time_s=3.5,
        grasp_attachment=np.eye(4),
        object_center_m=np.array([0.3, 0.0, 0.0]),
    )

    assert comparison["observed_frames"] == 4
    assert [(run["first_frame"], run["last_frame"]) for run in comparison["runs"]] == [
        (1, 3),
        (5, 5),
    ]
    assert comparison["runs"][0]["center_gap_median_m"] == pytest.approx(0.02)
    assert comparison["runs"][0]["gripper_relative_center_scatter_p90_m"] == pytest.approx(0)
    assert comparison["runs"][0]["gripper_excursion_from_first_m"] == pytest.approx(0.1)
    assert comparison["runs"][0]["gripper_turn_from_first_rad"] == pytest.approx(0)
    assert comparison["runs"][0]["gripper_relative_turn_from_first_p90_rad"] == pytest.approx(0)


def test_reference_center_gap_is_independent_of_remote_object_frame_origin():
    rotated = np.eye(4)
    rotated[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    rotated[:3, 3] = [1, -1, 0]

    comparison = compare_demo_held_pose(
        role="held",
        estimates=[{"held": _estimate(rotated, 0.5)}],
        arm_times_s=np.array([0.0, 1.0]),
        arm_poses=np.stack([np.eye(4), np.eye(4)]),
        grasp_time_s=0.25,
        release_time_s=0.75,
        grasp_attachment=np.eye(4),
        object_center_m=np.array([1.0, 0.0, 0.0]),
    )

    run = comparison["runs"][0]
    assert run["center_gap_median_m"] == pytest.approx(0)
    assert run["rotation_gap_median_rad"] == pytest.approx(np.pi / 2)
