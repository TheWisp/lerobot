"""FLASH controls arm joints while retaining a robot's full kinematic model."""

import numpy as np
import pytest

from lerobot.policies.flash.kinematics import SelectedJointKinematics
from lerobot.policies.flash.planner import JointPlanner, PlannerLimits
from lerobot.policies.flash.skill import Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


class FakeKinematics:
    def forward_kinematics(self, full):
        pose = np.eye(4)
        pose[0, 3] = full[0]
        return pose

    def inverse_kinematics(self, seed, target):
        assert seed.tolist() == [1.0, 7.0, 2.0]
        return np.array([target[0, 3], 7.0, 2.0])


def test_selected_joints_keep_full_model_order_and_fixed_values():
    adapter = SelectedJointKinematics(
        FakeKinematics(),
        ("shoulder", "passive", "elbow"),
        ("shoulder", "elbow"),
        {"passive": 7.0},
    )
    target = adapter.forward_kinematics(np.array([3.0, 2.0]))
    assert target[0, 3] == 3.0
    np.testing.assert_allclose(adapter.inverse_kinematics(np.array([1.0, 2.0]), target), [3.0, 2.0])
    with pytest.raises(ValueError, match="excluded joint"):
        SelectedJointKinematics(FakeKinematics(), ("a", "b"), ("a",), {})


def test_so107_recorded_pose_can_be_solved_with_six_arm_joints():
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES

    # Joint state from the regular LeRobot insertion episode's first frame.
    recorded = np.array([-2.8571, -98.0220, 88.5714, 14.9451, -58.3297, -13.8462, 79.5610])
    model = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    adapter = SelectedJointKinematics(model, MOTOR_NAMES, MOTOR_NAMES[:-1], {"gripper": float(recorded[-1])})
    target = model.forward_kinematics(recorded)
    solution = adapter.inverse_kinematics(recorded[:-1], target)
    reached = adapter.forward_kinematics(solution)
    np.testing.assert_allclose(reached, target, atol=1e-5)

    # L6_1 is upstream of gripper S7, so the spatial EE pose must not change
    # when that excluded gripper value changes during a contact interval.
    other_gripper = SelectedJointKinematics(model, MOTOR_NAMES, MOTOR_NAMES[:-1], {"gripper": 0.0})
    np.testing.assert_allclose(other_gripper.forward_kinematics(recorded[:-1]), target, atol=1e-9)

    reference = np.eye(4)
    interval = Interval(
        "demo_contact",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "object",
        (
            SpatialSample(0.0, target, reference, gripper=float(recorded[-1])),
            SpatialSample(0.1, target, reference, gripper=float(recorded[-1])),
        ),
        reference_version="object-v1",
    )
    live = PoseEstimate(
        "object", reference, 0.0, 0.0, Evidence.OBSERVED, "object-v1", (True, True, True), (True, True, True)
    )
    planner = JointPlanner(
        adapter, MOTOR_NAMES[:-1], "gripper", PlannerLimits((5000.0,) * 6, 0.002, 0.02, 1.0)
    )
    commands = planner.plan(
        interval,
        start_tick=0,
        count=1,
        epoch_s=0.0,
        period_s=1 / 30,
        interval_start_s=0.0,
        seed_deg=recorded[:-1],
        reference=live,
        attachment=None,
    )
    assert set(commands[0].action) == {f"{name}.pos" for name in MOTOR_NAMES}
