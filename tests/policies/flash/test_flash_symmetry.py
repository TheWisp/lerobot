"""Declared physical symmetry must remove unobservable tracker spin."""

import numpy as np

from lerobot.policies.flash.grasp_symmetry import select_grasp_symmetry
from lerobot.policies.flash.planner import JointPlanner, PlannerLimits
from lerobot.policies.flash.pose import Evidence, PoseEstimate
from lerobot.policies.flash.skill import (
    AttachmentAction,
    AttachmentTransition,
    EndpointRequirement,
    FlashSkill,
    Interval,
    MotionMode,
    Relation,
    SpatialSample,
)
from lerobot.policies.flash.symmetry import AxialSymmetry


def _x_pose(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


class _BoundedLinearKinematics:
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        return _x_pose(float(joints[0]) / 100)

    def inverse_kinematics(self, seed: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.array([min(float(target[0, 3]) * 100, 100)])


def test_grasp_symmetry_preview_honors_preferred_hover_and_required_carry():
    """The selector must use the same soft hover and IK rules as execution."""
    identity = np.eye(4)
    hover = Interval(
        "hover",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "held",
        (SpatialSample(0, _x_pose(0), identity), SpatialSample(1, _x_pose(1.2), identity)),
        speed_m_s=1,
        reference_version="held-v1",
        endpoint_requirement=EndpointRequirement.PREFERRED,
    )
    grasp = Interval(
        "grasp",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "held",
        (
            SpatialSample(0, _x_pose(0.8), identity),
            SpatialSample(1, _x_pose(0.8), identity, event_id="attach"),
        ),
        speed_m_s=1,
        reference_version="held-v1",
    )
    carry = Interval(
        "carry",
        MotionMode.CONNECTION,
        Relation.HELD_TO_TARGET,
        "target",
        (
            SpatialSample(0, _x_pose(0.8), identity, _x_pose(0)),
            SpatialSample(1, _x_pose(0.9), identity, _x_pose(0.1)),
        ),
        speed_m_s=1,
        reference_version="target-v1",
        held_reference_version="held-v1",
        held_role="held",
    )
    skill = FlashSkill(
        "demo",
        0,
        (hover, grasp, carry),
        attachment_transitions=(AttachmentTransition("attach", AttachmentAction.ATTACH, "held", "held-v1"),),
    )
    symmetry = AxialSymmetry("held", np.array([0, 0, 1]), np.zeros(3), identity)
    planner = JointPlanner(
        _BoundedLinearKinematics(),
        ("arm",),
        "gripper",
        PlannerLimits((1000,), 0.005, 0.05, 1),
    )
    poses = {
        role: PoseEstimate(role, identity, 0, 0, Evidence.OBSERVED, version, (True,) * 3, (True,) * 3)
        for role, version in (("held", "held-v1"), ("target", "target-v1"))
    }
    choice = select_grasp_symmetry(skill, symmetry, poses, planner, np.array([0.0]), 1 / 30)
    assert choice.angle_rad == 0
    assert not choice.exchange_ends


def _turn_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
