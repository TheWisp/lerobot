"""Build an independent SO-107 kinematics workspace for isolated planning."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

from .kinematics import SelectedJointKinematics
from .planner import JointPlanner, PlannerLimits


@dataclass(frozen=True)
class SO107PlannerFactory:
    calibration: dict
    arm: str
    arm_joint_names: tuple[str, ...]
    gripper_position: float
    limits: PlannerLimits
    joint_limits_deg: dict[str, tuple[float, float]]

    def __post_init__(self):
        if not set(self.arm_joint_names) <= set(self.joint_limits_deg):
            raise ValueError("Hardware planning requires measured limits for every controlled joint")

    def __call__(self) -> JointPlanner:
        kinematics = SelectedJointKinematics(
            make_calibrated_so107_kinematics(
                self.calibration, arm=self.arm, joint_limits_deg=self.joint_limits_deg
            ),
            tuple(MOTOR_NAMES),
            self.arm_joint_names,
            {"gripper": self.gripper_position},
        )
        return JointPlanner(kinematics, self.arm_joint_names, "gripper", self.limits)
