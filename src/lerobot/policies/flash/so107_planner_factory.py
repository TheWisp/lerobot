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

    def __call__(self) -> JointPlanner:
        kinematics = SelectedJointKinematics(
            make_calibrated_so107_kinematics(self.calibration, arm=self.arm),
            tuple(MOTOR_NAMES),
            self.arm_joint_names,
            {"gripper": self.gripper_position},
        )
        return JointPlanner(kinematics, self.arm_joint_names, "gripper", self.limits)
