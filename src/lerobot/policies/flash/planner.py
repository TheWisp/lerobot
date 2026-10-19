"""Kinematics-neutral conversion from spatial references to joint chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .execution import TimedAction
from .geometry import transform
from .skill import Interval, MotionMode
from .trajectory import (
    AttachmentState,
    ConnectionTrajectory,
    MissingEvidenceError,
    PoseEstimate,
    demonstrated_goal,
    spatial_goal,
)


class CartesianKinematics(Protocol):
    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray: ...

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray: ...


class InfeasibleTrajectoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannerLimits:
    joint_speed_deg_s: tuple[float, ...]
    position_error_m: float
    rotation_error_rad: float
    max_pose_age_s: float
    allow_predicted_reference: bool = False

    def __post_init__(self) -> None:
        values = (
            *self.joint_speed_deg_s,
            self.position_error_m,
            self.rotation_error_rad,
            self.max_pose_age_s,
        )
        if not values or not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("planner limits must be finite and positive")


class JointPlanner:
    """Solve a short timed chunk, seeding each IK from the previous command.

    The caller runs this outside the motor loop and publishes the resulting
    chunk atomically. The per-robot kinematics implementation owns link frames,
    calibration and joint mappings. No object identity is special-cased here.
    """

    def __init__(
        self,
        kinematics: CartesianKinematics,
        arm_joint_names: tuple[str, ...],
        gripper_joint_name: str,
        limits: PlannerLimits,
    ) -> None:
        if len(arm_joint_names) != len(limits.joint_speed_deg_s):
            raise ValueError("joint names and speed limits must match")
        self.kinematics = kinematics
        self.arm_joint_names = arm_joint_names
        self.gripper_joint_name = gripper_joint_name
        self.limits = limits

    def plan(
        self,
        interval: Interval,
        *,
        start_tick: int,
        count: int,
        epoch_s: float,
        period_s: float,
        interval_start_s: float,
        seed_deg: np.ndarray,
        reference: PoseEstimate,
        attachment: AttachmentState | None,
        connection: ConnectionTrajectory | None = None,
    ) -> list[TimedAction]:
        if count <= 0 or start_tick < 0 or period_s <= 0:
            raise ValueError("invalid chunk timing")
        seed = np.asarray(seed_deg, dtype=float).copy()
        if seed.shape != (len(self.arm_joint_names),) or not np.isfinite(seed).all():
            raise ValueError("invalid joint seed")
        actions = []
        for tick in range(start_tick, start_tick + count):
            due_s = epoch_s + tick * period_s
            if not reference.usable(due_s, self.limits.max_pose_age_s, self.limits.allow_predicted_reference):
                raise MissingEvidenceError(f"{reference.role} has no usable pose for tick {tick}")
            if (
                interval.reference_version is None
                or reference.reference_version != interval.reference_version
            ):
                raise MissingEvidenceError("live reference does not share the demonstrated object frame")
            if interval.mode == MotionMode.DEMONSTRATED:
                pose, grip = demonstrated_goal(
                    interval,
                    due_s - interval_start_s,
                    reference.pose,
                    self._attachment_pose(attachment, due_s, interval.held_reference_version),
                )
                complete = due_s >= interval_start_s + interval.demo_duration_s
            else:
                if connection is None:
                    raise ValueError("generated interval requires a connection trajectory")
                live_attachment = self._attachment_pose(attachment, due_s, interval.held_reference_version)
                current_endpoint = spatial_goal(
                    interval, interval.samples[-1], reference.pose, live_attachment
                )
                if (
                    np.linalg.norm(current_endpoint[:3, 3] - connection.goal_pose[:3, 3])
                    > self.limits.position_error_m
                    or np.arccos(
                        np.clip(
                            (np.trace(current_endpoint[:3, :3].T @ connection.goal_pose[:3, :3]) - 1) / 2,
                            -1,
                            1,
                        )
                    )
                    > self.limits.rotation_error_rad
                ):
                    raise InfeasibleTrajectoryError(
                        "connection target changed; retarget from current measured pose"
                    )
                pose = connection.at(due_s)
                grip = interval.samples[-1].gripper
                complete = connection.finished(due_s)
            solved = np.asarray(self.kinematics.inverse_kinematics(seed, transform(pose)), dtype=float)
            if solved.shape != seed.shape or not np.isfinite(solved).all():
                raise InfeasibleTrajectoryError(f"IK returned an invalid solution at tick {tick}")
            speed = np.abs(solved - seed) / period_s
            if np.any(speed > self.limits.joint_speed_deg_s):
                raise InfeasibleTrajectoryError(f"joint speed exceeds configured limit at tick {tick}")
            reached = transform(self.kinematics.forward_kinematics(solved))
            translation_error = np.linalg.norm(reached[:3, 3] - pose[:3, 3])
            angle_error = np.arccos(np.clip((np.trace(reached[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1))
            if (
                translation_error > self.limits.position_error_m
                or angle_error > self.limits.rotation_error_rad
            ):
                raise InfeasibleTrajectoryError(f"IK goal error exceeds configured tolerance at tick {tick}")
            command = {
                f"{name}.pos": float(value) for name, value in zip(self.arm_joint_names, solved, strict=True)
            }
            if grip is not None:
                command[f"{self.gripper_joint_name}.pos"] = float(grip)
            actions.append(TimedAction(tick, command, interval.name, interval_complete=complete))
            seed = solved
            if complete:
                break
        return actions

    def _attachment_pose(
        self, attachment: AttachmentState | None, due_s: float, required_version: str | None
    ) -> np.ndarray | None:
        if attachment is None:
            return None
        if not attachment.usable(due_s, self.limits.max_pose_age_s):
            raise MissingEvidenceError("held object attachment is stale or lost")
        if required_version is None or attachment.reference_version != required_version:
            raise MissingEvidenceError("held attachment does not share the demonstrated object frame")
        return attachment.gripper_from_object
