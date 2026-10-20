"""Kinematics-neutral conversion from spatial references to joint chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .execution import TimedAction
from .geometry import interpolate, transform
from .skill import EndpointRequirement, Interval, MotionMode, Relation
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


class _ConnectionSpeedExceededError(InfeasibleTrajectoryError):
    def __init__(self, tick: int, ratio: float) -> None:
        super().__init__(f"joint speed exceeds configured limit at tick {tick}")
        self.tick = tick
        self.ratio = ratio


@dataclass(frozen=True)
class PlannerLimits:
    joint_speed_deg_s: tuple[float, ...]
    position_error_m: float
    rotation_error_rad: float
    max_pose_age_s: float
    allow_predicted_reference: bool = False
    ik_refinement_attempts: int = 1
    max_attachment_age_s: float | None = None
    max_connection_ik_samples: int = 128
    max_connection_retimings: int = 8

    def __post_init__(self) -> None:
        values = (
            *self.joint_speed_deg_s,
            self.position_error_m,
            self.rotation_error_rad,
            self.max_pose_age_s,
        )
        if not values or not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("planner limits must be finite and positive")
        if (
            isinstance(self.ik_refinement_attempts, bool)
            or not isinstance(self.ik_refinement_attempts, int)
            or self.ik_refinement_attempts < 1
        ):
            raise ValueError("IK refinement attempts must be a positive integer")
        if self.max_attachment_age_s is not None and (
            not np.isfinite(self.max_attachment_age_s) or self.max_attachment_age_s <= 0
        ):
            raise ValueError("maximum attachment age must be finite and positive")
        if (
            isinstance(self.max_connection_ik_samples, bool)
            or not isinstance(self.max_connection_ik_samples, int)
            or self.max_connection_ik_samples < 2
        ):
            raise ValueError("maximum connection IK samples must be an integer of at least two")
        if (
            isinstance(self.max_connection_retimings, bool)
            or not isinstance(self.max_connection_retimings, int)
            or self.max_connection_retimings < 1
        ):
            raise ValueError("maximum connection retimings must be a positive integer")

    @property
    def effective_attachment_age_s(self) -> float:
        # A rigid grasp assumption remains in force until release or a
        # contradictory independent observation. Pose freshness belongs to
        # the target; elapsed wall time alone is not evidence of object slip.
        return float("inf") if self.max_attachment_age_s is None else self.max_attachment_age_s


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
        issued_events: frozenset[str] = frozenset(),
    ) -> list[TimedAction]:
        kwargs = {
            "start_tick": start_tick,
            "count": count,
            "epoch_s": epoch_s,
            "period_s": period_s,
            "interval_start_s": interval_start_s,
            "seed_deg": seed_deg,
            "reference": reference,
            "attachment": attachment,
            "connection": connection,
            "issued_events": issued_events,
        }
        if interval.mode != MotionMode.CONNECTION or connection is None:
            return self._plan_once(interval, **kwargs)
        for attempt in range(self.limits.max_connection_retimings):
            try:
                return self._plan_once(interval, **kwargs)
            except _ConnectionSpeedExceededError as exc:
                if attempt + 1 == self.limits.max_connection_retimings:
                    raise InfeasibleTrajectoryError(
                        "generated connection could not be timed within joint speed limits "
                        f"at tick {exc.tick} (ratio {exc.ratio:.6f}, duration {connection.duration_s:.3f} s)"
                    ) from exc
                first_due_s = epoch_s + start_tick * period_s
                if first_due_s > connection.start_time_s + 1e-9:
                    assert interval.speed_m_s is not None
                    remaining_s = max(0.0, connection.start_time_s + connection.duration_s - first_due_s)
                    connection.start_pose = transform(
                        self.kinematics.forward_kinematics(np.asarray(seed_deg, dtype=float))
                    )
                    connection.start_time_s = first_due_s
                    distance_m = np.linalg.norm(connection.goal_pose[:3, 3] - connection.start_pose[:3, 3])
                    connection.duration_s = max(remaining_s, distance_m / interval.speed_m_s)
                    connection.joint_retimed = False
                # The sampled joint-speed bound is only an estimate of the
                # discrete IK trajectory. Give a failed tick at least one
                # additional motor period; multiplying by a near-1 ratio
                # alone can repeat the same failure on every retry.
                connection.duration_s = connection.duration_s * exc.ratio + period_s
        raise AssertionError("connection retiming loop did not return")

    def _plan_once(
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
        issued_events: frozenset[str] = frozenset(),
    ) -> list[TimedAction]:
        if count <= 0 or start_tick < 0 or period_s <= 0:
            raise ValueError("invalid chunk timing")
        seed = np.asarray(seed_deg, dtype=float).copy()
        if seed.shape != (len(self.arm_joint_names),) or not np.isfinite(seed).all():
            raise ValueError("invalid joint seed")
        if interval.mode == MotionMode.DEMONSTRATED:
            first_local_s = epoch_s + start_tick * period_s - interval_start_s
            missed = [
                sample.event_id
                for sample in interval.samples
                if sample.event_id
                and sample.time_s - interval.samples[0].time_s <= first_local_s - period_s + 1e-9
                and sample.event_id not in issued_events
            ]
            if missed:
                raise InfeasibleTrajectoryError(
                    f"required semantic event was not issued: {', '.join(missed)}"
                )
        elif (
            connection is not None
            and interval.samples[-1].event_id
            and interval.samples[-1].event_id not in issued_events
            and connection.finished(epoch_s + (start_tick - 1) * period_s)
        ):
            raise InfeasibleTrajectoryError(
                f"required semantic event was not issued: {interval.samples[-1].event_id}"
            )
        if interval.mode == MotionMode.CONNECTION and connection is not None:
            self._retime_connection(connection, seed, period_s)
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
                local_time_s = due_s - interval_start_s
                pose, grip = demonstrated_goal(
                    interval,
                    local_time_s,
                    reference.pose,
                    (
                        self._attachment_pose(attachment, due_s, interval.held_reference_version)
                        if interval.relation == Relation.HELD_TO_TARGET
                        else None
                    ),
                )
                complete = due_s >= interval_start_s + interval.demo_duration_s
                matching_events = [
                    sample
                    for sample in interval.samples
                    if sample.event_id
                    and local_time_s - period_s
                    < sample.time_s - interval.samples[0].time_s
                    <= local_time_s + 1e-9
                ]
                if len(matching_events) > 1:
                    raise InfeasibleTrajectoryError(f"multiple semantic events fall on tick {tick}")
                event = matching_events[0] if matching_events else None
            else:
                if connection is None:
                    raise ValueError("generated interval requires a connection trajectory")
                live_attachment = (
                    self._attachment_pose(attachment, due_s, interval.held_reference_version)
                    if interval.relation == Relation.HELD_TO_TARGET
                    else None
                )
                current_endpoint = spatial_goal(
                    interval, interval.samples[-1], reference.pose, live_attachment
                )
                if (
                    np.linalg.norm(current_endpoint[:3, 3] - connection.requested_goal_pose[:3, 3])
                    > self.limits.position_error_m
                    or np.arccos(
                        np.clip(
                            (
                                np.trace(current_endpoint[:3, :3].T @ connection.requested_goal_pose[:3, :3])
                                - 1
                            )
                            / 2,
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
                complete = connection.finished(due_s)
                event = interval.samples[-1] if complete else None
                grip = (
                    interval.samples[0].gripper
                    if not complete and interval.samples[-1].event_id is not None
                    else interval.samples[-1].gripper
                )
            refinement_seed = seed
            for _ in range(self.limits.ik_refinement_attempts):
                solved = np.asarray(
                    self.kinematics.inverse_kinematics(refinement_seed, transform(pose)), dtype=float
                )
                if solved.shape != seed.shape or not np.isfinite(solved).all():
                    raise InfeasibleTrajectoryError(f"IK returned an invalid solution at tick {tick}")
                reached = transform(self.kinematics.forward_kinematics(solved))
                translation_error = np.linalg.norm(reached[:3, 3] - pose[:3, 3])
                angle_error = np.arccos(np.clip((np.trace(reached[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1))
                if (
                    translation_error <= self.limits.position_error_m
                    and angle_error <= self.limits.rotation_error_rad
                ):
                    break
                refinement_seed = solved
            speed = np.abs(solved - seed) / period_s
            if np.any(speed > self.limits.joint_speed_deg_s):
                if interval.mode == MotionMode.CONNECTION:
                    raise _ConnectionSpeedExceededError(
                        tick, float(np.max(speed / self.limits.joint_speed_deg_s))
                    )
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
            actions.append(
                TimedAction(
                    tick,
                    command,
                    interval.name,
                    interval_complete=complete,
                    event_id=event.event_id if event is not None else None,
                    requires_milestone=event.requires_milestone if event is not None else None,
                    may_skip_if_late=interval.mode == MotionMode.CONNECTION,
                )
            )
            seed = solved
            if complete:
                break
        return actions

    def _retime_connection(
        self, connection: ConnectionTrajectory, seed_deg: np.ndarray, period_s: float
    ) -> None:
        """Bound joint speed along a generated spatial path before publishing its first tick.

        Adaptive IK samples follow the same spatial interpolation that the
        motor ticks will use. A sample advances only when every joint moves no
        farther than it could in one motor period. The measured joint travel
        per fraction of the path supplies a lower bound on total duration.
        The finite sample budget rejects discontinuous or costly IK paths.
        """
        if connection.joint_retimed:
            return
        joint_speeds = np.asarray(self.limits.joint_speed_deg_s, dtype=float)
        max_step = joint_speeds * period_s
        previous = seed_deg.copy()
        fraction = 0.0
        step = 1.0
        required_duration_s = 0.0
        for _ in range(self.limits.max_connection_ik_samples):
            next_fraction = min(1.0, fraction + step)
            if next_fraction <= fraction:
                break
            target = interpolate(connection.start_pose, connection.goal_pose, next_fraction)
            refinement_seed = previous
            for _ in range(self.limits.ik_refinement_attempts):
                solved = np.asarray(self.kinematics.inverse_kinematics(refinement_seed, target), dtype=float)
                if solved.shape != previous.shape or not np.isfinite(solved).all():
                    raise InfeasibleTrajectoryError("generated connection IK returned an invalid solution")
                reached = transform(self.kinematics.forward_kinematics(solved))
                translation_error = np.linalg.norm(reached[:3, 3] - target[:3, 3])
                rotation_error = np.arccos(
                    np.clip((np.trace(reached[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)
                )
                if (
                    translation_error <= self.limits.position_error_m
                    and rotation_error <= self.limits.rotation_error_rad
                ):
                    break
                refinement_seed = solved
            if (
                translation_error > self.limits.position_error_m
                or rotation_error > self.limits.rotation_error_rad
            ):
                # A local IK solver can fail from a distant seed even when
                # nearby poses have a continuous solution. Subdivide the
                # spatial step before declaring the whole connection infeasible.
                step /= 2
                continue
            joint_step = np.abs(solved - previous)
            if np.any(joint_step > max_step + 1e-9):
                step /= 2
                continue
            actual_fraction = next_fraction - fraction
            required_duration_s = max(
                required_duration_s,
                float(np.max(joint_step / (joint_speeds * actual_fraction))),
            )
            previous = solved
            fraction = next_fraction
            if fraction >= 1.0 - 1e-9:
                # One motor period covers discretization at the sampled peaks.
                connection.duration_s = max(connection.duration_s, required_duration_s + period_s)
                connection.joint_retimed = True
                return
            step = min(1.0 - fraction, step * 2)
        if connection.interval.endpoint_requirement == EndpointRequirement.PREFERRED:
            # An annotated preferred waypoint has no semantic event. If the
            # bounded local search cannot finish, retain only the continuous
            # IK prefix that was actually checked. The next interval starts
            # from that measured endpoint and must satisfy its own limits.
            connection.goal_pose = interpolate(
                connection.start_pose, connection.requested_goal_pose, fraction
            )
            connection.achieved_fraction = fraction
            connection.duration_s = max(
                period_s,
                connection.duration_s * fraction,
                required_duration_s * fraction + period_s,
            )
            connection.joint_retimed = True
            return
        raise InfeasibleTrajectoryError("generated connection has no bounded continuous IK path")

    def _attachment_pose(
        self, attachment: AttachmentState | None, due_s: float, required_version: str | None
    ) -> np.ndarray | None:
        if attachment is None:
            return None
        if not attachment.usable(due_s, self.limits.effective_attachment_age_s):
            raise MissingEvidenceError("held object attachment is stale or lost")
        if required_version is None or attachment.reference_version != required_version:
            raise MissingEvidenceError("held attachment does not share the demonstrated object frame")
        return attachment.gripper_from_object
