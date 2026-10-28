"""Retarget a demonstration without tying it to the live frame count."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .geometry import held_object_goal, interpolate, pregrasp_goal, transform, turn_about_reference_point
from .skill import Interval, MotionMode, Relation, SpatialSample


class Evidence(str, Enum):
    OBSERVED = "observed"
    PREDICTED = "predicted"
    LOST = "lost"


def required_rotation_axes(
    point: np.ndarray,
    direction: np.ndarray | None,
) -> tuple[bool, bool, bool]:
    """Local rotations that can change a point or a directed feature.

    Ignore only floating-point roundoff when a vector lies on an axis. A
    nearly aligned feature still requires that axis to be observed.
    """
    vectors = (point,) if direction is None else (point, direction)
    epsilon = np.finfo(float).eps
    return tuple(
        any(
            np.linalg.norm(np.cross(axis, vector)) > 32 * epsilon * max(1.0, np.linalg.norm(vector))
            for vector in vectors
        )
        for axis in np.eye(3)
    )


@dataclass(frozen=True)
class PoseEstimate:
    """Pose in the robot-base frame, with times on the motor monotonic clock.

    Supported translation axes use the robot-base frame; supported rotation
    axes describe rotations about the object's local X, Y and Z axes. They
    mark task-relevant evidence, not a claim that all six axes were observed.
    A partial pose remains available when the task is invariant to its
    unsupported axes.
    """

    role: str
    pose: np.ndarray | None
    source_time_s: float
    available_time_s: float
    evidence: Evidence
    reference_version: str
    translation_supported: tuple[bool, bool, bool]
    rotation_supported: tuple[bool, bool, bool]
    direct_translation_observed: bool = False
    estimator: str = "unspecified"
    last_direct_observation_s: float | None = None
    symmetry_canonicalized: bool = False
    source_frame_id: int | None = None

    def __post_init__(self) -> None:
        if self.pose is not None:
            object.__setattr__(self, "pose", transform(self.pose))
        if self.evidence != Evidence.LOST and self.pose is None:
            raise ValueError("observed or predicted evidence needs a pose")
        if self.available_time_s < self.source_time_s:
            raise ValueError("availability cannot precede source time")
        if len(self.translation_supported) != 3 or len(self.rotation_supported) != 3:
            raise ValueError("pose support must specify three translation and three rotation axes")
        if not self.estimator:
            raise ValueError("pose estimator provenance must be nonempty")
        if self.last_direct_observation_s is not None and self.last_direct_observation_s > self.source_time_s:
            raise ValueError("direct observation cannot postdate pose source")
        if self.source_frame_id is not None and (
            isinstance(self.source_frame_id, bool)
            or not isinstance(self.source_frame_id, int)
            or self.source_frame_id < 0
        ):
            raise ValueError("pose source frame ID must be a nonnegative integer")

    @property
    def has_direct_translation(self) -> bool:
        """A current image measured the centre, even if orientation was constrained."""
        return self.evidence == Evidence.OBSERVED or self.direct_translation_observed

    def supports(
        self,
        translation: tuple[bool, bool, bool] = (True, True, True),
        rotation: tuple[bool, bool, bool] = (True, True, True),
    ) -> bool:
        return all(
            not required or supported
            for required, supported in zip(translation, self.translation_supported, strict=True)
        ) and all(
            not required or supported
            for required, supported in zip(rotation, self.rotation_supported, strict=True)
        )

    def usable(
        self,
        now_s: float,
        max_age_s: float,
        allow_prediction: bool,
        translation: tuple[bool, bool, bool] = (True, True, True),
        rotation: tuple[bool, bool, bool] = (True, True, True),
    ) -> bool:
        return (
            self.pose is not None
            and self.evidence != Evidence.LOST
            and (allow_prediction or self.evidence == Evidence.OBSERVED)
            and 0 <= now_s - self.source_time_s <= max_age_s
            and now_s >= self.available_time_s
            and self.supports(translation, rotation)
        )


@dataclass(frozen=True)
class AttachmentState:
    """``T_E_H`` for the current rollout, distinct from a world pose."""

    gripper_from_object: np.ndarray
    source_time_s: float
    available_time_s: float
    reference_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "gripper_from_object", transform(self.gripper_from_object))
        if self.available_time_s < self.source_time_s:
            raise ValueError("attachment availability cannot precede source time")

    def usable(self, now_s: float, max_age_s: float) -> bool:
        return self.available_time_s <= now_s and 0 <= now_s - self.source_time_s <= max_age_s


class MissingEvidenceError(RuntimeError):
    pass


def _interpolated_sample(interval: Interval, local_time_s: float) -> SpatialSample:
    """Sample source motion on the demonstrated local clock."""
    samples = interval.samples
    absolute = samples[0].time_s + np.clip(local_time_s, 0, interval.demo_duration_s)
    for left, right in zip(samples, samples[1:], strict=False):
        if absolute <= right.time_s:
            u = (absolute - left.time_s) / (right.time_s - left.time_s)
            return SpatialSample(
                time_s=float(absolute),
                demo_ee=interpolate(left.demo_ee, right.demo_ee, u),
                demo_reference=interpolate(left.demo_reference, right.demo_reference, u),
                demo_held=(
                    interpolate(left.demo_held, right.demo_held, u)
                    if left.demo_held is not None and right.demo_held is not None
                    else None
                ),
                gripper=(
                    left.gripper
                    if right.event_id is not None and absolute < right.time_s
                    else (
                        (1 - u) * left.gripper + u * right.gripper
                        if left.gripper is not None and right.gripper is not None
                        else None
                    )
                ),
            )
    return samples[-1]


def spatial_goal(
    interval: Interval,
    sample: SpatialSample,
    live_reference: np.ndarray,
    live_attachment: np.ndarray | None,
    turn_angle_rad: float = 0.0,
) -> np.ndarray:
    if interval.relation in (Relation.EE_TO_OBJECT, Relation.EE_TO_START):
        goal = pregrasp_goal(sample.demo_ee, sample.demo_reference, live_reference)
    else:
        if live_attachment is None or sample.demo_held is None:
            raise MissingEvidenceError("held-to-target motion needs a live gripper-to-object attachment")
        goal = held_object_goal(sample.demo_held, sample.demo_reference, live_reference, live_attachment)
    rule = interval.turn_at_point
    if rule is None:
        if turn_angle_rad != 0:
            raise ValueError("turn angle provided without a declared turn rule")
        return goal
    if not rule.min_angle_rad <= turn_angle_rad <= rule.max_angle_rad:
        raise ValueError("turn angle exceeds its declared permitted range")
    return turn_about_reference_point(
        goal, live_reference, rule.pivot_reference_m, rule.axis_reference, turn_angle_rad
    )


def demonstrated_goal(
    interval: Interval,
    local_time_s: float,
    live_reference: np.ndarray,
    live_attachment: np.ndarray | None = None,
    turn_angle_rad: float = 0.0,
) -> tuple[np.ndarray, float | None]:
    if interval.mode != MotionMode.DEMONSTRATED:
        raise ValueError("expected a demonstrated interval")
    sample = _interpolated_sample(interval, local_time_s)
    return spatial_goal(interval, sample, live_reference, live_attachment, turn_angle_rad), sample.gripper


@dataclass
class ConnectionTrajectory:
    """A generated free-space connection with distance and joint-speed timing.

    Retargeting starts a *remaining* path at the measured EE pose; downstream
    demonstrated intervals begin their own clock only after this path ends.
    """

    interval: Interval
    start_pose: np.ndarray
    goal_pose: np.ndarray
    start_time_s: float
    duration_s: float
    joint_retimed: bool = False
    requested_goal_pose: np.ndarray | None = None
    achieved_fraction: float = 1.0
    turn_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        self.start_pose = transform(self.start_pose)
        self.goal_pose = transform(self.goal_pose)
        self.requested_goal_pose = transform(
            self.goal_pose if self.requested_goal_pose is None else self.requested_goal_pose
        )
        if not 0 <= self.achieved_fraction <= 1:
            raise ValueError("connection achieved fraction must lie between zero and one")
        if not np.isfinite(self.turn_angle_rad):
            raise ValueError("connection turn angle must be finite")

    @classmethod
    def begin(
        cls,
        interval: Interval,
        current_ee: np.ndarray,
        live_reference: np.ndarray,
        live_attachment: np.ndarray | None,
        now_s: float,
        turn_angle_rad: float = 0.0,
    ) -> ConnectionTrajectory:
        if interval.mode != MotionMode.CONNECTION:
            raise ValueError("expected generated connection")
        endpoint = spatial_goal(
            interval, interval.samples[-1], live_reference, live_attachment, turn_angle_rad
        )
        distance = float(np.linalg.norm(transform(endpoint)[:3, 3] - transform(current_ee)[:3, 3]))
        return cls(
            interval,
            transform(current_ee),
            transform(endpoint),
            now_s,
            distance / interval.speed_m_s,
            requested_goal_pose=transform(endpoint),
            turn_angle_rad=turn_angle_rad,
        )

    def retarget(
        self,
        current_ee: np.ndarray,
        live_reference: np.ndarray,
        live_attachment: np.ndarray | None,
        now_s: float,
    ) -> None:
        replacement = self.begin(
            self.interval, current_ee, live_reference, live_attachment, now_s, self.turn_angle_rad
        )
        self.start_pose, self.goal_pose = replacement.start_pose, replacement.goal_pose
        self.requested_goal_pose = replacement.requested_goal_pose
        self.achieved_fraction = replacement.achieved_fraction
        self.start_time_s, self.duration_s = replacement.start_time_s, replacement.duration_s
        self.joint_retimed = False

    def at(self, now_s: float) -> np.ndarray:
        progress = 1.0 if self.duration_s == 0 else (now_s - self.start_time_s) / self.duration_s
        return interpolate(self.start_pose, self.goal_pose, progress)

    def finished(self, now_s: float) -> bool:
        return now_s >= self.start_time_s + self.duration_s
