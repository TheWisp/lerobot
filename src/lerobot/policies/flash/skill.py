"""Versioned spatial annotation attached to a regular LeRobot episode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .geometry import transform


class MotionMode(str, Enum):
    DEMONSTRATED = "demonstrated"
    CONNECTION = "connection"


class EndpointRequirement(str, Enum):
    """Whether a generated waypoint must be reached before advancing."""

    REQUIRED = "required"
    PREFERRED = "preferred"


class Relation(str, Enum):
    EE_TO_OBJECT = "ee_to_object"
    EE_TO_START = "ee_to_start"
    HELD_TO_TARGET = "held_to_target"


class AttachmentAction(str, Enum):
    ATTACH = "attach"
    DETACH = "detach"


@dataclass(frozen=True)
class TurnAtPoint:
    """Permitted turn about a measured point in the reference object's frame.

    The same rule ID across adjacent intervals means one causal choice, not a
    new independent rotation at every key frame. An angle of zero reproduces
    the demonstrated relation exactly.
    """

    rule_id: str
    pivot_reference_m: tuple[float, float, float]
    axis_reference: tuple[float, float, float]
    min_angle_rad: float
    max_angle_rad: float
    source_frame_index: int | None = None
    source_evidence: str = "unverified"

    def __post_init__(self) -> None:
        pivot = np.asarray(self.pivot_reference_m, dtype=float)
        axis = np.asarray(self.axis_reference, dtype=float)
        if not self.rule_id or pivot.shape != (3,) or axis.shape != (3,):
            raise ValueError("turn needs an ID, a reference-frame pivot and direction")
        if not np.isfinite(pivot).all() or not np.isfinite(axis).all():
            raise ValueError("turn pivot and direction must be finite")
        magnitude = float(np.linalg.norm(axis))
        if not np.isclose(magnitude, 1.0, atol=1e-5):
            raise ValueError("turn direction must be a unit vector")
        if not all(np.isfinite(x) for x in (self.min_angle_rad, self.max_angle_rad)):
            raise ValueError("turn angles must be finite")
        if self.min_angle_rad > 0 or self.max_angle_rad < 0 or self.min_angle_rad >= self.max_angle_rad:
            raise ValueError("turn range must include the demonstrated zero angle")
        if self.source_frame_index is not None and self.source_frame_index < 0:
            raise ValueError("turn source frame must be nonnegative")
        if self.source_evidence not in ("observed", "symmetry_invariant", "predicted", "unverified"):
            raise ValueError(
                "turn source evidence must be observed, symmetry invariant, predicted or unverified"
            )
        object.__setattr__(self, "pivot_reference_m", tuple(float(x) for x in pivot))
        object.__setattr__(self, "axis_reference", tuple(float(x) for x in axis))


@dataclass(frozen=True)
class AttachmentTransition:
    """Meaning of one demonstrated gripper event for a tracked object."""

    event_id: str
    action: AttachmentAction
    role: str
    reference_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", AttachmentAction(self.action))
        if not all((self.event_id, self.role, self.reference_version)):
            raise ValueError("attachment transition needs event, object role and frame version")


@dataclass(frozen=True)
class SpatialSample:
    time_s: float
    demo_ee: np.ndarray
    demo_reference: np.ndarray
    demo_held: np.ndarray | None = None
    gripper: float | None = None
    source_frame_index: int | None = None
    reference_evidence: str = "unverified"
    held_evidence: str | None = None
    event_id: str | None = None
    requires_milestone: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("sample time must be nonnegative and finite")
        object.__setattr__(self, "demo_ee", transform(self.demo_ee))
        object.__setattr__(self, "demo_reference", transform(self.demo_reference))
        if self.demo_held is not None:
            object.__setattr__(self, "demo_held", transform(self.demo_held))
        if self.requires_milestone is not None and not self.event_id:
            raise ValueError("sample milestone gate needs a semantic event")


@dataclass(frozen=True)
class Interval:
    name: str
    mode: MotionMode
    relation: Relation
    reference_role: str
    samples: tuple[SpatialSample, ...]
    speed_m_s: float | None = None
    entry_milestone: str | None = None
    completion_milestone: str | None = None
    recovery_intervals: tuple[str, ...] = ()
    reference_version: str | None = None
    held_reference_version: str | None = None
    held_role: str | None = None
    endpoint_requirement: EndpointRequirement = EndpointRequirement.REQUIRED
    turn_at_point: TurnAtPoint | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_requirement", EndpointRequirement(self.endpoint_requirement))
        if not self.name or not self.reference_role or len(self.samples) < 2:
            raise ValueError("interval needs a name, reference role and at least two samples")
        if any(b.time_s <= a.time_s for a, b in zip(self.samples, self.samples[1:], strict=False)):
            raise ValueError("sample times must increase within an interval")
        if self.relation == Relation.HELD_TO_TARGET and any(s.demo_held is None for s in self.samples):
            raise ValueError("held-to-target samples require held-object poses")
        if self.held_role is not None and self.relation != Relation.HELD_TO_TARGET:
            raise ValueError("held object role only applies to held-to-target intervals")
        if self.mode == MotionMode.CONNECTION and (self.speed_m_s is None or self.speed_m_s <= 0):
            raise ValueError("generated connections need a positive reference speed")
        events = [sample.event_id for sample in self.samples if sample.event_id]
        if len(events) != len(set(events)):
            raise ValueError("semantic event IDs must be unique within an interval")
        if self.mode == MotionMode.CONNECTION and any(sample.event_id for sample in self.samples[:-1]):
            raise ValueError("generated connection events may occur only at its endpoint")
        if self.endpoint_requirement == EndpointRequirement.PREFERRED and (
            self.mode != MotionMode.CONNECTION or any(sample.event_id for sample in self.samples)
        ):
            raise ValueError("a preferred endpoint must be a generated connection without semantic events")
        if self.turn_at_point is not None and self.relation == Relation.EE_TO_START:
            raise ValueError("a reference-frame turn needs an observed object reference")

    @property
    def demo_duration_s(self) -> float:
        return self.samples[-1].time_s - self.samples[0].time_s


@dataclass(frozen=True)
class PoseRelationCriterion:
    """Observed relation between two object frames or their functional features.

    A local point replaces the moving-frame origin for the position check. A
    local direction checks only its alignment, leaving rotation about that
    direction free. Task tolerance is distinct from pose-estimator support.
    """

    moving_role: str
    reference_role: str
    reference_from_moving: np.ndarray
    moving_version: str
    reference_version: str
    translation_tolerance_m: float
    rotation_tolerance_rad: float | None
    max_pose_age_s: float
    moving_point_m: tuple[float, float, float] | None = None
    moving_direction: tuple[float, float, float] | None = None
    direction_tolerance_rad: float | None = None

    def __post_init__(self) -> None:
        if not all((self.moving_role, self.reference_role, self.moving_version, self.reference_version)):
            raise ValueError("pose relation needs two named roles and object-frame versions")
        if self.moving_role == self.reference_role:
            raise ValueError("pose relation needs distinct object roles")
        object.__setattr__(self, "reference_from_moving", transform(self.reference_from_moving))
        limits = (self.translation_tolerance_m, self.max_pose_age_s)
        if self.rotation_tolerance_rad is not None:
            limits += (self.rotation_tolerance_rad,)
        if any(not np.isfinite(value) or value <= 0 for value in limits):
            raise ValueError("pose relation tolerances and maximum age must be finite and positive")
        if self.rotation_tolerance_rad is not None and self.direction_tolerance_rad is not None:
            raise ValueError("specify either full rotation or direction tolerance")
        if (self.moving_direction is None) != (self.direction_tolerance_rad is None):
            raise ValueError("a moving direction and its tolerance must be specified together")
        if self.direction_tolerance_rad is not None and (
            not np.isfinite(self.direction_tolerance_rad) or self.direction_tolerance_rad <= 0
        ):
            raise ValueError("direction tolerance must be finite and positive")
        if self.moving_point_m is not None:
            point = np.asarray(self.moving_point_m, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("moving point must have three finite local coordinates")
            object.__setattr__(self, "moving_point_m", tuple(float(value) for value in point))
        if self.moving_direction is not None:
            direction = np.asarray(self.moving_direction, dtype=float)
            if (
                direction.shape != (3,)
                or not np.isfinite(direction).all()
                or not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-5)
            ):
                raise ValueError("moving direction must be a unit local vector")
            object.__setattr__(self, "moving_direction", tuple(float(value) for value in direction))


@dataclass(frozen=True)
class MilestoneSpec:
    name: str
    depends_on_roles: tuple[str, ...]
    pose_relation: PoseRelationCriterion | None = None
    require_post_motion_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.depends_on_roles or any(not role for role in self.depends_on_roles):
            raise ValueError("milestone needs a name and dependent object roles")
        if self.pose_relation is not None and not {
            self.pose_relation.moving_role,
            self.pose_relation.reference_role,
        } <= set(self.depends_on_roles):
            raise ValueError("pose relation roles must be milestone dependencies")


@dataclass(frozen=True)
class FlashSkill:
    source_repo_id: str
    source_episode: int
    intervals: tuple[Interval, ...]
    schema_version: int = 1
    milestones: tuple[MilestoneSpec, ...] = ()
    attachment_transitions: tuple[AttachmentTransition, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or not self.source_repo_id
            or self.source_episode < 0
            or not self.intervals
        ):
            raise ValueError("invalid FLASH skill manifest")
        names = [interval.name for interval in self.intervals]
        if len(names) != len(set(names)):
            raise ValueError("interval names must be unique")
        event_ids = [
            sample.event_id
            for interval in self.intervals
            for sample in interval.samples
            if sample.event_id is not None
        ]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("semantic event IDs must be unique across the skill")
        transition_ids = [transition.event_id for transition in self.attachment_transitions]
        if len(transition_ids) != len(set(transition_ids)) or not set(transition_ids) <= set(event_ids):
            raise ValueError("attachment transitions must bind distinct demonstrated events")
        milestone_names = {milestone.name for milestone in self.milestones}
        if len(milestone_names) != len(self.milestones):
            raise ValueError("milestone names must be unique")
        referenced = {
            name
            for interval in self.intervals
            for name in (
                interval.entry_milestone,
                interval.completion_milestone,
                *(sample.requires_milestone for sample in interval.samples),
            )
            if name is not None
        }
        if not referenced <= milestone_names:
            raise ValueError(f"missing milestone definitions: {sorted(referenced - milestone_names)}")
        turns: dict[str, tuple[TurnAtPoint, str, str | None, int]] = {}
        for index, interval in enumerate(self.intervals):
            rule = interval.turn_at_point
            if rule is None:
                continue
            prior = turns.get(rule.rule_id)
            if prior is not None and (
                prior[:3] != (rule, interval.reference_role, interval.reference_version)
                or prior[3] != index - 1
            ):
                raise ValueError("a turn rule must stay identical across adjacent intervals of one reference")
            turns[rule.rule_id] = (rule, interval.reference_role, interval.reference_version, index)

    @classmethod
    def load(cls, path: str | Path) -> FlashSkill:
        """Load a sidecar; paths are resolved by the dataset consumer, not this file."""
        data = json.loads(Path(path).read_text())
        intervals = []
        for item in data["intervals"]:
            samples = tuple(
                SpatialSample(
                    time_s=sample["time_s"],
                    demo_ee=sample["demo_ee"],
                    demo_reference=sample["demo_reference"],
                    demo_held=sample.get("demo_held"),
                    gripper=sample.get("gripper"),
                    source_frame_index=sample.get("source_frame_index"),
                    reference_evidence=sample.get("reference_evidence", "unverified"),
                    held_evidence=sample.get("held_evidence"),
                    event_id=sample.get("event_id"),
                    requires_milestone=sample.get("requires_milestone"),
                )
                for sample in item["samples"]
            )
            intervals.append(
                Interval(
                    name=item["name"],
                    mode=MotionMode(item["mode"]),
                    relation=Relation(item["relation"]),
                    reference_role=item["reference_role"],
                    samples=samples,
                    speed_m_s=item.get("speed_m_s"),
                    entry_milestone=item.get("entry_milestone"),
                    completion_milestone=item.get("completion_milestone"),
                    recovery_intervals=tuple(item.get("recovery_intervals", [])),
                    reference_version=item.get("reference_version"),
                    held_reference_version=item.get("held_reference_version"),
                    held_role=item.get("held_role"),
                    endpoint_requirement=EndpointRequirement(
                        item.get("endpoint_requirement", EndpointRequirement.REQUIRED)
                    ),
                    turn_at_point=TurnAtPoint(**item["turn_at_point"]) if item.get("turn_at_point") else None,
                )
            )
        milestones = tuple(
            MilestoneSpec(
                item["name"],
                tuple(item["depends_on_roles"]),
                PoseRelationCriterion(**item["pose_relation"]) if item.get("pose_relation") else None,
                item.get("require_post_motion_evidence", False),
            )
            for item in data.get("milestones", [])
        )
        attachment_transitions = tuple(
            AttachmentTransition(
                item["event_id"],
                AttachmentAction(item["action"]),
                item["role"],
                item["reference_version"],
            )
            for item in data.get("attachment_transitions", [])
        )
        return cls(
            data["source_repo_id"],
            data["source_episode"],
            tuple(intervals),
            data["schema_version"],
            milestones,
            attachment_transitions,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source_repo_id": self.source_repo_id,
            "source_episode": self.source_episode,
            "attachment_transitions": [
                {
                    "event_id": transition.event_id,
                    "action": transition.action.value,
                    "role": transition.role,
                    "reference_version": transition.reference_version,
                }
                for transition in self.attachment_transitions
            ],
            "milestones": [
                {
                    "name": milestone.name,
                    "depends_on_roles": list(milestone.depends_on_roles),
                    "require_post_motion_evidence": milestone.require_post_motion_evidence,
                    "pose_relation": (
                        {
                            "moving_role": milestone.pose_relation.moving_role,
                            "reference_role": milestone.pose_relation.reference_role,
                            "reference_from_moving": milestone.pose_relation.reference_from_moving.tolist(),
                            "moving_version": milestone.pose_relation.moving_version,
                            "reference_version": milestone.pose_relation.reference_version,
                            "translation_tolerance_m": milestone.pose_relation.translation_tolerance_m,
                            "rotation_tolerance_rad": milestone.pose_relation.rotation_tolerance_rad,
                            "moving_point_m": milestone.pose_relation.moving_point_m,
                            "moving_direction": milestone.pose_relation.moving_direction,
                            "direction_tolerance_rad": milestone.pose_relation.direction_tolerance_rad,
                            "max_pose_age_s": milestone.pose_relation.max_pose_age_s,
                        }
                        if milestone.pose_relation is not None
                        else None
                    ),
                }
                for milestone in self.milestones
            ],
            "intervals": [
                {
                    "name": interval.name,
                    "mode": interval.mode.value,
                    "relation": interval.relation.value,
                    "reference_role": interval.reference_role,
                    "reference_version": interval.reference_version,
                    "held_reference_version": interval.held_reference_version,
                    "held_role": interval.held_role,
                    "endpoint_requirement": interval.endpoint_requirement.value,
                    "turn_at_point": (
                        {
                            "rule_id": interval.turn_at_point.rule_id,
                            "pivot_reference_m": interval.turn_at_point.pivot_reference_m,
                            "axis_reference": interval.turn_at_point.axis_reference,
                            "min_angle_rad": interval.turn_at_point.min_angle_rad,
                            "max_angle_rad": interval.turn_at_point.max_angle_rad,
                            "source_frame_index": interval.turn_at_point.source_frame_index,
                            "source_evidence": interval.turn_at_point.source_evidence,
                        }
                        if interval.turn_at_point is not None
                        else None
                    ),
                    "speed_m_s": interval.speed_m_s,
                    "entry_milestone": interval.entry_milestone,
                    "completion_milestone": interval.completion_milestone,
                    "recovery_intervals": list(interval.recovery_intervals),
                    "samples": [
                        {
                            "time_s": sample.time_s,
                            "demo_ee": sample.demo_ee.tolist(),
                            "demo_reference": sample.demo_reference.tolist(),
                            "demo_held": sample.demo_held.tolist() if sample.demo_held is not None else None,
                            "gripper": sample.gripper,
                            "source_frame_index": sample.source_frame_index,
                            "reference_evidence": sample.reference_evidence,
                            "held_evidence": sample.held_evidence,
                            "event_id": sample.event_id,
                            "requires_milestone": sample.requires_milestone,
                        }
                        for sample in interval.samples
                    ],
                }
                for interval in self.intervals
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
