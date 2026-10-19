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


class Relation(str, Enum):
    EE_TO_OBJECT = "ee_to_object"
    HELD_TO_TARGET = "held_to_target"


@dataclass(frozen=True)
class SpatialSample:
    time_s: float
    demo_ee: np.ndarray
    demo_reference: np.ndarray
    demo_held: np.ndarray | None = None
    gripper: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("sample time must be nonnegative and finite")
        object.__setattr__(self, "demo_ee", transform(self.demo_ee))
        object.__setattr__(self, "demo_reference", transform(self.demo_reference))
        if self.demo_held is not None:
            object.__setattr__(self, "demo_held", transform(self.demo_held))


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

    def __post_init__(self) -> None:
        if not self.name or not self.reference_role or len(self.samples) < 2:
            raise ValueError("interval needs a name, reference role and at least two samples")
        if any(b.time_s <= a.time_s for a, b in zip(self.samples, self.samples[1:], strict=False)):
            raise ValueError("sample times must increase within an interval")
        if self.relation == Relation.HELD_TO_TARGET and any(s.demo_held is None for s in self.samples):
            raise ValueError("held-to-target samples require held-object poses")
        if self.mode == MotionMode.CONNECTION and (self.speed_m_s is None or self.speed_m_s <= 0):
            raise ValueError("generated connections need a positive reference speed")

    @property
    def demo_duration_s(self) -> float:
        return self.samples[-1].time_s - self.samples[0].time_s


@dataclass(frozen=True)
class MilestoneSpec:
    name: str
    depends_on_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.depends_on_roles or any(not role for role in self.depends_on_roles):
            raise ValueError("milestone needs a name and dependent object roles")


@dataclass(frozen=True)
class FlashSkill:
    source_repo_id: str
    source_episode: int
    intervals: tuple[Interval, ...]
    schema_version: int = 1
    milestones: tuple[MilestoneSpec, ...] = ()

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
        milestone_names = {milestone.name for milestone in self.milestones}
        referenced = {
            name
            for interval in self.intervals
            for name in (interval.entry_milestone, interval.completion_milestone)
            if name is not None
        }
        if not referenced <= milestone_names:
            raise ValueError(f"missing milestone definitions: {sorted(referenced - milestone_names)}")

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
                )
            )
        milestones = tuple(
            MilestoneSpec(item["name"], tuple(item["depends_on_roles"]))
            for item in data.get("milestones", [])
        )
        return cls(
            data["source_repo_id"],
            data["source_episode"],
            tuple(intervals),
            data["schema_version"],
            milestones,
        )
