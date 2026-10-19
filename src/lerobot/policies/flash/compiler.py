"""Compile an annotated regular LeRobot episode into spatial skill intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .geometry import inverse, transform
from .skill import (
    AttachmentTransition,
    FlashSkill,
    Interval,
    MilestoneSpec,
    MotionMode,
    PoseRelationCriterion,
    Relation,
    SpatialSample,
)
from .trajectory import Evidence, PoseEstimate


class DatasetEpisode(Protocol):
    features: dict

    def __getitem__(self, index: int) -> dict[str, Any]: ...

    def episode_rows(self, episode: int) -> tuple[int, int]: ...


class ForwardKinematics(Protocol):
    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ActionEventAnnotation:
    frame_index: int
    event_id: str
    requires_milestone: str | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0 or not self.event_id:
            raise ValueError("action event needs a frame and ID")


@dataclass(frozen=True)
class IntervalAnnotation:
    name: str
    first_frame: int
    last_frame: int
    mode: MotionMode
    relation: Relation
    reference_role: str
    held_role: str | None = None
    speed_m_s: float | None = None
    entry_milestone: str | None = None
    completion_milestone: str | None = None
    recovery_intervals: tuple[str, ...] = ()
    allow_predicted_demo_pose: bool = False
    events: tuple[ActionEventAnnotation, ...] = ()

    def __post_init__(self) -> None:
        if self.first_frame < 0 or self.last_frame <= self.first_frame:
            raise ValueError("interval annotation needs ordered frame bounds")
        if self.relation == Relation.HELD_TO_TARGET and not self.held_role:
            raise ValueError("held-to-target interval needs a held object role")
        if any(not self.first_frame <= event.frame_index <= self.last_frame for event in self.events):
            raise ValueError("action events must lie within their interval")
        if len({event.frame_index for event in self.events}) != len(self.events):
            raise ValueError("only one semantic event may be annotated per demo frame")


@dataclass(frozen=True)
class PoseRelationAnnotation:
    """Human-selected demo frame and explicit task tolerance for one milestone."""

    name: str
    frame_index: int
    moving_role: str
    reference_role: str
    translation_tolerance_m: float
    rotation_tolerance_rad: float | None
    max_pose_age_s: float
    allow_predicted_demo_pose: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.frame_index < 0 or not self.moving_role or not self.reference_role:
            raise ValueError("pose milestone annotation needs a name, frame and two roles")


def _measured_pose(
    estimates: dict[str, PoseEstimate],
    role: str,
    frame_index: int,
    allow_prediction: bool,
    *,
    require_rotation: bool = True,
) -> PoseEstimate:
    estimate = estimates.get(role)
    if estimate is None or estimate.pose is None or not estimate.supports(rotation=(require_rotation,) * 3):
        raise ValueError(f"frame {frame_index}: {role} has no required metric pose axes")
    if estimate.evidence == Evidence.LOST or (
        estimate.evidence == Evidence.PREDICTED and not allow_prediction
    ):
        raise ValueError(f"frame {frame_index}: {role} pose is not observed")
    return estimate


def _joint_index(names: list[str] | tuple[str, ...] | None, joint: str, feature: str) -> int:
    if names is None:
        raise ValueError(f"{feature} has no named joints")
    matches = [index for index, name in enumerate(names) if name in (joint, f"{joint}.pos")]
    if len(matches) != 1:
        raise ValueError(f"{feature} must contain exactly one entry for {joint}")
    return matches[0]


def compile_skill(
    *,
    dataset: DatasetEpisode,
    source_repo_id: str,
    source_episode: int,
    kinematics: ForwardKinematics,
    arm_joint_names: tuple[str, ...],
    gripper_joint_name: str,
    poses_by_frame: dict[int, dict[str, PoseEstimate]],
    annotations: tuple[IntervalAnnotation, ...],
    milestones: tuple[MilestoneSpec, ...] = (),
    pose_relation_annotations: tuple[PoseRelationAnnotation, ...] = (),
    attachment_transitions: tuple[AttachmentTransition, ...] = (),
    joint_units: str = "degrees",
) -> FlashSkill:
    """Use measured robot state, accepted gripper action and pose sidecars.

    ``dataset`` is a normal ``LeRobotDataset(..., episodes=[episode])`` or a
    compatible local reader. Its frame indices here are episode-local. The
    caller supplies pose estimates registered into stable object frames;
    missing or unsupported demo geometry is rejected rather than invented.
    """
    if joint_units != "degrees":
        raise ValueError("FLASH kinematics currently requires motor-degree dataset state")
    state_names = dataset.features["observation.state"]["names"]
    action_names = dataset.features["action"]["names"]
    joint_indices = [_joint_index(state_names, name, "observation.state") for name in arm_joint_names]
    gripper_index = _joint_index(action_names, gripper_joint_name, "action")
    first_row, episode_length = dataset.episode_rows(source_episode)
    intervals = []
    for annotation in annotations:
        samples = []
        versions: dict[str, str] = {}
        events_by_frame = {event.frame_index: event for event in annotation.events}
        if annotation.last_frame >= episode_length:
            raise ValueError(f"{annotation.name}: frame bound exceeds episode {source_episode}")
        for frame_index in range(annotation.first_frame, annotation.last_frame + 1):
            frame = dataset[first_row + frame_index]
            estimates = poses_by_frame.get(frame_index, {})
            reference = _measured_pose(
                estimates, annotation.reference_role, frame_index, annotation.allow_predicted_demo_pose
            )
            held = (
                _measured_pose(
                    estimates, annotation.held_role, frame_index, annotation.allow_predicted_demo_pose
                )
                if annotation.held_role
                else None
            )
            for estimate in (reference, held):
                if estimate is None:
                    continue
                prior = versions.setdefault(estimate.role, estimate.reference_version)
                if prior != estimate.reference_version:
                    raise ValueError(f"frame {frame_index}: {estimate.role} object frame version changed")
            state = np.asarray(frame["observation.state"], dtype=float).reshape(-1)
            action = np.asarray(frame["action"], dtype=float).reshape(-1)
            joints = state[joint_indices]
            event = events_by_frame.get(frame_index)
            samples.append(
                SpatialSample(
                    time_s=float(frame["timestamp"]),
                    demo_ee=transform(kinematics.forward_kinematics(joints)),
                    demo_reference=reference.pose,
                    demo_held=held.pose if held is not None else None,
                    gripper=float(action[gripper_index]),
                    source_frame_index=frame_index,
                    reference_evidence=reference.evidence.value,
                    held_evidence=held.evidence.value if held is not None else None,
                    event_id=event.event_id if event is not None else None,
                    requires_milestone=event.requires_milestone if event is not None else None,
                )
            )
        speed = annotation.speed_m_s
        if annotation.mode == MotionMode.CONNECTION and speed is None:
            length = sum(
                float(np.linalg.norm(b.demo_ee[:3, 3] - a.demo_ee[:3, 3]))
                for a, b in zip(samples, samples[1:], strict=False)
            )
            speed = length / (samples[-1].time_s - samples[0].time_s)
            if speed <= 0:
                raise ValueError(f"{annotation.name}: cannot infer speed from a stationary demo interval")
        intervals.append(
            Interval(
                name=annotation.name,
                mode=annotation.mode,
                relation=annotation.relation,
                reference_role=annotation.reference_role,
                samples=tuple(samples),
                speed_m_s=speed,
                entry_milestone=annotation.entry_milestone,
                completion_milestone=annotation.completion_milestone,
                recovery_intervals=annotation.recovery_intervals,
                reference_version=versions[annotation.reference_role],
                held_reference_version=versions.get(annotation.held_role) if annotation.held_role else None,
                held_role=annotation.held_role,
            )
        )
    compiled_milestones = list(milestones)
    for annotation in pose_relation_annotations:
        if annotation.frame_index >= episode_length:
            raise ValueError(f"{annotation.name}: milestone frame exceeds episode {source_episode}")
        frame_poses = poses_by_frame.get(annotation.frame_index, {})
        reference = _measured_pose(
            frame_poses,
            annotation.reference_role,
            annotation.frame_index,
            annotation.allow_predicted_demo_pose,
        )
        moving = _measured_pose(
            frame_poses,
            annotation.moving_role,
            annotation.frame_index,
            annotation.allow_predicted_demo_pose,
            require_rotation=annotation.rotation_tolerance_rad is not None,
        )
        compiled_milestones.append(
            MilestoneSpec(
                annotation.name,
                (annotation.moving_role, annotation.reference_role),
                PoseRelationCriterion(
                    annotation.moving_role,
                    annotation.reference_role,
                    inverse(reference.pose) @ moving.pose,
                    moving.reference_version,
                    reference.reference_version,
                    annotation.translation_tolerance_m,
                    annotation.rotation_tolerance_rad,
                    annotation.max_pose_age_s,
                ),
            )
        )
    return FlashSkill(
        source_repo_id,
        source_episode,
        tuple(intervals),
        milestones=tuple(compiled_milestones),
        attachment_transitions=attachment_transitions,
    )
