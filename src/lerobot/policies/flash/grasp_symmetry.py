"""Choose one declared physical symmetry branch for an entire grasp prefix.

The scene estimator keeps its canonical object frame. Only pregrasp goals use
the equivalent pose; after the attach event, the measured canonical
gripper-to-object relation handles whichever branch was actually grasped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import ceil, pi

import numpy as np

from .geometry import inverse, transform
from .planner import InfeasibleTrajectoryError, JointPlanner
from .skill import AttachmentAction, EndpointRequirement, FlashSkill, MotionMode, Relation
from .symmetry import AxialSymmetry
from .trajectory import AttachmentState, ConnectionTrajectory, Evidence, MissingEvidenceError, PoseEstimate


@dataclass(frozen=True)
class GraspSymmetryChoice:
    role: str
    equivalent_transform: np.ndarray
    angle_rad: float
    exchange_ends: bool


def _pregrasp_prefix(skill: FlashSkill, role: str):
    attach_events = {
        item.event_id
        for item in skill.attachment_transitions
        if item.action == AttachmentAction.ATTACH and item.role == role
    }
    prefix = []
    for interval in skill.intervals:
        if interval.relation != Relation.EE_TO_OBJECT or interval.reference_role != role:
            break
        prefix.append(interval)
        if any(sample.event_id in attach_events for sample in interval.samples):
            break
    return tuple(prefix)


def _candidate_angles(rotation_tolerance_rad: float):
    """Cover continuous symmetry to within the planner's angular tolerance."""
    step = 2 * rotation_tolerance_rad
    yield 0.0
    for index in range(1, ceil(pi / step) + 1):
        angle = min(index * step, pi)
        yield -angle
        if angle < pi:
            yield angle


def _preview_intervals(skill: FlashSkill, role: str):
    """Check the grasp and its first required carry, not only the approach."""
    prefix = _pregrasp_prefix(skill, role)
    if not prefix:
        return ()
    attach_ids = {
        item.event_id
        for item in skill.attachment_transitions
        if item.action == AttachmentAction.ATTACH and item.role == role
    }
    if not any(sample.event_id in attach_ids for interval in prefix for sample in interval.samples):
        return prefix
    preview = list(prefix)
    for interval in skill.intervals[len(prefix) :]:
        preview.append(interval)
        if (
            interval.relation == Relation.HELD_TO_TARGET
            and interval.held_role == role
            and interval.endpoint_requirement == EndpointRequirement.REQUIRED
        ):
            break
    return tuple(preview)


def _preview_candidate(
    intervals,
    skill: FlashSkill,
    symmetry: AxialSymmetry,
    live_poses: Mapping[str, PoseEstimate],
    equivalent: np.ndarray,
    planner: JointPlanner,
    seed_deg: np.ndarray,
    period_s: float,
) -> bool:
    """Exercise the runtime planner with the candidate and predicted attachment.

    This establishes bounded IK continuity only. No geometric collision model
    is available, so it cannot certify that the free-space path misses props.
    """
    canonical = live_poses[symmetry.role].pose
    if canonical is None:
        return False
    joints = np.asarray(seed_deg, dtype=float).copy()
    attachment = None
    # A static startup snapshot is valid throughout this *simulated* path.
    preview = JointPlanner(
        planner.kinematics,
        planner.arm_joint_names,
        planner.gripper_joint_name,
        replace(planner.limits, max_pose_age_s=1e6, max_attachment_age_s=None),
    )
    transitions = {item.event_id: item for item in skill.attachment_transitions}
    for interval in intervals:
        if interval.relation == Relation.EE_TO_OBJECT and interval.reference_role == symmetry.role:
            reference_pose = canonical @ equivalent
        elif interval.relation == Relation.EE_TO_START:
            reference_pose = transform(planner.kinematics.forward_kinematics(joints))
        else:
            observed = live_poses.get(interval.reference_role)
            if observed is None or observed.pose is None:
                return False
            reference_pose = observed.pose
        reference = PoseEstimate(
            interval.reference_role,
            reference_pose,
            0.0,
            0.0,
            Evidence.OBSERVED,
            interval.reference_version,
            (True,) * 3,
            (True,) * 3,
        )
        connection = (
            ConnectionTrajectory.begin(
                interval,
                planner.kinematics.forward_kinematics(joints),
                reference_pose,
                attachment.gripper_from_object
                if attachment is not None and interval.relation == Relation.HELD_TO_TARGET
                else None,
                0.0,
            )
            if interval.mode == MotionMode.CONNECTION
            else None
        )
        duration = connection.duration_s if connection is not None else interval.demo_duration_s
        try:
            # Retiming can lengthen a generated path after the initial count.
            # Re-run from the same start until the complete endpoint is included.
            for _ in range(planner.limits.max_connection_retimings + 1):
                count = max(2, ceil(duration / period_s) + 1)
                actions = preview.plan(
                    interval,
                    start_tick=0,
                    count=count,
                    epoch_s=0.0,
                    period_s=period_s,
                    interval_start_s=0.0,
                    seed_deg=joints,
                    reference=reference,
                    attachment=attachment,
                    connection=connection,
                )
                if actions and actions[-1].interval_complete:
                    break
                if connection is None or connection.duration_s <= duration:
                    return False
                duration = connection.duration_s
            else:
                return False
        except (InfeasibleTrajectoryError, MissingEvidenceError, ValueError):
            return False
        joints = np.array([actions[-1].action[f"{name}.pos"] for name in planner.arm_joint_names])
        for action in actions:
            transition = transitions.get(action.event_id)
            if (
                transition is not None
                and transition.action == AttachmentAction.ATTACH
                and transition.role == symmetry.role
            ):
                event_joints = np.array([action.action[f"{name}.pos"] for name in planner.arm_joint_names])
                attachment = AttachmentState(
                    inverse(planner.kinematics.forward_kinematics(event_joints)) @ canonical,
                    0.0,
                    0.0,
                    transition.reference_version,
                )
    return True


def select_grasp_symmetry(
    skill: FlashSkill,
    symmetry: AxialSymmetry,
    live_poses: Mapping[str, PoseEstimate],
    planner: JointPlanner,
    seed_deg: np.ndarray,
    period_s: float,
) -> GraspSymmetryChoice:
    """Pick a physical equivalent that reaches the grasp and first required carry."""
    intervals = _preview_intervals(skill, symmetry.role)
    if not intervals:
        raise ValueError(f"{symmetry.role}: skill has no leading object-relative grasp prefix")
    if symmetry.role not in live_poses or live_poses[symmetry.role].pose is None:
        raise ValueError(f"{symmetry.role}: no pose for grasp symmetry selection")
    for angle in _candidate_angles(planner.limits.rotation_error_rad):
        for exchange in (False, True) if symmetry.ends_interchangeable else (False,):
            equivalent = symmetry.equivalent_transform(angle, exchange_ends=exchange)
            if _preview_candidate(
                intervals, skill, symmetry, live_poses, equivalent, planner, seed_deg, period_s
            ):
                return GraspSymmetryChoice(symmetry.role, equivalent, angle, exchange)
    raise InfeasibleTrajectoryError(
        f"{symmetry.role}: no declared axial/end-exchange grasp alternative reaches the grasp "
        "and first required held-to-target goal with bounded continuous IK"
    )
