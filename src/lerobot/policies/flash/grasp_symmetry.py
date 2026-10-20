"""Choose one declared physical symmetry branch for an entire grasp prefix.

The scene estimator keeps its canonical object frame. Only pregrasp goals use
the equivalent pose; after the attach event, the measured canonical
gripper-to-object relation handles whichever branch was actually grasped.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi

import numpy as np

from .geometry import interpolate, transform
from .planner import CartesianKinematics, InfeasibleTrajectoryError, PlannerLimits
from .skill import AttachmentAction, FlashSkill, MotionMode, Relation
from .symmetry import AxialSymmetry
from .trajectory import spatial_goal


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


def _feasible_prefix(
    prefix,
    live_pose: np.ndarray,
    symmetry: AxialSymmetry,
    kinematics: CartesianKinematics,
    seed_deg: np.ndarray,
    limits: PlannerLimits,
) -> bool:
    joints = seed_deg.copy()
    current_ee = transform(kinematics.forward_kinematics(joints))
    live_center_z = float((live_pose @ np.r_[symmetry.center_object_m, 1])[2])
    for interval in prefix:
        goals = [spatial_goal(interval, sample, live_pose, None) for sample in interval.samples]
        for sample, goal in zip(interval.samples, goals, strict=True):
            demo_center_z = float((sample.demo_reference @ np.r_[symmetry.center_object_m, 1])[2])
            demonstrated_clearance = sample.demo_ee[2, 3] - demo_center_z
            if goal[2, 3] - live_center_z < demonstrated_clearance - limits.position_error_m:
                return False
        if interval.mode == MotionMode.CONNECTION:
            end = goals[-1]
            distance = np.linalg.norm(end[:3, 3] - current_ee[:3, 3])
            angle = np.arccos(np.clip((np.trace(current_ee[:3, :3].T @ end[:3, :3]) - 1) / 2, -1, 1))
            count = max(2, ceil(distance / limits.position_error_m), ceil(angle / limits.rotation_error_rad))
            targets = [interpolate(current_ee, end, fraction) for fraction in np.linspace(0, 1, count + 1)]
        else:
            targets = goals
        prior_sample_joints = None
        for index, target in enumerate(targets):
            refinement_seed = joints
            for _ in range(limits.ik_refinement_attempts):
                solved = np.asarray(kinematics.inverse_kinematics(refinement_seed, target), dtype=float)
                if solved.shape != joints.shape or not np.isfinite(solved).all():
                    return False
                reached = transform(kinematics.forward_kinematics(solved))
                position_error = np.linalg.norm(reached[:3, 3] - target[:3, 3])
                rotation_error = np.arccos(
                    np.clip((np.trace(reached[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)
                )
                if position_error <= limits.position_error_m and rotation_error <= limits.rotation_error_rad:
                    break
                refinement_seed = solved
            if position_error > limits.position_error_m or rotation_error > limits.rotation_error_rad:
                return False
            if interval.mode == MotionMode.DEMONSTRATED and prior_sample_joints is not None:
                dt = interval.samples[index].time_s - interval.samples[index - 1].time_s
                if np.any(np.abs(solved - prior_sample_joints) / dt > limits.joint_speed_deg_s):
                    return False
            joints = solved
            if interval.mode == MotionMode.DEMONSTRATED:
                prior_sample_joints = solved
        current_ee = goals[-1]
    return True


def select_grasp_symmetry(
    skill: FlashSkill,
    symmetry: AxialSymmetry,
    live_pose: np.ndarray,
    kinematics: CartesianKinematics,
    seed_deg: np.ndarray,
    limits: PlannerLimits,
) -> GraspSymmetryChoice:
    """Find a continuous-IK alternative without lowering demo-relative clearance."""
    prefix = _pregrasp_prefix(skill, symmetry.role)
    if not prefix:
        raise ValueError(f"{symmetry.role}: skill has no leading object-relative grasp prefix")
    for angle in _candidate_angles(limits.rotation_error_rad):
        for exchange in (False, True) if symmetry.ends_interchangeable else (False,):
            equivalent = symmetry.equivalent_transform(angle, exchange_ends=exchange)
            if _feasible_prefix(
                prefix, transform(live_pose) @ equivalent, symmetry, kinematics, seed_deg, limits
            ):
                return GraspSymmetryChoice(symmetry.role, equivalent, angle, exchange)
    raise InfeasibleTrajectoryError(
        f"{symmetry.role}: no declared axial/end-exchange grasp alternative has a continuous IK path "
        "with demonstrated object-relative clearance"
    )
