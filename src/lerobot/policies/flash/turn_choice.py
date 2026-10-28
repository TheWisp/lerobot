"""Select one permitted reference-frame turn off the fixed-rate motor thread."""

from __future__ import annotations

from math import inf

import numpy as np
from scipy.optimize import minimize_scalar

from .geometry import transform
from .planner import InfeasibleTrajectoryError, JointPlanner
from .skill import Interval, MotionMode
from .trajectory import spatial_goal


def select_turn_angle(
    planner: JointPlanner,
    interval: Interval,
    seed_deg: np.ndarray,
    live_reference: np.ndarray,
    live_attachment: np.ndarray | None,
    period_s: float,
) -> float:
    """Prefer the reachable goal requiring the shortest feasible joint motion.

    Generated connections are checked along the same interpolated path and
    timing constraints as runtime. Demonstrated intervals still use endpoint
    IK here; the ordinary planner checks every sample before publishing.
    Collision freedom is not inferred from these kinematic checks.
    """
    rule = interval.turn_at_point
    if rule is None:
        return 0.0
    if rule.source_evidence not in ("observed", "symmetry_invariant"):
        raise InfeasibleTrajectoryError(
            f"{interval.name}: turn pivot has {rule.source_evidence} object-frame evidence; "
            "choose a frame with a directly observed pose before planning this adaptation"
        )
    seed = np.asarray(seed_deg, dtype=float)
    cache: dict[float, float] = {}

    def cost(angle: float) -> float:
        angle = float(angle)
        if angle in cache:
            return cache[angle]
        if len(cache) >= planner.limits.max_turn_ik_evaluations:
            return inf
        try:
            if interval.mode == MotionMode.CONNECTION:
                score = planner.probe_connection(
                    interval, seed, live_reference, live_attachment, period_s, angle
                )
            else:
                goal = spatial_goal(interval, interval.samples[-1], live_reference, live_attachment, angle)
                solved = np.asarray(planner.kinematics.inverse_kinematics(seed, goal), dtype=float)
                if solved.shape != seed.shape or not np.isfinite(solved).all():
                    raise ValueError("invalid IK result")
                reached = transform(planner.kinematics.forward_kinematics(solved))
                translation = np.linalg.norm(reached[:3, 3] - goal[:3, 3])
                rotation = np.arccos(np.clip((np.trace(reached[:3, :3].T @ goal[:3, :3]) - 1) / 2, -1, 1))
                if (
                    translation > planner.limits.position_error_m
                    or rotation > planner.limits.rotation_error_rad
                ):
                    raise ValueError("IK goal error")
                joint_speeds = np.asarray(planner.limits.joint_speed_deg_s, dtype=float)
                score = float(np.max(np.abs(solved - seed) / joint_speeds))
        except (ValueError, OverflowError, InfeasibleTrajectoryError):
            score = inf
        cache[angle] = score
        return score

    # Always include the demonstrated choice and both annotated limits. The
    # bounded minimizer then searches between them without a full angular grid.
    for angle in (0.0, rule.min_angle_rad, rule.max_angle_rad):
        cost(angle)
    remaining = planner.limits.max_turn_ik_evaluations - len(cache)
    if remaining > 0:
        result = minimize_scalar(
            cost,
            bounds=(rule.min_angle_rad, rule.max_angle_rad),
            method="bounded",
            options={"maxiter": remaining},
        )
        if np.isfinite(result.x) and len(cache) < planner.limits.max_turn_ik_evaluations:
            cost(float(result.x))
    feasible = [(score, abs(angle), angle) for angle, score in cache.items() if np.isfinite(score)]
    if not feasible:
        raise InfeasibleTrajectoryError(
            f"{interval.name}: no IK-supported turn in the annotated range; "
            "kinematic feasibility does not certify collision clearance"
        )
    return min(feasible)[2]
