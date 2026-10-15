"""Recompose a checked path from live object and attachment observations.

No hardware ownership or object classes live here. Phase indices never rewind;
correction steps hold the phase and gripper event until the new target is reached.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import motion as core


def slice_plan(plan: dict, start: int = 0, stop: int | None = None) -> dict:
    out = {
        **plan,
        "q": plan["q"][start:stop],
        "times": plan["times"][start:stop] - plan["times"][start],
        "stage": plan["stage"][start:stop],
    }
    if "live_path" in plan:
        out["live_path"] = {k: v[start:stop] for k, v in plan["live_path"].items()}
    return out


def keep_released(plan: dict, grip_index: int, closing: float) -> dict:
    """Keep the just-released gripper open through the remainder of this interaction.

    A line after a release may set its gripper to the demo's later, more closed
    opening *before* withdrawing. That can regrasp the object while the arm is
    still in contact. A later interaction with an explicit new grasp gets its
    own plan and can close again. Additional opening remains permitted.
    """
    q = np.asarray(plan["q"]).copy()
    released = float(q[0, grip_index])
    if closing > 0:
        q[:, grip_index] = np.minimum(q[:, grip_index], released)
    else:
        q[:, grip_index] = np.maximum(q[:, grip_index], released)
    out = {**plan, "q": q}
    if "live_path" in plan:
        path = {**plan["live_path"], "grips": q[:, grip_index].copy()}
        out["live_path"] = path
    return out


def targets(
    path: dict, motions: dict[str, np.ndarray], fix: np.ndarray, start_pose: np.ndarray | None = None
) -> np.ndarray:
    out = []
    anchor = None if start_pose is None else np.asarray(start_pose)[:3, 3].copy()
    constraints = path.get("constraint", ["pose"] * len(path["frames"]))
    for name, local, held, constraint in zip(
        path["frames"], path["local"], path["held"], constraints, strict=True
    ):
        pose = motions[name] @ local @ (fix if held else np.eye(4))
        if constraint == "orientation" and anchor is not None:
            pose[:3, 3] = anchor
        else:
            anchor = pose[:3, 3].copy()
        out.append(pose)
    return np.stack(out)


def checked_step(
    kin: Any,
    previous: np.ndarray,
    goal: np.ndarray,
    grip: float,
    hint: np.ndarray,
    grip_index: int,
    ranges: tuple | None,
    box: tuple,
    floor: float,
    dt: float,
    limits: tuple[float, float],
    grip_speed: float,
    preferred: np.ndarray | None = None,
) -> dict:
    """One bounded reference step, with the same reach/floor/joint checks as planning.

    The opening/closing event advances only with its pose phase. A changed target
    first inserts correction steps with the current gripper command preserved.
    """
    current = kin.forward_kinematics(previous)
    dist, angle = core.pose_residual(current, goal)
    ratio = min(1.0, limits[0] * dt / max(dist, 1e-12), limits[1] * dt / max(np.radians(angle), 1e-12))
    target = core.interp_rigid(current, goal, ratio)
    complete = ratio >= 1.0 - 1e-9
    g = previous[grip_index]
    if complete:
        g += float(np.clip(grip - g, -grip_speed * dt, grip_speed * dt))
        complete = abs(g - grip) < 1e-6
    seed_hint = np.asarray(hint if complete else np.zeros_like(hint))
    if complete and preferred is not None:
        e_m, e_deg = core.pose_residual(kin.forward_kinematics(preferred), target)
        arm_delta = np.delete(np.asarray(preferred) - previous, grip_index)
        if (
            e_m <= core.ACT_SOLVE_TOL_M
            and e_deg <= core.ACT_SOLVE_TOL_DEG
            and np.max(np.abs(arm_delta)) <= core.ACT_MAX_JOINT_STEP_DEG
        ):
            # Keep the checked solution when it still satisfies the live target.
            # Re-solving from scratch needlessly adds another solver-tolerance error.
            seed_hint = np.asarray(preferred) - previous
    solved = core.solve_plan_joints(
        kin,
        target[None],
        np.array([g]),
        seed_hint[None],
        previous,
        grip_index,
        *(ranges or (None, None)),
    )
    q = solved["q"][0]
    arm = np.arange(len(q)) != grip_index
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    bottom = min(lo[2], floor, current[2, 3])
    p = target[:3, 3]
    reason = ""
    if solved["residual_m"][0] > core.ACT_REACH_TOL_M or solved["residual_deg"][0] > core.ACT_REACH_TOL_DEG:
        reason = "updated pose is outside this arm configuration's reach"
    elif np.max(np.abs(q[arm] - previous[arm])) > core.ACT_MAX_JOINT_STEP_DEG:
        reason = "updated pose needs a joint jump"
    elif (
        p[2] < bottom - core.ACT_REACH_TOL_M
        or np.any(p[:2] < lo[:2] - core.ACT_REACH_TOL_M)
        or np.any(p > hi + core.ACT_REACH_TOL_M)
    ):
        reason = "updated pose leaves the checked workspace or floor"
    elif ratio < 1.0 - 1e-9 and np.max(np.abs(q[arm] - previous[arm])) < 1e-6:
        reason = "updated pose makes no progress in this arm configuration"
    return {
        "q": q,
        "pose": target,
        "goal": goal,
        "complete": complete,
        "reason": reason,
        "residual_mm": float(solved["residual_m"][0] * 1000),
    }


def checked_joint_samples(
    kin: Any,
    previous: np.ndarray,
    result: dict,
    ticks: int,
    grip_index: int,
    box: tuple,
    floor: float,
    max_joint_tick_deg: float,
    previous_sample: np.ndarray | None = None,
    max_joint_accel_deg: float | None = None,
) -> list[np.ndarray] | None:
    """Interpolate a solved step at motor rate only when its whole joint path is checked.

    FK of each intermediate joint command must stay near the corresponding
    Cartesian interpolation and inside the same workspace/floor as the solve.
    When the last command is known, prefer a constant-acceleration joint walk
    from its incoming velocity. This avoids a velocity jump at each IK chunk
    boundary. If that walk leaves the checked Cartesian corridor or exceeds
    the joint-rate/acceleration budget, retain the original checked linear
    interpolation. Returning None asks for a shorter checked Cartesian step.
    """
    if ticks < 1 or result["reason"]:
        return None
    endpoint = np.asarray(result["q"], dtype=float)
    previous = np.asarray(previous, dtype=float)
    arm = np.arange(len(endpoint)) != grip_index
    start_pose = kin.forward_kinematics(previous)
    end_pose = np.asarray(result.get("pose", kin.forward_kinematics(endpoint)))
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    bottom = min(lo[2], floor, start_pose[2, 3])

    def checked(samples: list[np.ndarray], *, acceleration: np.ndarray | None = None) -> bool:
        steps = np.diff(np.stack([previous, *samples]), axis=0)
        if np.max(np.abs(steps[:, arm])) > max_joint_tick_deg:
            return False
        if (
            acceleration is not None
            and max_joint_accel_deg is not None
            and np.max(np.abs(acceleration[arm])) > max_joint_accel_deg
        ):
            return False
        for k, q in enumerate(samples, start=1):
            u = k / ticks
            actual = kin.forward_kinematics(q)
            expected = core.interp_rigid(start_pose, end_pose, u)
            distance, angle = core.pose_residual(actual, expected)
            p = actual[:3, 3]
            if (
                distance > core.ACT_REACH_TOL_M
                or angle > core.ACT_REACH_TOL_DEG
                or p[2] < bottom - core.ACT_REACH_TOL_M
                or np.any(p[:2] < lo[:2] - core.ACT_REACH_TOL_M)
                or np.any(p > hi + core.ACT_REACH_TOL_M)
            ):
                return False
        return True

    if previous_sample is not None and ticks > 1:
        incoming = previous - np.asarray(previous_sample, dtype=float)
        acceleration = 2 * (endpoint - previous - ticks * incoming) / (ticks * (ticks + 1))
        velocities = np.stack([incoming + k * acceleration for k in range(1, ticks + 1)])
        velocities[:, grip_index] = (endpoint[grip_index] - previous[grip_index]) / ticks
        smoothed = [q.copy() for q in previous + np.cumsum(velocities, axis=0)]
        smoothed[-1] = endpoint.copy()  # exact phase/gripper endpoint, despite roundoff
        if checked(smoothed, acceleration=acceleration):
            return smoothed

    linear = [previous + (k / ticks) * (endpoint - previous) for k in range(1, ticks + 1)]
    return linear if checked(linear) else None


def unverified_object_shift(motion: np.ndarray, centre: np.ndarray, tolerance_m: float) -> float | None:
    """Measured object-centre motion too large to carry through an unverified pregrasp.

    A rotation about an origin away from the object may have a large transform
    translation while the object itself stays still. Compare its material
    centre, and allow the placement residual on each of the two observations.
    """
    centre = np.asarray(centre, dtype=float)
    motion = np.asarray(motion, dtype=float)
    shift = float(np.linalg.norm(motion[:3, :3] @ centre + motion[:3, 3] - centre))
    return shift if shift > 2 * tolerance_m else None


def choose_turn(
    kin: Any,
    path: dict,
    motions: dict,
    fix: np.ndarray,
    obj: str,
    centre: np.ndarray,
    turns: list[float],
    q_demo: np.ndarray,
    previous: np.ndarray,
    grip_index: int,
    ranges: tuple | None,
    box: tuple,
    limits: tuple,
    hz: float,
    axis: np.ndarray | None = None,
) -> dict:
    """Search permitted alternatives for the REMAINING phase, including its entry.

    No four-candidate cutoff: every sampled-feasible candidate is eligible. Rank
    retains the demo-posture preference, not a claim of global optimality.
    """
    selected = np.unique(np.linspace(0, len(q_demo) - 1, min(24, len(q_demo)), dtype=int))

    def poses(turn):
        return targets(
            path,
            {**motions, obj: core.turn_about(centre, turn, axis) @ motions[obj]},
            fix,
            kin.forward_kinematics(previous),
        )

    ranked = core.rank_landings(
        kin, turns, lambda a: poses(a)[selected], q_demo[selected], *(ranges or (None, None)), grip_index
    )
    attempts = []
    start = kin.forward_kinematics(previous)
    for cost, turn in ranked:
        goals = poses(turn)
        distance, angle = core.pose_residual(start, goals[0])
        n = max(1, int(np.ceil(max(distance / limits[0], np.radians(angle) / limits[1]) * hz)))
        bridge = np.stack([core.interp_rigid(start, goals[0], k / n) for k in range(1, n + 1)])
        all_poses = np.concatenate([bridge, goals])
        grips = np.r_[np.full(n, previous[grip_index]), path["grips"]]
        hints = np.vstack([np.zeros((n, len(previous))), path["hints"]])
        sol = core.solve_plan_joints(
            kin, all_poses, grips, hints, previous, grip_index, *(ranges or (None, None))
        )
        low, high = np.asarray(box[0]), np.asarray(box[1])
        floors = np.minimum(np.r_[np.full(n, path["floor"][0]), path["floor"]], low[2])
        floors = np.minimum(floors, start[2, 3])
        pos = all_poses[:, :3, 3]
        ok = bool(
            np.all(sol["residual_m"] <= core.ACT_REACH_TOL_M)
            and np.all(sol["residual_deg"] <= core.ACT_REACH_TOL_DEG)
            and np.all(sol["step_deg"] <= core.ACT_MAX_JOINT_STEP_DEG)
            and np.all(pos[:, 2] >= floors - core.ACT_REACH_TOL_M)
            and np.all(pos[:, :2] >= low[:2] - core.ACT_REACH_TOL_M)
            and np.all(pos <= high + core.ACT_REACH_TOL_M)
        )
        attempts.append(
            {
                "turn_deg": turn,
                "cost": cost,
                "ok": ok,
                "worst_residual_mm": float(sol["residual_m"].max() * 1000),
            }
        )
        if ok:
            return {"turn_deg": turn, "reachable": len(ranked), "attempts": attempts}
    return {"turn_deg": None, "reachable": len(ranked), "attempts": attempts}
