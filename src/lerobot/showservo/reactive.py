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
    continuous: bool = False,
    joint_ranges: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[np.ndarray] | None:
    """Interpolate a solved step at motor rate only when its whole joint path is checked.

    FK of each intermediate joint command must stay near the corresponding
    Cartesian interpolation and inside the same workspace/floor as the solve.
    Free moves use a cubic matching incoming velocity and braking at the real
    waypoint. Their path check permits eased progress along the Cartesian
    corridor, and rejects velocity/acceleration violations without a linear
    fallback. Recorded follow samples retain their timing: prefer the existing
    constant-acceleration walk, with checked linear interpolation as fallback.
    Returning None asks for a shorter checked Cartesian step.
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
    if joint_ranges is not None:
        joint_lo, joint_hi = (np.asarray(bound, dtype=float) for bound in joint_ranges)

    def checked(samples: list[np.ndarray], *, acceleration: np.ndarray | None = None) -> bool:
        steps = np.diff(np.stack([previous, *samples]), axis=0)
        if np.max(np.abs(steps[:, arm])) > max_joint_tick_deg:
            return False
        if joint_ranges is not None:
            commands = np.asarray(samples)[:, arm]
            if np.any(commands < joint_lo[arm] - 1e-6) or np.any(commands > joint_hi[arm] + 1e-6):
                return False
        if continuous and previous_sample is not None and max_joint_accel_deg is not None:
            incoming = previous - np.asarray(previous_sample, dtype=float)
            if (
                np.max(np.abs(np.diff(np.vstack([incoming, steps]), axis=0)[:, arm]))
                > max_joint_accel_deg + 1e-9
            ):
                return False
        if (
            acceleration is not None
            and max_joint_accel_deg is not None
            and np.max(np.abs(acceleration[arm])) > max_joint_accel_deg
        ):
            return False
        for k, q in enumerate(samples, start=1):
            u = k / ticks
            if continuous:
                chord = endpoint[arm] - previous[arm]
                u = float(
                    np.clip(np.dot(q[arm] - previous[arm], chord) / max(np.dot(chord, chord), 1e-12), 0, 1)
                )
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

    if continuous and previous_sample is not None and ticks > 1:
        # A free waypoint is a trajectory endpoint, not a row of independent
        # motor goals. Match incoming velocity, and brake at the real endpoint.
        # Interior chunks retain a through velocity; no linear fallback may
        # silently discard continuity when this curve fails its path checks.
        incoming = previous - np.asarray(previous_sample, dtype=float)
        outgoing = np.zeros_like(incoming) if result["complete"] else (endpoint - previous) / ticks
        samples = []
        for k in range(1, ticks + 1):
            u = k / ticks
            q = (
                (2 * u**3 - 3 * u**2 + 1) * previous
                + (u**3 - 2 * u**2 + u) * ticks * incoming
                + (-2 * u**3 + 3 * u**2) * endpoint
                + (u**3 - u**2) * ticks * outgoing
            )
            q[grip_index] = previous[grip_index] + u * (endpoint[grip_index] - previous[grip_index])
            samples.append(q)
        samples[-1] = endpoint.copy()
        return samples if checked(samples) else None

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


def checked_queue_blend(
    kin: Any,
    previous: np.ndarray,
    old: list[np.ndarray],
    new: list[np.ndarray],
    grip_index: int,
    box: tuple,
    floor: float,
    max_joint_tick_deg: float,
    blend_ticks: int,
    previous_sample: np.ndarray | None = None,
    max_joint_accel_deg: float | None = None,
) -> list[np.ndarray] | None:
    """Ease a replacement chunk away from its unsent predecessor, RTC-style.

    The first overlap sample mostly follows the old command; the last follows
    the new one. Never blend a gripper event. The entire proposed joint walk
    must still satisfy the joint-rate, Cartesian-corridor and workspace checks
    used for a newly solved chunk. A failed check leaves the caller's original
    checked replacement available, rather than sending an unchecked average.
    """
    count = min(blend_ticks, len(old), len(new))
    if count < 2:
        return None
    old = [np.asarray(q, dtype=float) for q in old[:count]]
    new = [np.asarray(q, dtype=float) for q in new]
    if any(abs(a[grip_index] - b[grip_index]) > 1e-6 for a, b in zip(old, new, strict=False)):
        return None
    result = [q.copy() for q in new]
    for k in range(count):
        weight_new = (k + 1) / count
        result[k] = (1.0 - weight_new) * old[k] + weight_new * new[k]
        result[k][grip_index] = new[k][grip_index]
    arm = np.arange(len(previous)) != grip_index
    steps = np.diff(np.stack([previous, *result]), axis=0)
    if np.max(np.abs(steps[:, arm])) > max_joint_tick_deg + 1e-9:
        return None
    if previous_sample is not None and max_joint_accel_deg is not None:
        velocities = np.diff(np.stack([previous_sample, previous, *result]), axis=0)
        if np.max(np.abs(np.diff(velocities, axis=0)[:, arm])) > max_joint_accel_deg + 1e-9:
            return None
    start = kin.forward_kinematics(previous)
    end = kin.forward_kinematics(new[-1])
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    bottom = min(lo[2], floor, start[2, 3])
    for k, q in enumerate(result, start=1):
        actual = kin.forward_kinematics(q)
        expected = core.interp_rigid(start, end, k / len(result))
        distance, angle = core.pose_residual(actual, expected)
        p = actual[:3, 3]
        if (
            distance > core.ACT_REACH_TOL_M
            or angle > core.ACT_REACH_TOL_DEG
            or p[2] < bottom - core.ACT_REACH_TOL_M
            or np.any(p[:2] < lo[:2] - core.ACT_REACH_TOL_M)
            or np.any(p > hi + core.ACT_REACH_TOL_M)
        ):
            return None
    return result


def unverified_object_shift(motion: np.ndarray, points: np.ndarray, tolerance_m: float) -> float | None:
    """Measured object-centre motion too large to carry through an unverified pregrasp.

    A rotation about an origin away from the object may have a large transform
    translation while the object itself stays still. Compare its material
    centre, and allow the placement residual on each of the two observations.
    """
    centre = np.asarray(points, dtype=float).reshape(-1, 3).mean(axis=0)
    motion = np.asarray(motion, dtype=float)
    shift = float(np.linalg.norm(motion[:3, :3] @ centre + motion[:3, 3] - centre))
    return shift if shift > 2 * tolerance_m else None


def nearby_pregrasp_shift(
    motion: np.ndarray, points: np.ndarray, fingertip: np.ndarray, tolerance_m: float
) -> float | None:
    """An object moved before grasp while the gripper was within its measured extent."""
    shift = unverified_object_shift(motion, points, tolerance_m)
    if shift is None:
        return None
    motion = np.asarray(motion, dtype=float)
    current = np.asarray(points, dtype=float).reshape(-1, 3) @ motion[:3, :3].T + motion[:3, 3]
    return shift if nearby_object_extent(current, fingertip, tolerance_m) else None


def nearby_object_extent(points: np.ndarray, fingertip: np.ndarray, tolerance_m: float) -> bool:
    """Whether the tip enters the observed object's bounding sphere.

    This deliberately conservative proximity check is used only to stop an
    unverified pregrasp; a successful check does not certify gripper clearance.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    centre = points.mean(axis=0)
    radius = float(np.linalg.norm(points - centre, axis=1).max())
    return bool(np.linalg.norm(np.asarray(fingertip) - centre) <= radius + tolerance_m)


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
