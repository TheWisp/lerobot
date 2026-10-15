"""One replaceable command ahead of the motor loop, recomposed from live poses."""

from __future__ import annotations

import asyncio
import functools
import time

import numpy as np

from lerobot.showservo import motion as core, reactive

from . import runtime


async def stream(
    plan,
    kin,
    motions,
    correction,
    interrupted,
    observed,
    reselect,
    limits,
    speed,
    settled=None,
    revision=None,
    best_effort_stopped=None,
):
    """Return (reason, last command). Phase advances only after its command is consumed.

    Planning/IK never owns the motor bus. A stale solve cannot replace a command
    already consumed, and the existing servo lag/release guards keep their state.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from .. import jog

    gi = MOTOR_NAMES.index("gripper")
    path = plan["live_path"]
    qs, phases, due, goals = [plan["q"][0].copy()], [0], [0.0], [None]
    compositions = [None]

    def as_dict(q):
        return dict(zip(MOTOR_NAMES, map(float, q), strict=True))

    jog.play_joints([as_dict(qs[0])], [0.0])
    recorded, last_phase, progressing = -1, 0, time.monotonic()
    loop = asyncio.get_running_loop()
    settling = []
    orientation_anchor = None
    orientation_stage = None
    arm = np.arange(len(qs[0])) != gi
    best_phase, best_seen, best_progress_at = None, None, time.monotonic()

    def stop_best_effort(reason):
        try:
            held = jog.playback_stop_at_observed()
        except RuntimeError as error:
            return str(error), qs[max(0, recorded)]
        if best_effort_stopped is not None:
            best_effort_stopped(plan["stage"][-1], reason, held)
        return "", np.array([held[name] for name in MOTOR_NAMES], dtype=float)

    while True:
        await asyncio.sleep(min(runtime.ACT_TICK_S, 1.0 / jog.HZ))
        why = interrupted()
        if why:
            jog.playback_stop()
            return why, qs[max(0, recorded)]
        done, stamps = jog.playback_state()
        for k in range(recorded + 1, done + 1):
            observed(phases[k], qs[k], stamps[k], goals[k], compositions[k])
        recorded = done
        if done < 0:
            continue
        phase = phases[done]
        if phase != last_phase:
            last_phase, progressing = phase, time.monotonic()
        if time.monotonic() - progressing > runtime.ACT_STEP_TIMEOUT_S:
            jog.playback_stop()
            return (
                f"{plan['stage'][min(phase + 1, len(plan['q']) - 1)]}: live target did not converge within the step timeout",
                qs[done],
            )
        i = min(phase + 1, len(plan["q"]) - 1)
        motion, fix = motions(), correction()
        goal = reactive.targets({k: v[i : i + 1] for k, v in path.items()}, motion, fix)[0]
        constraint = path.get("constraint", ["pose"] * len(plan["q"]))[i]
        free_terminal = (
            constraint == "best_effort"
            and i == len(plan["q"]) - 1
            and abs(float(path["grips"][i] - path["grips"][phase])) < core.GRIP_STILL_UNITS
        )
        if free_terminal:
            cur = jog.current_tip_and_anchor()
            if cur is None:
                return "the arm went away", qs[done]
            actual = np.array([cur[2][m] for m in MOTOR_NAMES], dtype=float)
            now = time.monotonic()
            if best_phase != i or best_seen is None:
                best_phase, best_seen, best_progress_at = i, actual, now
            elif np.max(np.abs(actual[arm] - best_seen[arm])) >= runtime.ACT_STILL_DEG:
                best_seen, best_progress_at = actual, now
            lag = float(np.max(np.abs(actual[arm] - qs[done][arm])))
            remaining_m, remaining_deg = core.pose_residual(kin.forward_kinematics(actual), goal)
            still_aiming = remaining_m > core.ACT_REACH_TOL_M or remaining_deg > core.ACT_REACH_TOL_DEG
            if (
                done > 0
                and (lag > jog.PLAYBACK_LAG_DEG or still_aiming)
                and now - best_progress_at >= 0.8 * jog.PLAYBACK_HOLD_MAX_S
            ):
                return stop_best_effort("no observed joint progress")
        else:
            best_phase, best_seen = None, None
        if constraint == "orientation":
            # A rotation/carry frame constrains attitude, not the recorded XY/Z.
            # Capture position once on entry; taking each new command as the
            # anchor would turn small IK errors into translational drift.
            if orientation_stage != plan["stage"][i]:
                orientation_stage = plan["stage"][i]
                orientation_anchor = kin.forward_kinematics(qs[done])[:3, 3].copy()
            goal[:3, 3] = orientation_anchor
        else:
            orientation_stage = orientation_anchor = None
        local = path["local"][i]
        if constraint == "orientation":
            attachment = fix if path["held"][i] else np.eye(4)
            local = np.linalg.inv(motion[path["frames"][i]]) @ goal @ np.linalg.inv(attachment)
        # Snapshot before the asynchronous solve. A new observation may arrive
        # while IK runs or while this command waits for the bus; logging current
        # state at consumption would falsely attribute it to the newer estimate.
        composition = {
            "at": time.time(),
            "phase": i,
            "frame": path["frames"][i],
            "motion": np.asarray(motion[path["frames"][i]]).tolist(),
            "local": np.asarray(local).tolist(),
            "nominal_local": np.asarray(path["local"][i]).tolist(),
            "constraint": constraint,
            "held": bool(path["held"][i]),
            "fix": np.asarray(fix).tolist(),
            "hold_revision": None if revision is None else revision(),
        }
        if phase == len(plan["q"]) - 1:
            error_m, error_deg = core.pose_residual(kin.forward_kinematics(qs[done]), goal)
            previous_goal = goals[done] if goals[done] is not None else kin.forward_kinematics(qs[done])
            changed_m, changed_deg = core.pose_residual(previous_goal, goal)
            if (
                error_m <= core.ACT_REACH_TOL_M
                and error_deg <= core.ACT_REACH_TOL_DEG
                and changed_m <= core.ACT_SOLVE_TOL_M
                and changed_deg <= core.ACT_SOLVE_TOL_DEG
            ):
                cur = jog.current_tip_and_anchor()
                if cur is None:
                    return "the arm went away", qs[done]
                actual = np.array([cur[2][m] for m in MOTOR_NAMES])[arm]
                lag = float(np.max(np.abs(actual - qs[done][arm])))
                now = time.monotonic()
                settling = [(t, q) for t, q in settling if now - t <= runtime.ACT_STILL_S] + [(now, actual)]
                still = (
                    now - settling[0][0] >= 0.8 * runtime.ACT_STILL_S
                    and np.ptp([q for _, q in settling], axis=0).max() < runtime.ACT_STILL_DEG
                )
                if lag <= runtime.ACT_SETTLE_DEG or (still and lag <= runtime.ACT_STALL_DEG):
                    if settled is not None and lag > runtime.ACT_SETTLE_DEG:
                        settled(lag)
                    return "", qs[done]
                if still:
                    return f"the end: the arm stopped {lag:.0f} deg short of it", qs[done]
                continue
            settling.clear()
        dt = max(float(plan["times"][i] - plan["times"][phase]), 1.0 / jog.HZ)
        result = await loop.run_in_executor(
            runtime._ACT_EXECUTOR,
            functools.partial(
                reactive.checked_step,
                kin,
                qs[done],
                goal,
                float(path["grips"][i]),
                path["hints"][i],
                gi,
                jog.servo_ranges(),
                jog.workspace_box(),
                float(path["floor"][i]),
                dt,
                (limits[0] * speed, limits[1] * speed),
                jog.GRIP_UNITS_S * speed,
                preferred=plan["q"][i],
            ),
        )
        if result["reason"]:
            if free_terminal:
                return stop_best_effort(result["reason"])
            # Remove a pending command before an expensive alternative search.
            # It may have been computed against an older accepted observation.
            if not jog.update_playback_next(done, as_dict(qs[done]), due[done] + dt):
                continue
            slot = done + 1
            entry = (qs[done].copy(), phase, due[done] + dt, None, None)
            if slot == len(qs):
                for array, value in zip((qs, phases, due, goals, compositions), entry, strict=True):
                    array.append(value)
            else:
                qs[slot], phases[slot], due[slot], goals[slot], compositions[slot] = entry
            changed = await reselect(plan, i, motion, fix, qs[done])
            if not changed:
                jog.playback_stop()
                return result["reason"] + "; no checked permitted alternative", qs[done]
            continue
        # Gripper events follow the completed phase, never an intermediate correction.
        next_phase = i if result["complete"] else phase
        next_due = due[done] + dt
        if jog.update_playback_next(done, as_dict(result["q"]), next_due):
            slot = done + 1
            entry = (result["q"], next_phase, next_due, goal, composition)
            if slot == len(qs):
                for array, value in zip((qs, phases, due, goals, compositions), entry, strict=True):
                    array.append(value)
            else:
                qs[slot], phases[slot], due[slot], goals[slot], compositions[slot] = entry
