"""A short, replaceable horizon of checked commands ahead of the motor loop."""

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
    diagnostics=None,
):
    """Return (reason, last command). Phase advances only after its command is consumed.

    Planning/IK never owns the motor bus. A short checked horizon keeps the bus
    moving while a solve or observation is late. New poses replace only unsent
    samples; the existing servo lag/release guards keep their state.
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
    horizon = int(jog.HZ)  # One motor-second ahead; every unsent sample stays replaceable.
    if diagnostics is not None:
        diagnostics.update(solve_ms=[], tail_replacements=0, starved_polls=0, max_queue_samples=0)

    def changed_pending(done, motion, fix):
        for composition in compositions[done + 1 :]:
            if composition is None:
                continue
            old_motion = np.asarray(composition["motion"])
            distance, angle = core.pose_residual(old_motion, motion[composition["frame"]])
            if distance > core.ACT_SOLVE_TOL_M or angle > core.ACT_SOLVE_TOL_DEG:
                return True
            if composition["held"]:
                distance, angle = core.pose_residual(np.asarray(composition["fix"]), fix)
                if distance > core.ACT_SOLVE_TOL_M or angle > core.ACT_SOLVE_TOL_DEG:
                    return True
        return False

    def stop_best_effort(reason):
        try:
            held = jog.playback_stop_at_observed()
        except RuntimeError as error:
            return str(error), qs[max(0, recorded)]
        if best_effort_stopped is not None:
            best_effort_stopped(plan["stage"][-1], reason, held)
        return "", np.array([held[name] for name in MOTOR_NAMES], dtype=float)

    while True:
        # Poll faster than a motor tick so a consumed prefix can be replenished
        # before the next tick, without coupling planning to camera cadence.
        await asyncio.sleep(min(runtime.ACT_TICK_S, 0.25 / jog.HZ))
        why = interrupted()
        if why:
            jog.playback_stop()
            return why, qs[max(0, recorded)]
        done, stamps = jog.playback_state()
        if diagnostics is not None:
            depth = len(qs) - done - 1
            diagnostics["max_queue_samples"] = max(diagnostics["max_queue_samples"], depth)
            if depth == 0 and done >= 0 and phases[done] < len(plan["q"]) - 1:
                diagnostics["starved_polls"] += 1
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
        motion, fix = motions(), correction()
        replacing = changed_pending(done, motion, fix)
        # Keep the old checked tail moving while the replacement is solved.
        # The swap below is atomic and only succeeds if this head has not
        # advanced; a changed head is replanned on the next producer turn.
        if replacing and orientation_stage is not None and plan["stage"][phases[done]] != orientation_stage:
            orientation_stage = orientation_anchor = None
        pending = 0 if replacing else len(qs) - done - 1
        i = min(phase + 1, len(plan["q"]) - 1)
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
        if phase == len(plan["q"]) - 1 and pending == 0 and not replacing:
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
        if pending and phases[-1] == len(plan["q"]) - 1:
            continue
        base = done if replacing else len(qs) - 1
        phase = phases[base]
        i = min(phase + 1, len(plan["q"]) - 1)
        if (
            pending
            and path["held"][i]
            and (
                abs(float(path["grips"][i] - qs[base][gi])) > 1e-6
                or any(abs(float(qs[k][gi] - qs[k - 1][gi])) > 1e-6 for k in range(done + 1, len(qs)))
            )
        ):
            # A held-object release must remain at the queue head so a new
            # object/attachment observation can retarget the final approach
            # before opening. An unheld grasp close may be queued: its samples
            # are still consumed in order, and the grasp is verified afterward.
            continue
        if base != done:
            goal = reactive.targets({k: v[i : i + 1] for k, v in path.items()}, motion, fix)[0]
            constraint = path.get("constraint", ["pose"] * len(plan["q"]))[i]
            free_terminal = (
                constraint == "best_effort"
                and i == len(plan["q"]) - 1
                and abs(float(path["grips"][i] - path["grips"][phase])) < core.GRIP_STILL_UNITS
            )
            if constraint == "orientation":
                if orientation_stage != plan["stage"][i]:
                    orientation_stage = plan["stage"][i]
                    orientation_anchor = kin.forward_kinematics(qs[base])[:3, 3].copy()
                goal[:3, 3] = orientation_anchor
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
        # One IK solve covers at least two motor ticks. The motor receives
        # interpolated, FK-checked samples; it does not wait for one solve per
        # tick when the solver is slower than 30 Hz.
        nominal_ticks = max(2, min(6, int(np.ceil((plan["times"][i] - plan["times"][phase]) * jog.HZ))))
        if pending + nominal_ticks > horizon:
            continue
        box, floor = jog.workspace_box(), float(path["floor"][i])
        solve_args = (
            kin,
            qs[base],
            goal,
            float(path["grips"][i]),
            path["hints"][i],
            gi,
            jog.servo_ranges(),
            box,
            floor,
        )

        async def solve(
            step_ticks,
            args=solve_args,
            move_limits=(limits[0] * speed, limits[1] * speed),
            grip_rate=jog.GRIP_UNITS_S * speed,
            preferred=plan["q"][i],
        ):
            return await loop.run_in_executor(
                runtime._ACT_EXECUTOR,
                functools.partial(
                    reactive.checked_step,
                    *args,
                    step_ticks / jog.HZ,
                    move_limits,
                    grip_rate,
                    preferred=preferred,
                ),
            )

        # Cartesian speed alone does not bound joint speed near a changing IK
        # configuration. Give a checked step enough motor ticks to respect the
        # same joint slew used by ready/park, scaled by the requested act speed.
        joint_tick = min(jog.PLAYBACK_LAG_DEG, jog.RAMP_DEG_S * speed / jog.HZ)
        step_ticks = nominal_ticks
        ticks = nominal_ticks
        room_needed = False
        samples = None
        for _ in range(4):
            solve_started = time.perf_counter()
            result = await solve(step_ticks)
            if diagnostics is not None:
                diagnostics["solve_ms"].append((time.perf_counter() - solve_started) * 1000.0)
            if result["reason"]:
                break
            arm_delta = np.delete(np.asarray(result["q"]) - qs[base], gi)
            ticks = max(nominal_ticks, int(np.ceil(np.max(np.abs(arm_delta)) / joint_tick)))
            if pending + ticks > horizon:
                room_needed = True
                break
            samples = reactive.checked_joint_samples(
                kin,
                qs[base],
                result,
                ticks,
                gi,
                box,
                floor,
                joint_tick,
                previous_sample=qs[max(0, base - 1)],
                max_joint_accel_deg=jog.RAMP_DEG_S * speed / jog.HZ,
            )
            if samples is not None:
                break
            # A joint chord that misses the Cartesian corridor needs a
            # smaller Cartesian target, never an unchecked single-tick jump.
            step_ticks /= 2
        if room_needed:
            continue
        if not result["reason"] and samples is None:
            result = {**result, "reason": "updated pose has no checked smooth joint path"}
        dt = ticks / jog.HZ
        if result["reason"]:
            # The earlier checked prefix may still be moving. Its last sample
            # is the correct start for a reach search or best-effort stop;
            # searching from the current read would skip those phases.
            if base != done:
                continue
            if free_terminal:
                return stop_best_effort(result["reason"])
            # An infeasible future step cannot leave a stale queue moving while
            # the alternative search runs. Hold the last consumed command.
            if not jog.replace_playback_tail(done, [as_dict(qs[done])], [due[done] + dt]):
                continue
            slot = done + 1
            entry = (qs[done].copy(), phases[done], due[done] + dt, None, None)
            for array, value in zip((qs, phases, due, goals, compositions), entry, strict=True):
                del array[slot:]
                array.append(value)
            changed = await reselect(plan, i, motion, fix, qs[done])
            if not changed:
                jog.playback_stop()
                return result["reason"] + "; no checked permitted alternative", qs[done]
            continue
        # Gripper events follow the completed phase, never an intermediate correction.
        next_phase = i if result["complete"] else phase
        due_samples = [due[base] + (k + 1) / jog.HZ for k in range(ticks)]
        commands = [as_dict(q) for q in samples]
        accepted = (
            jog.replace_playback_tail(done, commands, due_samples)
            if replacing
            else jog.append_playback_chunk(base, commands, due_samples)
        )
        if accepted:
            if replacing:
                if diagnostics is not None:
                    diagnostics["tail_replacements"] += 1
                del qs[done + 1 :]
                del phases[done + 1 :]
                del due[done + 1 :]
                del goals[done + 1 :]
                del compositions[done + 1 :]
            for k, (q, at) in enumerate(zip(samples, due_samples, strict=True)):
                entry = (q, next_phase if k == ticks - 1 else phase, at, goal, composition)
                for array, value in zip((qs, phases, due, goals, compositions), entry, strict=True):
                    array.append(value)
