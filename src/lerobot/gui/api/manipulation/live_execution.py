"""A short, replaceable horizon of checked commands ahead of the motor loop."""

from __future__ import annotations

import asyncio
import functools
import time
from collections import deque

import numpy as np

from lerobot.showservo import motion as core, reactive

from . import runtime

# In the measured insertion trace, checking all 30 buffered slots took roughly
# 0.2 s per retarget. Eight slots cover 267 ms of motor execution while their
# solves take about 40 ms; the normal producer replenishes the queue afterward.
RETARGET_HORIZON_S = 0.25


def _retarget_timed_tail(
    kin,
    plan,
    start,
    before_start,
    old_q,
    old_phases,
    old_due,
    old_goals,
    old_compositions,
    motions,
    fix,
    limits,
    speed,
    ranges,
    box,
    gi,
    revision,
    frozen_ticks,
):
    """Re-solve an unsent chunk in its existing motor slots.

    As with RTC's inference-delay prefix, retain the old checked commands that
    may execute during this job. Every later slot keeps its recorded phase and
    gripper state. If a changed target needs more time, use the ordinary
    incremental replan instead of silently dilating the recorded sequence.
    """
    from .. import jog

    path = plan["live_path"]
    if not old_q or frozen_ticks >= len(old_q):
        return None
    samples = [np.asarray(q, dtype=float).copy() for q in old_q[:frozen_ticks]]
    phases = list(old_phases[:frozen_ticks])
    goals = list(old_goals[:frozen_ticks])
    compositions = list(old_compositions[:frozen_ticks])
    previous = samples[-1] if samples else np.asarray(start, dtype=float).copy()
    previous_sample = samples[-2] if len(samples) > 1 else np.asarray(before_start, dtype=float).copy()
    joint_tick = min(jog.PLAYBACK_LAG_DEG, jog.RAMP_DEG_S * speed / jog.HZ)
    k = frozen_ticks
    while k < len(old_q):
        original = old_compositions[k]
        if original is None or original["constraint"] != "pose":
            return None
        i = int(original["phase"])
        end = k + 1
        while end < len(old_q) and old_compositions[end] is not None and old_compositions[end]["phase"] == i:
            end += 1
        ticks = end - k
        if any(abs(float(q[gi] - previous[gi])) > 1e-6 for q in old_q[k:end]):
            # The release/grasp event cannot be moved by a queue splice.
            return None
        frame = original["frame"]
        local = np.asarray(original["local"], dtype=float)
        attachment = fix if original["held"] else np.eye(4)
        goal = np.asarray(motions[frame]) @ local @ attachment
        result = reactive.checked_step(
            kin,
            previous,
            goal,
            float(old_q[end - 1][gi]),
            path["hints"][i],
            gi,
            ranges,
            box,
            float(path["floor"][i]),
            ticks / jog.HZ,
            (limits[0] * speed, limits[1] * speed),
            jog.GRIP_UNITS_S * speed,
            preferred=plan["q"][i],
        )
        if result["reason"] or result["complete"] != (old_phases[end - 1] == i):
            return None
        checked = reactive.checked_joint_samples(
            kin,
            previous,
            result,
            ticks,
            gi,
            box,
            float(path["floor"][i]),
            joint_tick,
            previous_sample=previous_sample,
            max_joint_accel_deg=jog.RAMP_DEG_S * speed / jog.HZ,
            continuous=path.get("move", ["follow"] * len(plan["q"]))[i] in {"line", "lift"},
            joint_ranges=ranges,
        )
        if checked is None:
            return None
        if any(abs(float(q[gi] - old_q[k + j][gi])) > 1e-6 for j, q in enumerate(checked)):
            return None
        new_composition = {
            **original,
            "at": time.time(),
            "motion": np.asarray(motions[frame]).tolist(),
            "fix": np.asarray(fix).tolist(),
            "hold_revision": revision,
        }
        samples.extend(checked)
        phases.extend(old_phases[k:end])
        goals.extend([goal] * ticks)
        compositions.extend([new_composition] * ticks)
        previous_sample, previous = checked[-2] if len(checked) > 1 else previous, checked[-1]
        k = end
    return samples, phases, list(old_due), goals, compositions


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
    queue_horizon_s=1.0,
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
    constraints = path.get("constraint", ["pose"] * len(plan["q"]))
    qs, phases, due, goals = [plan["q"][0].copy()], [0], [0.0], [None]
    compositions = [None]

    def as_dict(q):
        return dict(zip(MOTOR_NAMES, map(float, q), strict=True))

    def next_target_phase(phase, current):
        i = min(phase + 1, len(plan["q"]) - 1)
        moves = path.get("move")
        if moves is None or moves[i] not in {"line", "lift"}:
            return i
        # Generated samples along a free line are validation points, not
        # independent stop targets. Preserve real stage/gripper boundaries.
        if abs(float(path["grips"][i] - current[gi])) > 1e-6:
            return i
        while i + 1 < len(plan["q"]):
            k = i + 1
            if (
                plan["stage"][k] != plan["stage"][i]
                or moves[k] != moves[i]
                or path["frames"][k] != path["frames"][i]
                or path["held"][k] != path["held"][i]
                or abs(float(path["grips"][k] - current[gi])) > 1e-6
                or constraints[k] != constraints[i]
            ):
                break
            i = k
        return i

    jog.play_joints([as_dict(qs[0])], [0.0])
    recorded, last_phase, progressing = -1, 0, time.monotonic()
    loop = asyncio.get_running_loop()
    settling = []
    orientation_anchor = None
    orientation_stage = None
    arm = np.arange(len(qs[0])) != gi
    best_phase, best_seen, best_progress_at = None, None, time.monotonic()
    horizon = max(2, int(round(queue_horizon_s * jog.HZ)))
    unchanged_snapshot = None
    solve_seconds = deque(maxlen=10)
    if diagnostics is not None:
        diagnostics.update(
            solve_ms=[],
            tail_replacements=0,
            starved_polls=0,
            max_queue_samples=0,
            stale_replacements=0,
            chunk_replacements=0,
            chunk_trimmed_samples=0,
            chunk_rejected=0,
            chunk_solve_ms=[],
            retarget_horizon_s=RETARGET_HORIZON_S,
            queue_horizon_s=queue_horizon_s,
        )

    def changed_pending(done, motion, fix):
        nonlocal unchanged_snapshot
        snapshot = (tuple((name, value.tobytes()) for name, value in motion.items()), fix.tobytes())
        # The producer polls faster than perception. Once the pending queue
        # agrees, consuming it or appending samples using the same snapshot
        # cannot create a disagreement. Avoid rescanning it every 8 ms.
        if snapshot == unchanged_snapshot:
            return False
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
        unchanged_snapshot = snapshot
        return False

    def stop_best_effort(reason):
        try:
            held = jog.playback_stop_at_observed()
        except RuntimeError as error:
            return str(error), qs[max(0, recorded)]
        if best_effort_stopped is not None:
            best_effort_stopped(plan["stage"][-1], reason, held)
        return "", np.array([held[name] for name in MOTOR_NAMES], dtype=float)

    poll_delay = 0.0
    while True:
        # Poll faster than a motor tick so a consumed prefix can be replenished
        # before the next tick, without coupling planning to camera cadence.
        await asyncio.sleep(poll_delay)
        poll_delay = min(runtime.ACT_TICK_S, 0.25 / jog.HZ)
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
        retarget_count = min(len(qs) - done - 1, max(4, int(round(RETARGET_HORIZON_S * jog.HZ))))
        if replacing and retarget_count >= 4:
            # Match LeRobot RTC's timing model: the old queue executes while a
            # whole checked chunk is produced. Keep a predicted execution
            # prefix, then trim the *actual* number of commands consumed before
            # atomically installing the new unsent tail.
            old_count = retarget_count
            expected_solve_s = old_count * (np.median(solve_seconds) if solve_seconds else 0.002)
            frozen_ticks = min(old_count - 1, max(1, int(np.ceil(2 * expected_solve_s * jog.HZ))))
            chunk_started = time.perf_counter()
            candidate = await loop.run_in_executor(
                runtime._ACT_EXECUTOR,
                functools.partial(
                    _retarget_timed_tail,
                    kin,
                    plan,
                    qs[done],
                    qs[max(0, done - 1)],
                    qs[done + 1 : done + 1 + old_count],
                    phases[done + 1 : done + 1 + old_count],
                    due[done + 1 : done + 1 + old_count],
                    goals[done + 1 : done + 1 + old_count],
                    compositions[done + 1 : done + 1 + old_count],
                    motion,
                    fix,
                    limits,
                    speed,
                    jog.servo_ranges(),
                    jog.workspace_box(),
                    gi,
                    None if revision is None else revision(),
                    frozen_ticks,
                ),
            )
            if diagnostics is not None:
                diagnostics["chunk_solve_ms"].append((time.perf_counter() - chunk_started) * 1000.0)
            if candidate is not None:
                new_q, new_phases, new_due, new_goals, new_compositions = candidate
                installed = False
                for _ in range(3):
                    current_done, _ = jog.playback_state()
                    consumed = current_done - done
                    if consumed < 0 or consumed >= len(new_q):
                        break
                    join_phase = (
                        new_compositions[consumed]["phase"]
                        if new_compositions[consumed] is not None
                        else new_phases[consumed]
                    )
                    # Slicing alone is not enough for an IK trajectory: the
                    # splice must still join the actually sent old command
                    # within the same joint-speed, acceleration and workspace
                    # checks as the original samples.
                    join = reactive.checked_joint_samples(
                        kin,
                        qs[current_done],
                        {"reason": "", "q": new_q[consumed], "pose": kin.forward_kinematics(new_q[consumed])},
                        1,
                        gi,
                        jog.workspace_box(),
                        float(path["floor"][join_phase]),
                        min(jog.PLAYBACK_LAG_DEG, jog.RAMP_DEG_S * speed / jog.HZ),
                        previous_sample=qs[max(0, current_done - 1)],
                        max_joint_accel_deg=jog.RAMP_DEG_S * speed / jog.HZ,
                        continuous=True,
                        joint_ranges=jog.servo_ranges(),
                    )
                    if join is None:
                        break
                    if consumed + 1 < len(new_q):
                        second_phase = (
                            new_compositions[consumed + 1]["phase"]
                            if new_compositions[consumed + 1] is not None
                            else new_phases[consumed + 1]
                        )
                        second = reactive.checked_joint_samples(
                            kin,
                            new_q[consumed],
                            {
                                "reason": "",
                                "q": new_q[consumed + 1],
                                "pose": kin.forward_kinematics(new_q[consumed + 1]),
                            },
                            1,
                            gi,
                            jog.workspace_box(),
                            float(path["floor"][second_phase]),
                            min(jog.PLAYBACK_LAG_DEG, jog.RAMP_DEG_S * speed / jog.HZ),
                            previous_sample=qs[current_done],
                            max_joint_accel_deg=jog.RAMP_DEG_S * speed / jog.HZ,
                            continuous=True,
                            joint_ranges=jog.servo_ranges(),
                        )
                        if second is None:
                            break
                    if jog.replace_playback_tail(
                        current_done,
                        [as_dict(q) for q in new_q[consumed:]],
                        new_due[consumed:],
                    ):
                        del qs[current_done + 1 :]
                        del phases[current_done + 1 :]
                        del due[current_done + 1 :]
                        del goals[current_done + 1 :]
                        del compositions[current_done + 1 :]
                        qs.extend(new_q[consumed:])
                        phases.extend(new_phases[consumed:])
                        due.extend(new_due[consumed:])
                        goals.extend(new_goals[consumed:])
                        compositions.extend(new_compositions[consumed:])
                        unchanged_snapshot = (
                            tuple((name, value.tobytes()) for name, value in motion.items()),
                            fix.tobytes(),
                        )
                        if diagnostics is not None:
                            diagnostics["tail_replacements"] += 1
                            diagnostics["chunk_replacements"] += 1
                            diagnostics["chunk_trimmed_samples"] += consumed
                        poll_delay = 0.0
                        installed = True
                        break
                if installed:
                    continue
            if diagnostics is not None:
                diagnostics["chunk_rejected"] += 1
        # Keep the old checked tail moving while the replacement is solved.
        # The swap below is atomic and only succeeds if this head has not
        # advanced; a changed head is replanned on the next producer turn.
        if replacing and orientation_stage is not None and plan["stage"][phases[done]] != orientation_stage:
            orientation_stage = orientation_anchor = None
        pending = 0 if replacing else len(qs) - done - 1
        i = next_target_phase(phase, qs[done])
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
        i = next_target_phase(phase, qs[base])
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
        # Preserve one-tick demo phases when IK keeps up. A two-tick minimum
        # doubled every small contact/hold sample even with 2 ms solves.
        # Sustained slower IK can still use a longer checked interpolation;
        # isolated spikes are absorbed by the queue, not baked into its speed.
        compute_ticks = 1 if not solve_seconds else int(np.ceil(np.median(solve_seconds) * jog.HZ))
        nominal_ticks = max(
            1,
            min(
                6, max(compute_ticks, int(np.ceil((plan["times"][i] - plan["times"][phase]) * jog.HZ - 1e-8)))
            ),
        )
        continuous = (
            path.get("move", ["follow"] * len(plan["q"]))[i] in {"line", "lift"}
            and abs(float(path["grips"][i] - qs[base][gi])) < 1e-6
        )
        if continuous:
            nominal_ticks = min(horizon, max(nominal_ticks, 6))
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
            solve_seconds.append(time.perf_counter() - solve_started)
            if diagnostics is not None:
                diagnostics["solve_ms"].append((time.perf_counter() - solve_started) * 1000.0)
            if (
                continuous
                and np.max(np.abs(np.delete(np.asarray(result["q"]) - qs[base], gi)))
                > core.ACT_MAX_JOINT_STEP_DEG
            ):
                step_ticks /= 2
                continue
            if result["reason"]:
                break
            arm_delta = np.delete(np.asarray(result["q"]) - qs[base], gi)
            # A cubic easing from rest peaks at 1.5 times its mean speed.
            # Reserve that room before checking the complete joint curve.
            peak = 1.5 if continuous else 1.0
            ticks = max(nominal_ticks, int(np.ceil(peak * np.max(np.abs(arm_delta)) / joint_tick)))
            if ticks > horizon:
                step_ticks *= min(0.5, horizon / ticks)
                continue
            if pending + ticks > horizon:
                room_needed = True
                break
            samples = await loop.run_in_executor(
                runtime._ACT_EXECUTOR,
                functools.partial(
                    reactive.checked_joint_samples,
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
                    continuous=continuous,
                    joint_ranges=solve_args[6],
                ),
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
            # Fill available capacity immediately. The executor already
            # yields; adding an idle poll after every solve wastes production
            # time precisely when the queue needs its reserve rebuilt.
            poll_delay = 0.0
            if replacing:
                unchanged_snapshot = None
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
        elif replacing and diagnostics is not None:
            diagnostics["stale_replacements"] += 1
