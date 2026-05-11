#!/usr/bin/env python
"""Decoupled motor-rate probe: send goal_position at one rate, read state at another.

Background: the previous "200 Hz" trajectory_replay experiment was bus-bound
because we were doing both sync_read (14 motors) AND sync_write (14 motors)
on every loop iteration. The motor reads aren't actually needed at the
send rate — for open-loop trajectory replay, we just need ACTIONS to land
on the bus at the target rate. State can be sampled at a much lower rate
for analysis.

This script runs:
  - sync_write goal_position at ``--send-fps`` (default 200 Hz)
  - sync_read present_position at ``--read-fps`` (default 30 Hz)
  - logs (timestamp, action[14], state[14]) to a CSV
  - exits when the trajectory is exhausted

After the run, use scripts/analyze_control_lag.py with --csv-path to compute
per-joint lag and plateau-σ. Or just import and analyze inline.

Hardware footprint: motors only, no cameras, no dataset, no observation
pipeline. Minimal latency overhead so we can isolate the motor-bus ceiling
from everything else.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.robots.bi_so107_follower import BiSO107FollowerConfig
from lerobot.robots.bi_so107_follower.bi_so107_follower import BiSO107Follower
from lerobot.robots.safe_trajectory import validate_trajectory
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


def load_trajectory(path: Path) -> dict:
    traj = json.loads(path.read_text())
    validate_trajectory(traj)
    if not traj["timestamps"]:
        raise ValueError(f"trajectory at {path} is empty")
    return traj


def compute_action(
    elapsed: float,
    traj: dict,
    lookahead_s: float = 0.0,
    chunk_size: int | None = None,
) -> dict[str, float]:
    """Resolve action vector for ``elapsed`` with optional lookahead + chunking.

    - ``lookahead_s``: query the trajectory at ``elapsed + lookahead_s``
      instead of ``elapsed``. Used to cancel motor response lag.
    - ``chunk_size``: pretend the trajectory was delivered in chunks of N
      consecutive frames. When the query lands past the active chunk's
      last frame, extrapolate from the chunk's own forward-diff velocity
      rather than peek at the next chunk — modelling a chunked policy.

    With ``chunk_size = None`` and ``lookahead_s = 0`` this is the
    original "full visibility, no lookahead" baseline.
    """
    timestamps = traj["timestamps"]
    positions = traj["positions"]
    joints = traj["joints"]
    n = len(timestamps)
    query_t = elapsed + lookahead_s

    # Chunk window is set by *elapsed* (when chunks "arrive"), not query_t.
    if chunk_size and chunk_size > 0:
        # Find frame bracketing elapsed (clamped to last frame).
        clamp_elapsed = min(elapsed, timestamps[-1])
        cur_idx = 0
        while cur_idx + 1 < n and timestamps[cur_idx + 1] <= clamp_elapsed:
            cur_idx += 1
        chunk_start = (cur_idx // chunk_size) * chunk_size
        chunk_end = min(chunk_start + chunk_size - 1, n - 1)
    else:
        chunk_start = 0
        chunk_end = n - 1

    # Left clamp.
    if query_t <= timestamps[chunk_start]:
        return dict(zip(joints, positions[chunk_start], strict=True))

    chunk_end_t = timestamps[chunk_end]

    if query_t <= chunk_end_t:
        # Interpolation inside the visible chunk.
        idx = chunk_start
        while idx + 1 <= chunk_end and timestamps[idx + 1] <= query_t:
            idx += 1
        if idx + 1 <= chunk_end:
            t0, t1 = timestamps[idx], timestamps[idx + 1]
            dt = t1 - t0
            alpha = (query_t - t0) / dt if dt > 1e-9 else 0.0
            p0, p1 = positions[idx], positions[idx + 1]
            frame = [a + (b - a) * alpha for a, b in zip(p0, p1, strict=True)]
        else:
            frame = positions[idx]
        return dict(zip(joints, frame, strict=True))

    # Extrapolation past chunk end using last-two-frame forward diff.
    if chunk_end == chunk_start:
        frame = positions[chunk_end]
    else:
        t_prev = timestamps[chunk_end - 1]
        t_last = timestamps[chunk_end]
        dt = t_last - t_prev
        p_prev = positions[chunk_end - 1]
        p_last = positions[chunk_end]
        if dt < 1e-9:
            frame = list(p_last)
        else:
            ahead = query_t - t_last
            frame = [last + (last - prev) * (ahead / dt) for prev, last in zip(p_prev, p_last, strict=True)]
    return dict(zip(joints, frame, strict=True))


def ground_truth_action(elapsed: float, traj: dict) -> dict[str, float]:
    """Trajectory value at exactly ``elapsed`` (pure interpolation, no lookahead).

    Used as the accuracy reference: chunked/lookahead actions deviate from
    GT by design; the question is whether the *motor's observed state*
    ends up closer to GT.
    """
    return compute_action(elapsed, traj, lookahead_s=0.0, chunk_size=None)


def compute_velocity(elapsed: float, traj: dict) -> dict[str, float]:
    """Centered-difference velocity at the bracketing trajectory frames.

    Returns velocity in (position_unit / second). For Feetech Goal_Velocity
    the bus's sync_write expects raw motor units; the velocity feed-forward
    experiment scales these centrally before writing.
    """
    timestamps = traj["timestamps"]
    positions = traj["positions"]
    joints = traj["joints"]
    n = len(timestamps)

    if elapsed <= timestamps[0] or elapsed >= timestamps[-1]:
        return dict.fromkeys(joints, 0.0)

    idx = 0
    while idx + 1 < n and timestamps[idx + 1] <= elapsed:
        idx += 1

    # Use the centered difference where possible; forward/backward at the ends.
    lo = max(0, idx - 1)
    hi = min(n - 1, idx + 2)
    dt = timestamps[hi] - timestamps[lo]
    if dt < 1e-9:
        return dict.fromkeys(joints, 0.0)
    p_lo, p_hi = positions[lo], positions[hi]
    return {j: (p_hi[i] - p_lo[i]) / dt for i, j in enumerate(joints)}


def main(args: argparse.Namespace) -> int:
    init_logging()
    traj_path = Path(args.trajectory).expanduser()
    traj = load_trajectory(traj_path)
    duration = float(traj["timestamps"][-1])

    cfg = BiSO107FollowerConfig(
        id="white",
        left_arm_port=args.left_port,
        right_arm_port=args.right_port,
        left_arm_disable_torque_on_disconnect=False,
        right_arm_disable_torque_on_disconnect=False,
        left_arm_use_degrees=False,
        right_arm_use_degrees=False,
        cameras={},  # no cameras
    )
    robot = BiSO107Follower(cfg)
    robot.connect(calibrate=False)

    # Override P_Coefficient after connect: configure_motors() sets P=16
    # unconditionally; stiffer P shortens the motor's response τ but adds
    # shakiness. Sweep this alongside --lookahead-ms to study the
    # τ-vs-lookahead tradeoff.
    if args.p_coefficient is not None:
        for arm in (robot.left_arm, robot.right_arm):
            for motor in arm.bus.motors:
                arm.bus.write("P_Coefficient", motor, args.p_coefficient)
        print(f"overrode P_Coefficient = {args.p_coefficient} on all motors")

    # Override D_Coefficient: configure_motors() sets D=32. Higher D damps
    # the position-loop ringing that high P causes — main lever for
    # killing the steady-state jitter that visually shows up at P>=48.
    if args.d_coefficient is not None:
        for arm in (robot.left_arm, robot.right_arm):
            for motor in arm.bus.motors:
                arm.bus.write("D_Coefficient", motor, args.d_coefficient)
        print(f"overrode D_Coefficient = {args.d_coefficient} on all motors")

    # Override small-error responsiveness floor: dead zones + startup force.
    # configure_motors() doesn't touch these, so they're at firmware
    # defaults (a few units of deadband + a small startup force). Zeroing
    # them removes a deadband-induced latency for sub-threshold motions
    # without affecting the PID loop — useful at low P where the upstream
    # P=16 default is otherwise sluggish for tiny corrections.
    for arg_name, reg_name in (
        ("cw_dead_zone", "CW_Dead_Zone"),
        ("ccw_dead_zone", "CCW_Dead_Zone"),
        ("minimum_startup_force", "Minimum_Startup_Force"),
    ):
        val = getattr(args, arg_name)
        if val is None:
            continue
        for arm in (robot.left_arm, robot.right_arm):
            for motor in arm.bus.motors:
                arm.bus.write(reg_name, motor, val)
        print(f"overrode {reg_name} = {val} on all motors")

    send_period = 1.0 / args.send_fps
    read_period = 1.0 / args.read_fps
    log: list[tuple[float, float, list[float], list[float]]] = []  # (t_send, t_read, action, state)

    # First state read so the first log row has a state too (the loop
    # reads on multiples of read_period after t=0).
    initial_obs = robot.get_observation()
    joints = traj["joints"]
    last_state = [float(initial_obs.get(j, 0.0)) for j in joints]
    last_read_t = 0.0

    print(
        f"sending at {args.send_fps} Hz, reading at {args.read_fps} Hz, "
        f"trajectory {duration:.2f}s ({len(traj['timestamps'])} frames)"
    )

    start = time.perf_counter()
    next_send_t = start
    overruns = 0
    n_sends = 0
    n_reads = 0

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= duration:
                break

            # Send action (interpolated, with optional lookahead + chunk sim)
            action = compute_action(
                elapsed,
                traj,
                lookahead_s=args.lookahead_ms / 1000.0,
                chunk_size=args.chunk_size,
            )
            t_send_start = time.perf_counter()
            robot.send_action(action)

            # Optional velocity feed-forward: write Goal_Velocity (register 46)
            # alongside Goal_Position. Centered-difference velocity from the
            # trajectory, scaled by --vff-scale (Feetech motor units per
            # position-unit-per-second — calibrate experimentally).
            if args.velocity_ff:
                vel = compute_velocity(elapsed, traj)
                scale = args.vff_scale
                left_vel = {
                    k.removeprefix("left_").removesuffix(".pos"): int(v * scale)
                    for k, v in vel.items()
                    if k.startswith("left_")
                }
                right_vel = {
                    k.removeprefix("right_").removesuffix(".pos"): int(v * scale)
                    for k, v in vel.items()
                    if k.startswith("right_")
                }
                robot.left_arm.bus.sync_write("Goal_Velocity", left_vel)
                robot.right_arm.bus.sync_write("Goal_Velocity", right_vel)

            t_send_end = time.perf_counter()
            n_sends += 1

            # Read state at lower rate. Record t_read at the moment the
            # bus traffic completes (post-sync_read). That's our best
            # estimate of when these samples were captured — within ~3 ms
            # of the actual rotor sample for a 14-motor sync_read at 1 Mbps.
            if elapsed - last_read_t >= read_period:
                obs = robot.get_observation()
                last_state = [float(obs.get(j, 0.0)) for j in joints]
                last_read_t = time.perf_counter() - start
                n_reads += 1

            log.append((elapsed, last_read_t, [action[j] for j in joints], list(last_state)))
            _ = t_send_start, t_send_end  # kept for potential per-iter timing

            # Sleep to next send deadline
            next_send_t += send_period
            now = time.perf_counter()
            sleep_for = next_send_t - now
            if sleep_for > 0:
                precise_sleep(sleep_for)
            else:
                overruns += 1
                # Reset the deadline so we don't accumulate debt.
                next_send_t = time.perf_counter()
    finally:
        # Reset Goal_Velocity to 0 (firmware default = unlimited) on every
        # motor BEFORE doing the park-to-rest move. configure_motors() doesn't
        # reset this register, so values written during --velocity-ff
        # experiments would persist into subsequent runs of any program. A
        # small lingering Goal_Velocity ceiling would slow down the park
        # move (or any future move) silently.
        try:
            for arm in (robot.left_arm, robot.right_arm):
                if not arm.bus.is_connected:
                    continue
                zero = dict.fromkeys(arm.bus.motors, 0)
                arm.bus.sync_write("Goal_Velocity", zero)
        except Exception as e:
            print(f"warning: Goal_Velocity reset failed: {e}")

        # Park back at trajectory[0] to keep the arm at a safe rest pose
        # before disconnect (mirrors the auto-reset in lerobot-record).
        try:
            from lerobot.robots.rest_position import move_to_rest_position

            start_pose = {j: float(traj["positions"][0][i]) for i, j in enumerate(joints)}
            move_to_rest_position(robot, start_pose, duration_s=2.0)
        except Exception as e:
            print(f"warning: park-to-start failed: {e}")
        robot.disconnect()

    duration_actual = time.perf_counter() - start
    effective_send_hz = n_sends / duration_actual
    effective_read_hz = n_reads / duration_actual
    print()
    print(f"actual: {duration_actual:.2f}s")
    print(f"sends:  {n_sends}  ({effective_send_hz:.1f} Hz effective)  overruns: {overruns}")
    print(f"reads:  {n_reads}  ({effective_read_hz:.1f} Hz effective)")

    # Write CSV
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["t_send", "t_read"] + [f"action.{j}" for j in joints] + [f"state.{j}" for j in joints]
        w.writerow(header)
        for t_send, t_read, a, s in log:
            w.writerow([t_send, t_read, *a, *s])
    print(f"wrote {out_path}")

    # Per-row time arrays.
    t_send_arr = np.array([row[0] for row in log], dtype=np.float64)
    t_read_arr = np.array([row[1] for row in log], dtype=np.float64)
    action_arr = np.array([row[2] for row in log], dtype=np.float64)
    state_arr = np.array([row[3] for row in log], dtype=np.float64)  # stair-cased
    dt = 1.0 / args.send_fps

    # Ground-truth trajectory at each row's send time.
    gt_arr = np.array(
        [[ground_truth_action(t, traj)[j] for j in joints] for t in t_send_arr],
        dtype=np.float64,
    )

    # Build a bias-corrected state column: re-attach each read's value to
    # its actual read time (not the row's send time), then resample onto
    # send timestamps via linear interpolation. This removes the
    # stair-casing artifact that adds ~16 ms of apparent lag.
    # Take unique read events: rows where t_read changes from the prior row.
    read_mask = np.concatenate(([True], np.diff(t_read_arr) > 1e-9))
    t_read_uniq = t_read_arr[read_mask]
    state_uniq = state_arr[read_mask]
    state_interp_arr = np.empty_like(state_arr)
    for j_idx in range(state_arr.shape[1]):
        state_interp_arr[:, j_idx] = np.interp(t_send_arr, t_read_uniq, state_uniq[:, j_idx])

    # 1. Per-joint cross-correlation lag of STATE vs GROUND TRUTH.
    # This is the "effective motor lag" — independent of the action
    # transformation (lookahead, chunking). With lookahead working it
    # should drop versus the baseline.
    max_lag = int(0.3 / dt)  # search up to 300 ms

    def _lag_table(state_col: np.ndarray, label: str) -> list[float]:
        """Per-joint cross-corr of state vs GT. Prints and returns confident lags."""
        print()
        print(f"--- per-joint state-vs-GT lag ({label}) ---")
        print(f"{'joint':<26}{'lag_fr':>8}{'lag_ms':>9}{'corr':>8}")
        confident_local: list[float] = []
        for j_idx_, j_name_ in enumerate(joints):
            g = gt_arr[:, j_idx_]
            s = state_col[:, j_idx_]
            best_k, best_c = 0, -np.inf
            for k in range(max_lag + 1):
                gg = g[: len(g) - k] if k > 0 else g
                ss = s[k:] if k > 0 else s
                gg_c = gg - gg.mean()
                ss_c = ss - ss.mean()
                ng, ns = float(np.linalg.norm(gg_c)), float(np.linalg.norm(ss_c))
                if ng < 1e-12 or ns < 1e-12:
                    continue
                c = float(np.dot(gg_c, ss_c) / (ng * ns))
                if c > best_c:
                    best_c = c
                    best_k = k
            lag_ms_ = best_k * dt * 1e3
            marker = "*" if best_c >= 0.95 else " "
            if best_c >= 0.95:
                confident_local.append(lag_ms_)
            print(f"{marker} {j_name_:<24}{best_k:>8d}{lag_ms_:>9.1f}{best_c:>8.3f}")
        if confident_local:
            print(
                f"mean lag (corr >= 0.95): {np.mean(confident_local):.1f} ms  "
                f"({len(confident_local)}/{len(joints)} joints)"
            )
        return confident_local

    # Stair-cased (what the loop actually sees row-by-row): includes the
    # ~16 ms read-rate bias because each row's "state" is held from the
    # previous read until a new read arrives.
    conf_stair = _lag_table(state_arr, "raw row-stair, includes read-rate bias")
    # Read-time corrected: state interpolated at each row's send time using
    # the actual times reads happened — removes the staircase bias.
    conf_interp = _lag_table(state_interp_arr, "read-time corrected")
    if conf_stair and conf_interp:
        bias = float(np.mean(conf_stair)) - float(np.mean(conf_interp))
        print(f"read-rate bias removed by interp: {bias:.1f} ms")

    # 2. Tracking accuracy: RMSE of state vs ground truth, per joint and overall.
    # This is the headline accuracy number.
    print()
    print("--- per-joint state-vs-GT RMSE (tracking accuracy) ---")
    print(f"{'joint':<26}{'rmse':>10}")
    rmse_per_joint = np.sqrt(np.mean((state_arr - gt_arr) ** 2, axis=0))
    for j_name, r in zip(joints, rmse_per_joint, strict=True):
        print(f"  {j_name:<24}{r:>10.4f}")
    print(f"  {'OVERALL':<24}{float(np.sqrt(np.mean(rmse_per_joint**2))):>10.4f}")

    # 3. Smoothness: action-side and state-side jerk (mean |second diff|).
    # If chunk extrapolation injects kinks at boundaries, action_jerk rises.
    # If the motor amplifies them into visible shake, state_jerk rises.
    print()
    print("--- smoothness: mean |second diff| (lower = smoother) ---")
    print(f"{'joint':<26}{'action':>12}{'state':>12}")
    action_jerk = np.mean(np.abs(np.diff(action_arr, n=2, axis=0)), axis=0)
    state_jerk = np.mean(np.abs(np.diff(state_arr, n=2, axis=0)), axis=0)
    for j_name, aj, sj in zip(joints, action_jerk, state_jerk, strict=True):
        print(f"  {j_name:<24}{aj:>12.5f}{sj:>12.5f}")

    # 4. Plateau jitter: state velocity std during quasi-stationary GT periods.
    # The averaged jerk metric dilutes rest-state shake with the bulk of
    # the trajectory's motion. Looking at state *velocity* (not position)
    # during quiet GT periods isolates high-frequency oscillation —
    # exactly what the eye sees as jitter — from steady-state tracking
    # offset (slow motors that don't quite reach setpoint at rest).
    #
    # Method: classify a row as "quiet" if |d/dt GT| < 30th-percentile.
    # Report two numbers per joint:
    #   • track_σ  = std(state - GT) over quiet rows — total residual
    #     (large at low P due to slow tracking; small at high P)
    #   • jitter_σ = std(state_velocity) over quiet rows — pure shake
    #     (small at low P because slow motor doesn't oscillate; rises
    #      with P as the position loop rings)
    print()
    print("--- plateau metrics during quiet GT periods ---")
    gt_vel = np.gradient(gt_arr, dt, axis=0)
    state_vel = np.gradient(state_arr, dt, axis=0)
    print(f"{'joint':<26}{'track_σ':>10}{'jitter_σ':>12}{'n_quiet':>10}")
    track_sigmas: list[float] = []
    jitter_sigmas: list[float] = []
    for j_idx_, j_name_ in enumerate(joints):
        v = np.abs(gt_vel[:, j_idx_])
        thresh = float(np.percentile(v, 30))
        quiet = v <= thresh
        if quiet.sum() < 20:
            print(f"  {j_name_:<24}{'—':>10}{'—':>12}{int(quiet.sum()):>10}")
            continue
        track_sigma = float(np.std(state_arr[quiet, j_idx_] - gt_arr[quiet, j_idx_]))
        jitter_sigma = float(np.std(state_vel[quiet, j_idx_]))
        track_sigmas.append(track_sigma)
        jitter_sigmas.append(jitter_sigma)
        print(f"  {j_name_:<24}{track_sigma:>10.4f}{jitter_sigma:>12.3f}{int(quiet.sum()):>10}")
    if track_sigmas:
        print(
            f"mean track_σ: {float(np.mean(track_sigmas)):.4f}  "
            f"mean jitter_σ: {float(np.mean(jitter_sigmas)):.3f}"
        )

    # 5. Chunk-boundary discontinuity (only meaningful when chunked).
    if args.chunk_size and args.chunk_size > 0:
        # Boundaries occur when (cur_idx // chunk_size) advances. Detect them
        # by finding rows where the active chunk's start index increments.
        traj_ts = np.array(traj["timestamps"], dtype=np.float64)
        # For each logged row, which trajectory frame brackets its elapsed time?
        frame_idx = np.searchsorted(traj_ts, t_send_arr, side="right") - 1
        frame_idx = np.clip(frame_idx, 0, len(traj_ts) - 1)
        chunk_idx = frame_idx // args.chunk_size
        boundary_rows = np.where(np.diff(chunk_idx) > 0)[0] + 1
        if len(boundary_rows) > 0:
            # |action[t_boundary] - action[t_boundary - 1]| per joint.
            jumps = np.abs(action_arr[boundary_rows] - action_arr[boundary_rows - 1])
            print()
            print(f"--- chunk-boundary action jumps ({len(boundary_rows)} boundaries) ---")
            print(f"{'joint':<26}{'mean':>10}{'p95':>10}{'max':>10}")
            for j_idx, j_name in enumerate(joints):
                col = jumps[:, j_idx]
                print(
                    f"  {j_name:<24}{float(col.mean()):>10.4f}"
                    f"{float(np.percentile(col, 95)):>10.4f}{float(col.max()):>10.4f}"
                )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectory", required=True, help="Path to .trajectory.json")
    parser.add_argument("--left-port", default="/dev/ttyACM0")
    parser.add_argument("--right-port", default="/dev/ttyACM2")
    parser.add_argument("--send-fps", type=int, default=200, help="Action send rate (Hz)")
    parser.add_argument("--read-fps", type=int, default=30, help="State read rate (Hz)")
    parser.add_argument(
        "--velocity-ff",
        action="store_true",
        help="Also write Goal_Velocity (register 46) per iteration alongside Goal_Position. "
        "Tests whether feeding the motor an explicit velocity target reduces tracking lag.",
    )
    parser.add_argument(
        "--vff-scale",
        type=float,
        default=100.0,
        help="Scale factor: Goal_Velocity_raw = trajectory_velocity * vff_scale. "
        "Needs calibration to motor velocity units; default 100 is a starting guess.",
    )
    parser.add_argument(
        "--lookahead-ms",
        type=float,
        default=0.0,
        help="Predictive lookahead: query the trajectory at elapsed + lookahead_ms "
        "instead of elapsed. Used to cancel motor response lag (~60 ms on STS3215 "
        "at P=48). Default 0 = no lookahead (baseline).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Simulate chunked-policy delivery: pretend the trajectory arrives in "
        "non-overlapping chunks of N consecutive frames. Queries past the active "
        "chunk's last frame extrapolate from the chunk's own velocity, mirroring "
        "what a chunked policy faces. Default None = full visibility (interp only).",
    )
    parser.add_argument(
        "--p-coefficient",
        type=int,
        default=None,
        help="Override the per-motor position P-gain after connect. Default None = "
        "leave at configure_motors() value (P=16). Higher P shortens motor response "
        "τ at the cost of shakiness; useful for sweeping lookahead-vs-τ.",
    )
    parser.add_argument(
        "--d-coefficient",
        type=int,
        default=None,
        help="Override the per-motor position D-gain after connect. Default None = "
        "leave at configure_motors() value (D=32). Higher D damps the position-loop "
        "ringing introduced by high P — main lever for reducing visible jitter.",
    )
    parser.add_argument(
        "--cw-dead-zone",
        type=int,
        default=None,
        help="Override CW_Dead_Zone (reg 26). Default None = leave at firmware "
        "default (a small deadband around the setpoint). Setting to 0 removes the "
        "deadband — motor responds to even single-unit position errors.",
    )
    parser.add_argument(
        "--ccw-dead-zone",
        type=int,
        default=None,
        help="Override CCW_Dead_Zone (reg 27). See --cw-dead-zone.",
    )
    parser.add_argument(
        "--minimum-startup-force",
        type=int,
        default=None,
        help="Override Minimum_Startup_Force (reg 24). Default None = firmware "
        "default. Setting to 0 removes the static-friction kickstart floor — "
        "motor moves even for tiny torque commands. Pairs with --cw/--ccw-dead-zone=0 "
        "for the 'high small-step responsiveness' profile from the motor-sensitivity-fix branch.",
    )
    parser.add_argument(
        "--output",
        default="outputs/probe_motor_send_rate/last.csv",
        help="CSV output path (joined under the cwd unless absolute)",
    )
    sys.exit(main(parser.parse_args()))
