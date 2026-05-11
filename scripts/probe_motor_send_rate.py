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


def interpolate_action(elapsed: float, traj: dict) -> dict[str, float]:
    """Linear interpolation between bracketing trajectory frames."""
    timestamps = traj["timestamps"]
    positions = traj["positions"]
    joints = traj["joints"]

    if elapsed >= timestamps[-1]:
        frame = positions[-1]
    elif elapsed <= timestamps[0]:
        frame = positions[0]
    else:
        # Linear scan with a static cursor — fine for thousands of frames.
        idx = 0
        while idx + 1 < len(timestamps) and timestamps[idx + 1] <= elapsed:
            idx += 1
        t0, t1 = timestamps[idx], timestamps[idx + 1]
        alpha = (elapsed - t0) / (t1 - t0)
        p0, p1 = positions[idx], positions[idx + 1]
        frame = [a + (b - a) * alpha for a, b in zip(p0, p1, strict=True)]

    return dict(zip(joints, frame, strict=True))


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

    send_period = 1.0 / args.send_fps
    read_period = 1.0 / args.read_fps
    log: list[tuple[float, list[float], list[float] | None]] = []

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

            # Send action (interpolated)
            action = interpolate_action(elapsed, traj)
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

            # Read state at lower rate
            if elapsed - last_read_t >= read_period:
                obs = robot.get_observation()
                last_state = [float(obs.get(j, 0.0)) for j in joints]
                last_read_t = elapsed
                n_reads += 1

            log.append((elapsed, [action[j] for j in joints], list(last_state)))
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
        header = ["t"] + [f"action.{j}" for j in joints] + [f"state.{j}" for j in joints]
        w.writerow(header)
        for t, a, s in log:
            w.writerow([t, *a, *s])
    print(f"wrote {out_path}")

    # Inline cross-correlation analysis (so we don't need a separate tool).
    print()
    print("--- per-joint cross-correlation lag ---")
    action_arr = np.array([row[1] for row in log], dtype=np.float64)
    state_arr = np.array([row[2] for row in log], dtype=np.float64)
    dt = 1.0 / args.send_fps
    max_lag = int(0.3 / dt)  # search up to 300 ms
    print(f"{'joint':<26}{'lag_fr':>8}{'lag_ms':>9}{'corr':>8}")
    confident = []
    for j_idx, j_name in enumerate(joints):
        a = action_arr[:, j_idx]
        s = state_arr[:, j_idx]
        best_k, best_c = 0, -np.inf
        for k in range(max_lag + 1):
            aa = a[: len(a) - k] if k > 0 else a
            ss = s[k:] if k > 0 else s
            aa_c = aa - aa.mean()
            ss_c = ss - ss.mean()
            na, ns = float(np.linalg.norm(aa_c)), float(np.linalg.norm(ss_c))
            if na < 1e-12 or ns < 1e-12:
                continue
            c = float(np.dot(aa_c, ss_c) / (na * ns))
            if c > best_c:
                best_c = c
                best_k = k
        lag_ms = best_k * dt * 1e3
        marker = "*" if best_c >= 0.95 else " "
        if best_c >= 0.95:
            confident.append(lag_ms)
        print(f"{marker} {j_name:<24}{best_k:>8d}{lag_ms:>9.1f}{best_c:>8.3f}")
    if confident:
        print(
            f"\nmean lag (corr >= 0.95): {np.mean(confident):.1f} ms  ({len(confident)}/{len(joints)} joints)"
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
        "--output",
        default="outputs/probe_motor_send_rate/last.csv",
        help="CSV output path (joined under the cwd unless absolute)",
    )
    sys.exit(main(parser.parse_args()))
