"""
Autonomous workspace explorer for the SO-107 right arm.

Visits a grid of test poses; at each pose, probes all 6 cardinal directions
(±X, ±Y, ±Z) with a short Jacobian-driven motion and measures actual EE
response. Returns to test pose between probes.

Output: a CSV mapping (pose -> direction -> aligned_mm, efficiency) plus a
console heatmap.

Safety: motion is slow (rate=10Hz), small (step=2mm/tick × 10 ticks per probe),
and respects max_relative_target via the bus.  Ctrl-C aborts cleanly.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.explore_workspace \\
        --port /dev/ttyACM2 --id right_white
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from ..kinematics import MOTOR_NAMES, RIGHT_ARM_MAP, So107Kinematics
from .teleop_sim import JacobianVelocityController

logging.basicConfig(level=logging.WARNING)

# Pose grid: only vary the two "long" joints (shoulder_lift, elbow_flex) which
# define arm reach. Keep others near 0 / safe for now.
# Motor degrees per the calibration's _normalize formula.
POSE_GRID_SHOULDER_LIFT = [-90.0, -60.0, -30.0, 0.0]  # within ±105.4 limit
POSE_GRID_ELBOW_FLEX = [0.0, 30.0, 60.0, 90.0]  # within ±94.5 limit

DIRECTIONS = [
    ("+X", np.array([+1.0, 0.0, 0.0])),
    ("-X", np.array([-1.0, 0.0, 0.0])),
    ("+Y", np.array([0.0, +1.0, 0.0])),
    ("-Y", np.array([0.0, -1.0, 0.0])),
    ("+Z", np.array([0.0, 0.0, +1.0])),
    ("-Z", np.array([0.0, 0.0, -1.0])),
]

PROBE_TICKS = 10  # per direction
STEP_MM = 2.0  # per tick
RATE_HZ = 10.0  # slower than teleop so motion is visibly safe
SETTLE_S = 0.5  # wait between commands and measurements
MAX_RELATIVE_TARGET = 5.0


def read_motors(robot: SO107Follower) -> dict[str, float]:
    obs = robot.bus.sync_read("Present_Position")
    return {n: float(obs[n]) for n in MOTOR_NAMES}


def safe_move_to(
    robot: SO107Follower,
    target: dict[str, float],
    max_step_deg: float = 3.0,
    timeout_s: float = 30.0,
    close_enough_deg: float = 3.0,  # "good enough" tolerance for probing
    reached_deg: float = 0.5,  # tight tolerance => exit immediately
    stall_window: int = 40,
) -> bool:
    """Drive motors toward `target` at max_step_deg/tick.

    Returns True if motors got within `close_enough_deg` of target (probing
    from a slightly-off pose is fine). Returns False only if they couldn't
    get even close (true hardware stall).
    """
    period = 1.0 / RATE_HZ
    t_start = time.monotonic()
    last_progress_mp = None
    no_progress_ticks = 0
    while True:
        cur = read_motors(robot)
        gap = {n: target[n] - cur[n] for n in MOTOR_NAMES}
        max_gap = max(abs(v) for v in gap.values())
        # Tight reached.
        if max_gap < reached_deg:
            return True

        timed_out = time.monotonic() - t_start > timeout_s
        # Stall detection: no motor moved 0.3° between consecutive checks.
        if last_progress_mp is None:
            last_progress_mp = dict(cur)
            stalled = False
        else:
            moved = any(abs(cur[n] - last_progress_mp[n]) > 0.3 for n in MOTOR_NAMES)
            if moved:
                last_progress_mp = dict(cur)
                no_progress_ticks = 0
                stalled = False
            else:
                no_progress_ticks += 1
                stalled = no_progress_ticks >= stall_window

        if timed_out or stalled:
            # "Close enough" treated as success — probe from where we ended up.
            if max_gap < close_enough_deg:
                reason = "timeout" if timed_out else "stall"
                print(f"    {reason} but close enough (max gap {max_gap:.2f}°); proceeding to probe")
                return True
            reason = "TIMEOUT" if timed_out else "STALL"
            print(f"    safe_move_to {reason} (max gap {max_gap:.2f}°). full state:")
            for n in MOTOR_NAMES:
                progress = (cur[n] - last_progress_mp[n]) if last_progress_mp else 0
                tag = "STUCK" if abs(gap[n]) > 1.0 and abs(progress) < 0.3 else ""
                print(f"      {n:14s}: at {cur[n]:+.2f}, want {target[n]:+.2f}, gap {gap[n]:+.2f}  {tag}")
            return False

        action = {}
        for n in MOTOR_NAMES:
            d = max(-max_step_deg, min(max_step_deg, gap[n]))
            action[f"{n}.pos"] = cur[n] + d
        robot.send_action(action)
        time.sleep(period)


def probe(
    robot: SO107Follower,
    kin: So107Kinematics,
    jac: JacobianVelocityController,
    direction: np.ndarray,
    joint_limits: dict[str, tuple[float, float]],
    n_ticks: int = PROBE_TICKS,
) -> dict:
    """From the current pose, run n_ticks of Jacobian motion in direction; measure
    actual EE motion. Does NOT return to start (caller does that)."""
    period = 1.0 / RATE_HZ
    step_m = STEP_MM / 1000.0

    start_mp = read_motors(robot)
    start_T = kin.fk_from_motors(start_mp)
    target_q = dict(start_mp)

    for _ in range(n_ticks):
        ee_step = direction * step_m
        new_q = jac.step(target_q, ee_step, position_only=True, method="transpose")
        # Lead-buffer + joint-limit clamps.
        mp = read_motors(robot)
        lead_buffer = MAX_RELATIVE_TARGET * 2.0
        for n in MOTOR_NAMES:
            if n == "gripper":
                continue
            gap = new_q[n] - mp[n]
            if gap > lead_buffer:
                new_q[n] = mp[n] + lead_buffer
            elif gap < -lead_buffer:
                new_q[n] = mp[n] - lead_buffer
            lo, hi = joint_limits[n]
            if new_q[n] > hi:
                new_q[n] = hi
            elif new_q[n] < lo:
                new_q[n] = lo
        target_q = new_q
        action = {f"{n}.pos": target_q[n] for n in MOTOR_NAMES}
        robot.send_action(action)
        time.sleep(period)

    time.sleep(SETTLE_S)
    end_mp = read_motors(robot)
    end_T = kin.fk_from_motors(end_mp)
    delta = end_T[:3, 3] - start_T[:3, 3]
    aligned_mm = float(np.dot(delta, direction)) * 1000.0
    total_mm = float(np.linalg.norm(delta)) * 1000.0
    commanded_mm = n_ticks * STEP_MM
    return {
        "aligned_mm": aligned_mm,
        "total_mm": total_mm,
        "commanded_mm": commanded_mm,
        "efficiency": aligned_mm / commanded_mm if commanded_mm else 0,
        "directional_purity": aligned_mm / total_mm if total_mm > 0.1 else 0,
        "start_mp": start_mp,
        "end_mp": end_mp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument("--id", default="right_white")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="probe current pose only, skip the grid",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Workspace explorer — AUTONOMOUS MOTION. Ctrl-C to abort cleanly.")
    print("=" * 70)
    print(
        f"Pose grid: {len(POSE_GRID_SHOULDER_LIFT)}x{len(POSE_GRID_ELBOW_FLEX)} = "
        f"{len(POSE_GRID_SHOULDER_LIFT) * len(POSE_GRID_ELBOW_FLEX)} poses x 6 directions"
    )
    print(f"Per-probe: {PROBE_TICKS} ticks @ {STEP_MM}mm/tick = {PROBE_TICKS * STEP_MM}mm commanded")
    print(f"Rate: {RATE_HZ}Hz  Safety cap: {MAX_RELATIVE_TARGET}°/tick")
    input("Press Enter to start (or Ctrl-C to abort): ")

    config = SO107FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        cameras={},
        max_relative_target=MAX_RELATIVE_TARGET,
    )
    robot = SO107Follower(config)
    robot.connect(calibrate=False)
    print(f"\nConnected to {robot}")

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)
    jac = JacobianVelocityController(joint_map=RIGHT_ARM_MAP)

    # Calibrated joint limits (with safety margin).
    joint_limits: dict[str, tuple[float, float]] = {}
    for name, c in robot.bus.calibration.items():
        mid = (c.range_min + c.range_max) / 2
        deg_min = (c.range_min - mid) * 360.0 / 4095
        deg_max = (c.range_max - mid) * 360.0 / 4095
        joint_limits[name] = (deg_min + 1.0, deg_max - 1.0)
    print("Joint limits (deg):")
    for n, (lo, hi) in joint_limits.items():
        print(f"  {n:14s}: [{lo:+7.2f}, {hi:+7.2f}]")

    aborted = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: aborted.update(v=True))

    log_path = Path(f"/tmp/so107_explore_{dt.datetime.now():%Y%m%d_%H%M%S}.csv")
    log_f = log_path.open("w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(
        [
            "shoulder_lift_deg",
            "elbow_flex_deg",
            "direction",
            "commanded_mm",
            "aligned_mm",
            "total_mm",
            "efficiency",
            "directional_purity",
        ]
    )

    try:
        if args.probe_only:
            print("\n--- probing current pose ---")
            current = read_motors(robot)
            print("  pose: " + ", ".join(f"{n}={current[n]:+.1f}" for n in MOTOR_NAMES))
            home_pose = dict(current)
            for label, d in DIRECTIONS:
                if aborted["v"]:
                    break
                r = probe(robot, kin, jac, d, joint_limits)
                safe_move_to(robot, home_pose)
                print(
                    f"  {label}: aligned {r['aligned_mm']:+6.2f}mm "
                    f"(eff {r['efficiency'] * 100:5.1f}%, purity {r['directional_purity'] * 100:5.1f}%)"
                )
                log_writer.writerow(
                    [
                        current["shoulder_lift"],
                        current["elbow_flex"],
                        label,
                        r["commanded_mm"],
                        r["aligned_mm"],
                        r["total_mm"],
                        r["efficiency"],
                        r["directional_purity"],
                    ]
                )
                log_f.flush()
        else:
            n_poses = len(POSE_GRID_SHOULDER_LIFT) * len(POSE_GRID_ELBOW_FLEX)
            pose_idx = 0
            results = []  # for heatmap
            for sl in POSE_GRID_SHOULDER_LIFT:
                for ef in POSE_GRID_ELBOW_FLEX:
                    if aborted["v"]:
                        break
                    pose_idx += 1
                    target_pose = {
                        "shoulder_pan": 0.0,
                        "shoulder_lift": sl,
                        "elbow_flex": ef,
                        "forearm_roll": 0.0,
                        "wrist_flex": 0.0,
                        "wrist_roll": 0.0,
                        "gripper": 0.0,
                    }
                    print(
                        f"\n[{pose_idx}/{n_poses}] moving to shoulder_lift={sl:+.0f} elbow_flex={ef:+.0f} ..."
                    )
                    if not safe_move_to(robot, target_pose):
                        print("    SKIPPING pose — couldn't reach it.")
                        # Log failure for visibility.
                        for label, _ in DIRECTIONS:
                            log_writer.writerow([sl, ef, label, 0, 0, 0, 0, 0])
                        log_f.flush()
                        continue
                    time.sleep(SETTLE_S)
                    home_pose = read_motors(robot)
                    pose_results = {}
                    for label, d in DIRECTIONS:
                        if aborted["v"]:
                            break
                        r = probe(robot, kin, jac, d, joint_limits)
                        pose_results[label] = r
                        safe_move_to(robot, home_pose)
                        log_writer.writerow(
                            [
                                sl,
                                ef,
                                label,
                                r["commanded_mm"],
                                r["aligned_mm"],
                                r["total_mm"],
                                r["efficiency"],
                                r["directional_purity"],
                            ]
                        )
                        log_f.flush()
                    results.append((sl, ef, pose_results))
                    print(
                        "  results: "
                        + "  ".join(
                            f"{lbl}={pose_results[lbl]['efficiency'] * 100:4.0f}%" for lbl, _ in DIRECTIONS
                        )
                    )
                if aborted["v"]:
                    break

            # Print heatmap.
            print("\n" + "=" * 70)
            print("Efficiency heatmap (aligned EE motion / commanded, %)")
            print("=" * 70)
            for label, _ in DIRECTIONS:
                print(f"\nDirection {label}:")
                print(f"{'elbow→':>14s}  " + "  ".join(f"{ef:>+5.0f}" for ef in POSE_GRID_ELBOW_FLEX))
                print(f"{'shoulder_lift':>14s}")
                for sl in POSE_GRID_SHOULDER_LIFT:
                    cells = []
                    for ef in POSE_GRID_ELBOW_FLEX:
                        match = next((r for s, e, r in results if s == sl and e == ef), None)
                        if match and label in match:
                            eff = match[label]["efficiency"] * 100
                            cells.append(f"{eff:>5.0f}")
                        else:
                            cells.append("  -- ")
                    print(f"{sl:>13.0f}   " + "  ".join(cells))
    finally:
        log_f.close()
        print(f"\nLog saved to {log_path}")
        with contextlib.suppress(Exception):
            robot.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
