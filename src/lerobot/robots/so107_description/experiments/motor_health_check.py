"""
Minimal motor-health check: bypass IK entirely. For each motor in turn,
command a small ±degree delta and see if the motor physically moves.

If many motors fail this, the issue is at the hardware/servo level, not IK.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.motor_health_check \\
        --port /dev/ttyACM2 --id right_white
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from ..kinematics import MOTOR_NAMES

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument("--id", default="right_white")
    parser.add_argument("--probe-deg", type=float, default=5.0, help="commanded delta size per nudge")
    parser.add_argument("--ticks-per-probe", type=int, default=10)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args()

    config = SO107FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        cameras={},
        max_relative_target=args.probe_deg + 1,  # don't clip our probe
    )
    robot = SO107Follower(config)
    robot.connect(calibrate=False)
    print(f"\nConnected to {robot}\n")

    period = 1.0 / args.rate_hz

    def read_all() -> dict:
        obs = robot.bus.sync_read("Present_Position")
        return {n: float(obs[n]) for n in MOTOR_NAMES}

    start = read_all()
    print("Initial pose:")
    for n in MOTOR_NAMES:
        print(f"  {n:14s}: {start[n]:+8.2f}°")

    print(
        f"\nProbing each motor by ±{args.probe_deg}° (others held at current). "
        f"{args.ticks_per_probe} ticks each."
    )
    print(
        f"{'motor':14s}  {'before':>8s}  {'cmd_target':>10s}  {'after':>8s}  {'moved':>7s}  "
        f"{'compliance':>11s}"
    )

    for probed_motor in MOTOR_NAMES:
        # Snapshot all motors before probe
        before = read_all()
        target = dict(before)
        target[probed_motor] = before[probed_motor] + args.probe_deg

        # Hold all other motors at their current value; nudge probed motor.
        for _ in range(args.ticks_per_probe):
            action = {f"{n}.pos": target[n] for n in MOTOR_NAMES}
            robot.send_action(action)
            time.sleep(period)

        time.sleep(0.3)
        after = read_all()
        moved = after[probed_motor] - before[probed_motor]
        compliance = moved / args.probe_deg * 100
        flag = " ✓" if abs(compliance) > 80 else (" weak" if abs(compliance) > 30 else " STUCK")
        print(
            f"{probed_motor:14s}  {before[probed_motor]:+8.2f}  "
            f"{target[probed_motor]:+10.2f}  {after[probed_motor]:+8.2f}  "
            f"{moved:+7.2f}°  {compliance:>10.1f}%{flag}"
        )

        # Return to before
        for _ in range(args.ticks_per_probe + 5):
            action = {f"{n}.pos": before[n] for n in MOTOR_NAMES}
            robot.send_action(action)
            time.sleep(period)

    print("\nDone.")
    robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
