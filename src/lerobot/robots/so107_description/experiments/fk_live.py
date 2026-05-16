"""
Diagnostic: hand-move the arm, watch FK xyz prediction in real time.

What this tells us:
    - Move arm STRAIGHT UP physically → FK z should increase monotonically.
      If z stays flat or x/y change → URDF Z axis isn't aligned with reality.
    - Move arm STRAIGHT FORWARD (away from base) → one axis should change
      monotonically. If multiple axes change at similar rates → frame is
      rotated.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.fk_live \\
        --port /dev/ttyACM2 --id right_white

Torque is OFF (hand-movable). Ctrl-C to exit.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import numpy as np

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from ..kinematics import MOTOR_NAMES, RIGHT_ARM_MAP, So107Kinematics

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument("--id", default="right_white")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    args = parser.parse_args()

    config = SO107FollowerConfig(port=args.port, id=args.id, use_degrees=True, cameras={})
    robot = SO107Follower(config)
    robot.connect(calibrate=False)
    robot.bus.disable_torque()
    print(f"\nConnected to {robot}. Torque DISABLED — hand-move the arm.\n")

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)

    aborted = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: aborted.update(v=True))

    print("Move the arm and watch which FK axis changes.")
    print("Especially: move STRAIGHT UP, then STRAIGHT FORWARD, then SIDE TO SIDE.")
    print("If FK matches reality, each pure physical direction should drive one xyz axis.\n")

    period = 1.0 / args.rate_hz
    last_ee = None
    last_print = 0.0
    while not aborted["v"]:
        t0 = time.monotonic()
        obs = robot.bus.sync_read("Present_Position")
        mp = {n: float(obs[n]) for n in MOTOR_NAMES}
        T = kin.fk_from_motors(mp)
        ee = T[:3, 3] * 1000  # mm

        # Compute instantaneous velocity for clarity.
        v_str = ""
        if last_ee is not None:
            dt = t0 - last_print
            if dt > 0.05:
                dee = ee - last_ee
                speed = np.linalg.norm(dee) / dt  # mm/s
                # Dominant axis.
                dom = ("x", "y", "z")[int(np.argmax(np.abs(dee)))]
                sign = "+" if dee[int(np.argmax(np.abs(dee)))] > 0 else "-"
                v_str = f"  Δ=({dee[0]:+5.1f},{dee[1]:+5.1f},{dee[2]:+5.1f})mm  dominant={sign}{dom}  speed={speed:5.0f}mm/s"
                last_ee = ee.copy()
                last_print = t0
        else:
            last_ee = ee.copy()
            last_print = t0

        motors_str = "  ".join(f"{n[:5]}={mp[n]:+6.1f}" for n in MOTOR_NAMES[:6])
        print(
            f"\rxyz=({ee[0]:+7.1f},{ee[1]:+7.1f},{ee[2]:+7.1f})mm  {motors_str}{v_str}      ",
            end="",
            flush=True,
        )

        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    print("\n\nDone.")
    robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
