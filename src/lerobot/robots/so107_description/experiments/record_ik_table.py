"""
Record (motor_pos, FK_EE) pairs by hand-moving the arm through the workspace.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.record_ik_table \\
        --port /dev/ttyACM2 --id right_white
    # Hand-move the arm through every region you want teleop to be able to reach.
    # Ctrl-C when done. CSV is saved to /tmp/.

Modes:
    --hand (default) — torque OFF, you physically move the arm
    --teleop         — torque ON, assumes you're driving from a leader elsewhere
                       (use this if you have leader-follower set up via the GUI)

What gets logged per tick:
    t, motor_<7 names>, ee_x, ee_y, ee_z   (in meters, base frame, via FK from
                                            current calibrated bridge)

Tip: cover the workspace by tracing slow lassos that visit every region you
care about — corners, edges, the "useful" central volume — at multiple
orientations. ~3–5 minutes of waving around is enough.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import signal
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from ..kinematics import MOTOR_NAMES, RIGHT_ARM_MAP, So107Kinematics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument("--id", default="right_white")
    parser.add_argument(
        "--out", default=None, help="output CSV path (default: /tmp/so107_ik_table_<timestamp>.csv)"
    )
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="leave torque enabled (use this if leader-follower is moving the arm elsewhere)",
    )
    args = parser.parse_args()

    config = SO107FollowerConfig(port=args.port, id=args.id, use_degrees=True, cameras={})
    robot = SO107Follower(config)
    robot.connect(calibrate=False)
    print(f"\nConnected to {robot}")

    if not args.teleop:
        robot.bus.disable_torque()
        print("Torque DISABLED — hand-move the arm through the workspace.")
        print("Cover every region you want teleop to reach (corners, edges, center).")
    else:
        print("Torque ENABLED — assuming leader-follower or similar is driving the arm.")

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)

    out_path = Path(args.out or f"/tmp/so107_ik_table_{dt.datetime.now():%Y%m%d_%H%M%S}.csv")
    f = out_path.open("w", newline="")
    w = csv.writer(f)
    w.writerow(["t", *(f"motor_{n}" for n in MOTOR_NAMES), "ee_x", "ee_y", "ee_z"])

    print(f"\nRecording to {out_path}. Press Ctrl-C when done.\n")

    interrupted = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: interrupted.update(v=True))

    t0 = time.monotonic()
    period = 1.0 / args.rate_hz
    n_rows = 0
    last_print = 0.0
    last_ee = None
    distinct_workspace_cells: set = set()

    try:
        while not interrupted["v"]:
            tick_t0 = time.monotonic()
            obs = robot.bus.sync_read("Present_Position")
            mp = {n: float(obs[n]) for n in MOTOR_NAMES}
            T = kin.fk_from_motors(mp)
            ee = T[:3, 3]
            t = time.monotonic() - t0
            w.writerow(
                [
                    f"{t:.3f}",
                    *(f"{mp[n]:.3f}" for n in MOTOR_NAMES),
                    f"{ee[0]:.4f}",
                    f"{ee[1]:.4f}",
                    f"{ee[2]:.4f}",
                ]
            )
            f.flush()
            n_rows += 1

            # Track coverage at 1cm-grid granularity.
            cell = (int(ee[0] * 100), int(ee[1] * 100), int(ee[2] * 100))
            distinct_workspace_cells.add(cell)

            # Periodic progress.
            if t - last_print > 2.0:
                v = 0.0
                if last_ee is not None:
                    v = float(np.linalg.norm(ee - last_ee)) / (t - last_print) * 1000  # mm/s
                last_ee = ee.copy()
                last_print = t
                print(
                    f"  t={t:6.1f}s  rows={n_rows:5d}  "
                    f"workspace cells visited={len(distinct_workspace_cells):4d}  "
                    f"current xyz=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f})m  "
                    f"speed≈{v:.0f}mm/s"
                )

            elapsed = time.monotonic() - tick_t0
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        f.close()
        with contextlib.suppress(Exception):
            robot.disconnect()
        print(f"\nSaved {n_rows} samples to {out_path}")
        print(f"Distinct 1cm cells visited: {len(distinct_workspace_cells)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
