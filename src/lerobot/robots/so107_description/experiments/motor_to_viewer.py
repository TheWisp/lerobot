"""
Discovery helper for the motor->URDF bridge: stream live motor positions
through the bridge into the MeshCat viewer, so you can move the real arm by
hand and watch the URDF arm. Joints that disagree need (sign, offset) fixes.

Usage (right arm, defaults from the user's bi_so107_follower config):
    .venv/bin/python -m lerobot.robots.so107_description.motor_to_viewer \\
        --port /dev/ttyACM2 --id right_white

Procedure:
    1. Disconnect the leader / power down anything that holds the arm.
    2. Run this script. Torque is disabled, so the arm becomes hand-movable.
    3. Open the printed MeshCat URL in a browser.
    4. Slowly rotate ONE joint at a time on the physical arm.
       Watch which URDF joint moves and in which direction.
    5. For each motor, decide:
       * If the URDF joint moves the same way: sign = +1.
       * If it moves the opposite way: sign = -1.
       * The offset is the URDF angle when the motor reads 0 degrees.
    6. Edit RIGHT_ARM_MAP in kinematics.py with the values you discover.
    7. Re-run; if the URDF tracks the real arm 1:1, the bridge is correct.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from ..kinematics import MOTOR_NAMES, RIGHT_ARM_MAP, So107Kinematics, motor_pos_to_urdf_q

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2", help="serial port of the right arm")
    parser.add_argument("--id", default="right_white", help="robot id (used to locate calibration file)")
    parser.add_argument("--rate-hz", type=float, default=15.0, help="viewer update rate")
    args = parser.parse_args()

    # use_degrees=True is required: the bridge math assumes motor positions are in degrees.
    config = SO107FollowerConfig(port=args.port, id=args.id, use_degrees=True, cameras={})
    robot = SO107Follower(config)

    # Connect WITHOUT auto-calibration prompt. Calibration is loaded from disk.
    robot.connect(calibrate=False)
    print(f"\nConnected to {robot}")

    # Disable torque so the user can hand-move the arm.
    robot.bus.disable_torque()
    print("Torque disabled. You can now hand-move the arm.\n")

    # Build kinematics + open viewer.
    sk = So107Kinematics(joint_map=RIGHT_ARM_MAP)
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer

    from . import get_meshes_dir, get_urdf_path

    package_dirs = [str(get_meshes_dir()), str(get_meshes_dir().parent)]
    model, collision_model, visual_model = pin.buildModelsFromUrdf(str(get_urdf_path()), package_dirs)
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    print("\nMove ONE joint at a time. Watch which URDF joint moves in the browser.")
    print("If the URDF moves opposite to the real arm for joint J, set its sign = -1.")
    print("If the URDF is consistently offset by N degrees, set its offset_deg = -sign*N.")
    print("Ctrl-C to exit.\n")
    header = "  ".join(f"{n:>13s}" for n in MOTOR_NAMES) + "    tip xyz (m)"
    print(header)
    print("-" * len(header))

    # Clean Ctrl-C
    interrupted = {"v": False}

    def _sigint(*_: object) -> None:
        interrupted["v"] = True

    signal.signal(signal.SIGINT, _sigint)

    period = 1.0 / args.rate_hz
    try:
        while not interrupted["v"]:
            t0 = time.monotonic()
            obs = robot.bus.sync_read("Present_Position")
            motor_pos = {n: float(obs[n]) for n in MOTOR_NAMES}
            q_rad = motor_pos_to_urdf_q(motor_pos, RIGHT_ARM_MAP)
            viz.display(q_rad)
            T = sk.fk_from_motors(motor_pos)
            xyz = T[:3, 3]
            row = "  ".join(f"{motor_pos[n]:+8.2f} deg" for n in MOTOR_NAMES)
            print(f"\r{row}  ({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f})", end="", flush=True)
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        print("\n\nDisconnecting.")
        # Leave torque OFF so the arm stays soft; user re-enables when needed.
        robot.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
