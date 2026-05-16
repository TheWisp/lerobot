"""
Keyboard + IK teleop for the SO-107 right arm.

Keys (window must be focused):
    W / S       -Y / +Y    (forward / back in base frame)
    A / D       -X / +X    (left / right)
    Q / E       +Z / -Z    (up / down)
    [ / ]       close / open gripper
    R           reset target to current EE pose (in case it drifts)
    Esc         exit

Motion is integrated: holding a key keeps moving. The target is held in
base-frame SE(3); orientation is kept constant at whatever it was when teleop
started.

Run AFTER signs+offsets in RIGHT_ARM_MAP are calibrated. Safe defaults:
    --step-mm 2.0  --rate-hz 20  --max-relative-target 5

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.teleop_keyboard \\
        --port /dev/ttyACM2 --id right_white
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import logging
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from ..kinematics import MOTOR_NAMES, RIGHT_ARM_MAP, So107Kinematics
from .teleop_sim import JacobianVelocityController

logging.basicConfig(level=logging.WARNING)

# Base-frame direction per key. Sign convention: see the URDF axes (X right, Y forward, Z up).
KEY_DIR: dict[str, tuple[float, float, float]] = {
    "w": (0.0, -1.0, 0.0),
    "s": (0.0, +1.0, 0.0),
    "a": (-1.0, 0.0, 0.0),
    "d": (+1.0, 0.0, 0.0),
    "q": (0.0, 0.0, +1.0),
    "e": (0.0, 0.0, -1.0),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument("--id", default="right_white")
    parser.add_argument("--step-mm", type=float, default=2.0, help="EE position step per tick (mm)")
    parser.add_argument("--gripper-step", type=float, default=2.0, help="gripper step per tick (degrees)")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="safety cap: max motor-degree change per send. Lower = safer.",
    )
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=1.0,
        help="(unused with --strategy jacobian) IK orientation weight",
    )
    parser.add_argument(
        "--strategy",
        choices=["jacobian", "iterated"],
        default="jacobian",
        help="control law. jacobian = damped pseudoinverse (recommended); iterated = placo pose IK",
    )
    parser.add_argument(
        "--jacobian-damping",
        type=float,
        default=0.02,
        help="DLS damping for jacobian strategy. Higher = smoother but slower.",
    )
    parser.add_argument(
        "--keep-orientation",
        action="store_true",
        help="(only with --ik-method pinv) constrain EE orientation",
    )
    parser.add_argument(
        "--ik-method",
        choices=["transpose", "pinv"],
        default="transpose",
        help="transpose: J^T scaled to commanded step (default, robust to singularities). "
        "pinv: damped pseudoinverse (better orientation preservation, worse in singular regions).",
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help="path to CSV log file (default: /tmp/so107_teleop_<timestamp>.csv)",
    )
    args = parser.parse_args()

    step_m = args.step_mm / 1000.0

    # Robot config: torque ON, use degrees, motor-relative safety cap.
    config = SO107FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        cameras={},
        max_relative_target=args.max_relative_target,
    )
    robot = SO107Follower(config)
    robot.connect(calibrate=False)
    print(f"\nConnected to {robot}. Torque should be enabled.\n")

    # Read calibration to know each motor's true reachable degree range.
    # NEVER command past these or the servo wedges (then needs GUI Recover).
    JOINT_LIMITS: dict[str, tuple[float, float]] = {}
    for name, c in robot.bus.calibration.items():
        rmin = c.range_min
        rmax = c.range_max
        mid = (rmin + rmax) / 2
        max_res = 4095
        deg_min = (rmin - mid) * 360.0 / max_res
        deg_max = (rmax - mid) * 360.0 / max_res
        # Pull in slightly from the literal limits — safety margin so servo
        # never sees a value at the exact boundary.
        margin = 1.0
        JOINT_LIMITS[name] = (deg_min + margin, deg_max - margin)
    print("Joint limits (degrees) from calibration, with 1° safety margin:")
    for n, (lo, hi) in JOINT_LIMITS.items():
        print(f"  {n:14s}: [{lo:+7.2f}, {hi:+7.2f}]")

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)
    jac = JacobianVelocityController(joint_map=RIGHT_ARM_MAP)

    # Initial state: read motors. target_q starts AT motors (so no spurious motion
    # at startup). target_T = FK(motors) is what we'll mutate when keys are pressed.
    obs0 = robot.bus.sync_read("Present_Position")
    motor_pos = {n: float(obs0[n]) for n in MOTOR_NAMES}
    target_q = dict(motor_pos)  # joint-space target sent to motors
    target_T = kin.fk_from_motors(motor_pos).copy()  # SE(3) EE target
    gripper_target = motor_pos["gripper"]
    print(f"Initial EE xyz = {target_T[:3, 3]}, initial gripper = {gripper_target:+.1f}°")

    # CSV log setup.
    log_path = args.log_csv or f"/tmp/so107_teleop_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    log_path = Path(log_path)
    log_f = log_path.open("w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(
        [
            "t",
            "keys",
            *(f"motor_{n}" for n in MOTOR_NAMES),
            *(f"cmd_{n}" for n in MOTOR_NAMES),
            "ee_cur_x",
            "ee_cur_y",
            "ee_cur_z",
            "ee_tgt_x",
            "ee_tgt_y",
            "ee_tgt_z",
            "gripper_target",
        ]
    )
    t_start = time.monotonic()
    print(f"Logging to {log_path}")

    # Tk UI.
    import tkinter as tk
    from tkinter import font as tkfont, ttk

    FONT_SIZE = 14
    root = tk.Tk()
    root.title("SO-107 keyboard teleop")
    root.tk.call("tk", "scaling", 2.0)
    for fname in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkFixedFont"):
        with contextlib.suppress(tk.TclError):
            tkfont.nametofont(fname).configure(size=FONT_SIZE)
    mono = tkfont.Font(family="TkFixedFont", size=FONT_SIZE)

    status_strs = {
        "keys": tk.StringVar(value="(none)"),
        "motors": tk.StringVar(value="--"),
        "ee_cur": tk.StringVar(value="--"),
        "ee_tgt": tk.StringVar(value="--"),
        "gripper": tk.StringVar(value=f"{gripper_target:+.1f}°"),
        "info": tk.StringVar(value="Click window to focus. WASD QE move, [] gripper, R reset, Esc quit."),
    }
    for i, (label, var) in enumerate(
        [
            ("info", status_strs["info"]),
            ("keys held", status_strs["keys"]),
            ("motor pos", status_strs["motors"]),
            ("EE current", status_strs["ee_cur"]),
            ("EE target", status_strs["ee_tgt"]),
            ("gripper", status_strs["gripper"]),
        ]
    ):
        row = ttk.Frame(root, padding=(10, 4))
        row.grid(row=i, column=0, sticky="ew")
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var, font=mono, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

    pressed: set[str] = set()

    def on_keydown(event: tk.Event) -> None:
        c = event.keysym.lower()
        if c == "escape":
            root.destroy()
            return
        if c == "r":
            nonlocal target_T, target_q
            current_T = kin.fk_from_motors(_last_motor_pos)
            target_T = current_T.copy()
            target_q = dict(_last_motor_pos)
            print(f"target reset to current EE xyz = {target_T[:3, 3]}")
            return
        if c in KEY_DIR or c in ("bracketleft", "bracketright"):
            pressed.add(c)

    def on_keyup(event: tk.Event) -> None:
        pressed.discard(event.keysym.lower())

    root.bind("<KeyPress>", on_keydown)
    root.bind("<KeyRelease>", on_keyup)
    root.focus_set()

    # Control loop.
    _last_motor_pos = dict(motor_pos)

    def tick() -> None:
        nonlocal target_T, target_q, gripper_target, _last_motor_pos
        try:
            # 1. Read motors (status only — IK guess comes from target_q).
            obs = robot.bus.sync_read("Present_Position")
            mp = {n: float(obs[n]) for n in MOTOR_NAMES}
            _last_motor_pos = mp

            # 2. Compute EE step from held keys.
            ee_step = np.zeros(3)
            target_changed = False
            for k in list(pressed):
                if k in KEY_DIR:
                    ee_step += np.array(KEY_DIR[k]) * step_m
                    target_changed = True
                elif k == "bracketleft":
                    gripper_target -= args.gripper_step
                elif k == "bracketright":
                    gripper_target += args.gripper_step

            if target_changed:
                if args.strategy == "jacobian":
                    # Velocity-mode IK. Compute Jacobian from target_q (so commands
                    # accumulate smoothly and motors play catch-up), then clamp
                    # target_q to a small buffer ahead of real motors. This stops
                    # the unbounded drift that happens at saturation (target_q
                    # was 140° past reachable in the prior log) while preserving
                    # the motor PID's natural follow-through.
                    new_q = jac.step(
                        target_q,
                        ee_step,
                        damping=args.jacobian_damping,
                        position_only=not args.keep_orientation,
                        method=args.ik_method,
                    )
                    new_q["gripper"] = gripper_target
                    # Per-joint clamp: (a) ±lead_buffer around real motor position
                    # to prevent runaway at saturation, (b) calibrated joint
                    # range to prevent wedging the servo past its limit.
                    lead_buffer = args.max_relative_target * 2.0 if args.max_relative_target > 0 else 10.0
                    for n in MOTOR_NAMES:
                        if n == "gripper":
                            continue
                        gap = new_q[n] - mp[n]
                        if gap > lead_buffer:
                            new_q[n] = mp[n] + lead_buffer
                        elif gap < -lead_buffer:
                            new_q[n] = mp[n] - lead_buffer
                        # Hard joint limit from calibration.
                        lo, hi = JOINT_LIMITS[n]
                        if new_q[n] > hi:
                            new_q[n] = hi
                        elif new_q[n] < lo:
                            new_q[n] = lo
                    # Track target_T for display only.
                    target_T = kin.fk_from_motors(new_q).copy()
                else:
                    target_T[0, 3] += ee_step[0]
                    target_T[1, 3] += ee_step[1]
                    target_T[2, 3] += ee_step[2]
                    ik_guess = dict(target_q)
                    ik_guess["gripper"] = gripper_target
                    new_q = kin.ik_to_motors(ik_guess, target_T, orientation_weight=args.orientation_weight)
                    new_q["gripper"] = gripper_target
                target_q = new_q
            else:
                # No target change — keep gripper synced; arm joints unchanged.
                target_q["gripper"] = gripper_target

            # 3. Always send target_q. Motors track. If no keys held, this is a no-op
            # for the arm joints (commanded value didn't change).
            action = {f"{name}.pos": float(target_q[name]) for name in MOTOR_NAMES}
            robot.send_action(action)

            new_motors = target_q  # for status/log
            # 5. Status.
            cur_T = kin.fk_from_motors(mp)
            status_strs["keys"].set(", ".join(sorted(pressed)) or "(none)")
            status_strs["motors"].set("  ".join(f"{n[:5]}={mp[n]:+6.1f}" for n in MOTOR_NAMES))
            status_strs["ee_cur"].set(f"xyz = ({cur_T[0, 3]:+.3f}, {cur_T[1, 3]:+.3f}, {cur_T[2, 3]:+.3f}) m")
            status_strs["ee_tgt"].set(
                f"xyz = ({target_T[0, 3]:+.3f}, {target_T[1, 3]:+.3f}, {target_T[2, 3]:+.3f}) m"
            )
            status_strs["gripper"].set(f"{gripper_target:+.1f}°")

            # Saturation indicator: cmd-motor gap larger than max_relative_target
            # means motors aren't keeping up (or are at a physical limit).
            sat_threshold = args.max_relative_target * 1.5 if args.max_relative_target > 0 else 8.0
            sat = [n for n in MOTOR_NAMES if n != "gripper" and abs(target_q[n] - mp[n]) > sat_threshold]
            if sat:
                gaps = [(n, target_q[n] - mp[n]) for n in sat]
                status_strs["info"].set("SATURATED: " + ", ".join(f"{n}({g:+.1f}°)" for n, g in gaps))
            else:
                status_strs["info"].set("OK — WASD QE move, [] gripper, R reset, Esc quit")

            # 6. Log row.
            log_writer.writerow(
                [
                    f"{time.monotonic() - t_start:.3f}",
                    "|".join(sorted(pressed)),
                    *(f"{mp[n]:.3f}" for n in MOTOR_NAMES),
                    *(f"{new_motors[n]:.3f}" for n in MOTOR_NAMES),
                    f"{cur_T[0, 3]:.4f}",
                    f"{cur_T[1, 3]:.4f}",
                    f"{cur_T[2, 3]:.4f}",
                    f"{target_T[0, 3]:.4f}",
                    f"{target_T[1, 3]:.4f}",
                    f"{target_T[2, 3]:.4f}",
                    f"{gripper_target:.3f}",
                ]
            )
        except Exception as e:  # noqa: BLE001
            status_strs["info"].set(f"ERROR: {e}")
            print(f"tick error: {e}")

        root.after(int(1000 / args.rate_hz), tick)

    tick()
    try:
        root.mainloop()
    finally:
        with contextlib.suppress(Exception):
            robot.disconnect()
        log_f.close()
        print(f"\nLog written: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
