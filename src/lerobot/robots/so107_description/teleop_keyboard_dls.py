"""
Keyboard teleop using minimum-motion DLS (damped least squares) Jacobian IK.

Why this is different from teleop_keyboard_ikpy: instead of running SLSQP each
tick (which finds globally optimal but can hop between solution branches), we
take ONE Newton step from current motors toward target_ee. By construction
this is the minimum-norm joint delta. No branch hops, no weird pose choices,
smooth motion.

Sim-tuned defaults: step=2mm, damping=0.05, alpha=1.0. At 20Hz this gives
~20mm/sec EE motion with joint jumps ≤3°/tick.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.teleop_keyboard_dls \\
        --port /dev/ttyACM2 --id right_white
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pinocchio as pin

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from . import get_urdf_path
from .kinematics import (
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    So107Kinematics,
    motor_pos_to_urdf_q,
    urdf_q_to_motor_pos,
)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)

KEY_DIR: dict[str, tuple[float, float, float]] = {
    "w": (0.0, -1.0, 0.0),
    "s": (0.0, +1.0, 0.0),
    "a": (-1.0, 0.0, 0.0),
    "d": (+1.0, 0.0, 0.0),
    "q": (0.0, 0.0, +1.0),
    "e": (0.0, 0.0, -1.0),
}


def min_motion_step(
    model: pin.Model,
    data: pin.Data,
    frame_id: int,
    motor_pos: dict[str, float],
    target_ee: np.ndarray,
    damping: float,
) -> tuple[dict[str, float], float]:
    """One DLS Jacobian step toward target_ee. Returns (new_motor_pos, ee_err_mm)."""
    q = motor_pos_to_urdf_q(motor_pos, RIGHT_ARM_MAP)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
    Jp = J[:3, :]
    err = target_ee - data.oMf[frame_id].translation
    err_mm = float(np.linalg.norm(err)) * 1000
    A = Jp @ Jp.T + (damping**2) * np.eye(3)
    q_delta = Jp.T @ np.linalg.solve(A, err)
    q_new = q + q_delta
    return urdf_q_to_motor_pos(q_new, RIGHT_ARM_MAP), err_mm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument("--id", default="right_white")
    parser.add_argument("--step-mm", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--gripper-step", type=float, default=2.0)
    parser.add_argument("--press-cap-mm", type=float, default=80.0)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="EMA on target_q. 1.0 = no smoothing (recommended at step=2mm)",
    )
    args = parser.parse_args()

    step_m = args.step_mm / 1000.0
    config = SO107FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        cameras={},
        max_relative_target=args.max_relative_target,
    )
    robot = SO107Follower(config)
    robot.connect(calibrate=False)
    print(f"Connected to {robot}.\n")

    joint_limits: dict[str, tuple[float, float]] = {}
    for name, c in robot.bus.calibration.items():
        mid = (c.range_min + c.range_max) / 2
        deg_min = (c.range_min - mid) * 360.0 / 4095
        deg_max = (c.range_max - mid) * 360.0 / 4095
        joint_limits[name] = (deg_min + 1.0, deg_max - 1.0)

    # Load pinocchio model once.
    model = pin.buildModelFromUrdf(str(get_urdf_path()))
    data = model.createData()
    frame_id = model.getFrameId("L7_1")
    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)

    obs0 = robot.bus.sync_read("Present_Position")
    motor_pos = {n: float(obs0[n]) for n in MOTOR_NAMES}
    T0 = kin.fk_from_motors(motor_pos)
    target_ee = T0[:3, 3].copy()
    target_q = dict(motor_pos)
    press_anchor_ee: np.ndarray | None = None
    gripper_target = motor_pos["gripper"]
    print(f"Initial EE xyz = {target_ee}")

    log_path = Path(tempfile.gettempdir()) / f"so107_dls_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
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
            "ik_err_mm",
        ]
    )
    t_start = time.monotonic()
    print(f"Logging to {log_path}")

    import tkinter as tk
    from tkinter import font as tkfont, ttk

    FONT_SIZE = 14
    root = tk.Tk()
    root.title("SO-107 DLS teleop")
    root.tk.call("tk", "scaling", 2.0)
    for fname in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkFixedFont"):
        with contextlib.suppress(tk.TclError):
            tkfont.nametofont(fname).configure(size=FONT_SIZE)
    mono = tkfont.Font(family="TkFixedFont", size=FONT_SIZE)

    status = {
        k: tk.StringVar(value="--") for k in ("info", "keys", "motors", "ee_cur", "ee_tgt", "ik", "gripper")
    }
    status["info"].set("DLS min-motion IK. WASD QE move, [] gripper, R reset, Esc quit")
    for i, (label, var) in enumerate(
        [
            ("info", status["info"]),
            ("keys", status["keys"]),
            ("motors", status["motors"]),
            ("EE current", status["ee_cur"]),
            ("EE target", status["ee_tgt"]),
            ("IK", status["ik"]),
            ("gripper", status["gripper"]),
        ]
    ):
        row = ttk.Frame(root, padding=(10, 4))
        row.grid(row=i, column=0, sticky="ew")
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var, font=mono, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

    pressed: set[str] = set()

    def on_keydown(event: tk.Event) -> None:
        c = event.keysym.lower()
        nonlocal target_ee, target_q, press_anchor_ee
        if c == "escape":
            root.destroy()
            return
        if c == "r":
            current = kin.fk_from_motors(motor_pos)
            target_ee = current[:3, 3].copy()
            target_q = dict(motor_pos)
            press_anchor_ee = None
            print(f"  reset target EE to {target_ee}")
            return
        if c in KEY_DIR or c in ("bracketleft", "bracketright"):
            pressed.add(c)

    def on_keyup(event: tk.Event) -> None:
        pressed.discard(event.keysym.lower())

    root.bind("<KeyPress>", on_keydown)
    root.bind("<KeyRelease>", on_keyup)
    root.focus_set()

    def tick() -> None:
        nonlocal target_ee, target_q, gripper_target, motor_pos, press_anchor_ee
        try:
            obs = robot.bus.sync_read("Present_Position")
            mp = {n: float(obs[n]) for n in MOTOR_NAMES}
            motor_pos = mp
            current_T = kin.fk_from_motors(mp)
            current_ee = current_T[:3, 3]

            cmd_dir = np.zeros(3)
            for k in list(pressed):
                if k in KEY_DIR:
                    cmd_dir += np.array(KEY_DIR[k])
                elif k == "bracketleft":
                    gripper_target -= args.gripper_step
                elif k == "bracketright":
                    gripper_target += args.gripper_step
            cmd_norm = float(np.linalg.norm(cmd_dir))
            if cmd_norm > 1e-6:
                cmd_unit = cmd_dir / cmd_norm
                if press_anchor_ee is None:
                    press_anchor_ee = current_ee.copy()
                    target_ee = current_ee.copy()
                target_ee = target_ee + cmd_unit * step_m
                gap = target_ee - press_anchor_ee
                gap_norm = float(np.linalg.norm(gap))
                cap_m = args.press_cap_mm / 1000.0
                if gap_norm > cap_m:
                    target_ee = press_anchor_ee + gap * (cap_m / gap_norm)
            else:
                press_anchor_ee = None

            ik_status = "idle (holding target_q)"
            ik_err_mm = 0.0
            if cmd_norm > 1e-6:
                ik_motors, ik_err_mm = min_motion_step(model, data, frame_id, mp, target_ee, args.damping)
                a = max(0.0, min(1.0, args.smoothing_alpha))
                for n in MOTOR_NAMES:
                    if n != "gripper":
                        target_q[n] = a * ik_motors[n] + (1 - a) * target_q[n]
                ik_status = f"err {ik_err_mm:.2f}mm"
            target_q["gripper"] = gripper_target

            new_q = {}
            for n in MOTOR_NAMES:
                d = target_q[n] - mp[n]
                d = max(-args.max_relative_target * 2, min(args.max_relative_target * 2, d))
                v = mp[n] + d
                lo, hi = joint_limits[n]
                if v > hi:
                    v = hi
                elif v < lo:
                    v = lo
                new_q[n] = v

            action = {f"{n}.pos": new_q[n] for n in MOTOR_NAMES}
            robot.send_action(action)

            status["keys"].set(", ".join(sorted(pressed)) or "(none)")
            status["motors"].set("  ".join(f"{n[:5]}={mp[n]:+6.1f}" for n in MOTOR_NAMES))
            status["ee_cur"].set(f"xyz=({current_ee[0]:+.3f},{current_ee[1]:+.3f},{current_ee[2]:+.3f}) m")
            status["ee_tgt"].set(f"xyz=({target_ee[0]:+.3f},{target_ee[1]:+.3f},{target_ee[2]:+.3f}) m")
            status["ik"].set(ik_status)
            status["gripper"].set(f"{gripper_target:+.1f}°")

            log_writer.writerow(
                [
                    f"{time.monotonic() - t_start:.3f}",
                    "|".join(sorted(pressed)),
                    *(f"{mp[n]:.3f}" for n in MOTOR_NAMES),
                    *(f"{new_q[n]:.3f}" for n in MOTOR_NAMES),
                    f"{current_ee[0]:.4f}",
                    f"{current_ee[1]:.4f}",
                    f"{current_ee[2]:.4f}",
                    f"{target_ee[0]:.4f}",
                    f"{target_ee[1]:.4f}",
                    f"{target_ee[2]:.4f}",
                    f"{ik_err_mm:.3f}",
                ]
            )
        except Exception as e:  # noqa: BLE001
            status["info"].set(f"ERROR: {e}")
            print(f"tick error: {e}")

        root.after(int(1000 / args.rate_hz), tick)

    tick()
    try:
        root.mainloop()
    finally:
        with contextlib.suppress(Exception):
            robot.disconnect()
        log_f.close()
        print(f"\nLog: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
