"""
Keyboard teleop using a recorded IK lookup table (no IK math at runtime).

Each tick:
    target_ee = current_ee + step * key_direction
    nearest table entry = KDTree query on EE position
    target_motors = nearest entry's motor_pos
    motors slew toward target_motors (clamped by max_relative_target and
    calibrated joint limits)

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.teleop_keyboard_lookup \\
        --port /dev/ttyACM2 --id right_white \\
        --table /tmp/so107_ik_table_*.csv
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
from scipy.spatial import cKDTree

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from ..kinematics import MOTOR_NAMES, RIGHT_ARM_MAP, So107Kinematics

logging.basicConfig(level=logging.WARNING)

KEY_DIR: dict[str, tuple[float, float, float]] = {
    "w": (0.0, -1.0, 0.0),
    "s": (0.0, +1.0, 0.0),
    "a": (-1.0, 0.0, 0.0),
    "d": (+1.0, 0.0, 0.0),
    "q": (0.0, 0.0, +1.0),
    "e": (0.0, 0.0, -1.0),
}


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (motor_arr, ee_arr) from a recorded CSV. Drops duplicates within 5mm/2°."""
    motors_rows = []
    ee_rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            motors_rows.append([float(r[f"motor_{n}"]) for n in MOTOR_NAMES])
            ee_rows.append([float(r[f"ee_{a}"]) for a in "xyz"])
    motors = np.array(motors_rows)
    ee = np.array(ee_rows)
    # Dedupe: keep one sample per 5mm EE cube to reduce table size.
    cells = (ee * 1000 / 5).astype(int)
    seen: dict[tuple, int] = {}
    keep: list[int] = []
    for i, c in enumerate(cells):
        key = tuple(c)
        if key not in seen:
            seen[key] = i
            keep.append(i)
    return motors[keep], ee[keep]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument("--id", default="right_white")
    parser.add_argument("--table", required=True, help="path to recorded IK table CSV")
    parser.add_argument("--step-mm", type=float, default=4.0, help="EE target step per tick (mm)")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--gripper-step", type=float, default=2.0)
    parser.add_argument(
        "--search-radius-mm",
        type=float,
        default=30.0,
        help="radius around current EE to search for forward table entries",
    )
    args = parser.parse_args()

    table_path = Path(args.table)
    motors_arr, ee_arr = load_table(table_path)
    if len(ee_arr) < 100:
        print(f"WARNING: table has only {len(ee_arr)} unique entries — lookup will be coarse.")
    else:
        print(f"Loaded {len(ee_arr)} unique entries from {table_path.name}")

    tree = cKDTree(ee_arr)

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

    # Calibrated joint limits.
    joint_limits: dict[str, tuple[float, float]] = {}
    for name, c in robot.bus.calibration.items():
        mid = (c.range_min + c.range_max) / 2
        deg_min = (c.range_min - mid) * 360.0 / 4095
        deg_max = (c.range_max - mid) * 360.0 / 4095
        joint_limits[name] = (deg_min + 1.0, deg_max - 1.0)

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)
    step_m = args.step_mm / 1000.0

    # Initial state.
    obs0 = robot.bus.sync_read("Present_Position")
    motor_pos = {n: float(obs0[n]) for n in MOTOR_NAMES}
    T0 = kin.fk_from_motors(motor_pos)
    target_ee = T0[:3, 3].copy()
    target_q = dict(motor_pos)  # latched joint-space target (persists across ticks)
    press_anchor_ee: np.ndarray | None = None  # EE at moment of current key press
    gripper_target = motor_pos["gripper"]
    print(f"Initial EE xyz = {target_ee}")

    # Log.
    log_path = Path(f"/tmp/so107_lookup_{dt.datetime.now():%Y%m%d_%H%M%S}.csv")
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
            "lookup_dist_mm",
        ]
    )
    t_start = time.monotonic()
    print(f"Logging to {log_path}")

    import tkinter as tk
    from tkinter import font as tkfont, ttk

    FONT_SIZE = 14
    root = tk.Tk()
    root.title("SO-107 LOOKUP teleop")
    root.tk.call("tk", "scaling", 2.0)
    for fname in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkFixedFont"):
        with contextlib.suppress(tk.TclError):
            tkfont.nametofont(fname).configure(size=FONT_SIZE)
    mono = tkfont.Font(family="TkFixedFont", size=FONT_SIZE)

    status = {
        k: tk.StringVar(value="--")
        for k in ("info", "keys", "motors", "ee_cur", "ee_tgt", "lookup", "gripper")
    }
    status["info"].set(f"Lookup teleop ({len(ee_arr)} entries). WASD QE move, [] gripper, R reset, Esc quit")
    for i, (label, var) in enumerate(
        [
            ("info", status["info"]),
            ("keys", status["keys"]),
            ("motors", status["motors"]),
            ("EE current", status["ee_cur"]),
            ("EE target", status["ee_tgt"]),
            ("lookup gap", status["lookup"]),
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
            target_q = dict(motor_pos)  # re-anchor joint target to current state
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
            # 1. Read motors.
            obs = robot.bus.sync_read("Present_Position")
            mp = {n: float(obs[n]) for n in MOTOR_NAMES}
            motor_pos = mp
            current_T = kin.fk_from_motors(mp)
            current_ee = current_T[:3, 3]
            current_motor_vec = np.array([mp[n] for n in MOTOR_NAMES])

            # 2. LATCH target_ee. While ANY movement key is held, target_ee
            # accumulates step_m in the commanded direction. The cap is
            # anchored to the EE at the START of this press, NOT to the
            # live current_ee — otherwise motor-space nonlinearity that dips
            # current_ee mid-flight would drag target_ee down too, creating
            # a feedback loop where each tick's "forward" is from an ever-
            # lower starting point and the arm slides downward.
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
                # Cap travel from press anchor (per-press hard limit).
                gap = target_ee - press_anchor_ee
                gap_norm = float(np.linalg.norm(gap))
                cap_m = 0.05  # 50 mm max travel in one continuous press
                if gap_norm > cap_m:
                    target_ee = press_anchor_ee + gap * (cap_m / gap_norm)
            else:
                press_anchor_ee = None

            # 4. Find table entry nearest to latched target_ee. Direction
            # filter: must be forward of current EE (no going backward).
            # Only UPDATE target_q when there's an active command; otherwise
            # leave target_q alone so the slew keeps pushing motors back to
            # the last commanded position (anti-sag against gravity).
            best_aligned_mm = 0.0
            lookup_status = "idle (holding last commanded pose)"
            if cmd_norm > 1e-6:
                k = min(20, len(ee_arr))
                dists, idxs = tree.query(target_ee, k=k)
                if np.isscalar(idxs):
                    idxs = np.array([idxs])
                    dists = np.array([dists])
                best_i = None
                best_score = float("inf")
                for d_ee, i in zip(dists, idxs, strict=False):
                    disp = ee_arr[i] - current_ee
                    aligned = float(np.dot(disp, cmd_unit))
                    if aligned < 1e-4:
                        continue
                    score = d_ee * 1000
                    if score < best_score:
                        best_score = score
                        best_i = i
                        best_aligned_mm = aligned * 1000
                if best_i is not None:
                    target_motors_vec = motors_arr[best_i]
                    motor_jump = float(np.linalg.norm(motors_arr[best_i] - current_motor_vec))
                    lookup_status = f"fwd {best_aligned_mm:+.1f}mm  motor_jump {motor_jump:.1f}°  table_gap {best_score:.1f}mm"
                    # Update latched target_q from the lookup.
                    for k_i, n in enumerate(MOTOR_NAMES):
                        target_q[n] = float(target_motors_vec[k_i])
                else:
                    lookup_status = "no forward entry within search"
            # Always sync gripper target.
            target_q["gripper"] = gripper_target

            # 5. Slew motor commands toward target_q with safety clamps.
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

            # 6. Send.
            action = {f"{n}.pos": new_q[n] for n in MOTOR_NAMES}
            robot.send_action(action)

            # 7. Status + log.
            status["keys"].set(", ".join(sorted(pressed)) or "(none)")
            status["motors"].set("  ".join(f"{n[:5]}={mp[n]:+6.1f}" for n in MOTOR_NAMES))
            status["ee_cur"].set(f"xyz=({current_ee[0]:+.3f},{current_ee[1]:+.3f},{current_ee[2]:+.3f}) m")
            status["ee_tgt"].set(f"xyz=({target_ee[0]:+.3f},{target_ee[1]:+.3f},{target_ee[2]:+.3f}) m")
            status["lookup"].set(lookup_status)
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
                    f"{best_aligned_mm:.2f}",
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
