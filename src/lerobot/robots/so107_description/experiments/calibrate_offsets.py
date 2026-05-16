"""
Interactive offset tuning for the SO-107 motor->URDF bridge.

The URDF in the browser ALWAYS follows the real arm in real time, using
the current (sign, offset) mapping for each joint. For each joint, you
adjust an "offset" slider until the URDF link visually matches the real
link. When all 7 joints line up, save the resulting RIGHT_ARM_MAP.

Run AFTER signs have been set via motor_to_viewer.py. Signs are kept;
this tool only tunes offsets.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.calibrate_offsets \\
        --port /dev/ttyACM2 --id right_white
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from .. import get_meshes_dir, get_urdf_path
from ..kinematics import MOTOR_NAMES, RIGHT_ARM_MAP, JointMap, motor_pos_to_urdf_q

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM2")
    parser.add_argument("--id", default="right_white")
    parser.add_argument("--poll-hz", type=float, default=15.0)
    args = parser.parse_args()

    package_dirs = [str(get_meshes_dir()), str(get_meshes_dir().parent)]
    model, _, visual_model = pin.buildModelsFromUrdf(str(get_urdf_path()), package_dirs)
    viz = MeshcatVisualizer(model, _, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    print("\nMeshCat viewer opened.\n")

    robot = SO107Follower(SO107FollowerConfig(port=args.port, id=args.id, use_degrees=True, cameras={}))
    robot.connect(calibrate=False)
    robot.bus.disable_torque()
    print(f"Connected to {robot}. Torque disabled — arm is hand-movable.\n")

    import tkinter as tk
    from tkinter import font as tkfont, ttk

    FONT_SIZE = 14
    root = tk.Tk()
    root.title("SO-107 offset tuning — URDF tracks the real arm")
    root.tk.call("tk", "scaling", 2.0)
    for fname in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkFixedFont"):
        with contextlib.suppress(tk.TclError):
            tkfont.nametofont(fname).configure(size=FONT_SIZE)
    mono = tkfont.Font(family="TkFixedFont", size=FONT_SIZE)
    style = ttk.Style()
    style.configure("Big.TButton", font=("TkDefaultFont", FONT_SIZE), padding=(10, 6))
    style.configure("Big.TLabel", font=("TkDefaultFont", FONT_SIZE))

    # Mutable map; signs preserved from existing RIGHT_ARM_MAP, offsets tunable here.
    current_map: dict[str, JointMap] = dict(RIGHT_ARM_MAP)
    offset_vars = [tk.DoubleVar(value=current_map[n].offset_deg) for n in MOTOR_NAMES]
    motor_strs = [tk.StringVar(value="--") for _ in range(7)]
    urdf_strs = [tk.StringVar(value="--") for _ in range(7)]
    offset_strs = [tk.StringVar(value="") for _ in range(7)]

    def commit_offset(i: int) -> None:
        name = MOTOR_NAMES[i]
        jm = current_map[name]
        current_map[name] = JointMap(sign=jm.sign, offset_deg=offset_vars[i].get())
        offset_strs[i].set(f"{offset_vars[i].get():+8.2f}°")

    for i in range(7):
        commit_offset(i)  # initialize

    last_motor_pos: dict[str, float] = dict.fromkeys(MOTOR_NAMES, 0.0)

    def push_urdf() -> None:
        q_rad = motor_pos_to_urdf_q(last_motor_pos, current_map)
        viz.display(q_rad)
        for i, name in enumerate(MOTOR_NAMES):
            jm = current_map[name]
            urdf_deg = jm.sign * last_motor_pos[name] + jm.offset_deg
            urdf_strs[i].set(f"{urdf_deg:+7.1f}°")

    def poll_motors() -> None:
        try:
            obs = robot.bus.sync_read("Present_Position")
            for name in MOTOR_NAMES:
                last_motor_pos[name] = float(obs[name])
            for i, name in enumerate(MOTOR_NAMES):
                motor_strs[i].set(f"{last_motor_pos[name]:+7.1f}°")
            push_urdf()
        except Exception as e:  # noqa: BLE001
            for i in range(7):
                motor_strs[i].set("read error")
            print(f"motor read failed: {e}")
        root.after(int(1000 / args.poll_hz), poll_motors)

    def on_slider(i: int) -> None:
        commit_offset(i)
        push_urdf()

    # Header.
    hdr = ttk.Frame(root, padding=(10, 6))
    hdr.grid(row=0, column=0, sticky="ew")
    for label, width in (
        ("motor", 15),
        ("real (deg)", 11),
        ("URDF (deg)", 11),
        ("offset slider (deg)", 22),
        ("offset", 12),
    ):
        ttk.Label(hdr, text=label, width=width, style="Big.TLabel").pack(side=tk.LEFT, padx=4)

    for i, name in enumerate(MOTOR_NAMES):
        row = ttk.Frame(root, padding=(10, 4))
        row.grid(row=i + 1, column=0, sticky="ew")
        ttk.Label(row, text=name, width=15, style="Big.TLabel").pack(side=tk.LEFT)
        ttk.Label(row, textvariable=motor_strs[i], width=10, font=mono, anchor="e").pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Label(row, textvariable=urdf_strs[i], width=10, font=mono, anchor="e").pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Scale(
            row,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            length=420,
            variable=offset_vars[i],
            command=lambda *_, i=i: on_slider(i),
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, textvariable=offset_strs[i], width=10, font=mono, anchor="e").pack(
            side=tk.LEFT, padx=8
        )

    def print_and_exit() -> None:
        print("\n" + "=" * 72)
        print("RIGHT_ARM_MAP — paste into kinematics.py:")
        print("=" * 72)
        print("RIGHT_ARM_MAP: dict[str, JointMap] = {")
        for n in MOTOR_NAMES:
            jm = current_map[n]
            pad = " " * (14 - len(n))
            print(f'    "{n}":{pad} JointMap(sign={jm.sign:+.1f}, offset_deg={jm.offset_deg:+8.2f}),')
        print("}")
        root.destroy()

    btn_row = ttk.Frame(root, padding=(10, 12))
    btn_row.grid(row=10, column=0, sticky="ew")
    ttk.Button(btn_row, text="Print & exit", command=print_and_exit, style="Big.TButton").pack(side=tk.LEFT)
    ttk.Label(
        btn_row,
        text="Move the real arm — URDF follows. Drag offset sliders until URDF matches real arm.",
        style="Big.TLabel",
    ).pack(side=tk.LEFT, padx=14)

    poll_motors()
    root.mainloop()

    robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
