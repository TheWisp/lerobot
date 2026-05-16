"""
Visualize the SO-107 URDF in a browser via MeshCat (pinocchio's viewer).

Modes:
    .venv/bin/python -m lerobot.robots.so107_description.viz              # sliders (default)
    .venv/bin/python -m lerobot.robots.so107_description.viz --sweep      # auto-animate
    .venv/bin/python -m lerobot.robots.so107_description.viz --repl       # type joint values
    .venv/bin/python -m lerobot.robots.so107_description.viz --presets    # step through canned poses

The browser tab is at http://127.0.0.1:7000/static/

Note: URDF "zero pose" is just the math reference (all 7 joints = 0 rad), not a
physically reachable target. It's fine in this viewer because there are no servos
to disagree with.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from . import get_meshes_dir, get_urdf_path

JOINT_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
SWEEP_AMPLITUDE_DEG = 60.0  # ±60° from zero
SWEEP_PERIOD_S = 4.0  # seconds per full cycle per joint
FRAME_RATE_HZ = 30.0


def build_visualizer() -> tuple[MeshcatVisualizer, pin.Model, pin.Data]:
    urdf = str(get_urdf_path())
    package_dirs = [str(get_meshes_dir()), str(get_meshes_dir().parent)]
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf, package_dirs)
    print(f"loaded URDF: {urdf}")
    print(f"joints (nq={model.nq}): {JOINT_NAMES}")

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    data = model.createData()
    return viz, model, data


def fk_tip_xyz(model: pin.Model, data: pin.Data, q: np.ndarray) -> np.ndarray:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId("L7_1")].translation.copy()


def run_sweep(viz: MeshcatVisualizer, model: pin.Model, data: pin.Data) -> None:
    """Cycle through joints, moving each through ±SWEEP_AMPLITUDE_DEG. Ctrl-C to quit."""
    print(
        f"\nSweep mode: each joint S1..S7 oscillates ±{SWEEP_AMPLITUDE_DEG}° in turn "
        f"(one cycle = {SWEEP_PERIOD_S}s). Ctrl-C to exit."
    )
    dt = 1.0 / FRAME_RATE_HZ
    amp = math.radians(SWEEP_AMPLITUDE_DEG)
    try:
        while True:
            for joint_idx in range(7):
                t0 = time.monotonic()
                while True:
                    t = time.monotonic() - t0
                    if t > SWEEP_PERIOD_S:
                        break
                    q = np.zeros(7)
                    q[joint_idx] = amp * math.sin(2 * math.pi * t / SWEEP_PERIOD_S)
                    viz.display(q)
                    time.sleep(dt)
                print(f"  swept {JOINT_NAMES[joint_idx]} — tip xyz at apex: {fk_tip_xyz(model, data, q)}")
    except KeyboardInterrupt:
        print("\nbye")


def run_repl(viz: MeshcatVisualizer, model: pin.Model, data: pin.Data) -> None:
    """Type joint values to update the viewer. Examples:
    S2=-45 S3=30       # set named joints, leave others as last
    0 0 -45 30 0 0 0   # all 7 at once, in degrees
    reset              # back to zero
    quit               # exit
    """
    print("\nREPL mode. Examples:")
    print("  S2=-45 S3=30        (set named joints to degrees; others keep last value)")
    print("  0 0 -45 30 0 0 0    (all 7 joints in degrees)")
    print("  reset               (zero pose)")
    print("  quit                (exit)\n")

    q_deg = np.zeros(7)
    while True:
        try:
            line = input("joints (deg)> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            return
        if line == "reset":
            q_deg[:] = 0
        elif "=" in line:
            try:
                for tok in line.split():
                    name, val = tok.split("=")
                    if name not in JOINT_NAMES:
                        raise ValueError(f"unknown joint {name!r}, expected one of {JOINT_NAMES}")
                    q_deg[JOINT_NAMES.index(name)] = float(val)
            except (ValueError, IndexError) as e:
                print(f"  parse error: {e}")
                continue
        else:
            try:
                parts = [float(x) for x in line.split()]
                if len(parts) != 7:
                    print(f"  expected 7 values, got {len(parts)}")
                    continue
                q_deg[:] = parts
            except ValueError as e:
                print(f"  parse error: {e}")
                continue

        q_rad = np.deg2rad(q_deg)
        viz.display(q_rad)
        tip = fk_tip_xyz(model, data, q_rad)
        print(f"  q_deg = {q_deg.tolist()}")
        print(f"  L7_1 tip xyz = ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m")


def run_sliders(viz: MeshcatVisualizer, model: pin.Model, data: pin.Data) -> None:
    """Tkinter window with one slider per joint. Drag → arm updates in browser."""
    import tkinter as tk
    from tkinter import font as tkfont, ttk

    SLIDER_MIN_DEG = -180.0
    SLIDER_MAX_DEG = 180.0
    FONT_SIZE = 16
    SLIDER_LENGTH = 600  # pixels
    SLIDER_THICKNESS = 28  # pixels

    root = tk.Tk()
    root.title("SO-107 joints")

    # HiDPI / readability fixes.
    root.tk.call("tk", "scaling", 2.0)  # 2x logical DPI; tkinter ignores system DPI otherwise
    for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkFixedFont"):
        with contextlib.suppress(tk.TclError):
            tkfont.nametofont(name).configure(size=FONT_SIZE)
    mono = tkfont.Font(family="TkFixedFont", size=FONT_SIZE)

    style = ttk.Style()
    style.configure("Big.Horizontal.TScale", sliderlength=SLIDER_THICKNESS)
    style.configure("Big.TButton", font=("TkDefaultFont", FONT_SIZE), padding=(12, 8))
    style.configure("Big.TLabel", font=("TkDefaultFont", FONT_SIZE))

    q_deg = [tk.DoubleVar(value=0.0) for _ in range(7)]
    val_strs = [tk.StringVar(value="+0.0") for _ in range(7)]
    tip_var = tk.StringVar(value="L7_1 tip xyz = (computing...)")

    def update_pose() -> None:
        vals = np.array([v.get() for v in q_deg])
        for i, v in enumerate(vals):
            val_strs[i].set(f"{v:+7.1f}°")
        q_rad = np.deg2rad(vals)
        viz.display(q_rad)
        tip = fk_tip_xyz(model, data, q_rad)
        tip_var.set(f"L7_1 tip xyz = ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m")

    for i, jname in enumerate(JOINT_NAMES):
        row = ttk.Frame(root, padding=(12, 6))
        row.grid(row=i, column=0, sticky="ew")
        ttk.Label(row, text=jname, width=4, style="Big.TLabel").pack(side=tk.LEFT)
        ttk.Label(row, textvariable=val_strs[i], width=9, font=mono, anchor="e").pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Scale(
            row,
            from_=SLIDER_MIN_DEG,
            to=SLIDER_MAX_DEG,
            orient=tk.HORIZONTAL,
            length=SLIDER_LENGTH,
            variable=q_deg[i],
            command=lambda *_: update_pose(),
            style="Big.Horizontal.TScale",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def reset() -> None:
        for v in q_deg:
            v.set(0.0)
        update_pose()

    btn_row = ttk.Frame(root, padding=(12, 12))
    btn_row.grid(row=8, column=0, sticky="ew")
    ttk.Button(btn_row, text="Reset to zero", command=reset, style="Big.TButton").pack(side=tk.LEFT)
    ttk.Label(btn_row, textvariable=tip_var, font=mono).pack(side=tk.LEFT, padx=20)

    update_pose()
    print("\nSlider window opened. Close it to exit.")
    root.mainloop()


def run_presets(viz: MeshcatVisualizer, model: pin.Model, data: pin.Data) -> None:
    presets: list[tuple[str, list[float]]] = [
        ("zero pose (all 0)", [0, 0, 0, 0, 0, 0, 0]),
        ("S1 = 90deg", [90, 0, 0, 0, 0, 0, 0]),
        ("S2 = -45deg", [0, -45, 0, 0, 0, 0, 0]),
        ("S2=-45, S3=45 (home-ish)", [0, -45, 45, 0, 0, 0, 0]),
        ("S7 = 30deg", [0, 0, 0, 0, 0, 0, 30]),
    ]
    for label, q_deg in presets:
        q = np.deg2rad(q_deg)
        viz.display(q)
        tip = fk_tip_xyz(model, data, q)
        print(f"\n{label}")
        print(f"  L7_1 tip xyz = ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m")
        try:
            input("  Enter for next (Ctrl-C to quit)...")
        except (EOFError, KeyboardInterrupt):
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sweep", action="store_true", help="auto-sweep each joint")
    group.add_argument("--repl", action="store_true", help="type joint values interactively")
    group.add_argument("--presets", action="store_true", help="step through canned poses")
    parser.add_argument("--no-sliders", action="store_true", help="suppress default tkinter slider window")
    args = parser.parse_args()

    viz, model, data = build_visualizer()
    if args.sweep:
        run_sweep(viz, model, data)
    elif args.repl:
        run_repl(viz, model, data)
    elif args.presets:
        run_presets(viz, model, data)
    else:
        run_sliders(viz, model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
