"""
Simulation-only teleop for fast IK-strategy iteration. No real robot needed.

The "robot" is a Python motor-position vector that evolves toward commands
each tick (with optional max_relative_target clipping to mimic the real arm).
The same control loop as teleop_keyboard.py runs, and the URDF arm in meshcat
visualizes the simulated motor state in real time.

Strategies switchable via --strategy:
    one-shot     placo single-step IK, current behavior (baseline jitter)
    iterated     placo IK iterated to convergence (default)
    jacobian     damped pseudoinverse of Jacobian, velocity-based (no IK redundancy)

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.teleop_sim --strategy iterated
    .venv/bin/python -m lerobot.robots.so107_description.teleop_sim --strategy jacobian
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from .. import get_meshes_dir, get_urdf_path
from ..kinematics import (
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    So107Kinematics,
    motor_pos_to_urdf_q,
    urdf_q_to_motor_pos,
)

# EE direction in base frame per movement key.
KEY_DIR: dict[str, tuple[float, float, float]] = {
    "w": (0.0, -1.0, 0.0),
    "s": (0.0, +1.0, 0.0),
    "a": (-1.0, 0.0, 0.0),
    "d": (+1.0, 0.0, 0.0),
    "q": (0.0, 0.0, +1.0),
    "e": (0.0, 0.0, -1.0),
}


class JacobianVelocityController:
    """Velocity-mode IK via damped pseudoinverse of the Jacobian. No global redundancy."""

    def __init__(self, joint_map: dict, tip_frame: str = "L7_1"):
        self.joint_map = joint_map
        self.model = pin.buildModelFromUrdf(str(get_urdf_path()))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(tip_frame)
        assert self.model.nq == 7, f"unexpected nq={self.model.nq}"

    def step(
        self,
        motor_pos: dict,
        ee_step_xyz: np.ndarray,
        damping: float = 0.02,
        position_only: bool = True,
        method: str = "transpose",
    ) -> dict:
        """Given motor positions and a desired EE position step (3-vec, base-frame),
        return new motor positions advancing one tick.

        method:
          - "transpose": q_dot = J^T direction, scaled so predicted EE motion magnitude
            equals ||ee_step_xyz||. No matrix inversion, gracefully reduces motion
            in singular directions. Position-only by construction.
          - "pinv": damped pseudoinverse J^+. Constrains orientation if position_only=False;
            otherwise solves 3D position with damped least squares.
        """
        q_curr = motor_pos_to_urdf_q(motor_pos, self.joint_map)
        pin.computeJointJacobians(self.model, self.data, q_curr)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(self.model, self.data, self.frame_id, pin.LOCAL_WORLD_ALIGNED)

        if method == "transpose":
            # User's "apply force at EE, let physics solve" intuition.
            # q_dot direction = J_pos^T @ ee_dir; rescale so predicted EE motion = ee_step.
            Jp = J[:3, :]
            ee_step_norm = float(np.linalg.norm(ee_step_xyz))
            if ee_step_norm < 1e-9:
                return dict(motor_pos)
            ee_dir = ee_step_xyz / ee_step_norm
            q_unscaled = Jp.T @ ee_dir  # 7
            v_pred = Jp @ q_unscaled  # 3
            v_pred_norm = float(np.linalg.norm(v_pred))
            if v_pred_norm < 1e-9:
                # Singular in this direction — don't move (don't fight the wall).
                return dict(motor_pos)
            q_delta = (ee_step_norm / v_pred_norm) * q_unscaled
        elif position_only:
            Jp = J[:3, :]
            A = Jp @ Jp.T + (damping**2) * np.eye(3)
            q_delta = Jp.T @ np.linalg.solve(A, ee_step_xyz)
        else:
            v_ee = np.concatenate([ee_step_xyz, np.zeros(3)])
            A = J @ J.T + (damping**2) * np.eye(6)
            q_delta = J.T @ np.linalg.solve(A, v_ee)

        q_new = q_curr + q_delta
        return urdf_q_to_motor_pos(q_new, self.joint_map)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["one-shot", "iterated", "jacobian"], default="iterated")
    parser.add_argument("--step-mm", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="simulated motor clip per tick (deg). Set 0 to disable.",
    )
    parser.add_argument(
        "--motor-lag",
        type=float,
        default=0.7,
        help="0..1: fraction of (cmd - state) the motor closes per tick (1 = perfect)",
    )
    parser.add_argument(
        "--start-pose",
        default="calibrated",
        choices=["calibrated", "zero", "user_initial"],
        help="initial motor state for the sim",
    )
    parser.add_argument("--log-csv", default=None)
    args = parser.parse_args()
    step_m = args.step_mm / 1000.0

    # Pick initial sim motor state.
    if args.start_pose == "calibrated":
        # Matches the user's real-world rest pose at the time of the last log capture.
        motor_pos = {
            "shoulder_pan": 1.98,
            "shoulder_lift": -106.7,
            "elbow_flex": 99.6,
            "forearm_roll": -3.78,
            "wrist_flex": 70.0,
            "wrist_roll": -5.0,
            "gripper": 13.6,
        }
    elif args.start_pose == "zero":
        motor_pos = dict.fromkeys(MOTOR_NAMES, 0.0)
    else:  # user_initial — placeholder, customize if needed
        motor_pos = dict.fromkeys(MOTOR_NAMES, 0.0)

    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)
    jac = JacobianVelocityController(joint_map=RIGHT_ARM_MAP)

    # Meshcat viewer setup.
    package_dirs = [str(get_meshes_dir()), str(get_meshes_dir().parent)]
    model, _, vmodel = pin.buildModelsFromUrdf(str(get_urdf_path()), package_dirs)
    viz = MeshcatVisualizer(model, _, vmodel)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    target_q = dict(motor_pos)
    target_T = kin.fk_from_motors(motor_pos).copy()
    gripper_target = motor_pos["gripper"]
    print(f"strategy={args.strategy}  start tip xyz = {target_T[:3, 3]}")

    log_path = args.log_csv or f"/tmp/so107_sim_{args.strategy}_{dt.datetime.now():%H%M%S}.csv"
    log_path = Path(log_path)
    log_f = log_path.open("w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(
        [
            "t",
            "keys",
            "strategy",
            *(f"sim_{n}" for n in MOTOR_NAMES),
            *(f"cmd_{n}" for n in MOTOR_NAMES),
            "ee_cur_x",
            "ee_cur_y",
            "ee_cur_z",
            "ee_tgt_x",
            "ee_tgt_y",
            "ee_tgt_z",
        ]
    )
    t_start = time.monotonic()

    # Tk UI.
    import tkinter as tk
    from tkinter import font as tkfont, ttk

    FONT_SIZE = 14
    root = tk.Tk()
    root.title(f"SO-107 SIM teleop — strategy={args.strategy}")
    root.tk.call("tk", "scaling", 2.0)
    for fname in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkFixedFont"):
        with contextlib.suppress(tk.TclError):
            tkfont.nametofont(fname).configure(size=FONT_SIZE)
    mono = tkfont.Font(family="TkFixedFont", size=FONT_SIZE)
    status = {k: tk.StringVar(value="--") for k in ("keys", "motors", "ee_cur", "ee_tgt", "gripper", "info")}
    status["info"].set(
        f"strategy={args.strategy}  step={args.step_mm}mm  WASD QE move, [] gripper, R reset, Esc quit"
    )
    for i, (label, var) in enumerate(
        [
            ("info", status["info"]),
            ("keys", status["keys"]),
            ("sim motors", status["motors"]),
            ("EE current", status["ee_cur"]),
            ("EE target", status["ee_tgt"]),
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
        if c == "escape":
            root.destroy()
            return
        if c == "r":
            nonlocal target_T, target_q
            target_T = kin.fk_from_motors(motor_pos).copy()
            target_q = dict(motor_pos)
            print(f"reset target to current EE xyz = {target_T[:3, 3]}")
            return
        if c in KEY_DIR or c in ("bracketleft", "bracketright"):
            pressed.add(c)

    def on_keyup(event: tk.Event) -> None:
        pressed.discard(event.keysym.lower())

    root.bind("<KeyPress>", on_keydown)
    root.bind("<KeyRelease>", on_keyup)
    root.focus_set()

    def tick() -> None:
        nonlocal target_T, target_q, gripper_target, motor_pos
        try:
            # 1. Compute EE delta from held keys.
            ee_step = np.zeros(3)
            target_changed = False
            for k in list(pressed):
                if k in KEY_DIR:
                    ee_step += np.array(KEY_DIR[k]) * step_m
                    target_changed = True
                elif k == "bracketleft":
                    gripper_target -= 2.0
                elif k == "bracketright":
                    gripper_target += 2.0

            # 2. Compute target_q according to strategy.
            if args.strategy == "jacobian":
                # Pure velocity control — no global IK. Step from current sim state.
                if target_changed:
                    new_q = jac.step(motor_pos, ee_step)
                    new_q["gripper"] = gripper_target
                    target_q = new_q
                    # target_T tracks for display
                    target_T = kin.fk_from_motors(target_q).copy()
                else:
                    target_q["gripper"] = gripper_target
            else:
                # Pose-based IK (one-shot or iterated).
                if target_changed:
                    target_T[0, 3] += ee_step[0]
                    target_T[1, 3] += ee_step[1]
                    target_T[2, 3] += ee_step[2]
                    ik_guess = dict(target_q)
                    ik_guess["gripper"] = gripper_target
                    max_iters = 1 if args.strategy == "one-shot" else 20
                    new_q = kin.ik_to_motors(
                        ik_guess,
                        target_T,
                        orientation_weight=args.orientation_weight,
                        max_iters=max_iters,
                    )
                    new_q["gripper"] = gripper_target
                    target_q = new_q
                else:
                    target_q["gripper"] = gripper_target

            # 3. Simulate motor dynamics: clip and lag toward target_q.
            new_motor_pos = {}
            for n in MOTOR_NAMES:
                delta = target_q[n] - motor_pos[n]
                if args.max_relative_target > 0:
                    delta = max(-args.max_relative_target, min(args.max_relative_target, delta))
                new_motor_pos[n] = motor_pos[n] + args.motor_lag * delta
            motor_pos = new_motor_pos

            # 4. Update viewer.
            q_rad = motor_pos_to_urdf_q(motor_pos, RIGHT_ARM_MAP)
            viz.display(q_rad)

            # 5. Status + log.
            cur_T = kin.fk_from_motors(motor_pos)
            status["keys"].set(", ".join(sorted(pressed)) or "(none)")
            status["motors"].set("  ".join(f"{n[:5]}={motor_pos[n]:+6.1f}" for n in MOTOR_NAMES))
            status["ee_cur"].set(f"xyz=({cur_T[0, 3]:+.3f},{cur_T[1, 3]:+.3f},{cur_T[2, 3]:+.3f}) m")
            status["ee_tgt"].set(f"xyz=({target_T[0, 3]:+.3f},{target_T[1, 3]:+.3f},{target_T[2, 3]:+.3f}) m")
            status["gripper"].set(f"{gripper_target:+.1f}°")

            log_writer.writerow(
                [
                    f"{time.monotonic() - t_start:.3f}",
                    "|".join(sorted(pressed)),
                    args.strategy,
                    *(f"{motor_pos[n]:.3f}" for n in MOTOR_NAMES),
                    *(f"{target_q[n]:.3f}" for n in MOTOR_NAMES),
                    f"{cur_T[0, 3]:.4f}",
                    f"{cur_T[1, 3]:.4f}",
                    f"{cur_T[2, 3]:.4f}",
                    f"{target_T[0, 3]:.4f}",
                    f"{target_T[1, 3]:.4f}",
                    f"{target_T[2, 3]:.4f}",
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
        log_f.close()
        print(f"\nSim log: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
