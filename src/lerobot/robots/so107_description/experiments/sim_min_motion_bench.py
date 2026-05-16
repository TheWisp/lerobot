"""
Headless sim bench: minimum-motion DLS IK + all the latched patterns we built.

Runs a fixed motion script through the sim with various parameter combinations
and reports stability/smoothness metrics. Used to autonomously tune the IK
before deploying to the real arm.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.sim_min_motion_bench
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .. import get_urdf_path
from ..kinematics import (
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    So107Kinematics,
    motor_pos_to_urdf_q,
    urdf_q_to_motor_pos,
)

warnings.filterwarnings("ignore")

# Match the user's actual operating-pose start so sim results transfer.
START_MOTOR_POS = {
    "shoulder_pan": 0.7,
    "shoulder_lift": -78.6,
    "elbow_flex": 95.5,
    "forearm_roll": -11.3,
    "wrist_flex": 44.7,
    "wrist_roll": 3.6,
    "gripper": 13.6,
}

KEY_DIR = {
    "w": (0.0, -1.0, 0.0),
    "s": (0.0, +1.0, 0.0),
    "a": (-1.0, 0.0, 0.0),
    "d": (+1.0, 0.0, 0.0),
    "q": (0.0, 0.0, +1.0),
    "e": (0.0, 0.0, -1.0),
}

# Script: each line "(held_key, duration_ticks)". None = idle.
# Tries 4 cardinal directions with idle gaps to test hold-position.
SCRIPT = [
    (None, 10),  # 0.5s idle, anti-sag check
    ("q", 60),  # 3s up
    (None, 20),  # 1s idle, check hold
    ("e", 60),  # 3s down
    (None, 20),
    ("w", 40),  # 2s forward
    (None, 20),
    ("s", 40),  # 2s back
    (None, 20),
    ("d", 40),  # 2s right
    (None, 20),
    ("a", 40),  # 2s left
    (None, 30),
]


@dataclass
class Params:
    damping: float = 0.05
    smoothing_alpha: float = 1.0  # 1.0 = no smoothing
    step_mm: float = 3.0
    rate_hz: float = 20.0
    motor_lag: float = 0.7
    max_relative_target: float = 5.0
    press_cap_mm: float = 80.0


def min_motion_step(
    model, data, frame_id, motor_pos: dict, target_ee: np.ndarray, damping: float
) -> tuple[dict, float]:
    """One Newton step toward target_ee. Returns (new_motor_pos, ee_err_mm)."""
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


def run(params: Params) -> dict:
    model = pin.buildModelFromUrdf(str(get_urdf_path()))
    data = model.createData()
    frame_id = model.getFrameId("L7_1")
    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)

    motor_pos = dict(START_MOTOR_POS)
    initial_T = kin.fk_from_motors(motor_pos)
    target_ee = initial_T[:3, 3].copy()
    target_q = dict(motor_pos)
    press_anchor: np.ndarray | None = None

    step_m = params.step_mm / 1000.0
    rows = []
    tick_idx = 0
    for key, duration in SCRIPT:
        for _ in range(duration):
            tick_idx += 1
            # Pretend "tk read": find commanded direction
            cmd_dir = np.zeros(3)
            if key in KEY_DIR:
                cmd_dir = np.array(KEY_DIR[key])
            cmd_norm = float(np.linalg.norm(cmd_dir))

            current_T = kin.fk_from_motors(motor_pos)
            current_ee = current_T[:3, 3]

            # Latched target_ee with press anchor.
            if cmd_norm > 1e-6:
                cmd_unit = cmd_dir / cmd_norm
                if press_anchor is None:
                    press_anchor = current_ee.copy()
                    target_ee = current_ee.copy()
                target_ee = target_ee + cmd_unit * step_m
                gap = target_ee - press_anchor
                gap_norm = float(np.linalg.norm(gap))
                cap_m = params.press_cap_mm / 1000.0
                if gap_norm > cap_m:
                    target_ee = press_anchor + gap * (cap_m / gap_norm)
            else:
                press_anchor = None

            # IK: minimum-motion DLS step from CURRENT motors toward target_ee.
            # Update target_q only when a key is held (anti-sag).
            ik_err = 0.0
            if cmd_norm > 1e-6:
                ik_motors, ik_err = min_motion_step(
                    model, data, frame_id, motor_pos, target_ee, params.damping
                )
                # EMA smoothing on target_q.
                a = params.smoothing_alpha
                for n in MOTOR_NAMES:
                    if n != "gripper":
                        target_q[n] = a * ik_motors[n] + (1 - a) * target_q[n]

            # Simulate motor dynamics: slew toward target_q with max_relative_target cap.
            new_mp = {}
            for n in MOTOR_NAMES:
                d = target_q[n] - motor_pos[n]
                if params.max_relative_target > 0:
                    d = max(-params.max_relative_target, min(params.max_relative_target, d))
                new_mp[n] = motor_pos[n] + params.motor_lag * d
            motor_pos = new_mp

            rows.append(
                {
                    "t": tick_idx,
                    "key": key or "",
                    "motor": dict(motor_pos),
                    "target_q": dict(target_q),
                    "ee": current_ee.copy(),
                    "target_ee": target_ee.copy(),
                    "ik_err_mm": ik_err,
                }
            )

    # Metrics.
    motors_no_grip = [n for n in MOTOR_NAMES if n != "gripper"]
    idle_jumps, motion_jumps = [], []
    idle_ee_drift = 0.0
    aligned_by_key: dict[str, list[tuple[float, float]]] = {}  # key -> (aligned, commanded) pairs

    seg_starts = []
    prev_key = ""
    for i, r in enumerate(rows):
        if r["key"] != prev_key:
            seg_starts.append((i, r["key"]))
            prev_key = r["key"]

    # Per-segment direction-aligned EE motion.
    for idx in range(len(seg_starts)):
        s, k = seg_starts[idx]
        e = seg_starts[idx + 1][0] - 1 if idx + 1 < len(seg_starts) else len(rows) - 1
        if k in KEY_DIR:
            dir_v = np.array(KEY_DIR[k])
            d_ee = rows[e]["ee"] - rows[s]["ee"]
            aligned = float(np.dot(d_ee, dir_v)) * 1000
            commanded = (e - s + 1) * params.step_mm
            aligned_by_key.setdefault(k, []).append((aligned, commanded))

    for i in range(1, len(rows)):
        jumps = max(abs(rows[i]["target_q"][n] - rows[i - 1]["target_q"][n]) for n in motors_no_grip)
        if rows[i]["key"]:
            motion_jumps.append(jumps)
        elif not rows[i - 1]["key"]:
            idle_jumps.append(jumps)
            idle_ee_drift += float(np.linalg.norm(rows[i]["ee"] - rows[i - 1]["ee"])) * 1000

    # Direction switch: pose change between IDLE→active and active→IDLE
    # Motor change start->end per segment (for "minimum motion" check)
    per_key_eff = {}
    for k, vals in aligned_by_key.items():
        a_sum = sum(a for a, _ in vals)
        c_sum = sum(c for _, c in vals)
        per_key_eff[k] = (a_sum / c_sum * 100) if c_sum > 0 else 0.0

    return {
        "idle_jump_max_deg": max(idle_jumps) if idle_jumps else 0.0,
        "idle_ee_drift_mm": idle_ee_drift,
        "motion_jump_max_deg": max(motion_jumps) if motion_jumps else 0.0,
        "motion_jump_p99_deg": float(np.percentile(motion_jumps, 99)) if motion_jumps else 0.0,
        "motion_jump_median_deg": float(np.median(motion_jumps)) if motion_jumps else 0.0,
        "per_key_efficiency_pct": per_key_eff,
        "rows": rows,
    }


def report(label: str, m: dict) -> None:
    eff = m["per_key_efficiency_pct"]
    eff_str = " ".join(f"{k}={eff.get(k, 0):4.0f}%" for k in "wsadqe")
    print(
        f"  {label:30s}  idle_jump={m['idle_jump_max_deg']:5.2f}°  idle_drift={m['idle_ee_drift_mm']:5.2f}mm  "
        f"motion_jump max/p99/med={m['motion_jump_max_deg']:5.2f}/{m['motion_jump_p99_deg']:5.2f}/{m['motion_jump_median_deg']:5.2f}°  "
        f"eff: {eff_str}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="run a parameter sweep")
    args = parser.parse_args()

    print("Reference: min-motion DLS Jacobian IK + latched target_ee + press anchor + EMA smoothing.")
    print(f"Script: {SCRIPT}\n")
    print("Thresholds: idle_jump <0.5°, motion_jump_max <10°, idle_drift <1mm, eff > 50%\n")

    base = Params()
    if not args.sweep:
        report("baseline (default params)", run(base))
        return 0

    # Damping sweep
    print("--- damping sweep (alpha=1.0, step=3mm) ---")
    for d in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15]:
        p = Params(damping=d)
        report(f"damping={d:.3f}", run(p))

    # Smoothing sweep
    print("\n--- smoothing-alpha sweep (damping=0.05, step=3mm) ---")
    for a in [1.0, 0.5, 0.3, 0.2, 0.1]:
        p = Params(smoothing_alpha=a)
        report(f"alpha={a:.2f}", run(p))

    # Step size sweep
    print("\n--- step-mm sweep (damping=0.05, alpha=0.3) ---")
    for s in [1.0, 2.0, 3.0, 5.0, 8.0]:
        p = Params(step_mm=s, smoothing_alpha=0.3)
        report(f"step={s}mm", run(p))

    return 0


if __name__ == "__main__":
    sys.exit(main())
