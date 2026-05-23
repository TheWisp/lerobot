#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cartesian-IK tracking quality vs trajectory speed (bimanual SO-107).

Sweeps a 30 mm circle through the bimanual ``CartesianIKController`` +
``make_bimanual_ik_transform`` + ``PinkKinematics`` stack at a range of
waypoint counts (≡ trajectory speeds at the 30 Hz loop rate) and plots
the max position + orientation tracking error against peak EE speed.

Answers the question: how does the iterative IK lag scale with trajectory
speed in the operating regime the user actually inhabits? The shaded
"typical teleop" band on the plot is the rough speed range a Quest hand
covers in normal use.

Run from the repo root::

    .venv/bin/python benchmarks/cartesian_ik_tracking.py

Writes:
    src/lerobot/teleoperators/quest_vr/docs/cartesian_ik_tradeoff.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lerobot.robots.so107_description.cartesian_ik import (
    CartesianIKController,
    make_bimanual_ik_transform,
    make_so107_arm_kinematics,
)
from lerobot.robots.so107_description.joint_alignment import (
    LEFT_ARM_ALIGNMENT,
    MOTOR_NAMES,
    RIGHT_ARM_ALIGNMENT,
)

LOOP_HZ = 30.0
RADIUS_M = 0.030  # 30 mm circle, same trajectory as PR #9's test_pink_ik
URDF_SEED = np.array([0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0])
WAYPOINT_COUNTS = (1024, 512, 256, 128, 64, 32)
WARMUP_TICKS = 20
TYPICAL_TELEOP_CM_S = (1.0, 10.0)


def _motor_seed(alignment) -> np.ndarray:
    sign = np.array([alignment[m].sign for m in MOTOR_NAMES])
    offset = np.array([alignment[m].offset_deg for m in MOTOR_NAMES])
    return (URDF_SEED - offset) / sign


def _shape_deltas(ref_pose: np.ndarray, n: int) -> list[np.ndarray]:
    p = ref_pose[:3, 3]
    flat = np.array([p[0], p[1], 0.0])
    norm = float(np.linalg.norm(flat))
    inward = -flat / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])
    perp = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    perp /= np.linalg.norm(perp)
    r = RADIUS_M
    center = p + r * inward
    return [
        (center + r * (np.cos(2 * np.pi * i / n) * -inward + np.sin(2 * np.pi * i / n) * perp)) - p
        for i in range(n)
    ]


def _run_trajectory(n_waypoints: int) -> tuple[float, float]:
    """Drive both arms through ``n_waypoints`` of the 30 mm circle.

    Returns ``(max position drift [mm], max rotation drift [deg])`` over
    the trajectory, excluding a warm-up window so the IK can converge from
    its initial seed before we measure.
    """
    q_left = _motor_seed(LEFT_ARM_ALIGNMENT)
    q_right = _motor_seed(RIGHT_ARM_ALIGNMENT)
    gripper_idx = MOTOR_NAMES.index("gripper")
    grip_left = float(q_left[gripper_idx])
    grip_right = float(q_right[gripper_idx])

    left_kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    right_kin = make_so107_arm_kinematics(RIGHT_ARM_ALIGNMENT)
    ref_left = left_kin.forward_kinematics(q_left)
    ref_right = right_kin.forward_kinematics(q_right)

    def _ctrl(kin, q):
        return CartesianIKController(
            kinematics=kin,
            motor_names=list(MOTOR_NAMES),
            q_init=q,
            workspace_min=(-10.0, -10.0, -10.0),
            workspace_max=(10.0, 10.0, 10.0),
            max_ee_step_m=10.0,
        )

    transform = make_bimanual_ik_transform(_ctrl(left_kin, q_left), _ctrl(right_kin, q_right))
    dl = _shape_deltas(ref_left, n_waypoints)
    dr = _shape_deltas(ref_right, n_waypoints)

    pos_err: list[float] = []
    rot_err: list[float] = []
    for dl_i, dr_i in zip(dl, dr, strict=True):
        action = {
            "left_enabled": 1.0,
            "left_target_x": float(dl_i[0]),
            "left_target_y": float(dl_i[1]),
            "left_target_z": float(dl_i[2]),
            "left_target_wx": 0.0,
            "left_target_wy": 0.0,
            "left_target_wz": 0.0,
            "left_gripper_pos": grip_left,
            "right_enabled": 1.0,
            "right_target_x": float(dr_i[0]),
            "right_target_y": float(dr_i[1]),
            "right_target_z": float(dr_i[2]),
            "right_target_wx": 0.0,
            "right_target_wy": 0.0,
            "right_target_wz": 0.0,
            "right_gripper_pos": grip_right,
        }
        out = transform(action)
        q_left_out = np.array([out[f"left_{m}.pos"] for m in MOTOR_NAMES])
        fk_left = left_kin.forward_kinematics(q_left_out)
        exp_left = ref_left[:3, 3] + dl_i
        pos_err.append(float(np.linalg.norm(fk_left[:3, 3] - exp_left)))
        r_err = ref_left[:3, :3].T @ fk_left[:3, :3]
        cos_a = max(-1.0, min(1.0, (float(np.trace(r_err)) - 1.0) * 0.5))
        rot_err.append(float(np.degrees(np.arccos(cos_a))))

    return max(pos_err[WARMUP_TICKS:]) * 1000.0, max(rot_err[WARMUP_TICKS:])


def main() -> None:
    speeds: list[float] = []
    pos_errs: list[float] = []
    rot_errs: list[float] = []

    print(f"{'n_wp':>5}  {'speed':>10}  {'pos_drift':>10}  {'rot_drift':>10}")
    print(f"{'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for n in WAYPOINT_COUNTS:
        # Peak EE speed on the circle: circumference / cycle_period
        # = 2π r / (n / LOOP_HZ) cm/s -> ×100 for cm.
        speed_cm_s = 2 * np.pi * RADIUS_M * LOOP_HZ / n * 100.0
        pos_mm, rot_deg = _run_trajectory(n)
        speeds.append(speed_cm_s)
        pos_errs.append(pos_mm)
        rot_errs.append(rot_deg)
        print(f"{n:>5}  {speed_cm_s:>7.2f} cm/s  {pos_mm:>7.2f} mm  {rot_deg:>7.2f} deg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax in (ax1, ax2):
        ax.axvspan(*TYPICAL_TELEOP_CM_S, alpha=0.15, color="green", label="typical teleop")
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Peak EE speed (cm/s, log)")
        ax.legend(loc="upper left")

    ax1.plot(speeds, pos_errs, "o-", linewidth=2)
    ax1.set_ylabel("Max position drift (mm)")
    ax1.set_title("Position tracking — 30 mm circle, bimanual SO-107")

    ax2.plot(speeds, rot_errs, "o-", linewidth=2, color="C1")
    ax2.set_ylabel("Max rotation drift (deg)")
    ax2.set_title("Orientation tracking — 30 mm circle, bimanual SO-107")

    plt.tight_layout()
    out_path = Path("src/lerobot/teleoperators/quest_vr/docs/cartesian_ik_tradeoff.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nplot -> {out_path}")


if __name__ == "__main__":
    main()
