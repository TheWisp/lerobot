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

"""Cartesian-IK tracking quality trade-off (bimanual SO-107 stack).

Sweeps three trajectory shapes through the bimanual ``CartesianIKController``
+ ``make_bimanual_ik_transform`` + ``JointMappedKinematics`` + ``PinkKinematics``
stack across a range of waypoint counts (≡ peak EE speeds at a 30 Hz loop),
and plots max position + rotation drift against peak EE speed.

The shapes mirror ``benchmarks/pink_ik_tracking.py`` on the dependency
branch (PR #9) so the two plots are directly comparable: any drift not
present in the bare-IK plot is contributed by *this* PR's stack —
``CartesianIKController``, ``JointMappedKinematics``, the bimanual
transform — on top of the IK floor.

Both arms run their own copy of the shape per tick (each in its own
arm-natural plane). The reported drift is the left arm — bimanual
symmetry holds in tests, so reporting one keeps the plot directly
comparable to the bare-IK single-arm benchmark.

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
URDF_SEED = np.array([0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 0.0])
WAYPOINT_COUNTS = (1024, 512, 256, 128, 64, 32)
WARMUP_TICKS = 20
TYPICAL_TELEOP_CM_S = (1.0, 10.0)

CIRCLE_RADIUS_M = 0.030
SQUARE_SIDE_M = 0.050
LINE_AMP_M = 0.030  # ±A from the seed along inward; must stay inside reach


def _motor_seed(alignment) -> np.ndarray:
    sign = np.array([alignment[m].sign for m in MOTOR_NAMES])
    offset = np.array([alignment[m].offset_deg for m in MOTOR_NAMES])
    return (URDF_SEED - offset) / sign


def _plane_basis(ref_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = ref_pose[:3, 3]
    flat = np.array([p[0], p[1], 0.0])
    norm = float(np.linalg.norm(flat))
    inward = -flat / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])
    perp = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    perp /= np.linalg.norm(perp)
    return p, inward, perp


def _circle_deltas(ref: np.ndarray, n: int) -> list[np.ndarray]:
    p, inward, perp = _plane_basis(ref)
    r = CIRCLE_RADIUS_M
    center = p + r * inward
    return [
        (center + r * (np.cos(2 * np.pi * i / n) * -inward + np.sin(2 * np.pi * i / n) * perp)) - p
        for i in range(n)
    ]


def _square_deltas(ref: np.ndarray, n: int) -> list[np.ndarray]:
    p, inward, perp = _plane_basis(ref)
    s = SQUARE_SIDE_M
    corners = [p, p + s * inward, p + s * inward + s * perp, p + s * perp]
    per_edge = n // 4
    out: list[np.ndarray] = []
    for e in range(4):
        a, b = corners[e], corners[(e + 1) % 4]
        for k in range(per_edge):
            out.append((a + (b - a) * (k / per_edge)) - p)
    return out


def _line_deltas(ref: np.ndarray, n: int) -> list[np.ndarray]:
    _p, inward, _perp = _plane_basis(ref)
    return [LINE_AMP_M * np.sin(2 * np.pi * i / n) * inward for i in range(n)]


SHAPES = {
    "circle 30 mm r": (_circle_deltas, lambda n: 2 * np.pi * CIRCLE_RADIUS_M * LOOP_HZ / n),
    "square 50 mm s": (_square_deltas, lambda n: 4 * SQUARE_SIDE_M * LOOP_HZ / n),
    "line 60 mm amp": (_line_deltas, lambda n: 2 * np.pi * LINE_AMP_M * LOOP_HZ / n),
}


def _run_trajectory(shape_name: str, n_waypoints: int) -> tuple[float, float]:
    """Drive both arms through ``shape_name``; return left arm's drift."""
    make_deltas, _ = SHAPES[shape_name]

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
    dl = make_deltas(ref_left, n_waypoints)
    dr = make_deltas(ref_right, n_waypoints)

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
    results: dict[str, dict[str, list[float]]] = {
        name: {"speed": [], "pos": [], "rot": []} for name in SHAPES
    }

    print(f"{'shape':>16}  {'n_wp':>5}  {'speed':>11}  {'pos_drift':>10}  {'rot_drift':>10}")
    print(f"{'-' * 16}  {'-' * 5}  {'-' * 11}  {'-' * 10}  {'-' * 10}")
    for name, (_, speed_fn) in SHAPES.items():
        for n in WAYPOINT_COUNTS:
            speed_cm_s = speed_fn(n) * 100.0
            pos_mm, rot_deg = _run_trajectory(name, n)
            results[name]["speed"].append(speed_cm_s)
            results[name]["pos"].append(pos_mm)
            results[name]["rot"].append(rot_deg)
            print(f"{name:>16}  {n:>5}  {speed_cm_s:>7.2f} cm/s  {pos_mm:>7.2f} mm  {rot_deg:>7.2f} deg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"circle 30 mm r": "C0", "square 50 mm s": "C1", "line 60 mm amp": "C2"}
    markers = {"circle 30 mm r": "o", "square 50 mm s": "s", "line 60 mm amp": "^"}

    for ax in (ax1, ax2):
        ax.axvspan(*TYPICAL_TELEOP_CM_S, alpha=0.12, color="green", label="typical teleop")
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Peak EE speed (cm/s, log)")

    for name, data in results.items():
        ax1.plot(
            data["speed"], data["pos"], marker=markers[name], color=colors[name], label=name, linewidth=2
        )
        ax2.plot(
            data["speed"], data["rot"], marker=markers[name], color=colors[name], label=name, linewidth=2
        )
    ax1.set_ylabel("Max position drift (mm)")
    ax1.set_title("Position tracking — bimanual SO-107 stack (left arm)")
    ax2.set_ylabel("Max rotation drift (deg)")
    ax2.set_title("Orientation tracking — bimanual SO-107 stack (left arm)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_path = Path("src/lerobot/teleoperators/quest_vr/docs/cartesian_ik_tradeoff.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"\nplot -> {out_path}")


if __name__ == "__main__":
    main()
