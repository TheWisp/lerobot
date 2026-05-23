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

"""On-hardware Cartesian-IK trajectory benchmark (bimanual SO-107).

Drives the actual robot through the same three trajectories the math viz
uses — 50 mm heart, 60 mm circle, 50 mm square — and records what
hardware does on top of the IK floor.

Each shape is traced in the (forward, lateral) horizontal plane at the
seed's height, anchored at ``seed + SHAPE_OFFSET_FORWARD * forward +
SHAPE_OFFSET_UP * up`` so the closest-to-base point clears the desk and
the singular zone. The arm ramps smoothly from seed to that anchor over
``RAMP_TICKS``, traces the shape, and ramps back; only the shape portion
is plotted / scored. ``N_WAYPOINTS = 1024`` keeps per-tick joint steps
under ~0.5° so STS3215 motor following error doesn't dominate.

The bench uses the production ``CartesianIKController`` + bimanual
transform — the same code path the Quest VR teleop installs — so the
workspace clip, EE-step cap, and joint-step backstop are all engaged.

Writes:

- ``trajectory_traces.png`` — commanded EE trace vs achieved EE trace
  (from FK on observed motor positions, left arm). Residual contains
  motor following error, backlash, compliance under load, URDF-vs-
  measured geometry mismatch.
- ``run.npz`` — per-tick raw log; the plot is generated from this.

URDF render of the achieved motion: out of scope here. Use the GUI's
three.js + urdf-loader pipeline (PR #8, ``gui/static/urdf_viz.html``) to
replay the achieved-joint stream from ``run.npz``, then screen-record
the browser via ``ffmpeg x11grab``.

Staging:
- Position the arms at a comfortable mid-range pose, well inside the
  workspace box (``SO107_WORKSPACE_MIN/MAX`` in ``cartesian_ik.py``).
- The script anchors each trajectory at whatever joints it reads at the
  start of the shape — there is no required "home pose."
- Ctrl-C stops cleanly: the controller's last command holds until
  ``disconnect()`` releases torque per the follower's config.

Run from the repo root::

    .venv/bin/python benchmarks/cartesian_ik_hardware_traj.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lerobot.robots.bi_so107_follower import BiSO107Follower, BiSO107FollowerConfig
from lerobot.robots.so107_description.cartesian_ik import (
    SO107_WORKSPACE_MAX,
    SO107_WORKSPACE_MIN,
    CartesianIKController,
    make_bimanual_ik_transform,
)
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger("cartesian_ik_hardware_traj")

# --- User-editable ---------------------------------------------------------

# Path to the bi_so107_follower profile JSON the GUI uses for this rig.
PROFILE_PATH = Path.home() / ".config" / "lerobot" / "robots" / "white.json"

# --- Bench constants -------------------------------------------------------

OUT_DIR = Path("src/lerobot/teleoperators/quest_vr/docs/hardware")
LOOP_HZ = 30.0
# 1024 waypoints over 256 (math-viz default) keeps the per-tick joint step
# at <0.5 deg — well under what STS3215 servos track tightly under load.
# At 256 wp the achieved trace lagged ~20 mm steady-state (4-5 ticks of
# motor delay × the trajectory's tangential speed); going 4x slower drops
# that proportionally toward the math IK floor.
N_WAYPOINTS = 1024
WARMUP_TICKS = 30
SETTLE_TICKS = 15  # hold the starting pose between shapes
RAMP_TICKS = 60  # ~2 s linear ramp into / out of the trajectory anchor

# Push the trajectory anchor away from the staged seed so the closest-to-
# base point of any shape sits comfortably inside the reachable workspace
# and clears the desk. Each shape is traced *around* (seed + offset), with
# the arm ramped smoothly from seed to the anchor before the trace begins
# and back to seed after.
SHAPE_OFFSET_FORWARD_M = 0.05  # +5 cm away from base
SHAPE_OFFSET_UP_M = 0.05  # +5 cm above seed

SHAPES = [
    ("heart 50 mm", "heart", 0.050),
    ("circle 60 mm radius", "circle", 0.060),
    ("square 50 mm side", "square", 0.050),
]


# --- Shapes (mirror benchmarks/cartesian_ik_trajectory_viz.py) -------------


def _heart_unit(n: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, y], axis=1)
    pts -= pts[0]
    return pts / float((pts.max(axis=0) - pts.min(axis=0)).max())


def _plane_basis(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (seed EE, forward, lateral) — the *horizontal* plane at seed height.

    Same horizontal geometry as the math viz / PR #9's bench, but with
    ``forward`` flipped to point *away* from the base (math viz uses
    ``inward`` = toward base). Toward-base on hardware drives the arms
    into singular / out-of-reach configurations near the staging pose;
    forward stays in safely reachable air space and only asks
    ``shoulder_pan`` + wrist joints to do most of the work (the
    fast-tracking joints), leaving ``shoulder_lift`` quiet.
    """
    p = ref[:3, 3]
    flat = np.array([p[0], p[1], 0.0])
    norm = float(np.linalg.norm(flat))
    forward = flat / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    lateral = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    lateral /= np.linalg.norm(lateral)
    return p, forward, lateral


def _shape_deltas(ref: np.ndarray, shape: str, size_m: float, n: int) -> list[np.ndarray]:
    """Per-tick EE deltas tracing ``shape`` in the (forward, lateral) plane.

    All shapes start at the seed (t=0 delta = 0) so the first IK solve
    has no position jump. Trajectory stays at constant z (the seed's
    height) and extends forward / sideways — nothing back toward the
    base, nothing up or down. Re-stage the arms higher to raise the
    whole plane.
    """
    p, forward, lateral = _plane_basis(ref)
    if shape == "circle":
        r = size_m
        center = p + r * forward
        return [
            (center + r * (np.cos(2 * np.pi * i / n) * -forward + np.sin(2 * np.pi * i / n) * lateral)) - p
            for i in range(n)
        ]
    if shape == "heart":
        unit = _heart_unit(n)
        return [size_m * (u[0] * forward + u[1] * lateral) for u in unit]
    s = size_m
    corners = [p, p + s * forward, p + s * forward + s * lateral, p + s * lateral]
    per_edge = n // 4
    out: list[np.ndarray] = []
    for e in range(4):
        a, b = corners[e], corners[(e + 1) % 4]
        for k in range(per_edge):
            out.append((a + (b - a) * (k / per_edge)) - p)
    return out


# --- Profile loading -------------------------------------------------------


def _load_follower_config(profile_path: Path) -> BiSO107FollowerConfig:
    """Load the bi_so107_follower profile JSON (the format the GUI writes).

    Skips cameras: this bench drives motors and renders the URDF post-hoc;
    it does not consume camera frames. Avoiding camera init keeps RealSense
    threads from starting and shaves a couple of seconds off the run.
    """
    if not profile_path.exists():
        raise SystemExit(f"profile not found: {profile_path}")
    data = json.loads(profile_path.read_text())
    if data.get("type") != "bi_so107_follower":
        raise SystemExit(f"{profile_path} is not a bi_so107_follower profile (type={data.get('type')})")
    fields = dict(data.get("fields", {}))
    fields["cameras"] = {}
    return BiSO107FollowerConfig(**fields)


# --- One-tick action -------------------------------------------------------


def _make_action(dl: np.ndarray, dr: np.ndarray, grip_l: float, grip_r: float) -> dict:
    return {
        "left_enabled": 1.0,
        "left_target_x": float(dl[0]),
        "left_target_y": float(dl[1]),
        "left_target_z": float(dl[2]),
        "left_target_wx": 0.0,
        "left_target_wy": 0.0,
        "left_target_wz": 0.0,
        "left_gripper_pos": grip_l,
        "right_enabled": 1.0,
        "right_target_x": float(dr[0]),
        "right_target_y": float(dr[1]),
        "right_target_z": float(dr[2]),
        "right_target_wx": 0.0,
        "right_target_wy": 0.0,
        "right_target_wz": 0.0,
        "right_gripper_pos": grip_r,
    }


def _joints_from_obs(obs: dict, prefix: str) -> np.ndarray:
    return np.array([float(obs[f"{prefix}{m}.pos"]) for m in MOTOR_NAMES])


def _hold_pose(follower: BiSO107Follower, q_l: np.ndarray, q_r: np.ndarray, ticks: int) -> None:
    cmd = {f"left_{m}.pos": float(q_l[i]) for i, m in enumerate(MOTOR_NAMES)}
    cmd |= {f"right_{m}.pos": float(q_r[i]) for i, m in enumerate(MOTOR_NAMES)}
    period = 1.0 / LOOP_HZ
    for _ in range(ticks):
        t0 = time.time()
        follower.send_action(cmd)
        follower.get_observation()
        time.sleep(max(0.0, period - (time.time() - t0)))


# --- Per-shape drive -------------------------------------------------------


def _run_shape(
    follower: BiSO107Follower,
    transform,
    left_kin,
    right_kin,
    left_ctrl,
    right_ctrl,
    label: str,
    shape: str,
    size_m: float,
) -> dict:
    """Drive one shape; return per-tick log dict."""
    # Anchor the trajectory at whatever the arm has settled to *now* — the
    # commanded "home" the controller was originally seeded with can have
    # drifted by several degrees under gravity / soft hold, and re-seeding
    # from a stale home means the controller's clutch-engage FK reference
    # doesn't match the trajectory anchor → IK starts off-balance and the
    # joint-step backstop fires for many ticks.
    obs = follower.get_observation()
    q_l_start = _joints_from_obs(obs, "left_")
    q_r_start = _joints_from_obs(obs, "right_")
    for ctrl, q in ((left_ctrl, q_l_start), (right_ctrl, q_r_start)):
        ctrl._q_last = q.copy()
        ctrl._prev_enabled = False
        ctrl._ref = None
        ctrl._last_pos = None
    ref_l = left_kin.forward_kinematics(q_l_start)
    ref_r = right_kin.forward_kinematics(q_r_start)
    grip_idx = MOTOR_NAMES.index("gripper")
    grip_l = float(q_l_start[grip_idx])
    grip_r = float(q_r_start[grip_idx])

    # Per-arm shape offset = (forward, up) in the arm's natural frame. Shifts
    # the trajectory anchor away from base + up, so the trajectory's closest
    # point doesn't graze the desk or the singular region near full retract.
    def _offset(ref: np.ndarray) -> np.ndarray:
        _, forward, _ = _plane_basis(ref)
        return SHAPE_OFFSET_FORWARD_M * forward + SHAPE_OFFSET_UP_M * np.array([0.0, 0.0, 1.0])

    off_l = _offset(ref_l)
    off_r = _offset(ref_r)

    # Build the full per-tick delta list:
    #   [ramp_in (0 -> offset) | shape (offset + shape_delta) | ramp_out (offset -> 0)]
    # The bench's drift metric only uses the shape portion (shape_start ..
    # shape_end), so warmup is the ramp + the first WARMUP_TICKS of the shape.
    shape_l = [d + off_l for d in _shape_deltas(ref_l, shape, size_m, N_WAYPOINTS)]
    shape_r = [d + off_r for d in _shape_deltas(ref_r, shape, size_m, N_WAYPOINTS)]
    ramp_in_l = [off_l * (i + 1) / RAMP_TICKS for i in range(RAMP_TICKS)]
    ramp_in_r = [off_r * (i + 1) / RAMP_TICKS for i in range(RAMP_TICKS)]
    ramp_out_l = [off_l * (RAMP_TICKS - i - 1) / RAMP_TICKS for i in range(RAMP_TICKS)]
    ramp_out_r = [off_r * (RAMP_TICKS - i - 1) / RAMP_TICKS for i in range(RAMP_TICKS)]
    deltas_l = ramp_in_l + shape_l + ramp_out_l
    deltas_r = ramp_in_r + shape_r + ramp_out_r
    shape_start = RAMP_TICKS
    shape_end = RAMP_TICKS + N_WAYPOINTS

    period = 1.0 / LOOP_HZ
    cmd_ee_l, ach_ee_l, cmd_j_l, ach_j_l = [], [], [], []
    cmd_ee_r, ach_ee_r, cmd_j_r, ach_j_r = [], [], [], []
    misses = 0

    for dl, dr in zip(deltas_l, deltas_r, strict=True):
        t0 = time.time()
        joint_action = transform(_make_action(dl, dr, grip_l, grip_r))
        follower.send_action(joint_action)
        obs = follower.get_observation()

        ach_l = _joints_from_obs(obs, "left_")
        ach_r = _joints_from_obs(obs, "right_")
        cmd_l = np.array([joint_action[f"left_{m}.pos"] for m in MOTOR_NAMES])
        cmd_r = np.array([joint_action[f"right_{m}.pos"] for m in MOTOR_NAMES])

        cmd_ee_l.append(ref_l[:3, 3] + dl)
        cmd_ee_r.append(ref_r[:3, 3] + dr)
        ach_ee_l.append(left_kin.forward_kinematics(ach_l)[:3, 3])
        ach_ee_r.append(right_kin.forward_kinematics(ach_r)[:3, 3])
        cmd_j_l.append(cmd_l)
        cmd_j_r.append(cmd_r)
        ach_j_l.append(ach_l)
        ach_j_r.append(ach_r)

        elapsed = time.time() - t0
        if elapsed > period * 1.2:
            misses += 1
        time.sleep(max(0.0, period - elapsed))

    if misses:
        logger.warning("%s: %d ticks ran long (>20%% over %.0f ms)", label, misses, period * 1000)

    # Drift metric: only the shape portion (skip ramp ticks + first
    # WARMUP_TICKS of the shape so the IK has converged to steady state).
    cmd_arr = np.asarray(cmd_ee_l[shape_start + WARMUP_TICKS : shape_end])
    ach_arr = np.asarray(ach_ee_l[shape_start + WARMUP_TICKS : shape_end])
    pos_err = np.linalg.norm(cmd_arr - ach_arr, axis=1) * 1000.0
    logger.info(
        "%s: max left-arm hardware pos drift = %.2f mm (shape portion, post-warmup)",
        label,
        float(pos_err.max()),
    )

    return {
        "ref_l": ref_l,
        "ref_r": ref_r,
        "shape_start": np.int32(shape_start),
        "shape_end": np.int32(shape_end),
        "cmd_ee_l": np.asarray(cmd_ee_l),
        "ach_ee_l": np.asarray(ach_ee_l),
        "cmd_ee_r": np.asarray(cmd_ee_r),
        "ach_ee_r": np.asarray(ach_ee_r),
        "cmd_j_l": np.asarray(cmd_j_l),
        "ach_j_l": np.asarray(ach_j_l),
        "cmd_j_r": np.asarray(cmd_j_r),
        "ach_j_r": np.asarray(ach_j_r),
    }


# --- Output ----------------------------------------------------------------


def _save_npz(results: dict, path: Path) -> None:
    payload: dict[str, np.ndarray] = {}
    for label, log in results.items():
        slug = label.replace(" ", "_").replace("/", "_")
        for k, v in log.items():
            payload[f"{slug}__{k}"] = np.asarray(v)
    np.savez(path, **payload)


def _plot_results(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(13, 7), height_ratios=(2, 1))
    if len(results) == 1:
        axes = axes.reshape(2, 1)

    for col, (label, log) in enumerate(results.items()):
        ref = log["ref_l"]
        _, forward, lateral = _plane_basis(ref)
        # Plot the shape portion only (skip the ramp-in / ramp-out ticks).
        s = int(log["shape_start"])
        e = int(log["shape_end"])
        d_cmd = log["cmd_ee_l"][s:e] - ref[:3, 3]
        d_ach = log["ach_ee_l"][s:e] - ref[:3, 3]
        cmd_xy = np.stack([d_cmd @ forward, d_cmd @ lateral], axis=1) * 1000.0
        ach_xy = np.stack([d_ach @ forward, d_ach @ lateral], axis=1) * 1000.0
        err_mm = np.linalg.norm(cmd_xy - ach_xy, axis=1)
        max_err = float(err_mm[WARMUP_TICKS:].max())

        ax = axes[0, col]
        ax.plot(cmd_xy[:, 0], cmd_xy[:, 1], color="C0", linewidth=2.5, label="commanded")
        ax.plot(
            ach_xy[:, 0],
            ach_xy[:, 1],
            color="C3",
            linewidth=1.0,
            linestyle="--",
            label="achieved (hardware FK)",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("forward (mm)")
        if col == 0:
            ax.set_ylabel("lateral (mm)")
        ax.set_title(f"{label}\nmax left-arm drift: {max_err:.2f} mm")
        ax.legend(loc="upper right", fontsize=9)

        eax = axes[1, col]
        eax.plot(np.arange(len(err_mm)), err_mm, color="C2", linewidth=1.5)
        eax.set_xlim(0, len(err_mm))
        eax.set_ylim(0, max(1.0, max_err * 1.3))
        eax.grid(alpha=0.3)
        eax.set_xlabel("waypoint")
        if col == 0:
            eax.set_ylabel("drift (mm)")

    fig.suptitle("SO-107 bimanual stack — hardware trajectory traces (left arm)", fontsize=12)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close(fig)


# --- Main ------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_follower_config(PROFILE_PATH)
    follower = BiSO107Follower(cfg)
    logger.info("connecting bi_so107_follower (id=%s)…", cfg.id)
    follower.connect()

    results: dict = {}
    try:
        if follower._ik_kinematics is None:
            raise SystemExit("IK kinematics unavailable on the follower (is pin-pink installed?)")
        left_kin = follower._ik_kinematics["left"]
        right_kin = follower._ik_kinematics["right"]

        obs = follower.get_observation()
        home_l = _joints_from_obs(obs, "left_")
        home_r = _joints_from_obs(obs, "right_")
        logger.info("staged at q_left=%s", np.round(home_l, 2))
        logger.info("staged at q_right=%s", np.round(home_r, 2))

        left_ctrl = CartesianIKController(
            kinematics=left_kin,
            motor_names=list(MOTOR_NAMES),
            q_init=home_l,
            workspace_min=SO107_WORKSPACE_MIN,
            workspace_max=SO107_WORKSPACE_MAX,
        )
        right_ctrl = CartesianIKController(
            kinematics=right_kin,
            motor_names=list(MOTOR_NAMES),
            q_init=home_r,
            workspace_min=SO107_WORKSPACE_MIN,
            workspace_max=SO107_WORKSPACE_MAX,
        )
        transform = make_bimanual_ik_transform(left_ctrl, right_ctrl)

        for label, shape, size_m in SHAPES:
            logger.info("=== %s ===", label)
            # Settle the arms at the staged pose between shapes. The
            # controllers re-seed themselves from observed joints inside
            # _run_shape, so the trajectory anchor and the IK reference
            # always agree even if motors have drifted from the command.
            _hold_pose(follower, home_l, home_r, SETTLE_TICKS)
            results[label] = _run_shape(
                follower, transform, left_kin, right_kin, left_ctrl, right_ctrl, label, shape, size_m
            )
        # Return to staged pose at the end.
        _hold_pose(follower, home_l, home_r, SETTLE_TICKS)
    finally:
        follower.disconnect()
        logger.info("disconnected")

    if results:
        _save_npz(results, OUT_DIR / "run.npz")
        _plot_results(results, OUT_DIR / "trajectory_traces.png")
        logger.info("npz  -> %s", OUT_DIR / "run.npz")
        logger.info("plot -> %s", OUT_DIR / "trajectory_traces.png")
        # URDF render is intentionally out of scope: use the GUI's three.js +
        # urdf-loader pipeline (PR #8, `gui/static/urdf_viz.html`) to replay
        # the achieved-joint stream from run.npz, then screen-record with
        # ffmpeg x11grab per the existing recipe.


if __name__ == "__main__":
    main()
