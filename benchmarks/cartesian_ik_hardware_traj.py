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

Two phases:

1. **Drive** the actual robot through the same three trajectories the math
   viz uses — 50 mm heart, 60 mm circle, 50 mm square. The bench uses the
   production ``CartesianIKController`` + bimanual transform — the same
   path the Quest VR teleop installs — so the workspace clip, EE-step cap,
   and joint-step backstop are all engaged. Logs per-tick commanded EE,
   commanded joints, and *observed* motor positions.

2. **Render** the SO-107 URDF at each tick's *achieved* joints through
   pinocchio + meshcat, capturing one frame per tick into an MP4. The
   video is the achieved hardware motion shown as a URDF render — the
   visual analog of the math viz's commanded-vs-achieved plot, but for
   the robot's body.

Writes:

- ``trajectory_traces.png`` — commanded EE trace vs achieved EE trace
  (from FK on observed motor positions, left arm). Residual now contains
  motor following error, backlash, compliance under load, URDF-vs-measured
  geometry mismatch.
- ``run.npz`` — per-tick raw log; the plot is generated from this.
- ``run.mp4`` — left-arm URDF render of the achieved joints across all
  three shapes back-to-back.

Staging:
- Position the arms at a comfortable mid-range pose, well inside the
  workspace box (``SO107_WORKSPACE_MIN/MAX`` in ``cartesian_ik.py``).
- The script anchors each trajectory at whatever joint configuration it
  reads at the start of the shape — there is no required "home pose."
- Between shapes the arms hold the starting joints for ~0.5 s to settle.
- Ctrl-C stops cleanly: the controller's last command holds until
  ``disconnect()`` releases torque per the follower's config.

Phase 2 opens a meshcat browser tab. Keep that tab visible / focused
while the render runs (meshcat pauses ``requestAnimationFrame`` on
hidden tabs, which stalls ``captureImage``). The render is single-arm
(left); the plot covers both arms.

Run from the repo root::

    .venv/bin/python benchmarks/cartesian_ik_hardware_traj.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import draccus
import matplotlib.pyplot as plt
import numpy as np

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.bi_so107_follower import BiSO107Follower, BiSO107FollowerConfig
from lerobot.robots.so107_description.cartesian_ik import (
    SO107_WORKSPACE_MAX,
    SO107_WORKSPACE_MIN,
    CartesianIKController,
    make_bimanual_ik_transform,
)
from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger("cartesian_ik_hardware_traj")

# --- User-editable ---------------------------------------------------------

# Path to the bi_so107_follower profile JSON the GUI uses for this rig.
PROFILE_PATH = Path.home() / ".config" / "lerobot" / "robots" / "white.json"

# --- Bench constants -------------------------------------------------------

OUT_DIR = Path("src/lerobot/teleoperators/quest_vr/docs/hardware")
LOOP_HZ = 30.0
N_WAYPOINTS = 256
WARMUP_TICKS = 20
SETTLE_TICKS = 15  # hold the starting pose between shapes

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
    p = ref[:3, 3]
    flat = np.array([p[0], p[1], 0.0])
    norm = float(np.linalg.norm(flat))
    inward = -flat / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])
    perp = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    perp /= np.linalg.norm(perp)
    return p, inward, perp


def _shape_deltas(ref: np.ndarray, shape: str, size_m: float, n: int) -> list[np.ndarray]:
    p, inward, perp = _plane_basis(ref)
    if shape == "circle":
        r = size_m
        center = p + r * inward
        return [
            (center + r * (np.cos(2 * np.pi * i / n) * -inward + np.sin(2 * np.pi * i / n) * perp)) - p
            for i in range(n)
        ]
    if shape == "heart":
        unit = _heart_unit(n)
        return [size_m * (u[0] * inward + u[1] * perp) for u in unit]
    s = size_m
    corners = [p, p + s * inward, p + s * inward + s * perp, p + s * perp]
    per_edge = n // 4
    out: list[np.ndarray] = []
    for e in range(4):
        a, b = corners[e], corners[(e + 1) % 4]
        for k in range(per_edge):
            out.append((a + (b - a) * (k / per_edge)) - p)
    return out


# --- Profile loading -------------------------------------------------------


def _load_follower_config(profile_path: Path) -> BiSO107FollowerConfig:
    """Load the bi_so107_follower profile JSON (the format the GUI writes)."""
    if not profile_path.exists():
        raise SystemExit(f"profile not found: {profile_path}")
    data = json.loads(profile_path.read_text())
    if data.get("type") != "bi_so107_follower":
        raise SystemExit(f"{profile_path} is not a bi_so107_follower profile (type={data.get('type')})")
    fields = dict(data.get("fields", {}))
    cameras_raw = data.get("cameras", {})
    fields["cameras"] = {name: draccus.decode(cfg, CameraConfig) for name, cfg in cameras_raw.items()}
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
    label: str,
    shape: str,
    size_m: float,
) -> dict:
    """Drive one shape; return per-tick log dict."""
    # Anchor the trajectory at the arms' current joints (whatever the user
    # staged). The controller has just been re-seeded to these joints, so
    # FK(q_last) == FK(starting joints) == ref.
    obs = follower.get_observation()
    q_l_start = _joints_from_obs(obs, "left_")
    q_r_start = _joints_from_obs(obs, "right_")
    ref_l = left_kin.forward_kinematics(q_l_start)
    ref_r = right_kin.forward_kinematics(q_r_start)
    grip_idx = MOTOR_NAMES.index("gripper")
    grip_l = float(q_l_start[grip_idx])
    grip_r = float(q_r_start[grip_idx])

    deltas_l = _shape_deltas(ref_l, shape, size_m, N_WAYPOINTS)
    deltas_r = _shape_deltas(ref_r, shape, size_m, N_WAYPOINTS)

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

    pos_err = np.linalg.norm(np.asarray(cmd_ee_l) - np.asarray(ach_ee_l), axis=1) * 1000.0
    logger.info(
        "%s: max left-arm hardware pos drift = %.2f mm (after warmup)",
        label,
        float(pos_err[WARMUP_TICKS:].max()),
    )

    return {
        "ref_l": ref_l,
        "ref_r": ref_r,
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


def _render_urdf_video(results: dict, path: Path, fps: float = LOOP_HZ) -> None:
    """Replay the left arm's achieved joints through pinocchio + meshcat, capture an MP4.

    Loads the SO-107 URDF, opens a meshcat browser tab, displays the
    achieved joint configuration from every tick of every shape (back-to-
    back), grabs one image per tick via the meshcat WebSocket, and
    assembles them into an MP4 with imageio.

    Meshcat needs the browser tab to be open AND visible for the render
    loop to make progress (browsers throttle ``requestAnimationFrame`` on
    hidden tabs, which stalls the WebSocket image-request roundtrip). The
    function prints the URL and prompts for confirmation before starting.
    """
    try:
        import imageio.v2 as imageio
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        from lerobot.robots.so107_description import get_meshes_dir, get_urdf_path
    except Exception:
        logger.exception("URDF render skipped (imports failed)")
        return

    sign = np.array([LEFT_ARM_ALIGNMENT[m].sign for m in MOTOR_NAMES], dtype=float)
    offset = np.array([LEFT_ARM_ALIGNMENT[m].offset_deg for m in MOTOR_NAMES], dtype=float)

    def _motor_to_urdf_rad(q_motor_deg: np.ndarray) -> np.ndarray:
        return np.deg2rad(sign * q_motor_deg + offset)

    # Stitch all shapes' achieved-joint streams together — single render loop.
    all_q_urdf: list[np.ndarray] = []
    shape_starts: list[tuple[str, int]] = []
    for label, log in results.items():
        shape_starts.append((label, len(all_q_urdf)))
        for q in log["ach_j_l"]:
            all_q_urdf.append(_motor_to_urdf_rad(q))

    try:
        urdf_path = str(get_urdf_path())
        mesh_dir = str(get_meshes_dir())
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
        viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
    except Exception:
        logger.exception("URDF render skipped (pinocchio/meshcat setup failed)")
        return

    url = viz.viewer.url() if hasattr(viz.viewer, "url") else "http://127.0.0.1:7000/static/"
    logger.info("meshcat viewer at %s", url)
    logger.info("Open / focus the meshcat tab (a browser tab should have opened).")
    try:
        input("Press Enter to start the URDF render → MP4 (Ctrl-C aborts) ... ")
    except (EOFError, KeyboardInterrupt):
        logger.info("URDF render cancelled by user")
        return

    # Display the seed pose once so meshcat has something to render before
    # we start measuring captureImage roundtrips.
    viz.display(all_q_urdf[0])
    time.sleep(0.5)

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("rendering %d frames -> %s", len(all_q_urdf), path)
    writer = imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=8)
    try:
        for idx, q in enumerate(all_q_urdf):
            viz.display(q)
            img = viz.captureImage()
            writer.append_data(img)
            if idx and idx % 64 == 0:
                logger.info("  rendered %d / %d frames", idx, len(all_q_urdf))
    except Exception:
        logger.exception("URDF render aborted mid-loop")
    finally:
        writer.close()
    logger.info("MP4 written: %s", path)


def _plot_results(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, len(results), figsize=(13, 7), height_ratios=(2, 1))
    if len(results) == 1:
        axes = axes.reshape(2, 1)

    for col, (label, log) in enumerate(results.items()):
        ref = log["ref_l"]
        _, inward, perp = _plane_basis(ref)
        # Project commanded + achieved onto the arm's (inward, perp) plane (mm).
        d_cmd = log["cmd_ee_l"] - ref[:3, 3]
        d_ach = log["ach_ee_l"] - ref[:3, 3]
        cmd_xy = np.stack([d_cmd @ inward, d_cmd @ perp], axis=1) * 1000.0
        ach_xy = np.stack([d_ach @ inward, d_ach @ perp], axis=1) * 1000.0
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
        ax.set_xlabel("inward (mm)")
        if col == 0:
            ax.set_ylabel("perp (mm)")
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
            # Settle at the staged pose, then re-seed the controllers so each
            # shape gets a fresh clutch-engage (FK ref latched, last_pos reset).
            _hold_pose(follower, home_l, home_r, SETTLE_TICKS)
            for ctrl, q in ((left_ctrl, home_l), (right_ctrl, home_r)):
                ctrl._q_last = q.copy()
                ctrl._prev_enabled = False
                ctrl._ref = None
                ctrl._last_pos = None
            results[label] = _run_shape(follower, transform, left_kin, right_kin, label, shape, size_m)
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
        _render_urdf_video(results, OUT_DIR / "run.mp4")


if __name__ == "__main__":
    main()
