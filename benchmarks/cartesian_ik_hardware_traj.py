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
from lerobot.robots.bi_so107_follower_predictive import (
    BiSO107FollowerPredictive,
    BiSO107FollowerPredictiveConfig,
)
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
from lerobot.teleoperators.scripted_ee import (
    ScriptedBimanualEETeleop,
    ScriptedBimanualEETeleopConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger("cartesian_ik_hardware_traj")

# --- User-editable ---------------------------------------------------------

# Path to the bi_so107_follower profile JSON the GUI uses for this rig.
# Picks plain ``bi_so107_follower`` or ``bi_so107_follower_predictive``
# automatically based on the profile's ``type`` field.
PROFILE_PATH = Path.home() / ".config" / "lerobot" / "robots" / "white_pred.json"

# --- Bench constants -------------------------------------------------------

OUT_DIR = Path("src/lerobot/teleoperators/quest_vr/docs/hardware")
LOOP_HZ = 30.0
# 256 waypoints matches the math-viz / PR #9 trajectory test default and
# gives peak EE speeds (~4-5 cm/s) representative of a hand-paced Quest
# teleop. The plain follower needed 1024 wp to hide its motor following
# lag, but the predictive controller compensates for motor τ via the
# 80 ms intent lookahead — slowing the trajectory there just measures
# the gravity floor, not the dynamic tracking the controller exists to
# fix.
N_WAYPOINTS = 256
WARMUP_TICKS = 20
SETTLE_TICKS = 15  # hold the starting pose between shapes
RAMP_TICKS = 30  # ~1 s linear ramp into / out of the trajectory anchor

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


def _load_follower(profile_path: Path) -> BiSO107Follower:
    """Build a BiSO107Follower (plain or predictive) from a GUI profile JSON.

    Skips cameras: this bench drives motors and renders the URDF post-hoc;
    it does not consume camera frames. Avoiding camera init keeps RealSense
    threads from starting and shaves a couple of seconds off the run.
    Profile ``type`` selects the variant — ``bi_so107_follower`` for the
    plain follower (30 Hz push), ``bi_so107_follower_predictive`` for the
    200 Hz controller-thread follower with intent lookahead.
    """
    if not profile_path.exists():
        raise SystemExit(f"profile not found: {profile_path}")
    data = json.loads(profile_path.read_text())
    fields = dict(data.get("fields", {}))
    fields["cameras"] = {}
    # The profile JSON serialises ``calibration_dir`` as a string, but the
    # config dataclass field is ``Path | None`` and bare dataclass
    # construction does not auto-convert. The GUI's loader uses
    # draccus.decode which handles this; this bench skips that for
    # simplicity, so the conversion has to be explicit here.
    if isinstance(fields.get("calibration_dir"), str):
        fields["calibration_dir"] = Path(fields["calibration_dir"])
    type_ = data.get("type")
    if type_ == "bi_so107_follower":
        return BiSO107Follower(BiSO107FollowerConfig(**fields))
    if type_ == "bi_so107_follower_predictive":
        return BiSO107FollowerPredictive(BiSO107FollowerPredictiveConfig(**fields))
    raise SystemExit(f"{profile_path} unsupported follower type: {type_!r}")


# --- One-tick action -------------------------------------------------------


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
    scripted: ScriptedBimanualEETeleop,
    left_kin,
    right_kin,
    label: str,
) -> dict:
    """Loop until the scripted teleop reports ``is_exhausted``; log per-tick.

    The teleop self-paces its trajectory by wall clock — the bench's loop
    just polls ``get_action()`` / ``send_action()`` / ``get_observation()``
    at ``LOOP_HZ`` and logs. The teleop's ``get_action_raw()`` returns
    the current Cartesian EE delta in robot base frame; the follower's
    ``attach_teleop`` has already installed the IK transform (plain
    follower) or the Cartesian-adapter-cache transform (predictive
    follower), so ``get_action()`` returns motor-joint dicts in both
    paths.
    """
    obs = follower.get_observation()
    q_l_start = _joints_from_obs(obs, "left_")
    q_r_start = _joints_from_obs(obs, "right_")
    ref_l = left_kin.forward_kinematics(q_l_start)
    ref_r = right_kin.forward_kinematics(q_r_start)

    period = 1.0 / LOOP_HZ
    cmd_ee_l, ach_ee_l, cmd_j_l, ach_j_l = [], [], [], []
    cmd_ee_r, ach_ee_r, cmd_j_r, ach_j_r = [], [], [], []
    cart_deltas_l: list[np.ndarray] = []
    cart_deltas_r: list[np.ndarray] = []
    misses = 0
    empty_get_action = 0

    # Note: the trajectory has ramp_in / shape / ramp_out structure (see
    # ScriptedBimanualEETeleop.connect). We don't index into the teleop's
    # internal phase markers — instead we log the *raw* per-tick Cartesian
    # delta and figure out which ticks are the shape portion from the
    # teleop's config.
    cfg = scripted.config
    shape_start = cfg.ramp_ticks
    shape_end = cfg.ramp_ticks + cfg.n_waypoints

    while not scripted.is_exhausted:
        t0 = time.time()
        raw = scripted.get_action_raw()
        cart_l = np.array([raw["left_target_x"], raw["left_target_y"], raw["left_target_z"]], dtype=float)
        cart_r = np.array([raw["right_target_x"], raw["right_target_y"], raw["right_target_z"]], dtype=float)
        joint_action = scripted.get_action()
        if joint_action:
            follower.send_action(joint_action)
        else:
            empty_get_action += 1
        obs = follower.get_observation()

        ach_l = _joints_from_obs(obs, "left_")
        ach_r = _joints_from_obs(obs, "right_")
        cmd_l = (
            np.array([joint_action[f"left_{m}.pos"] for m in MOTOR_NAMES])
            if joint_action
            else np.full(len(MOTOR_NAMES), np.nan)
        )
        cmd_r = (
            np.array([joint_action[f"right_{m}.pos"] for m in MOTOR_NAMES])
            if joint_action
            else np.full(len(MOTOR_NAMES), np.nan)
        )

        cmd_ee_l.append(ref_l[:3, 3] + cart_l)
        cmd_ee_r.append(ref_r[:3, 3] + cart_r)
        ach_ee_l.append(left_kin.forward_kinematics(ach_l)[:3, 3])
        ach_ee_r.append(right_kin.forward_kinematics(ach_r)[:3, 3])
        cmd_j_l.append(cmd_l)
        cmd_j_r.append(cmd_r)
        ach_j_l.append(ach_l)
        ach_j_r.append(ach_r)
        cart_deltas_l.append(cart_l)
        cart_deltas_r.append(cart_r)

        elapsed = time.time() - t0
        if elapsed > period * 1.2:
            misses += 1
        time.sleep(max(0.0, period - elapsed))

    if empty_get_action:
        logger.warning(
            "%s: %d ticks got an empty get_action() (adapter cache not warm yet)",
            label,
            empty_get_action,
        )
    if misses:
        logger.warning("%s: %d ticks ran long (>20%% over %.0f ms)", label, misses, period * 1000)

    # Clamp the shape window to whatever the loop actually captured (the
    # while-loop may stop one tick before the ramp-out completes).
    n_captured = len(cmd_ee_l)
    shape_end_clamped = min(shape_end, n_captured)
    cmd_arr = np.asarray(cmd_ee_l[shape_start + WARMUP_TICKS : shape_end_clamped])
    ach_arr = np.asarray(ach_ee_l[shape_start + WARMUP_TICKS : shape_end_clamped])
    pos_err = np.linalg.norm(cmd_arr - ach_arr, axis=1) * 1000.0
    # Split total drift into IK floor vs hardware tracking — same split the
    # plot draws. Lets the run log show the breakdown directly without
    # needing to load the npz.
    cmd_j_arr = np.asarray(cmd_j_l[shape_start + WARMUP_TICKS : shape_end_clamped])
    ach_j_arr = np.asarray(ach_j_l[shape_start + WARMUP_TICKS : shape_end_clamped])
    if len(cmd_j_arr) and not np.any(np.isnan(cmd_j_arr)):
        fk_cmd_arr = np.array([left_kin.forward_kinematics(q)[:3, 3] for q in cmd_j_arr])
        fk_ach_arr = np.array([left_kin.forward_kinematics(q)[:3, 3] for q in ach_j_arr])
        ik_floor_mm = np.linalg.norm(fk_cmd_arr - cmd_arr, axis=1) * 1000.0
        motor_residual_mm = np.linalg.norm(fk_ach_arr - fk_cmd_arr, axis=1) * 1000.0
        z_state_mm = (fk_ach_arr[:, 2] - cmd_arr[:, 2]) * 1000.0
        z_action_mm = (fk_cmd_arr[:, 2] - cmd_arr[:, 2]) * 1000.0
        logger.info(
            "%s: total max %.2f mm  =  IK floor max %.2f mm  +  motor max %.2f mm",
            label,
            float(pos_err.max()),
            float(ik_floor_mm.max()),
            float(motor_residual_mm.max()),
        )
        logger.info(
            "%s: Δz  FK(action) mean %+5.2f / max |%4.2f|  |  FK(state) mean %+5.2f / max |%4.2f| mm",
            label,
            float(z_action_mm.mean()),
            float(np.abs(z_action_mm).max()),
            float(z_state_mm.mean()),
            float(np.abs(z_state_mm).max()),
        )
    else:
        logger.info(
            "%s: total max %.2f mm (no IK split — bench loop produced empty actions)",
            label,
            float(pos_err.max()) if len(pos_err) else float("nan"),
        )

    return {
        "ref_l": ref_l,
        "ref_r": ref_r,
        "shape_start": np.int32(shape_start),
        "shape_end": np.int32(shape_end_clamped),
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


def _plot_results(results: dict, path: Path, left_kin=None) -> None:
    """Two rows per shape: (1) 3D commanded vs achieved trace, (2) Z over waypoint.

    The earlier 2D-in-plane projection silently hid the gravity-induced
    Z sag — the achieved trace looked like a slightly-smaller-than-
    commanded shape, but the commanded shape and the achieved shape lived
    on different planes. The 3D row shows both traces in (forward,
    lateral, z) so the sag is geometrically visible; the Z row shows the
    sag magnitude in mm over the trajectory so it's also numerically
    readable.
    """
    n = len(results)
    fig = plt.figure(figsize=(5 * n, 9))
    gs = fig.add_gridspec(2, n, height_ratios=(2.2, 1.0))

    # Pre-pass: compute global axis ranges so all shape panels use the same
    # scale — otherwise a shape with smaller Z sag visually misleads the
    # eye into thinking it has less drift than a shape with larger Z range
    # plotted on its own auto-scaled axis.
    all_forward: list[float] = []
    all_lateral: list[float] = []
    all_z: list[float] = []
    all_dz: list[float] = []
    for log in results.values():
        ref = log["ref_l"]
        _, fwd, lat = _plane_basis(ref)
        s = int(log["shape_start"])
        e = int(log["shape_end"])
        d_cmd = log["cmd_ee_l"][s:e] - ref[:3, 3]
        d_ach = log["ach_ee_l"][s:e] - ref[:3, 3]
        for d in (d_cmd, d_ach):
            all_forward.extend((d @ fwd) * 1000.0)
            all_lateral.extend((d @ lat) * 1000.0)
            all_z.extend(d[:, 2] * 1000.0)
        all_dz.extend(((d_ach - d_cmd)[:, 2]) * 1000.0)
        if left_kin is not None and log.get("cmd_j_l") is not None:
            fk = np.array([left_kin.forward_kinematics(q)[:3, 3] for q in log["cmd_j_l"][s:e]])
            d_act = fk - ref[:3, 3]
            all_forward.extend((d_act @ fwd) * 1000.0)
            all_lateral.extend((d_act @ lat) * 1000.0)
            all_z.extend(d_act[:, 2] * 1000.0)
            all_dz.extend(((fk - log["cmd_ee_l"][s:e])[:, 2]) * 1000.0)

    def _pad(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
        if hi <= lo:
            return lo - 1, hi + 1
        span = hi - lo
        return lo - span * frac, hi + span * frac

    forward_lim = _pad(min(all_forward), max(all_forward))
    lateral_lim = _pad(min(all_lateral), max(all_lateral))
    z_lim = _pad(min(all_z), max(all_z))
    dz_lim = _pad(min(all_dz), max(all_dz))

    for col, (label, log) in enumerate(results.items()):
        ref = log["ref_l"]
        _, forward, lateral = _plane_basis(ref)
        s = int(log["shape_start"])
        e = int(log["shape_end"])
        d_cmd = log["cmd_ee_l"][s:e] - ref[:3, 3]
        d_ach = log["ach_ee_l"][s:e] - ref[:3, 3]
        cmd_xyz = np.stack([d_cmd @ forward, d_cmd @ lateral, d_cmd[:, 2]], axis=1) * 1000.0
        ach_xyz = np.stack([d_ach @ forward, d_ach @ lateral, d_ach[:, 2]], axis=1) * 1000.0

        # FK of the commanded joints — what the IK *intended* the EE to be.
        # Gap between this and ``cmd_xyz`` (the target the IK was asked to
        # hit) is the IK QP convergence floor — typically ~0 since the IK
        # is essentially perfect. Gap between this and ``ach_xyz`` (FK on
        # motor encoders) is pure hardware tracking — motor following
        # error + gravity sag — without any IK-math contribution.
        fk_action_xyz: np.ndarray | None = None
        if left_kin is not None and log.get("cmd_j_l") is not None:
            cmd_j = log["cmd_j_l"][s:e]
            fk_action = np.array([left_kin.forward_kinematics(q)[:3, 3] for q in cmd_j])
            d_act = fk_action - ref[:3, 3]
            fk_action_xyz = np.stack([d_act @ forward, d_act @ lateral, d_act[:, 2]], axis=1) * 1000.0

        err_3d_mm = np.linalg.norm(cmd_xyz - ach_xyz, axis=1)
        err_inplane_mm = np.linalg.norm(cmd_xyz[:, :2] - ach_xyz[:, :2], axis=1)
        z_drift_mm = ach_xyz[:, 2] - cmd_xyz[:, 2]
        warm = WARMUP_TICKS

        # Row 1: 3D — commanded target, FK(action) (= IK output), FK(state).
        ax3d = fig.add_subplot(gs[0, col], projection="3d")
        ax3d.plot(
            cmd_xyz[:, 0],
            cmd_xyz[:, 1],
            cmd_xyz[:, 2],
            color="C0",
            linewidth=2.2,
            label="commanded EE target",
        )
        if fk_action_xyz is not None:
            ax3d.plot(
                fk_action_xyz[:, 0],
                fk_action_xyz[:, 1],
                fk_action_xyz[:, 2],
                color="C2",
                linewidth=1.2,
                linestyle=":",
                label="FK(action) — what IK asked the motors to do",
            )
        ax3d.plot(
            ach_xyz[:, 0],
            ach_xyz[:, 1],
            ach_xyz[:, 2],
            color="C3",
            linewidth=1.0,
            linestyle="--",
            label="FK(state) — where the motors landed",
        )
        ax3d.set_xlabel("forward (mm)")
        ax3d.set_ylabel("lateral (mm)")
        ax3d.set_zlabel("z (mm)")
        ax3d.set_xlim(*forward_lim)
        ax3d.set_ylim(*lateral_lim)
        ax3d.set_zlim(*z_lim)
        title_lines = [
            label,
            f"FK(state) − target: max 3D {float(err_3d_mm[warm:].max()):.1f} mm "
            f"(in-plane {float(err_inplane_mm[warm:].max()):.1f}, "
            f"|z| {float(np.abs(z_drift_mm[warm:]).max()):.1f})",
        ]
        if fk_action_xyz is not None:
            ik_err = np.linalg.norm(fk_action_xyz - cmd_xyz, axis=1)
            motor_err = np.linalg.norm(ach_xyz - fk_action_xyz, axis=1)
            title_lines.append(
                f"IK math floor: max {float(ik_err[warm:].max()):.2f} mm  |  "
                f"motor tracking: max {float(motor_err[warm:].max()):.1f} mm"
            )
        ax3d.set_title("\n".join(title_lines), fontsize=9)
        ax3d.legend(loc="upper right", fontsize=8)
        ax3d.view_init(elev=18, azim=-65)

        # Row 2: Z over waypoint — commanded (flat 0), FK(action), FK(state).
        zax = fig.add_subplot(gs[1, col])
        x = np.arange(len(cmd_xyz))
        zax.axhline(0.0, color="C0", linewidth=2.0, label="commanded z target (constant)")
        if fk_action_xyz is not None:
            zax.plot(
                x,
                fk_action_xyz[:, 2] - cmd_xyz[:, 2],
                color="C2",
                linewidth=1.5,
                linestyle=":",
                label="FK(action) z − commanded z (IK floor)",
            )
        zax.plot(x, z_drift_mm, color="C3", linewidth=1.5, label="FK(state) z − commanded z")
        zax.set_xlim(0, len(cmd_xyz))
        zax.set_ylim(*dz_lim)
        zax.grid(alpha=0.3)
        zax.set_xlabel("waypoint")
        if col == 0:
            zax.set_ylabel("Δz vs commanded (mm)")
        zax.legend(loc="lower right", fontsize=7)

    fig.suptitle("SO-107 bimanual stack — hardware trajectory traces (left arm, 3D)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Main ------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    follower = _load_follower(PROFILE_PATH)
    logger.info(
        "loaded profile %s -> follower %s (id=%s)",
        PROFILE_PATH.name,
        type(follower).__name__,
        follower.config.id,
    )
    # Log the predictive controller's tuneable knobs so any deviation from
    # the production GUI run is visible side-by-side.
    if isinstance(follower, BiSO107FollowerPredictive):
        c = follower.config
        logger.info(
            "predictive cfg: control_rate_hz=%.0f  lookahead_ms=%.1f (max %.1f)  "
            "velocity_estimator=%s  velocity_window_ms=%.1f  alpha=%.3f  adaptive=%s",
            c.control_rate_hz,
            c.lookahead_ms,
            c.max_lookahead_ms,
            c.velocity_estimator,
            c.velocity_window_ms,
            c.alpha,
            c.adaptive,
        )
    logger.info(
        "follower fields: max_relative_target left=%s right=%s  use_degrees l=%s r=%s  dry_run=%s",
        follower.config.left_arm_max_relative_target,
        follower.config.right_arm_max_relative_target,
        follower.config.left_arm_use_degrees,
        follower.config.right_arm_use_degrees,
        getattr(follower.config, "dry_run", False),
    )

    logger.info("connecting…")
    follower.connect()
    logger.info("connected. is_connected=%s  is_calibrated=%s", follower.is_connected, follower.is_calibrated)

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

        for label, shape, size_m in SHAPES:
            logger.info("=== %s ===", label)
            # Settle the arms at the staged pose, then build a fresh
            # ScriptedBimanualEETeleop for this shape and attach it. The
            # follower's attach_teleop installs the IK path (plain ->
            # in-process transform; predictive -> Cartesian adapter).
            # The teleop self-paces the trajectory by wall clock and
            # flips ``is_exhausted`` after ramp_in + shape + ramp_out.
            _hold_pose(follower, home_l, home_r, SETTLE_TICKS)
            scripted = ScriptedBimanualEETeleop(
                ScriptedBimanualEETeleopConfig(
                    shape=shape,
                    size_m=size_m,
                    n_waypoints=N_WAYPOINTS,
                    ramp_ticks=RAMP_TICKS,
                    loop_hz=LOOP_HZ,
                    offset_forward_m=SHAPE_OFFSET_FORWARD_M,
                    offset_up_m=SHAPE_OFFSET_UP_M,
                )
            )
            scripted.connect()
            follower.attach_teleop(scripted)
            try:
                results[label] = _run_shape(follower, scripted, left_kin, right_kin, label)
            finally:
                follower.attach_teleop(None)
                scripted.disconnect()
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
