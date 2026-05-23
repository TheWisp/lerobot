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

"""Static-hold action↔state diagnostic on the bimanual SO-107 hardware.

Drives the predictive follower to the far-reach point of the circle/heart
shape (forward = +``HOLD_FORWARD_M``, lateral = 0 relative to the staged
seed) via a smooth ramp, then **holds** there at constant intent for
``HOLD_TICKS`` ticks. Logs ``action`` and ``observation.state`` per
motor per tick.

Answers two questions:

1. **Calibration symmetry.** At steady state (last fraction of the hold),
   ``action == observation.state`` to within ~1 motor-unit on every
   motor? If yes, calibration is symmetric and the action↔state gap we
   see during *motion* is purely dynamic motor following. If a motor
   shows a persistent offset, it has a calibration asymmetry worth
   chasing.
2. **Static gravity / friction floor.** If state catches up to action
   eventually, how fast (settle time, per motor)? If some motors never
   catch up, the residual is static gravity / stiction at that pose —
   the "real" hardware floor at this extreme reach.

Writes:
    src/lerobot/teleoperators/quest_vr/docs/hardware/static_hold.png
    src/lerobot/teleoperators/quest_vr/docs/hardware/static_hold.npz

Run from the repo root::

    .venv/bin/python benchmarks/cartesian_ik_static_hold.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", force=True)
logger = logging.getLogger("cartesian_ik_static_hold")

# --- User-editable ---------------------------------------------------------

PROFILE_PATH = Path.home() / ".config" / "lerobot" / "robots" / "white_pred.json"

# --- Constants -------------------------------------------------------------

OUT_DIR = Path("src/lerobot/teleoperators/quest_vr/docs/hardware")
LOOP_HZ = 30.0
RAMP_TICKS = 30  # 1 s slow ramp to the held pose
HOLD_TICKS = 300  # 10 s static hold at the extreme — long enough that any
# slow motor convergence is visible.
SETTLE_TICKS = 15

# Distance forward (away from base) of the held EE point, relative to the
# staged seed. Matches the circle's far side at size_m = HOLD_FORWARD_M.
# Set to roughly the circle's far reach (2 × radius = 120 mm) shifted by
# the bench's standard +5 cm forward + 5 cm up anchor.
HOLD_FORWARD_M = 0.12
SHAPE_OFFSET_FORWARD_M = 0.05
SHAPE_OFFSET_UP_M = 0.05

# Last fraction of the hold to call "steady state" — averages here are
# the per-motor settled positions used for the calibration-symmetry check.
STEADY_FRACTION = 0.25


def _load_follower(profile_path: Path) -> BiSO107Follower:
    if not profile_path.exists():
        raise SystemExit(f"profile not found: {profile_path}")
    data = json.loads(profile_path.read_text())
    fields = dict(data.get("fields", {}))
    fields["cameras"] = {}
    if isinstance(fields.get("calibration_dir"), str):
        fields["calibration_dir"] = Path(fields["calibration_dir"])
    type_ = data.get("type")
    if type_ == "bi_so107_follower":
        return BiSO107Follower(BiSO107FollowerConfig(**fields))
    if type_ == "bi_so107_follower_predictive":
        return BiSO107FollowerPredictive(BiSO107FollowerPredictiveConfig(**fields))
    raise SystemExit(f"{profile_path}: unsupported type {type_!r}")


def _joints_from_obs(obs: dict, prefix: str) -> np.ndarray:
    return np.array([float(obs[f"{prefix}{m}.pos"]) for m in MOTOR_NAMES])


def _load_deg_per_unit(follower: BiSO107Follower) -> dict[str, np.ndarray]:
    """Per-arm conversion: 1 motor-unit (RANGE_M100_100) → joint degrees.

    The bus normalises ``range_max - range_min`` raw ticks into a [-100,
    +100] motor-unit interval, so 1 unit = (range_max - range_min) / 200
    ticks. With the STS3215's 4096 ticks per revolution that's
    ``(range_max - range_min) / 200 * 360 / 4096`` degrees per unit.
    Calibration files live under ``follower.calibration_dir`` named
    ``{arm.id}.json``.
    """
    out: dict[str, np.ndarray] = {}
    for arm_name, sub in (("left", follower.left_arm), ("right", follower.right_arm)):
        cal_path = sub.calibration_fpath
        if not cal_path.exists():
            out[arm_name] = np.full(len(MOTOR_NAMES), 1.0)  # rough fallback
            continue
        cal = json.loads(cal_path.read_text())
        deg_per_unit = np.array(
            [((cal[m]["range_max"] - cal[m]["range_min"]) / 200.0) * 360.0 / 4096.0 for m in MOTOR_NAMES],
            dtype=float,
        )
        out[arm_name] = deg_per_unit
    return out


def _hold_pose(follower: BiSO107Follower, q_l: np.ndarray, q_r: np.ndarray, ticks: int) -> None:
    cmd = {f"left_{m}.pos": float(q_l[i]) for i, m in enumerate(MOTOR_NAMES)}
    cmd |= {f"right_{m}.pos": float(q_r[i]) for i, m in enumerate(MOTOR_NAMES)}
    period = 1.0 / LOOP_HZ
    for _ in range(ticks):
        t0 = time.time()
        follower.send_action(cmd)
        follower.get_observation()
        time.sleep(max(0.0, period - (time.time() - t0)))


def _drive_hold(
    follower: BiSO107Follower,
    scripted: ScriptedBimanualEETeleop,
) -> dict[str, np.ndarray]:
    """Run the static-hold trajectory; return per-tick action + state logs."""
    period = 1.0 / LOOP_HZ
    cmd_l, cmd_r = [], []
    obs_l, obs_r = [], []
    while not scripted.is_exhausted:
        t0 = time.time()
        action = scripted.get_action()
        if action:
            follower.send_action(action)
        obs = follower.get_observation()
        if action:
            cmd_l.append([action[f"left_{m}.pos"] for m in MOTOR_NAMES])
            cmd_r.append([action[f"right_{m}.pos"] for m in MOTOR_NAMES])
        else:
            cmd_l.append([np.nan] * len(MOTOR_NAMES))
            cmd_r.append([np.nan] * len(MOTOR_NAMES))
        obs_l.append(_joints_from_obs(obs, "left_"))
        obs_r.append(_joints_from_obs(obs, "right_"))
        time.sleep(max(0.0, period - (time.time() - t0)))
    return {
        "cmd_l": np.asarray(cmd_l),
        "cmd_r": np.asarray(cmd_r),
        "obs_l": np.asarray(obs_l),
        "obs_r": np.asarray(obs_r),
    }


def _plot(log: dict[str, np.ndarray], deg_per_unit_by_arm: dict[str, np.ndarray], path: Path) -> None:
    """Per-motor action vs state over time, both arms side by side.

    Each subplot title contextualises the steady-state Δ three ways:
    raw motor units, degrees of joint rotation, and Δ as a fraction of
    the motion the joint actually performed during this hold — so a
    reader can tell at a glance whether the offset is small, medium, or
    large relative to the commanded movement.
    """
    fig, axes = plt.subplots(len(MOTOR_NAMES), 2, figsize=(14, 2.0 * len(MOTOR_NAMES)), sharex=True)
    ramp_end = RAMP_TICKS
    hold_end = RAMP_TICKS + HOLD_TICKS
    n_steady = int(HOLD_TICKS * STEADY_FRACTION)
    steady_start = hold_end - n_steady

    for arm_idx, (arm, cmd_key, obs_key) in enumerate(
        (("left", "cmd_l", "obs_l"), ("right", "cmd_r", "obs_r"))
    ):
        cmd = log[cmd_key]
        obs = log[obs_key]
        dpu = deg_per_unit_by_arm[arm]
        for i, name in enumerate(MOTOR_NAMES):
            ax = axes[i, arm_idx]
            x = np.arange(len(cmd))
            ax.plot(x, cmd[:, i], color="C0", linewidth=1.8, label="action")
            ax.plot(x, obs[:, i], color="C3", linewidth=1.0, linestyle="--", label="state")
            ax.axvspan(ramp_end, hold_end, alpha=0.08, color="gray", label="hold" if i == 0 else None)
            ax.axvspan(steady_start, hold_end, alpha=0.18, color="green", label="steady" if i == 0 else None)
            steady_gap_u = float(np.nanmean(obs[steady_start:hold_end, i] - cmd[steady_start:hold_end, i]))
            steady_gap_deg = steady_gap_u * dpu[i]
            motion_u = float(abs(np.nanmean(cmd[steady_start:hold_end, i]) - cmd[0, i]))
            # Only meaningful when the joint actually moved a non-trivial
            # amount during this hold. Below 5 motor-units the ratio is a
            # divide-by-near-zero artefact and only confuses the reader.
            ratio_str = (
                f"Δ/motion = {abs(steady_gap_u) / motion_u * 100.0:.0f}%"
                if motion_u >= 5.0
                else "Δ/motion = — (joint static)"
            )
            ax.set_title(
                f"{arm} {name}   steady Δ={steady_gap_u:+.2f} u ({steady_gap_deg:+.2f}°)   "
                f"motion this hold: {motion_u:.1f} u   →   {ratio_str}",
                fontsize=9,
            )
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right", fontsize=7)
            if i == len(MOTOR_NAMES) - 1:
                ax.set_xlabel("tick (30 Hz; 1 tick ≈ 33 ms)")

    fig.suptitle(
        f"Static-hold @ +{HOLD_FORWARD_M * 1000:.0f} mm forward, +{SHAPE_OFFSET_UP_M * 1000:.0f} mm up "
        f"(ramp {RAMP_TICKS}t → hold {HOLD_TICKS}t @ 30 Hz; green = last {int(STEADY_FRACTION * 100)}%; "
        f"1 motor unit ≈ 1° on STS3215 arm joints)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    follower = _load_follower(PROFILE_PATH)
    logger.info("loaded profile %s -> %s", PROFILE_PATH.name, type(follower).__name__)
    if isinstance(follower, BiSO107FollowerPredictive):
        c = follower.config
        logger.info(
            "predictive cfg: control_rate_hz=%.0f  lookahead_ms=%.1f  velocity_estimator=%s  adaptive=%s",
            c.control_rate_hz,
            c.lookahead_ms,
            c.velocity_estimator,
            c.adaptive,
        )
    follower.connect()
    logger.info("connected")

    try:
        obs = follower.get_observation()
        home_l = _joints_from_obs(obs, "left_")
        home_r = _joints_from_obs(obs, "right_")
        logger.info("staged at q_left=%s", np.round(home_l, 2))
        logger.info("staged at q_right=%s", np.round(home_r, 2))

        _hold_pose(follower, home_l, home_r, SETTLE_TICKS)
        scripted = ScriptedBimanualEETeleop(
            ScriptedBimanualEETeleopConfig(
                shape="static_hold",
                size_m=HOLD_FORWARD_M,
                n_waypoints=HOLD_TICKS,
                ramp_ticks=RAMP_TICKS,
                loop_hz=LOOP_HZ,
                offset_forward_m=SHAPE_OFFSET_FORWARD_M,
                offset_up_m=SHAPE_OFFSET_UP_M,
            )
        )
        scripted.connect()
        follower.attach_teleop(scripted)
        logger.info(
            "driving static_hold @ +%.0f mm forward + %.0f mm up — ramp %dt + hold %dt (%.1f s)",
            HOLD_FORWARD_M * 1000,
            SHAPE_OFFSET_UP_M * 1000,
            RAMP_TICKS,
            HOLD_TICKS,
            HOLD_TICKS / LOOP_HZ,
        )
        log = _drive_hold(follower, scripted)
        follower.attach_teleop(None)
        scripted.disconnect()

        _hold_pose(follower, home_l, home_r, SETTLE_TICKS)
    finally:
        follower.disconnect()
        logger.info("disconnected")

    # Steady-state summary: per-motor mean(state − action) over the last
    # STEADY_FRACTION of the hold. Reported in three views so the
    # absolute magnitude is interpretable:
    #   * motor units (raw, what the dataset stores) — 1 unit ≈ 1°
    #   * degrees of joint rotation (using the per-arm calibration's
    #     range_min/range_max to convert from RANGE_M100_100)
    #   * fraction of the joint's commanded motion during this hold
    #     (i.e. Δ ÷ |action_at_hold − action_at_seed|) — tells you what
    #     fraction of the actual commanded movement is lost to sag
    hold_end = RAMP_TICKS + HOLD_TICKS
    n_steady = int(HOLD_TICKS * STEADY_FRACTION)
    steady_start = hold_end - n_steady

    # Per-arm motor-unit-to-degrees conversion: 200 motor units span the
    # calibrated mechanical range (which for the STS3215 is encoded as
    # ticks, 4096 ticks per full revolution).
    deg_per_unit_by_arm = _load_deg_per_unit(follower)

    logger.info("=== steady-state Δ(state − action), last %d ticks ===", n_steady)
    for arm, cmd_key, obs_key in (("left", "cmd_l", "obs_l"), ("right", "cmd_r", "obs_r")):
        cmd = log[cmd_key]
        obs = log[obs_key]
        deltas = np.nanmean(obs[steady_start:hold_end] - cmd[steady_start:hold_end], axis=0)
        stds = np.nanstd(obs[steady_start:hold_end] - cmd[steady_start:hold_end], axis=0)
        # Commanded motion = |action at hold| − |action at seed|. Use
        # the first commanded value (post-ramp-in) vs the last (steady).
        cmd_motion = np.abs(np.nanmean(cmd[steady_start:hold_end], axis=0) - cmd[0])
        deg_per_unit = deg_per_unit_by_arm[arm]
        for i, name in enumerate(MOTOR_NAMES):
            d_units = float(deltas[i])
            d_deg = d_units * deg_per_unit[i]
            motion = float(cmd_motion[i])
            if motion >= 5.0:
                ratio_str = f"Δ/motion = {abs(d_units) / motion * 100.0:>3.0f}%"
            else:
                ratio_str = "Δ/motion = —    "  # joint barely moved this hold
            logger.info(
                "  %-5s  %-14s  Δ=%+6.2f u (%+5.2f°)   |  motion %5.1f u   |  %s   |  std %4.2f",
                arm,
                name,
                d_units,
                d_deg,
                motion,
                ratio_str,
                stds[i],
            )

    np.savez(OUT_DIR / "static_hold.npz", **log)
    _plot(log, deg_per_unit_by_arm, OUT_DIR / "static_hold.png")
    logger.info("npz  -> %s", OUT_DIR / "static_hold.npz")
    logger.info("plot -> %s", OUT_DIR / "static_hold.png")


if __name__ == "__main__":
    main()
