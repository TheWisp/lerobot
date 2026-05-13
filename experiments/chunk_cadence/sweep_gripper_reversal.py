#!/usr/bin/env python
"""Test gripper overshoot under rapid direction reversal.

Step-velocity tests show capability, not stability. This test directly
measures overshoot caused by momentum on direction reversal — the
"human wiggles the gripper" failure mode for high-P configs.

Two metrics per config per reversal-timing:
  * **overshoot_past_start**: motor passes its starting position by how
    much before re-settling? (Reversal happens mid-flight; the motor
    must kill momentum and return. If P is too high, it shoots past
    the home position.)
  * **settle_time**: how long to stay within ±2 units of final goal.

Per trial:
  1. Park at 20.0 (target_low)
  2. Goal = 80.0  (target_high — motor accelerates up)
  3. At t = reverse_at_ms, Goal = 20.0  (snap reversal)
  4. Sample for 1.2 s total, then compute overshoot + settle

Reversal timings tested: 80ms (early — motor barely started), 150ms
(mid-acceleration), 250ms (near or past peak velocity).

Configs: P=16 (baseline), 24, 32, 48 — others fixed to current committed
defaults (D=32, MT=1000, PC=400, MVL=254, Acc=0).

Safety: same as sweep_gripper_registers — within 0..100 range, brief
duty cycles, baseline registers restored on exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots.bi_so107_follower import BiSO107Follower, BiSO107FollowerConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


TARGET_LOW = 20.0
TARGET_HIGH = 80.0
REVERSAL_MS = [80, 150, 250]
TRIAL_DURATION_S = 1.2
SAMPLE_HZ = 500.0
SETTLE_S = 0.8


BASELINE_REGS: dict[str, int] = {
    "P_Coefficient": 16,
    "I_Coefficient": 0,
    "D_Coefficient": 32,
    "Max_Torque_Limit": 1000,
    "Protection_Current": 400,
    "Overload_Torque": 25,
    "Maximum_Velocity_Limit": 254,
    "Acceleration": 0,
}


@dataclass
class ConfigVariant:
    name: str
    overrides: dict[str, int]

    def regs(self) -> dict[str, int]:
        return {**BASELINE_REGS, **self.overrides}


VARIANTS: list[ConfigVariant] = [
    ConfigVariant("P_16", {}),
    ConfigVariant("P_24", {"P_Coefficient": 24}),
    ConfigVariant("P_32", {"P_Coefficient": 32}),
    ConfigVariant("P_48", {"P_Coefficient": 48}),
]


def build_follower(profile_path: Path) -> BiSO107Follower:
    profile = json.loads(profile_path.read_text())
    fields = dict(profile.get("fields", {}))
    if "calibration_dir" in fields and fields["calibration_dir"] is not None:
        fields["calibration_dir"] = Path(fields["calibration_dir"]).expanduser()
    fields.pop("cameras", None)
    cfg = BiSO107FollowerConfig(**{k: v for k, v in fields.items() if k != "type"})
    return BiSO107Follower(cfg)


def apply_regs(bus: Any, motor: str, regs: dict[str, int]) -> None:
    for key, val in regs.items():
        bus.write(key, motor, val)


def reversal_trial(bus: Any, motor: str, reverse_at_ms: float) -> tuple[np.ndarray, float]:
    """Drive Goal=HIGH then flip to Goal=LOW at reverse_at_ms. Return
    samples (n, 2)=[t, pos] and the actual reversal time observed.
    """
    period = 1.0 / SAMPLE_HZ
    reverse_at_s = reverse_at_ms / 1000.0
    samples: list[tuple[float, float]] = []
    flipped = False
    actual_flip_t = float("nan")

    bus.write("Goal_Position", motor, TARGET_HIGH)
    t0 = time.perf_counter()
    next_t = t0
    while time.perf_counter() - t0 < TRIAL_DURATION_S:
        if not flipped and time.perf_counter() - t0 >= reverse_at_s:
            bus.write("Goal_Position", motor, TARGET_LOW)
            actual_flip_t = time.perf_counter() - t0
            flipped = True
        pos = bus.read("Present_Position", motor, normalize=True)
        samples.append((time.perf_counter() - t0, float(pos)))
        next_t += period
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_t = time.perf_counter()
    return np.array(samples, dtype=np.float64), actual_flip_t


def analyze_reversal(samples: np.ndarray, flip_t: float) -> dict[str, float]:
    """Compute peak position reached + overshoot_past_start + settle_time."""
    t = samples[:, 0]
    q = samples[:, 1]
    if len(t) < 10:
        return {
            "peak_q": float("nan"),
            "overshoot_past_start": float("nan"),
            "settle_time_s": float("nan"),
        }
    start_q = float(q[0])
    peak_q = float(q.max())
    # After flip, the motor decelerates and reverses toward TARGET_LOW.
    # Overshoot = how far past TARGET_LOW does the motor swing on return?
    post_flip_mask = t >= flip_t + 0.05  # 50 ms grace after flip
    if not post_flip_mask.any():
        overshoot_past_start = 0.0
    else:
        q_post = q[post_flip_mask]
        # Motor is going LOW. Overshoot = how far below TARGET_LOW it dips.
        min_q_post = float(q_post.min())
        overshoot_past_start = max(0.0, TARGET_LOW - min_q_post)
    # Settle time: from flip, how long to reach |q − TARGET_LOW| ≤ 2 and
    # stay within ±2 for ≥ 50 ms.
    settle_time = float("nan")
    settled_since: float | None = None
    for ti, qi in zip(t, q, strict=True):
        if ti < flip_t:
            continue
        within = abs(qi - TARGET_LOW) <= 2.0
        if within:
            if settled_since is None:
                settled_since = ti
            elif ti - settled_since >= 0.05:
                settle_time = settled_since - flip_t
                break
        else:
            settled_since = None
    return {
        "peak_q": peak_q,
        "start_q": start_q,
        "overshoot_past_start": overshoot_past_start,
        "settle_time_s": settle_time,
    }


def run_variant_arm(bus: Any, motor: str, variant: ConfigVariant) -> list[dict[str, Any]]:
    apply_regs(bus, motor, variant.regs())
    bus.write("Goal_Position", motor, TARGET_LOW)
    time.sleep(SETTLE_S)
    rows: list[dict[str, Any]] = []
    for reverse_at_ms in REVERSAL_MS:
        # 2 trials per reversal-timing (medians of 2 with hardware that
        # behaves consistently is plenty for picking among variants).
        trial_metrics = []
        for _ in range(2):
            samples, flip_t = reversal_trial(bus, motor, reverse_at_ms)
            trial_metrics.append(analyze_reversal(samples, flip_t))
            bus.write("Goal_Position", motor, TARGET_LOW)
            time.sleep(SETTLE_S)
        peak_q = statistics.median(m["peak_q"] for m in trial_metrics)
        overshoot = statistics.median(m["overshoot_past_start"] for m in trial_metrics)
        settle_times = [m["settle_time_s"] for m in trial_metrics if not np.isnan(m["settle_time_s"])]
        settle = statistics.median(settle_times) if settle_times else float("nan")
        rows.append(
            {
                "reverse_at_ms": reverse_at_ms,
                "peak_q": peak_q,
                "overshoot": overshoot,
                "settle_s": settle,
            }
        )
    return rows


@contextmanager
def safe_session(robot: BiSO107Follower):
    try:
        yield
    finally:
        logger.info("Restoring baseline registers + parking grippers at 50")
        try:
            for arm_bus in (robot.left_arm.bus, robot.right_arm.bus):
                apply_regs(arm_bus, "gripper", BASELINE_REGS)
                arm_bus.write("Goal_Position", "gripper", 50.0)
            time.sleep(SETTLE_S)
        except Exception as e:
            logger.error("Failed to restore registers: %s", e)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--profile",
        default=str(Path.home() / ".config/lerobot/robots/white.json"),
    )
    p.add_argument("--output", default=None)
    args = p.parse_args()

    init_logging()
    robot = build_follower(Path(args.profile).expanduser())
    robot.connect(calibrate=False)
    results: list[dict[str, Any]] = []
    try:
        with safe_session(robot):
            for variant in VARIANTS:
                logger.info("=== variant: %s | overrides=%s", variant.name, variant.overrides)
                for arm_label, arm_bus in (("left", robot.left_arm.bus), ("right", robot.right_arm.bus)):
                    rows = run_variant_arm(arm_bus, "gripper", variant)
                    for r in rows:
                        logger.info(
                            "  %s gripper @%3dms: peak=%5.1f overshoot=%5.2f settle=%.3fs",
                            arm_label,
                            r["reverse_at_ms"],
                            r["peak_q"],
                            r["overshoot"],
                            r["settle_s"],
                        )
                        results.append({"variant": variant.name, "arm": arm_label, **r})
    finally:
        robot.disconnect()

    print()
    print(f"{'variant':<8} {'arm':<6} {'rev_ms':>7} {'peak':>7} {'overshoot':>10} {'settle_s':>9}")
    print("-" * 52)
    for r in results:
        print(
            f"{r['variant']:<8} {r['arm']:<6} {r['reverse_at_ms']:>7} "
            f"{r['peak_q']:>7.1f} {r['overshoot']:>10.2f} {r['settle_s']:>9.3f}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
