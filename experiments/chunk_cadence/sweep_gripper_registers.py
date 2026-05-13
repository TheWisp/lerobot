#!/usr/bin/env python
"""Autonomously sweep Feetech gripper registers to find the config that
narrows the intent→state velocity gap on SO-107 grippers.

Method
------
For each gripper (left, right) on a bimanual follower, drive a STEP input
between two positions inside the historical 0..100 range and sample
Present_Position at ~200 Hz. Compute peak achievable state velocity per
trial. Repeat across configurations.

Each trial is a single open→close→open cycle (3 step inputs). The peak
state |dq/dt| during the transient (in normalized units per second) is
the primary metric: higher = motor reaches a higher peak velocity = the
intent→state gap narrows when the leader commands above that speed.

Configs swept (each varies one register from a baseline; baseline = what
``so_follower.configure()`` writes today):

  baseline:        P=16, D=32, MT=1000, PC=400, MVL=254, Acc=0
  P_high:          P=24, 32, 48   (more aggressive position loop)
  Acc_high:        Acc=1, 4, 16   (re-introduce acceleration ramp)
  MVL_low:         MVL=100, 150   (probe whether 254 was real bottleneck)
  MT_low:          MT=500         (revert torque doubling)
  PC_low:          PC=250         (revert current doubling)

Safety:
  - Driven range: 20..80 (well inside historical 0..100, never hits
    mechanical stops with margin).
  - Step transients are brief (≤1 s); between trials we wait 0.7 s so
    the motor settles without prolonged high-load condition.
  - On exit (normal or KeyboardInterrupt) we restore the so_follower.py
    baseline registers and park both grippers at 50.

Usage:
  experiments/chunk_cadence/sweep_gripper_registers.py
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


SWEEP_LOW = 20.0  # normalized 0..100
SWEEP_HIGH = 80.0
SETTLE_S = 0.7  # rest between steps; long enough for motor to fully stop
TRANSIENT_S = 1.0  # measurement window per step
SAMPLE_HZ = 500.0  # per-arm Present_Position read rate (single-motor read ≈0.3 ms)


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
    ConfigVariant("baseline", {}),
    ConfigVariant("P_24", {"P_Coefficient": 24}),
    ConfigVariant("P_32", {"P_Coefficient": 32}),
    ConfigVariant("P_48", {"P_Coefficient": 48}),
    ConfigVariant("Acc_1", {"Acceleration": 1}),
    ConfigVariant("Acc_4", {"Acceleration": 4}),
    ConfigVariant("Acc_16", {"Acceleration": 16}),
    ConfigVariant("MVL_100", {"Maximum_Velocity_Limit": 100}),
    ConfigVariant("MVL_150", {"Maximum_Velocity_Limit": 150}),
    ConfigVariant("MT_500", {"Max_Torque_Limit": 500}),
    ConfigVariant("PC_250", {"Protection_Current": 250}),
    ConfigVariant("P32_Acc0", {"P_Coefficient": 32, "Acceleration": 0}),
    ConfigVariant("P48_Acc0", {"P_Coefficient": 48, "Acceleration": 0}),
]


def build_follower(profile_path: Path) -> BiSO107Follower:
    profile = json.loads(profile_path.read_text())
    fields = dict(profile.get("fields", {}))
    if "calibration_dir" in fields and fields["calibration_dir"] is not None:
        fields["calibration_dir"] = Path(fields["calibration_dir"]).expanduser()
    # Disable cameras for this test (we only need motors).
    fields.pop("cameras", None)
    cfg = BiSO107FollowerConfig(**{k: v for k, v in fields.items() if k != "type"})
    return BiSO107Follower(cfg)


def apply_regs(bus: Any, motor: str, regs: dict[str, int]) -> None:
    for key, val in regs.items():
        bus.write(key, motor, val)


def step_and_sample(bus: Any, motor: str, target: float, duration_s: float) -> np.ndarray:
    """Issue Goal_Position step then sample (t, present_pos) at SAMPLE_HZ.

    Returns array shape (n, 2): columns [t_seconds, present_pos_normalized].
    """
    period = 1.0 / SAMPLE_HZ
    bus.write("Goal_Position", motor, target)
    t0 = time.perf_counter()
    next_t = t0
    samples: list[tuple[float, float]] = []
    while time.perf_counter() - t0 < duration_s:
        pos = bus.read("Present_Position", motor, normalize=True)
        samples.append((time.perf_counter() - t0, float(pos)))
        next_t += period
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_t = time.perf_counter()
    return np.array(samples, dtype=np.float64)


def peak_velocity(samples: np.ndarray) -> tuple[float, float]:
    """Return (peak |dq/dt|, time-to-90% travel) from a step transient trace.

    peak_v: max |Δq|/Δt over a 3-sample window (smooths read jitter).
    t90: time after step at which 90% of the |q[-1] − q[0]| travel is reached.
    """
    t = samples[:, 0]
    q = samples[:, 1]
    if len(t) < 6:
        return 0.0, float("nan")
    # 3-sample window: (q[i+2] - q[i-1]) / (t[i+2] - t[i-1]) — equivalent
    # to averaging instantaneous velocity over a small interval to ignore
    # single-sample read noise.
    dq = q[3:] - q[:-3]
    dt = t[3:] - t[:-3]
    inst_v = np.abs(dq / np.where(dt > 1e-6, dt, 1e-6))
    peak_v = float(inst_v.max())
    travel = q[-1] - q[0]
    if abs(travel) < 1.0:
        return peak_v, float("nan")
    target_pct = 0.9
    target = q[0] + target_pct * travel
    direction = np.sign(travel)
    idx = np.argmax(direction * (q - target) >= 0) if direction != 0 else 0
    t90 = float(t[idx]) if idx > 0 else float("nan")
    return peak_v, t90


def run_variant(bus: Any, motor: str, variant: ConfigVariant) -> dict[str, float]:
    apply_regs(bus, motor, variant.regs())
    # Park at LOW, then trigger close→open→close. Three transients = three
    # independent samples per gripper per variant.
    bus.write("Goal_Position", motor, SWEEP_LOW)
    time.sleep(SETTLE_S)

    transients: list[tuple[float, float]] = []
    for target in (SWEEP_HIGH, SWEEP_LOW, SWEEP_HIGH):
        samples = step_and_sample(bus, motor, target, TRANSIENT_S)
        peak_v, t90 = peak_velocity(samples)
        transients.append((peak_v, t90))
        time.sleep(SETTLE_S)
    # Median over 3 trials to reject one-off transient outliers.
    peak_vs = [p for p, _ in transients]
    t90s = [t for _, t in transients if not np.isnan(t)]
    return {
        "peak_v_med": statistics.median(peak_vs),
        "peak_v_max": max(peak_vs),
        "t90_med": statistics.median(t90s) if t90s else float("nan"),
    }


@contextmanager
def safe_session(robot: BiSO107Follower):
    """Always restore baseline registers + park grippers, even on Ctrl-C."""
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
        help="Bimanual follower profile JSON path",
    )
    p.add_argument("--output", default=None, help="Write results JSON to this path")
    args = p.parse_args()

    init_logging()
    robot = build_follower(Path(args.profile).expanduser())
    # calibrate=False: motors already calibrated; configure() applies
    # so_follower.py baseline which we'll overwrite per variant anyway.
    robot.connect(calibrate=False)

    results: list[dict[str, Any]] = []
    try:
        with safe_session(robot):
            for variant in VARIANTS:
                logger.info("=== variant: %s | overrides=%s", variant.name, variant.overrides)
                for arm_label, arm_bus in (("left", robot.left_arm.bus), ("right", robot.right_arm.bus)):
                    metrics = run_variant(arm_bus, "gripper", variant)
                    logger.info(
                        "  %s gripper: peak_v_med=%.1f  peak_v_max=%.1f  t90_med=%.3fs",
                        arm_label,
                        metrics["peak_v_med"],
                        metrics["peak_v_max"],
                        metrics["t90_med"],
                    )
                    results.append(
                        {
                            "variant": variant.name,
                            "overrides": variant.overrides,
                            "arm": arm_label,
                            **metrics,
                        }
                    )
    finally:
        robot.disconnect()

    # Pretty-print results table.
    print()
    print(f"{'variant':<14} {'arm':<6} {'peak_v_med':>10} {'peak_v_max':>10} {'t90_med_s':>10}")
    print("-" * 56)
    for r in results:
        print(
            f"{r['variant']:<14} {r['arm']:<6} "
            f"{r['peak_v_med']:>10.1f} {r['peak_v_max']:>10.1f} "
            f"{r['t90_med']:>10.3f}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
