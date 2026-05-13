#!/usr/bin/env python
"""Empirically measure the leader arm's position-stream noise at rest.

Connect to the high-rate leader, leave it alone (don't touch), sample
its cached pose for several seconds, and report per-joint position
std plus aggregate. This anchors the synthetic ``intent_noise_deg``
value used in backtest_teleop.py to a real-world number.

Caveats:
  * Measures the noise floor at rest. Active teleop adds hand tremor
    (8-12 Hz, joint-geometry dependent) on top of this floor.
  * Torque state is left as whatever the leader profile sets. For
    typical teleop the leader has torque off; for these arms at rest
    that's fine (gravity-balanced design).

Usage:
  experiments/chunk_cadence/measure_leader_noise.py [--duration 5] [--rate 100]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.teleoperators.bi_so107_leader_highrate import (
    BiSO107LeaderHighRate,
    BiSO107LeaderHighRateConfig,
)
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def build_leader(profile_path: Path) -> BiSO107LeaderHighRate:
    profile = json.loads(profile_path.read_text())
    fields = dict(profile.get("fields", {}))
    if "calibration_dir" in fields and fields["calibration_dir"] is not None:
        fields["calibration_dir"] = Path(fields["calibration_dir"]).expanduser()
    cfg = BiSO107LeaderHighRateConfig(**{k: v for k, v in fields.items() if k != "type"})
    return BiSO107LeaderHighRate(cfg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--profile",
        default=str(Path.home() / ".config/lerobot/teleops/blue_highrate.json"),
        help="Leader profile JSON path",
    )
    p.add_argument("--duration", type=float, default=5.0, help="Recording duration (s)")
    p.add_argument("--rate", type=float, default=100.0, help="Sampling rate (Hz)")
    p.add_argument("--warmup", type=float, default=0.5, help="Background thread warmup (s)")
    args = p.parse_args()

    init_logging()
    leader = build_leader(Path(args.profile).expanduser())
    leader.connect(calibrate=False)
    try:
        logger.info("Warming up background read thread for %.1fs", args.warmup)
        time.sleep(args.warmup)

        joint_names: list[str] = []
        samples: list[np.ndarray] = []
        sample_period = 1.0 / args.rate
        start = time.perf_counter()
        next_t = start
        logger.info("Sampling leader pose @ %.0fHz for %.1fs — leave the arm alone", args.rate, args.duration)
        while time.perf_counter() - start < args.duration:
            pose = leader.get_action()
            if not joint_names:
                joint_names = list(pose.keys())
            samples.append(np.array([pose[j] for j in joint_names], dtype=np.float64))
            next_t += sample_period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.perf_counter()
        elapsed = time.perf_counter() - start
    finally:
        leader.disconnect()

    arr = np.stack(samples)  # (n_samples, n_joints)
    logger.info("Captured %d samples over %.1fs (~%.0fHz effective)", len(arr), elapsed, len(arr) / elapsed)

    print()
    print(f"{'joint':<30s} {'mean':>10s} {'std':>10s} {'p2p':>10s}")
    print("-" * 64)
    for i, j in enumerate(joint_names):
        col = arr[:, i]
        print(f"{j:<30s} {col.mean():>10.3f} {col.std():>10.4f} {np.ptp(col):>10.4f}")
    print("-" * 64)
    print(f"{'AGGREGATE':<30s} {'':>10s} {arr.std(axis=0).mean():>10.4f}")
    print()
    print(
        f"Noise floor per-joint std (mean across joints): {arr.std(axis=0).mean():.4f} deg\n"
        f"For comparison: intent_noise_deg=0.1 in the chunk-cadence experiment\n"
        f"corresponds to {0.1 / arr.std(axis=0).mean():.1f}× this floor."
    )


if __name__ == "__main__":
    main()
