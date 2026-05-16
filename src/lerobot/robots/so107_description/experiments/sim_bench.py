"""
Headless benchmark: run each IK strategy through a fixed motion sequence and
report stability metrics. Used to iterate on IK strategy choices without
involving the real arm or human-in-the-loop testing.

Usage:
    .venv/bin/python -m lerobot.robots.so107_description.sim_bench
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from ..kinematics import (
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    So107Kinematics,
)
from .teleop_sim import KEY_DIR, JacobianVelocityController

# Same starting pose the user has at rest, from their log.
START_MOTOR_POS = {
    "shoulder_pan": 1.98,
    "shoulder_lift": -106.7,
    "elbow_flex": 99.6,
    "forearm_roll": -3.78,
    "wrist_flex": 70.0,
    "wrist_roll": -5.0,
    "gripper": 13.6,
}

# Motion script: list of (key, duration_ticks) tuples. None = idle.
# At 20Hz, each tick = 50ms.
DEFAULT_SCRIPT: list[tuple[str | None, int]] = [
    (None, 10),  # 0.5s idle at start
    ("d", 20),  # 1s of +X
    (None, 10),  # 0.5s idle
    ("w", 20),  # 1s of -Y (forward)
    (None, 10),
    ("q", 20),  # 1s of +Z (up) — the broken key
    (None, 10),
    ("e", 20),  # 1s of -Z (down)
    (None, 10),
    ("a", 20),  # 1s of -X
    (None, 10),
    ("s", 20),  # 1s of +Y
    (None, 30),
]


@dataclass
class BenchMetrics:
    strategy: str
    max_cmd_jump_idle_deg: float  # worst tick-to-tick command jump while idle
    max_cmd_jump_motion_deg: float  # worst tick-to-tick command jump while a key is held
    p99_cmd_jump_motion_deg: float
    ee_drift_idle_mm: float  # EE motion across idle segments (should be ~0)
    ee_motion_per_tick_mm: float  # mean EE motion per tick during keypress (should ~= step_mm)
    ee_orientation_drift_deg: float  # rotation angle between initial and final EE orientation
    rows: list[dict]  # full per-tick records, for plotting if needed

    def report(self) -> None:
        ok = (
            "OK"
            if (
                self.max_cmd_jump_idle_deg < 0.5
                and self.max_cmd_jump_motion_deg < 10
                and self.ee_drift_idle_mm < 1
            )
            else "BAD"
        )
        print(
            f"  [{ok}] {self.strategy:18s}  "
            f"idle jump max={self.max_cmd_jump_idle_deg:6.2f}°  "
            f"motion jump max={self.max_cmd_jump_motion_deg:7.2f}° p99={self.p99_cmd_jump_motion_deg:6.2f}°  "
            f"idle EE drift={self.ee_drift_idle_mm:5.2f}mm  "
            f"EE/tick motion={self.ee_motion_per_tick_mm:5.2f}mm  "
            f"orient drift={self.ee_orientation_drift_deg:5.1f}°"
        )


def run_strategy(
    strategy: str,
    script: list[tuple[str | None, int]],
    *,
    step_mm: float = 2.0,
    orientation_weight: float = 1.0,
    max_iters: int = 20,
    motor_lag: float = 1.0,
    max_relative_target: float = 0.0,
    jacobian_damping: float = 0.05,
) -> BenchMetrics:
    """Run one strategy through the script and return metrics.

    motor_lag=1, max_relative_target=0 => perfect motors (no realism). Isolates the IK behavior.
    """
    step_m = step_mm / 1000.0
    kin = So107Kinematics(joint_map=RIGHT_ARM_MAP)
    jac = JacobianVelocityController(joint_map=RIGHT_ARM_MAP)
    motor_pos = dict(START_MOTOR_POS)
    target_q = dict(motor_pos)
    target_T = kin.fk_from_motors(motor_pos).copy()
    initial_T = target_T.copy()
    gripper_target = motor_pos["gripper"]

    rows: list[dict] = []
    tick_idx = 0
    for key, duration in script:
        for _ in range(duration):
            tick_idx += 1
            # 1. EE step from current "key held".
            ee_step = np.array(KEY_DIR[key]) * step_m if (key in KEY_DIR) else np.zeros(3)
            target_changed = key in KEY_DIR

            # 2. Strategy-specific target_q update.
            if strategy == "jacobian":
                if target_changed:
                    new_q = jac.step(motor_pos, ee_step, damping=jacobian_damping)
                    new_q["gripper"] = gripper_target
                    target_q = new_q
                    target_T = kin.fk_from_motors(target_q).copy()
            else:
                if target_changed:
                    target_T[0, 3] += ee_step[0]
                    target_T[1, 3] += ee_step[1]
                    target_T[2, 3] += ee_step[2]
                    ik_guess = dict(target_q)
                    ik_guess["gripper"] = gripper_target
                    mi = 1 if strategy == "one-shot" else max_iters
                    new_q = kin.ik_to_motors(
                        ik_guess,
                        target_T,
                        orientation_weight=orientation_weight,
                        max_iters=mi,
                    )
                    new_q["gripper"] = gripper_target
                    target_q = new_q

            # 3. Simulated motor dynamics.
            new_mp = {}
            for n in MOTOR_NAMES:
                d = target_q[n] - motor_pos[n]
                if max_relative_target > 0:
                    d = max(-max_relative_target, min(max_relative_target, d))
                new_mp[n] = motor_pos[n] + motor_lag * d
            motor_pos = new_mp

            cur_T = kin.fk_from_motors(motor_pos)
            rows.append(
                {
                    "t": tick_idx,
                    "key": key or "",
                    "motor_pos": dict(motor_pos),
                    "target_q": dict(target_q),
                    "ee": cur_T[:3, 3].copy(),
                    "ee_R": cur_T[:3, :3].copy(),
                }
            )

    # Metrics.
    def cmd_jumps(rows: list[dict]) -> list[float]:
        out = []
        for i in range(1, len(rows)):
            jumps = [
                abs(rows[i]["target_q"][n] - rows[i - 1]["target_q"][n])
                for n in MOTOR_NAMES
                if n != "gripper"
            ]
            out.append(max(jumps))
        return out

    [r for r in rows if not r["key"]]
    [r for r in rows if r["key"]]

    # For "idle jump", we need consecutive idle rows. Compute over full timeline filtered to idle.
    idle_jumps: list[float] = []
    for i in range(1, len(rows)):
        if not rows[i]["key"] and not rows[i - 1]["key"]:
            jumps = [
                abs(rows[i]["target_q"][n] - rows[i - 1]["target_q"][n])
                for n in MOTOR_NAMES
                if n != "gripper"
            ]
            idle_jumps.append(max(jumps))

    motion_jumps: list[float] = []
    for i in range(1, len(rows)):
        if rows[i]["key"]:
            jumps = [
                abs(rows[i]["target_q"][n] - rows[i - 1]["target_q"][n])
                for n in MOTOR_NAMES
                if n != "gripper"
            ]
            motion_jumps.append(max(jumps))

    # Idle EE drift: sum of EE position deltas across consecutive idle pairs.
    idle_ee_drift = 0.0
    for i in range(1, len(rows)):
        if not rows[i]["key"] and not rows[i - 1]["key"]:
            idle_ee_drift += float(np.linalg.norm(rows[i]["ee"] - rows[i - 1]["ee"]))

    # EE motion per tick during motion.
    ee_motion_per_tick = []
    for i in range(1, len(rows)):
        if rows[i]["key"]:
            ee_motion_per_tick.append(float(np.linalg.norm(rows[i]["ee"] - rows[i - 1]["ee"])))

    # Orientation drift between initial and final EE rotation.
    R_init = initial_T[:3, :3]
    R_final = rows[-1]["ee_R"]
    # Angle of relative rotation.
    R_rel = R_init.T @ R_final
    cos = (np.trace(R_rel) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    orient_drift = float(np.degrees(np.arccos(cos)))

    return BenchMetrics(
        strategy=strategy,
        max_cmd_jump_idle_deg=max(idle_jumps) if idle_jumps else 0.0,
        max_cmd_jump_motion_deg=max(motion_jumps) if motion_jumps else 0.0,
        p99_cmd_jump_motion_deg=float(np.percentile(motion_jumps, 99)) if motion_jumps else 0.0,
        ee_drift_idle_mm=idle_ee_drift * 1000.0,
        ee_motion_per_tick_mm=float(np.mean(ee_motion_per_tick) * 1000.0) if ee_motion_per_tick else 0.0,
        ee_orientation_drift_deg=orient_drift,
        rows=rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies", default="one-shot,iterated,jacobian", help="comma-separated strategies to bench"
    )
    parser.add_argument("--damping", type=float, default=0.05, help="Jacobian damping parameter")
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--step-mm", type=float, default=2.0)
    args = parser.parse_args()

    print(
        f"Benchmark — step={args.step_mm}mm, orient_weight={args.orientation_weight}, jac damping={args.damping}"
    )
    print(f"Script: {DEFAULT_SCRIPT}\n")

    strategies = args.strategies.split(",")
    metrics: dict[str, BenchMetrics] = {}
    for s in strategies:
        m = run_strategy(
            s,
            DEFAULT_SCRIPT,
            step_mm=args.step_mm,
            orientation_weight=args.orientation_weight,
            jacobian_damping=args.damping,
        )
        metrics[s] = m

    print("\nResults:\n  (OK: idle jump <0.5°, motion jump <10°, idle drift <1mm)\n")
    for s in strategies:
        metrics[s].report()

    print("\nThresholds for 'stable':")
    print("  idle jump < 0.5°/tick     (no motion when no keys)")
    print("  motion jump < 10°/tick    (no IK branch flips)")
    print("  idle drift < 1mm          (no slow EE drift when idle)")
    print("  ee_motion_per_tick ~= step_mm  (actual motion matches commanded)")
    print("  orientation drift small   (EE doesn't twist when commanded position-only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
