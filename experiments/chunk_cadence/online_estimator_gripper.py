#!/usr/bin/env python
# ruff: noqa: N803, N806
"""Autonomously test velocity estimators on the gripper with a small-motion
synthetic signal, measuring real state oscillation.

The offline backtest (estimator_offline_test.py) showed that the current
``lsq_quad`` is catastrophic on the "small deliberate motion + persistent
hand tremor" regime — 3.8 deg motor_cmd jitter on a ±3 deg signal. This
script confirms whether that translates to real state oscillation by:

  1. Generating a synthetic intent stream that mimics the failure mode:
     0.5 Hz · 3 unit amplitude (deliberate small motion) + 10 Hz · 0.5
     unit amplitude (hand tremor). Gripper-unit scale (0..100).
  2. For each candidate estimator, computing motor_cmd at each tick and
     writing it to Goal_Position on the gripper via direct bus access.
  3. Concurrently sampling Present_Position at high rate and computing:
       * state_jit_hf: std(state - savgol(state)) over the trial — the
         user-felt "wiggle" magnitude.
       * state_pk_pk: peak-to-peak of state minus the savgol-smoothed
         intent — captures both bias and HF.

Estimators tested:
  * lsq_quad      — current production
  * forward_diff  — best on real-teleop (prior offline finding)
  * kalman_q1000  — best on the synthetic small-motion regime
  * amp_gated_lp4 — most consistent across regimes

Safety:
  * Drives only the gripper motor, within [TARGET_CENTER ± 5] = 45..55
    (well inside historical 0..100 range, never near mechanical stops).
  * 30 s per estimator per arm (= 4 estimators · 2 arms = ~4 min).
  * Restores baseline registers + parks both grippers at 50 on exit
    (normal or KeyboardInterrupt).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from lerobot.robots.bi_so107_follower import BiSO107Follower, BiSO107FollowerConfig
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


CENTER = 50.0
MOTION_AMP = 3.0  # ±3 units around center
MOTION_FREQ_HZ = 0.5
TREMOR_AMP = 0.5
TREMOR_FREQ_HZ = 10.0
TRIAL_DURATION_S = 30.0
SEND_HZ = 100.0  # write rate (Feetech bus isn't thread-safe; sample state in-loop)
L_MS = 110.0  # production lookahead default
WINDOW_MS = 70.0


GRIPPER_BASELINE: dict[str, int] = {
    "P_Coefficient": 24,
    "I_Coefficient": 0,
    "D_Coefficient": 32,
    "Max_Torque_Limit": 500,
    "Protection_Current": 250,
    "Overload_Torque": 25,
    "Maximum_Velocity_Limit": 254,
    "Acceleration": 0,
}


def gen_intent_signal(duration_s: float, rate_hz: float, seed: int = 42) -> np.ndarray:
    """(n,) sequence of leader gripper positions modelling small motion + tremor."""
    rng = np.random.default_rng(seed)
    phase_m = rng.uniform(0.0, 2 * np.pi)
    phase_t = rng.uniform(0.0, 2 * np.pi)
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    return (
        CENTER
        + MOTION_AMP * np.sin(2 * np.pi * MOTION_FREQ_HZ * t + phase_m)
        + TREMOR_AMP * np.sin(2 * np.pi * TREMOR_FREQ_HZ * t + phase_t)
    )


# ─────────────────────────────────────────────────────────────────────
# Estimator factories — return a callable that maps a rolling window to
# the *motor_cmd* (already includes leader_pos + L · V_est).
# ─────────────────────────────────────────────────────────────────────


class _KalmanCV:
    def __init__(self, q_accel: float = 1000.0, r_pos: float = 0.01):
        self.q_accel = q_accel
        self.r_pos = r_pos
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self._last_t: float | None = None

    def update(self, t: float, z: float) -> tuple[float, float]:
        if self.x is None:
            self.x = np.array([z, 0.0])
            self.P = np.eye(2)
            self._last_t = t
            return float(self.x[0]), float(self.x[1])
        dt = max(t - self._last_t, 1e-9)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q_accel * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        H = np.array([1.0, 0.0])
        y = z - H @ x_pred
        S = H @ P_pred @ H + self.r_pos
        K = P_pred @ H / S
        self.x = x_pred + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
        self._last_t = t
        return float(self.x[0]), float(self.x[1])


def make_lsq_quad_cmd(L_s: float, window_size: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def cmd(t_win: np.ndarray, p_win: np.ndarray) -> float:
        if len(t_win) < 3:
            return float(p_win[-1])
        t_rel = t_win - t_win[-1]
        A = np.stack([np.ones_like(t_rel), t_rel, t_rel * t_rel], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, p_win, rcond=None)
        except np.linalg.LinAlgError:
            return float(p_win[-1])
        a, b, _c = coef
        return float(a + b * L_s)

    cmd.window_size = window_size  # type: ignore[attr-defined]
    cmd.stateful = False  # type: ignore[attr-defined]
    return cmd


def make_forward_diff_cmd(L_s: float, window_size: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def cmd(t_win: np.ndarray, p_win: np.ndarray) -> float:
        if len(p_win) < 2:
            return float(p_win[-1])
        dt = max(float(t_win[-1] - t_win[-2]), 1e-9)
        v = (p_win[-1] - p_win[-2]) / dt
        return float(p_win[-1] + v * L_s)

    cmd.window_size = window_size  # type: ignore[attr-defined]
    cmd.stateful = False  # type: ignore[attr-defined]
    return cmd


def make_kalman_cmd(L_s: float, q_accel: float) -> Callable[[np.ndarray, np.ndarray], float]:
    state = _KalmanCV(q_accel=q_accel)

    def cmd(t_win: np.ndarray, p_win: np.ndarray) -> float:
        pos, vel = state.update(float(t_win[-1]), float(p_win[-1]))
        return float(pos + vel * L_s)

    cmd.window_size = 1  # type: ignore[attr-defined]
    cmd.stateful = True  # type: ignore[attr-defined]
    return cmd


def make_amp_gated_lp_cmd(
    L_s: float, window_size: int, fc_hz: float = 4.0, amp_lo: float = 1.0, amp_hi: float = 3.0
) -> Callable[[np.ndarray, np.ndarray], float]:
    def cmd(t_win: np.ndarray, p_win: np.ndarray) -> float:
        leader_pos = float(p_win[-1])
        if len(p_win) < 3:
            return leader_pos
        amplitude = float(p_win.max() - p_win.min())
        if amplitude <= amp_lo:
            return leader_pos
        # Lowpass V_est
        dt = float(np.median(np.diff(t_win)))
        rc = 1.0 / (2 * np.pi * fc_hz)
        ema_alpha = dt / (rc + dt)
        diffs = np.diff(p_win) / np.diff(t_win)
        v_smooth = diffs[0]
        for i in range(1, len(diffs)):
            v_smooth = ema_alpha * diffs[i] + (1.0 - ema_alpha) * v_smooth
        if amplitude >= amp_hi:
            return leader_pos + v_smooth * L_s
        gate = (amplitude - amp_lo) / (amp_hi - amp_lo)
        return leader_pos + gate * v_smooth * L_s

    cmd.window_size = window_size  # type: ignore[attr-defined]
    cmd.stateful = False  # type: ignore[attr-defined]
    return cmd


def build_estimators(L_s: float, dt_s: float) -> dict[str, Callable]:
    win_n = max(3, int(round(WINDOW_MS / 1000.0 / dt_s)))
    return {
        "lsq_quad": make_lsq_quad_cmd(L_s, win_n),
        "forward_diff": make_forward_diff_cmd(L_s, win_n),
        "kalman_q1000": make_kalman_cmd(L_s, q_accel=1000.0),
        "amp_gated_lp4": make_amp_gated_lp_cmd(L_s, win_n, fc_hz=4.0),
    }


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────


def apply_regs(bus: Any, motor: str, regs: dict[str, int]) -> None:
    for key, val in regs.items():
        bus.write(key, motor, val)


def run_trial(bus: Any, motor: str, estimator: Callable, intent_seq: np.ndarray) -> dict[str, Any]:
    """Push motor_cmd derived from intent through the bus; sample state
    in the same thread (Feetech bus isn't thread-safe — concurrent
    access from a reader thread aborts with 'Port is in use')."""
    dt_s = 1.0 / SEND_HZ
    window_size = getattr(estimator, "window_size", 1)
    is_stateful = getattr(estimator, "stateful", False)

    bus.write("Goal_Position", motor, CENTER)
    time.sleep(0.7)

    cmd_log: list[tuple[float, float, float]] = []  # (t, intent, motor_cmd)
    state_log: list[tuple[float, float]] = []
    intent_history: list[tuple[float, float]] = []

    t0 = time.perf_counter()
    for i, intent_val in enumerate(intent_seq):
        target_t = t0 + i * dt_s
        sleep_for = target_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        now = time.perf_counter()
        intent_history.append((now, float(intent_val)))
        if is_stateful:
            t_win = np.array([intent_history[-1][0]])
            p_win = np.array([intent_history[-1][1]])
        else:
            win = intent_history[-window_size:]
            t_win = np.array([x[0] for x in win])
            p_win = np.array([x[1] for x in win])
        motor_cmd = estimator(t_win, p_win)
        motor_cmd = max(40.0, min(60.0, motor_cmd))
        bus.write("Goal_Position", motor, motor_cmd)
        # Read state right after write — Feetech bus is half-duplex but
        # serial reads chase writes within ~1 ms.
        pos = bus.read("Present_Position", motor, normalize=True)
        sample_t = time.perf_counter() - t0
        cmd_log.append((now - t0, float(intent_val), float(motor_cmd)))
        state_log.append((sample_t, float(pos)))

    cmd_arr = np.array(cmd_log, dtype=np.float64)
    state_arr = np.array(state_log, dtype=np.float64)

    # Trim startup transient: first 0.5 s of data
    cmd_arr = cmd_arr[cmd_arr[:, 0] > 0.5]
    state_arr = state_arr[state_arr[:, 0] > 0.5]

    # Motor_cmd jitter (HF) — match the offline metric.
    if len(cmd_arr) > 16:
        cmd_smooth = savgol_filter(cmd_arr[:, 2], 15, 3)
        cmd_hf = float(np.std(cmd_arr[:, 2] - cmd_smooth))
    else:
        cmd_hf = float("nan")
    # State HF jitter — the operator-felt "wiggle" magnitude.
    if len(state_arr) > 32:
        state_smooth = savgol_filter(state_arr[:, 1], 31, 3)
        state_hf = float(np.std(state_arr[:, 1] - state_smooth))
    else:
        state_hf = float("nan")
    # Peak-to-peak state range vs the *intended* slow motion only (drop the
    # high-freq target ripple). Tells us if state is following the deliberate
    # motion or oscillating beyond it.
    if len(cmd_arr) > 16:
        intent_smooth_window = max(int(round(SEND_HZ * 1.0)), 5)  # ~1s window
        if intent_smooth_window % 2 == 0:
            intent_smooth_window += 1
        intent_slow = savgol_filter(cmd_arr[:, 1], intent_smooth_window, 2)
        intent_p2p = float(intent_slow.max() - intent_slow.min())
    else:
        intent_p2p = float("nan")
    state_p2p = float(state_arr[:, 1].max() - state_arr[:, 1].min()) if len(state_arr) > 1 else float("nan")
    excess_p2p = state_p2p - intent_p2p  # negative = state under-shoots intent

    return {
        "motor_cmd_hf": cmd_hf,
        "state_hf": state_hf,
        "intent_slow_p2p": intent_p2p,
        "state_p2p": state_p2p,
        "excess_p2p": excess_p2p,
        "n_cmd": int(len(cmd_arr)),
        "n_state": int(len(state_arr)),
    }


@contextmanager
def safe_session(robot: BiSO107Follower):
    try:
        yield
    finally:
        logger.info("Restoring baseline registers + parking grippers at 50")
        try:
            for arm_bus in (robot.left_arm.bus, robot.right_arm.bus):
                apply_regs(arm_bus, "gripper", GRIPPER_BASELINE)
                arm_bus.write("Goal_Position", "gripper", 50.0)
            time.sleep(0.7)
        except Exception as e:
            logger.error("Failed to restore registers: %s", e)


def build_follower(profile_path: Path) -> BiSO107Follower:
    profile = json.loads(profile_path.read_text())
    fields = dict(profile.get("fields", {}))
    if "calibration_dir" in fields and fields["calibration_dir"] is not None:
        fields["calibration_dir"] = Path(fields["calibration_dir"]).expanduser()
    fields.pop("cameras", None)
    cfg = BiSO107FollowerConfig(**{k: v for k, v in fields.items() if k != "type"})
    return BiSO107Follower(cfg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--profile",
        default=str(Path.home() / ".config/lerobot/robots/white.json"),
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    init_logging()
    robot = build_follower(Path(args.profile).expanduser())
    robot.connect(calibrate=False)
    L_s = L_MS / 1000.0
    dt_s = 1.0 / SEND_HZ
    intent_seq = gen_intent_signal(TRIAL_DURATION_S, SEND_HZ)
    estimators = build_estimators(L_s, dt_s)

    results: list[dict] = []
    try:
        with safe_session(robot):
            for est_name, est_fn in estimators.items():
                logger.info("=== estimator: %s", est_name)
                for arm_label, arm_bus in (
                    ("left", robot.left_arm.bus),
                    ("right", robot.right_arm.bus),
                ):
                    # Re-create stateful estimators per arm so trials don't share state.
                    if getattr(est_fn, "stateful", False):
                        est_fn_arm = build_estimators(L_s, dt_s)[est_name]
                    else:
                        est_fn_arm = est_fn
                    apply_regs(arm_bus, "gripper", GRIPPER_BASELINE)
                    metrics = run_trial(arm_bus, "gripper", est_fn_arm, intent_seq)
                    logger.info(
                        "  %s gripper: cmd_hf=%.3f  state_hf=%.3f  intent_p2p=%.2f  state_p2p=%.2f  excess=%+.2f",
                        arm_label,
                        metrics["motor_cmd_hf"],
                        metrics["state_hf"],
                        metrics["intent_slow_p2p"],
                        metrics["state_p2p"],
                        metrics["excess_p2p"],
                    )
                    results.append({"estimator": est_name, "arm": arm_label, **metrics})
    finally:
        robot.disconnect()

    print()
    print(
        f"{'estimator':<14} {'arm':<6} {'cmd_hf':>8} {'state_hf':>9} {'intent_p2p':>11} {'state_p2p':>10} {'excess_p2p':>11}"
    )
    print("-" * 76)
    for r in results:
        print(
            f"{r['estimator']:<14} {r['arm']:<6} "
            f"{r['motor_cmd_hf']:>8.3f} {r['state_hf']:>9.3f} "
            f"{r['intent_slow_p2p']:>11.2f} {r['state_p2p']:>10.2f} "
            f"{r['excess_p2p']:>+11.2f}"
        )
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
