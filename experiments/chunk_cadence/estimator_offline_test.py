#!/usr/bin/env python
# ruff: noqa: N803, N806
"""Offline velocity-estimator comparison on real teleop motion logs.

Extends scripts/backtest_velocity_estimators.py with:
  * Adaptive-window LSQ        — shorter window during high |accel|
  * Kalman filter (CV model)   — constant-velocity state with adaptive Q
  * Savitzky-Golay derivative  — closed-form polynomial derivative
  * Holt's double-exp smoothing— level + trend with two smoothing constants

Plus a SECOND metric beyond the existing prediction-RMSE: **output jitter**
— std(motor_cmd − savgol(motor_cmd)) over the same trace. Lower output
jitter means the controller's emitted commands have less HF noise that
the motor has to absorb.

For each estimator we therefore report:
  * `pred_rmse_deg`  — RMSE of intent(now+L) prediction vs the ground-
                       truth future leader sample (what the existing
                       scripts/backtest already does).
  * `motor_jit_deg`  — std of motor_cmd minus its own savgol-smoothed
                       version (what experiments/chunk_cadence/analyze.py
                       computes on backtest traces).

Tradeoff: an estimator with low prediction RMSE but high output jitter
isn't actually better in production — the motor doesn't care if our
prediction is closer to truth, it just sees a noisier command and the
operator sees more shake.

Input: a motion log (.npz) recorded by lerobot.utils.latency.motion —
specifically the ``intent`` array carrying per-tick leader pose. We
treat this as the controller's input stream and evaluate each
estimator on it post-hoc.

Usage:
  experiments/chunk_cadence/estimator_offline_test.py
      outputs/teleop/motion_20260513_175836.npz
      --L-ms 110 --window-ms 70
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

# ─────────────────────────────────────────────────────────────────────
# Estimators
# ─────────────────────────────────────────────────────────────────────

EstimatorFn = Callable[[np.ndarray, np.ndarray, float], float]


def _lsq_linear_v(t_win: np.ndarray, p_win: np.ndarray) -> float:
    t_c = t_win - t_win.mean()
    denom = float((t_c * t_c).sum())
    if denom < 1e-12:
        return 0.0
    return float((t_c @ (p_win - p_win.mean())) / denom)


def _lsq_quad_end_v(t_win: np.ndarray, p_win: np.ndarray) -> float:
    if len(t_win) < 3:
        return _lsq_linear_v(t_win, p_win)
    t_rel = t_win - t_win[-1]
    A = np.stack([np.ones_like(t_rel), t_rel, t_rel * t_rel], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(A, p_win, rcond=None)
    except np.linalg.LinAlgError:
        return _lsq_linear_v(t_win, p_win)
    return float(coef[1])


def lsq_linear(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    v = _lsq_linear_v(t_win, p_win)
    return float(p_win[-1] + v * L)


def lsq_quad_end(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    v = _lsq_quad_end_v(t_win, p_win)
    return float(p_win[-1] + v * L)


def forward_diff(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    if len(p_win) < 2:
        return float(p_win[-1])
    dt = float(t_win[-1] - t_win[-2])
    if abs(dt) < 1e-9:
        return float(p_win[-1])
    v = (p_win[-1] - p_win[-2]) / dt
    return float(p_win[-1] + v * L)


def held(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    return float(p_win[-1])


def savgol_deriv(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    """Savitzky-Golay derivative at the window END.

    Order-2 polynomial fit; deriv=1 returns dp/dt at every point. We
    take the END value (== "now"). Mathematically the same family as
    lsq_quad_end (both fit a polynomial and read the derivative at a
    point), but uses scipy's optimized convolution-based implementation.

    Subtle: savgol_filter assumes uniform sampling. Real motion logs
    have small dt jitter (~1-2 ms at 30 Hz). For the controller this
    will always be uniform (200 Hz tick), so the assumption matches
    deployment.
    """
    n = len(p_win)
    if n < 4:
        return _lsq_linear_v(t_win, p_win) * L + float(p_win[-1])
    dt = float(np.median(np.diff(t_win)))
    polyorder = min(2, n - 2)
    window_length = n if n % 2 == 1 else n - 1
    if window_length < polyorder + 2:
        return _lsq_linear_v(t_win, p_win) * L + float(p_win[-1])
    deriv = savgol_filter(p_win, window_length, polyorder, deriv=1, delta=dt, mode="interp")
    v_now = float(deriv[-1])
    return float(p_win[-1] + v_now * L)


def adaptive_window(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    """LSQ-quad with window length scaled DOWN by recent acceleration.

    Hypothesis (per discussion): during steady motion the window-based
    estimator is most accurate with a long window (averages noise);
    during transitions the long window is the wrong window (averages
    across two regimes), so a short window is better. We score
    "transition-ness" by the magnitude of recent second-difference of
    position and rescale the effective window.

    Concretely: full window n_full when |accel_norm| ≤ acc_lo, shrinks
    linearly to n_min when |accel_norm| ≥ acc_hi, where ``acc_norm`` is
    the second-difference of p_win normalised by its full-window stdev.

    n_min = 3 (minimum for quad fit). n_full = len(p_win).
    """
    n = len(p_win)
    if n < 6:
        return lsq_quad_end(t_win, p_win, L)
    # Second-difference magnitude in the last few samples vs the full
    # window's variation. Captures "is the signal changing direction".
    d2 = np.diff(p_win, n=2)
    recent_d2 = float(np.abs(d2[-3:]).mean())
    full_std = float(p_win.std())
    if full_std < 1e-9:
        return lsq_quad_end(t_win, p_win, L)
    acc_norm = recent_d2 / full_std
    # Tuning constants. acc_lo: signal is essentially steady; acc_hi:
    # signal is in clear transition. Below the values that the recorded
    # leader_noise sweep produces during transitions.
    acc_lo, acc_hi = 0.05, 0.25
    if acc_norm <= acc_lo:
        n_eff = n
    elif acc_norm >= acc_hi:
        n_eff = 3
    else:
        # Linear interp between full and min.
        frac = (acc_hi - acc_norm) / (acc_hi - acc_lo)
        n_eff = max(3, int(round(3 + (n - 3) * frac)))
    return lsq_quad_end(t_win[-n_eff:], p_win[-n_eff:], L)


# Kalman filter: stateful, so we wrap it per-joint.
class KalmanCVTracker:
    """Constant-velocity Kalman filter, scalar position observation.

    State: [position, velocity]. Transition with dt:
        x[k+1] = F @ x[k] + w,  F = [[1, dt], [0, 1]]
        z[k]   = H @ x[k] + v,  H = [[1, 0]]
    Process noise w ~ N(0, Q) with Q proportional to a piecewise-constant
    white-acceleration model. q_accel is the only knob.
    """

    def __init__(self, q_accel: float = 100.0, r_pos: float = 0.01) -> None:
        self.q_accel = q_accel  # variance of accel-driving white noise (deg²/s⁴)
        self.r_pos = r_pos  # variance of position observation noise (deg²)
        self.x: np.ndarray | None = None  # state
        self.P: np.ndarray | None = None  # covariance
        self._last_t: float | None = None

    def update(self, t: float, z: float) -> tuple[float, float]:
        """Step the filter with observation z at time t. Return (pos, vel)."""
        if self.x is None or self._last_t is None:
            self.x = np.array([z, 0.0], dtype=np.float64)
            self.P = np.eye(2, dtype=np.float64)
            self._last_t = t
            return float(self.x[0]), float(self.x[1])
        dt = max(t - self._last_t, 1e-9)
        # Predict
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        # White-acceleration process noise covariance.
        Q = self.q_accel * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=np.float64)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        # Update
        H = np.array([1.0, 0.0])
        y = z - H @ x_pred
        S = H @ P_pred @ H + self.r_pos
        K = P_pred @ H / S
        self.x = x_pred + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
        self._last_t = t
        return float(self.x[0]), float(self.x[1])


def make_kalman_stateful(q_accel: float, r_pos: float) -> Callable:
    """Return an estimator function that owns its own per-joint state.

    The backtest harness calls estimators per joint per index. We need
    state across calls. The harness handles per-joint dispatch via the
    `make_per_joint_estimator` wrapper below.
    """

    def fn(t_win: np.ndarray, p_win: np.ndarray, L: float, tracker: KalmanCVTracker) -> float:
        # Replay the last sample only (we trust prior state).
        pos, vel = tracker.update(float(t_win[-1]), float(p_win[-1]))
        return float(pos + vel * L)

    fn._needs_state = True  # type: ignore[attr-defined]
    fn._state_factory = lambda: KalmanCVTracker(q_accel=q_accel, r_pos=r_pos)  # type: ignore[attr-defined]
    return fn


def holt_smoothing(
    t_win: np.ndarray, p_win: np.ndarray, L: float, alpha: float = 0.5, beta: float = 0.25
) -> float:
    """Holt's double exponential smoothing: jointly estimate level + trend.

    s[k] = α p[k] + (1−α)(s[k−1] + b[k−1])
    b[k] = β (s[k] − s[k−1]) + (1−β) b[k−1]
    predict(L) = s[k] + b[k] · L / dt_avg

    The β knob acts like an automatic per-sample velocity filter. Larger
    α = trust new sample more; larger β = trust new trend more.
    """
    if len(p_win) < 2:
        return float(p_win[-1])
    s = float(p_win[0])
    b = float(p_win[1] - p_win[0])
    for i in range(1, len(p_win)):
        s_new = alpha * float(p_win[i]) + (1.0 - alpha) * (s + b)
        b = beta * (s_new - s) + (1.0 - beta) * b
        s = s_new
    dt_avg = float(np.mean(np.diff(t_win))) if len(t_win) > 1 else (1.0 / 200.0)
    # Holt's trend is per-sample; convert to per-second by dividing by dt.
    v_per_s = b / max(dt_avg, 1e-9)
    return float(s + v_per_s * L)


def lowpass_v_est(t_win: np.ndarray, p_win: np.ndarray, L: float, fc_hz: float = 4.0) -> float:
    """Forward-diff V_est with a 1st-order discrete IIR lowpass applied
    over the window. Cutoff fc_hz attenuates the human hand tremor band
    (≈8-12 Hz) while passing real motion (≤2 Hz dominant).

    α = dt / (RC + dt), RC = 1 / (2π fc). For dt=33ms, fc=4Hz, α≈0.46
    (gentle smoothing). For dt=5ms (200Hz), α≈0.11 (more samples → same
    effective filter).
    """
    if len(p_win) < 3:
        return forward_diff(t_win, p_win, L)
    dt = float(np.median(np.diff(t_win)))
    rc = 1.0 / (2 * np.pi * fc_hz)
    alpha = dt / (rc + dt)
    # Compute per-sample forward differences, then EMA them.
    diffs = np.diff(p_win) / np.diff(t_win)
    v_smooth = diffs[0]
    for i in range(1, len(diffs)):
        v_smooth = alpha * diffs[i] + (1.0 - alpha) * v_smooth
    return float(p_win[-1] + v_smooth * L)


def amp_gated_fwd_diff(
    t_win: np.ndarray,
    p_win: np.ndarray,
    L: float,
    amp_lo: float = 1.0,
    amp_hi: float = 3.0,
) -> float:
    """Forward-diff with lookahead scaled by recent motion amplitude.

    amplitude = max(p_win) - min(p_win). When amplitude < amp_lo, alpha=0
    (no prediction, motor_cmd = leader_pos). When amplitude > amp_hi,
    alpha=1 (full lookahead). Linear ramp between. Defaults assume the
    gripper's 0-100 scale and the arm's ~degree scale produce similar
    "small motion" thresholds — for unified-scale signals it works
    cleanly; for mixed-scale it should be applied per joint.
    """
    if len(p_win) < 2:
        return float(p_win[-1])
    amplitude = float(p_win.max() - p_win.min())
    if amplitude <= amp_lo:
        alpha = 0.0
    elif amplitude >= amp_hi:
        alpha = 1.0
    else:
        alpha = (amplitude - amp_lo) / (amp_hi - amp_lo)
    dt = float(t_win[-1] - t_win[-2])
    if abs(dt) < 1e-9:
        return float(p_win[-1])
    v = (p_win[-1] - p_win[-2]) / dt
    return float(p_win[-1] + alpha * v * L)


def amp_gated_lowpass(
    t_win: np.ndarray,
    p_win: np.ndarray,
    L: float,
    fc_hz: float = 4.0,
    amp_lo: float = 1.0,
    amp_hi: float = 3.0,
) -> float:
    """Combine options: lowpass V_est AND scale by amplitude."""
    base = lowpass_v_est(t_win, p_win, L, fc_hz=fc_hz)
    leader_pos = float(p_win[-1])
    amplitude = float(p_win.max() - p_win.min())
    if amplitude <= amp_lo:
        return leader_pos
    if amplitude >= amp_hi:
        return base
    alpha = (amplitude - amp_lo) / (amp_hi - amp_lo)
    return leader_pos + alpha * (base - leader_pos)


ESTIMATORS_STATELESS: dict[str, EstimatorFn] = {
    "lsq_linear": lsq_linear,
    "lsq_quad (current)": lsq_quad_end,
    "forward_diff": forward_diff,
    "savgol_deriv": savgol_deriv,
    "adaptive_window": adaptive_window,
    "lowpass_4Hz": lambda t, p, L: lowpass_v_est(t, p, L, fc_hz=4.0),
    "lowpass_3Hz": lambda t, p, L: lowpass_v_est(t, p, L, fc_hz=3.0),
    "amp_gated_fd": amp_gated_fwd_diff,
    "amp_gated_lp4": lambda t, p, L: amp_gated_lowpass(t, p, L, fc_hz=4.0),
    "holt_a05_b02": lambda t, p, L: holt_smoothing(t, p, L, 0.5, 0.2),
    "holt_a07_b03": lambda t, p, L: holt_smoothing(t, p, L, 0.7, 0.3),
    "held": held,
}

ESTIMATORS_STATEFUL: dict[str, Callable] = {
    "kalman_q100": make_kalman_stateful(q_accel=100.0, r_pos=0.01),
    "kalman_q1000": make_kalman_stateful(q_accel=1000.0, r_pos=0.01),
    "kalman_q10000": make_kalman_stateful(q_accel=10000.0, r_pos=0.01),
}


# ─────────────────────────────────────────────────────────────────────
# Backtest harness
# ─────────────────────────────────────────────────────────────────────


def _savgol_smooth(x: np.ndarray, window: int = 15, polyorder: int = 3) -> np.ndarray:
    """Per-column savgol for the OUTPUT-JITTER metric. Same shape as
    experiments/chunk_cadence/analyze.py for cross-comparability."""
    w = min(window, len(x) - (1 - len(x) % 2))
    if w < polyorder + 2:
        return x.copy()
    if w % 2 == 0:
        w -= 1
    return savgol_filter(x, w, polyorder, axis=0)


def evaluate_estimator(
    t: np.ndarray,
    p: np.ndarray,
    estimator_name: str,
    L_s: float,
    window_s: float,
) -> dict[str, float]:
    """For each tick, run the estimator on the rolling window and predict
    intent at t+L. Compute prediction error and output jitter.

    `p` shape (n_samples, n_joints).
    """
    dt = float(np.median(np.diff(t)))
    window_frames = max(3, int(round(window_s / dt)))
    L_frames = int(round(L_s / dt))
    n_samples, n_joints = p.shape

    # Build motor_cmd[i, j] for all valid i,j.
    motor_cmd = np.full_like(p, np.nan)

    stateful = ESTIMATORS_STATEFUL.get(estimator_name)
    if stateful is not None:
        # Per-joint state, replayed sample-by-sample.
        trackers = [stateful._state_factory() for _ in range(n_joints)]  # type: ignore[attr-defined]
        for i in range(n_samples):
            for j in range(n_joints):
                # Stateful estimators only need the latest sample, but we
                # still pass a 1-element window for API uniformity.
                t_win = t[max(0, i - window_frames + 1) : i + 1]
                p_win = p[max(0, i - window_frames + 1) : i + 1, j]
                pred = stateful(t_win, p_win, L_s, trackers[j])
                motor_cmd[i, j] = pred
    else:
        est_fn = ESTIMATORS_STATELESS[estimator_name]
        for i in range(window_frames - 1, n_samples):
            t_win = t[i - window_frames + 1 : i + 1]
            for j in range(n_joints):
                p_win = p[i - window_frames + 1 : i + 1, j]
                motor_cmd[i, j] = est_fn(t_win, p_win, L_s)

    # Prediction error: motor_cmd[i] should approximate p[i + L_frames].
    valid_start = window_frames - 1 if stateful is None else 0
    valid_end = n_samples - L_frames
    pred = motor_cmd[valid_start:valid_end]
    truth = p[valid_start + L_frames : valid_end + L_frames]
    mask = np.isfinite(pred)
    err = pred - truth
    err_flat = err[mask].ravel()

    # Output jitter on motor_cmd: same metric as analyze.py.
    cmd_valid = motor_cmd[valid_start:]
    cmd_valid = cmd_valid[np.isfinite(cmd_valid).all(axis=1)]
    if len(cmd_valid) >= 16:
        smoothed = _savgol_smooth(cmd_valid)
        cmd_jit = float(np.std(cmd_valid - smoothed, axis=0).mean())
    else:
        cmd_jit = float("nan")

    return {
        "pred_rmse_deg": float(np.sqrt((err_flat * err_flat).mean())),
        "pred_p95_abs_deg": float(np.percentile(np.abs(err_flat), 95)),
        "motor_jit_deg": cmd_jit,
        "n_predictions": int(err_flat.size),
    }


def load_motion_log(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (t, intent, joint_names) from a motion logger .npz file."""
    d = np.load(path, allow_pickle=False)
    return d["t"], d["intent"], [str(j) for j in d["joint_names"]]


def synth_small_motion_with_tremor(
    duration_s: float = 60.0,
    rate_hz: float = 30.0,
    n_joints: int = 7,
    tremor_amp_deg: float = 0.5,
    tremor_freq_hz: float = 10.0,
    motion_amp_deg: float = 3.0,
    motion_freq_hz: float = 0.5,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthesise a 'small deliberate motion + always-on hand tremor' signal.

    Models the exact failure mode the user described: leader moves 2-3
    deg slowly (motion_amp_deg @ motion_freq_hz), with a persistent
    10 Hz tremor (tremor_amp_deg) layered on top. Independent random
    phase per joint so the tremor isn't synchronised across DOFs.

    Returns (t, intent, joint_names) matching motion_log shape so the
    same backtest harness applies.
    """
    rng = np.random.default_rng(rng_seed)
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    intent = np.empty((len(t), n_joints), dtype=np.float64)
    for j in range(n_joints):
        # Per-joint random phase for both motion and tremor — avoids
        # creating an artificial synchrony that would let the estimator
        # "cheat" by averaging across joints (we evaluate per-joint, but
        # still — independent phases is the realistic case).
        phase_m = rng.uniform(0.0, 2 * np.pi)
        phase_t = rng.uniform(0.0, 2 * np.pi)
        # Add a small DC offset per joint so the magnitude of motion
        # relative to the absolute position varies realistically.
        offset = rng.uniform(-5.0, 5.0)
        intent[:, j] = (
            offset
            + motion_amp_deg * np.sin(2 * np.pi * motion_freq_hz * t + phase_m)
            + tremor_amp_deg * np.sin(2 * np.pi * tremor_freq_hz * t + phase_t)
        )
    return t, intent, [f"joint_{i}" for i in range(n_joints)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "motion_log",
        nargs="?",
        type=Path,
        help="Path to motion_*.npz file (omit + --synth to skip)",
    )
    p.add_argument("--L-ms", type=float, default=110.0, help="Lookahead horizon (ms)")
    p.add_argument("--window-ms", type=float, default=70.0, help="Estimator window (ms)")
    p.add_argument(
        "--synth",
        action="store_true",
        help="Also run on a synthetic small-motion + tremor signal",
    )
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    L_s = args.L_ms / 1000.0
    window_s = args.window_ms / 1000.0

    signals: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
    if args.motion_log is not None:
        if not args.motion_log.exists():
            print(f"Not found: {args.motion_log}", file=sys.stderr)
            return 1
        t, p_arr, _ = load_motion_log(args.motion_log)
        signals.append((f"real_teleop ({args.motion_log.name})", (t, p_arr)))
    if args.synth or args.motion_log is None:
        t, p_arr, _ = synth_small_motion_with_tremor()
        signals.append(
            (
                "synth_small_motion (3deg @ 0.5Hz + 0.5deg @ 10Hz tremor)",
                (t, p_arr),
            )
        )

    all_results: list[dict] = []
    for sig_name, (t, p_arr) in signals:
        fps = 1.0 / float(np.median(np.diff(t)))
        print(
            f"\n=== {sig_name} ===\n  duration: {t[-1]:.1f}s @ {fps:.0f}Hz, "
            f"{p_arr.shape[1]} joints | L={args.L_ms:.0f}ms window={args.window_ms:.0f}ms"
        )
        rows: list[dict] = []
        all_names = list(ESTIMATORS_STATELESS.keys()) + list(ESTIMATORS_STATEFUL.keys())
        for name in all_names:
            m = evaluate_estimator(t, p_arr, name, L_s, window_s)
            rows.append({"signal": sig_name, "estimator": name, **m})
        rows.sort(key=lambda r: r["motor_jit_deg"])
        print(f"  {'estimator':<22} {'pred_RMSE':>10} {'pred_p95':>10} {'motor_jit':>10} {'n_pred':>10}")
        print("  " + "-" * 64)
        for r in rows:
            print(
                f"  {r['estimator']:<22} "
                f"{r['pred_rmse_deg']:>10.4f} "
                f"{r['pred_p95_abs_deg']:>10.4f} "
                f"{r['motor_jit_deg']:>10.4f} "
                f"{r['n_predictions']:>10d}"
            )
        all_results.extend(rows)

    if args.output_json:
        args.output_json.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
