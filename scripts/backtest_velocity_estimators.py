# ruff: noqa: I001, N803, N806
"""Backtest velocity-estimator choices for the predictive controller.

Question: when the controller extrapolates ``intent(now + L)`` from a
window of past intent samples, which estimator gives the smallest
prediction error? Specifically does the suspected LSQ-linear centering
bias actually inflate error vs. unbiased alternatives like quadratic
LSQ or causal forward-difference?

Test signals:

  1. The recorded ``white.trajectory.json`` — real hand-guided leader
     motion at 30 fps, upsampled to the controller's 200 Hz tick rate
     via linear interpolation (what the trajectory_replay teleop
     actually does at runtime). Slow motion — represents the bottom
     of the operator's frequency spectrum.

  2. Synthetic sinusoids at 1, 3, 5, 8 Hz. Lets us see the bias
     directly as a function of dominant motion frequency. Real teleop
     has mixed spectrum but typically 2-5 Hz dominant.

Estimators compared:

  * lsq_linear      — fit p(t)=a+bt over window, target = intent(now) + b·L
                      [current production behaviour; suspected centering bias]
  * lsq_quad_end    — fit p(t)=a+bt+ct² over window, evaluate SLOPE at
                      t=now (end of window), target = intent(now) + v_now·L
                      [proposed unbiased fix]
  * forward_diff    — v(now) = (p[n] − p[n−1]) / dt, target = p[n] + v·L
                      [simplest unbiased; expected noisier]
  * smoothed_fd     — forward-diff of EMA-smoothed signal (α=0.3)
                      [middle ground]
  * held            — target = intent(now), no extrapolation
                      [baseline — what no controller would do]

For each (signal, estimator) pair we report RMSE of the prediction at
``now + L=80 ms`` over the full timeline. Lower is better.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def lsq_linear(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    """Current production behaviour. p_win shape (n,)."""
    t_c = t_win - t_win.mean()
    denom = (t_c * t_c).sum()
    if denom < 1e-12:
        return float(p_win[-1])
    b = (t_c @ (p_win - p_win.mean())) / denom
    return float(p_win[-1] + b * L)


def lsq_quad_end(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    """Quadratic LSQ; slope evaluated at most-recent sample (window end).

    Fit p(t) = a + b·t + c·t² in t_rel = t - t_now. At t_rel=0, slope = b.
    """
    t_now = t_win[-1]
    t_rel = t_win - t_now  # ≤ 0
    A = np.stack([np.ones_like(t_rel), t_rel, t_rel * t_rel], axis=1)
    coef, *_ = np.linalg.lstsq(A, p_win, rcond=None)
    # coef = [a, b, c]. Slope at t_rel=0 is b. Value at t_rel=0 is a.
    a, b, _c = coef
    return float(a + b * L)


def forward_diff(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    """v(now) = (p[-1] - p[-2]) / dt."""
    if len(p_win) < 2:
        return float(p_win[-1])
    dt = t_win[-1] - t_win[-2]
    v = (p_win[-1] - p_win[-2]) / max(dt, 1e-9)
    return float(p_win[-1] + v * L)


def smoothed_fd(t_win: np.ndarray, p_win: np.ndarray, L: float, alpha: float = 0.3) -> float:
    """EMA-smooth signal first, then forward difference. Reduces noise."""
    smoothed = np.empty_like(p_win)
    smoothed[0] = p_win[0]
    for i in range(1, len(p_win)):
        smoothed[i] = alpha * p_win[i] + (1.0 - alpha) * smoothed[i - 1]
    if len(p_win) < 2:
        return float(smoothed[-1])
    dt = t_win[-1] - t_win[-2]
    v = (smoothed[-1] - smoothed[-2]) / max(dt, 1e-9)
    return float(p_win[-1] + v * L)


def held(t_win: np.ndarray, p_win: np.ndarray, L: float) -> float:
    """No extrapolation — predict no change."""
    return float(p_win[-1])


ESTIMATORS = {
    "lsq_linear (current)": lsq_linear,
    "lsq_quad_end (proposed)": lsq_quad_end,
    "forward_diff": forward_diff,
    "smoothed_fd": smoothed_fd,
    "held (baseline)": held,
}


# ---------------------------------------------------------------------------
# Test signals
# ---------------------------------------------------------------------------


def signal_recorded_trajectory(path: Path, sample_rate_hz: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """Load white.trajectory.json, upsample to ``sample_rate_hz`` via
    linear interpolation (what trajectory_replay does at runtime).

    Returns (t, signal) where signal shape is (n_samples, n_joints).
    """
    data = json.loads(path.read_text())
    ts = np.asarray(data["timestamps"], dtype=np.float64)
    ps = np.asarray(data["positions"], dtype=np.float64)  # (n_frames, n_joints)
    duration = float(ts[-1])
    new_ts = np.arange(0.0, duration, 1.0 / sample_rate_hz)
    new_ps = np.empty((len(new_ts), ps.shape[1]), dtype=np.float64)
    for j in range(ps.shape[1]):
        new_ps[:, j] = np.interp(new_ts, ts, ps[:, j])
    return new_ts, new_ps


def signal_sinusoid(
    freq_hz: float,
    amplitude: float = 10.0,
    duration_s: float = 5.0,
    sample_rate_hz: float = 200.0,
    n_joints: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic sinusoid, all joints same phase."""
    t = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    p = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return t, np.stack([p] * n_joints, axis=1)


# ---------------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------------


def backtest_signal(
    t: np.ndarray,
    p: np.ndarray,
    estimator_fn,
    L: float,
    window_s: float,
) -> dict:
    """Run an estimator over a (t, p) signal. Returns RMSE / max error / etc.

    For each index i where both [i-window, i] and [i+L] are in bounds,
    compute predicted_p = estimator(t[i-window:i+1], p[..., i-window:i+1], L)
    and compare against true p[..., i + L_frames].
    """
    dt = t[1] - t[0]
    window_frames = max(2, int(round(window_s / dt)))
    L_frames = int(round(L / dt))
    n_samples, n_joints = p.shape
    errors = []
    for i in range(window_frames, n_samples - L_frames):
        t_win = t[i - window_frames + 1 : i + 1]
        for j in range(n_joints):
            pred = estimator_fn(t_win, p[i - window_frames + 1 : i + 1, j], L)
            true = p[i + L_frames, j]
            errors.append(pred - true)
    errors = np.asarray(errors)
    return {
        "rmse": float(np.sqrt((errors * errors).mean())),
        "p95_abs": float(np.percentile(np.abs(errors), 95)),
        "max_abs": float(np.abs(errors).max()),
        "n_samples": len(errors),
    }


def run_all(
    trajectory_path: Path,
    L_s: float = 0.080,
    window_s: float = 0.070,
    sample_rate_hz: float = 200.0,
) -> None:
    signals = [
        (
            f"recorded (white.trajectory.json @ {sample_rate_hz:.0f}Hz)",
            signal_recorded_trajectory(trajectory_path, sample_rate_hz),
        ),
        ("sinusoid 1 Hz", signal_sinusoid(1.0, sample_rate_hz=sample_rate_hz)),
        ("sinusoid 3 Hz", signal_sinusoid(3.0, sample_rate_hz=sample_rate_hz)),
        ("sinusoid 5 Hz", signal_sinusoid(5.0, sample_rate_hz=sample_rate_hz)),
        ("sinusoid 8 Hz", signal_sinusoid(8.0, sample_rate_hz=sample_rate_hz)),
    ]

    print(
        f"\nBacktest: predict intent(now + {L_s * 1000:.0f}ms) from "
        f"{window_s * 1000:.0f}ms window at {sample_rate_hz:.0f}Hz\n"
    )

    for sig_name, (t, p) in signals:
        results = {}
        for est_name, est_fn in ESTIMATORS.items():
            results[est_name] = backtest_signal(t, p, est_fn, L_s, window_s)
        # Print per-signal table
        print(f"=== {sig_name} ===")
        header = f"{'estimator':<28} {'RMSE':>12} {'p95 |err|':>12} {'max |err|':>12}"
        print(header)
        print("-" * len(header))
        # Sort by RMSE ascending for visual clarity
        for est_name, r in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
            print(f"{est_name:<28} {r['rmse']:>12.4f} {r['p95_abs']:>12.4f} {r['max_abs']:>12.4f}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("~/.config/lerobot/robots/white.trajectory.json").expanduser(),
    )
    parser.add_argument("--L_ms", type=float, default=80.0)
    parser.add_argument("--window_ms", type=float, default=70.0)
    parser.add_argument("--rate_hz", type=float, default=200.0)
    args = parser.parse_args()
    if not args.trajectory.exists():
        print(f"Trajectory not found: {args.trajectory}", file=sys.stderr)
        return 1
    run_all(
        args.trajectory,
        L_s=args.L_ms / 1000.0,
        window_s=args.window_ms / 1000.0,
        sample_rate_hz=args.rate_hz,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
