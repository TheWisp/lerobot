#!/usr/bin/env python
"""Analyze chunk-cadence backtest traces.

Reads .npz traces produced by `backtest_chunk_cadence.py`, computes:
  * per-config jitter = std(motor_cmd − savgol(motor_cmd, w=15, p=3)),
    per joint, then aggregated. Reading B from the design — each config's
    own smoothing so we measure pure HF jitter without conflating the
    intentional time shift.
  * action-to-state lag = cross-correlation peak lag between motor_cmd
    and state, per joint. Negative = motor_cmd leads state (good for the
    lookahead config).
  * chunk-boundary jump = |motor_cmd[k] − motor_cmd[k−1]| sampled at
    every tick whose emission_idx is the first tick of a new emission.

Usage:
  scripts/analyze_chunk_cadence.py outputs/chunk_cadence/trace_*.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def _savgol(x: np.ndarray, window: int = 15, polyorder: int = 3) -> np.ndarray:
    """Per-column Savitzky-Golay smoothing. Clamps the window to the signal length."""
    w = min(window, len(x) - (1 - len(x) % 2))
    if w < polyorder + 2:
        return x.copy()
    if w % 2 == 0:
        w -= 1
    return savgol_filter(x, w, polyorder, axis=0)


def _xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Lag k where a most resembles b. Returns: negative = a leads b.

    Implementation: per-segment Pearson correlation at each candidate lag,
    not the global-mean-subtracted scipy.signal.correlate form. With
    monotonically trending signals (e.g. an arm joint sweeping from 100 to
    −50 deg), the global-mean form is dragged toward lag=0 because the
    overall trend dominates the inner product and a small temporal shift
    barely budges it. Per-segment Pearson uses each lag's overlap region
    as its own sample — local mean, local std, local inner product — so
    the temporal alignment of the SHAPE wins over the trend amplitude.

    For SO-107 measurements this matters: motor τ ≈ 80-100 ms (≈ 2-3
    frames at 30 Hz) gets reported as 0 frames by the trend-biased form,
    but cleanly as -3 frames by per-segment Pearson.
    """
    n = len(a)
    if n < 2 * max_lag + 5:
        return 0
    best_lag = 0
    best_corr = -2.0
    for k in range(-max_lag, max_lag + 1):
        if k < 0:
            a_s, b_s = a[: n + k], b[-k:]
        elif k > 0:
            a_s, b_s = a[k:], b[: n - k]
        else:
            a_s, b_s = a, b
        m = len(a_s)
        if m < 5:
            continue
        a_c = a_s - a_s.mean()
        b_c = b_s - b_s.mean()
        denom = a_c.std() * b_c.std()
        if denom < 1e-9:
            continue
        c = float((a_c * b_c).mean() / denom)
        if c > best_corr:
            best_corr = c
            best_lag = k
    return best_lag


def analyze(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    t = d["t"]
    motor_lookahead = d["motor_cmd_lookahead"]
    motor_no_lookahead = d["motor_cmd_no_lookahead"]
    state = d["state"]
    joint_names = [str(j) for j in d["joint_names"]]
    emission_idx = d["emission_idx"]
    fps_est = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 30.0

    # Jitter: each config its own smoothing (Reading B).
    smooth_lookahead = _savgol(motor_lookahead)
    smooth_no_lookahead = _savgol(motor_no_lookahead)
    jitter_lookahead = np.std(motor_lookahead - smooth_lookahead, axis=0)
    jitter_no_lookahead = np.std(motor_no_lookahead - smooth_no_lookahead, axis=0)

    # Action-to-state lag, per joint. The lookahead's motor_cmd should lead
    # state by ~lookahead_ms / 1000 * fps frames; the no-lookahead's
    # motor_cmd should lag state by τ_motor / 1000 * fps frames. Joints with
    # almost no motion get a lag of 0 reported but are excluded from the
    # cross-joint median — for those, the correlation peak is dominated by
    # noise and the lag estimate is meaningless. Threshold of 1 deg state
    # std means roughly "joint moved more than servo step resolution".
    max_lag = max(1, int(0.5 * fps_est))  # ±500 ms search window
    # Higher threshold than "any motion": below ~20 deg state std, the
    # cross-correlation peak is dominated by the bias/noise rather than
    # the true temporal alignment and the per-joint lag becomes spurious.
    # Reported numbers use only joints that clearly moved during the run.
    motion_mask = state.std(axis=0) > 20.0
    lag_lookahead = [
        _xcorr_lag(motor_lookahead[:, j], state[:, j], max_lag) for j in range(motor_lookahead.shape[1])
    ]
    lag_no_lookahead = [
        _xcorr_lag(motor_no_lookahead[:, j], state[:, j], max_lag) for j in range(motor_no_lookahead.shape[1])
    ]
    lag_lookahead_filtered = [lag for lag, m in zip(lag_lookahead, motion_mask, strict=True) if m]
    lag_no_lookahead_filtered = [lag for lag, m in zip(lag_no_lookahead, motion_mask, strict=True) if m]

    # Chunk-boundary jumps: |motor_cmd at the first tick after an emission
    # − motor_cmd at the tick before|. Only measured on the lookahead
    # config — that's where the boundary jumps actually happen at runtime.
    new_emission_mask = np.zeros_like(emission_idx, dtype=bool)
    new_emission_mask[1:] = emission_idx[1:] != emission_idx[:-1]
    if new_emission_mask.any():
        boundary_jumps = np.abs(
            motor_lookahead[new_emission_mask] - motor_lookahead[np.roll(new_emission_mask, -1)]
        )
        # Roll-by-one trick wraps around; mask the wraparound out.
        if new_emission_mask[-1]:
            boundary_jumps = boundary_jumps[:-1]
        jump_p50 = np.percentile(np.linalg.norm(boundary_jumps, axis=1), 50) if len(boundary_jumps) else 0.0
        jump_p95 = np.percentile(np.linalg.norm(boundary_jumps, axis=1), 95) if len(boundary_jumps) else 0.0
    else:
        jump_p50 = jump_p95 = 0.0

    fps = float(fps_est)
    return {
        "file": str(npz_path),
        "joints": joint_names,
        "fps": fps,
        "n_ticks": int(len(t)),
        "duration_s": float(t[-1]),
        "n_emissions": int(emission_idx.max() + 1) if len(emission_idx) else 0,
        "jitter_lookahead_deg": jitter_lookahead.tolist(),
        "jitter_no_lookahead_deg": jitter_no_lookahead.tolist(),
        "jitter_lookahead_mean_deg": float(jitter_lookahead.mean()),
        "jitter_no_lookahead_mean_deg": float(jitter_no_lookahead.mean()),
        "lag_lookahead_frames": lag_lookahead,
        "lag_no_lookahead_frames": lag_no_lookahead,
        "n_moving_joints": int(motion_mask.sum()),
        "lag_lookahead_ms_median": float(np.median(lag_lookahead_filtered) / fps * 1000)
        if lag_lookahead_filtered
        else 0.0,
        "lag_no_lookahead_ms_median": float(np.median(lag_no_lookahead_filtered) / fps * 1000)
        if lag_no_lookahead_filtered
        else 0.0,
        "boundary_jump_p50_deg": float(jump_p50),
        "boundary_jump_p95_deg": float(jump_p95),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+", help="Trace .npz files")
    p.add_argument("--output-json", default=None, help="Write summary to this file")
    args = p.parse_args()

    rows = []
    for tr in args.traces:
        rows.append(analyze(Path(tr)))

    # Pretty-print summary.
    print(f"{'file':<60} {'jit_la':>8} {'jit_nl':>8} {'lag_la_ms':>10} {'lag_nl_ms':>10} {'jump95':>8}")
    for r in rows:
        print(
            f"{Path(r['file']).name:<60} "
            f"{r['jitter_lookahead_mean_deg']:>8.3f} "
            f"{r['jitter_no_lookahead_mean_deg']:>8.3f} "
            f"{r['lag_lookahead_ms_median']:>10.1f} "
            f"{r['lag_no_lookahead_ms_median']:>10.1f} "
            f"{r['boundary_jump_p95_deg']:>8.3f}"
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
