# ruff: noqa: N802, N803, N806
# (Research script: L is the standard math symbol for lookahead; A/S/T are
# frequency-domain transforms; uppercase is intentional.)
"""Backtest different lookahead-estimation methods against recorded data.

Question: is the cross-correlation-based adaptive lookahead in
``proto_decoupled_teleop.py`` actually optimal, or do alternatives give a
better-tracking estimate of the motor's effective latency τ?

The recorded (action, state) pairs in any teleop dataset already contain
the answer: the motor was given ``action[t]`` and ended up at
``state[t]`` some τ later. We can measure τ four ways and compare:

  1. xcorr     — cross-correlation (current method); maximize Pearson r
  2. l2        — direct L2 minimization of state[t+L] vs action[t]
  3. phase     — FFT-based phase-shift per dominant frequency
  4. xcorr_pj  — cross-correlation, per-joint (current AVERAGES joints)

We report:
  - Per-method optimal L per episode (median, IQR across episodes)
  - Per-joint optimal L (does τ vary across joints?)
  - Method agreement (does L2 agree with xcorr?)
  - Residual tracking error AFTER applying the optimal L from each method

Usage:
    python scripts/backtest_lookahead.py \
        --repo-id thewisp/cylinder_ring_assembly \
        --root ~/.cache/huggingface/lerobot/thewisp/cylinder_ring_assembly \
        --max-episodes 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset


def load_episode_action_state(ds: LeRobotDataset, ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (action, state) arrays for one episode, shape (T, n_joints)."""
    ep = ds.meta.episodes[ep_idx]
    start = ep["dataset_from_index"]
    end = ep["dataset_to_index"]
    actions = []
    states = []
    for i in range(start, end):
        item = ds[i]
        actions.append(item["action"].cpu().numpy())
        states.append(item["observation.state"].cpu().numpy())
    return np.stack(actions), np.stack(states)


# ── Method 1: cross-correlation (current) ────────────────────────────────────


def xcorr_lag(
    action: np.ndarray,
    state: np.ndarray,
    dt: float,
    max_lag_s: float = 0.3,
    min_amplitude: float = 1.0,
) -> dict:
    """Per-joint signed lag via Pearson r maximization, plus the averaged figure.

    A joint is "confident" iff BOTH:
      - peak Pearson r ≥ 0.95 (waveform match), AND
      - state-side amplitude ≥ ``min_amplitude`` (signal isn't just noise)

    The amplitude gate is what the original `cross_corr_lag` was
    missing — a near-stationary joint can still hit 0.95 correlation
    against encoder noise alignment, and its noise-driven lag then
    drags the mean.

    Returns:
        per_joint, per_joint_corr, per_joint_amp, mean_confident,
        weighted_mean (amplitude-weighted average over confident joints).
    """
    max_lag = int(max_lag_s / dt)
    n_j = action.shape[1]
    per_joint = np.full(n_j, np.nan)
    per_joint_corr = np.zeros(n_j)
    per_joint_amp = np.zeros(n_j)
    for j in range(n_j):
        a = action[:, j]
        s = state[:, j]
        per_joint_amp[j] = float(s.std())
        best_k, best_c = 0, -np.inf
        for k in range(-max_lag, max_lag + 1):
            if k >= 0:
                aa = a[: len(a) - k] if k > 0 else a
                ss = s[k:] if k > 0 else s
            else:
                m = -k
                aa = a[m:]
                ss = s[: len(s) - m]
            aa_c = aa - aa.mean()
            ss_c = ss - ss.mean()
            na, ns = float(np.linalg.norm(aa_c)), float(np.linalg.norm(ss_c))
            if na < 1e-12 or ns < 1e-12:
                continue
            c = float(np.dot(aa_c, ss_c) / (na * ns))
            if c > best_c:
                best_c, best_k = c, k
        per_joint[j] = best_k * dt
        per_joint_corr[j] = best_c
    confident_mask = (per_joint_corr >= 0.95) & (per_joint_amp >= min_amplitude)
    if confident_mask.any():
        mean_confident = float(np.mean(per_joint[confident_mask]))
        weights = per_joint_amp[confident_mask]
        weighted_mean = float(np.sum(per_joint[confident_mask] * weights) / weights.sum())
    else:
        mean_confident = np.nan
        weighted_mean = np.nan
    return {
        "per_joint": per_joint,
        "per_joint_corr": per_joint_corr,
        "per_joint_amp": per_joint_amp,
        "mean_confident": mean_confident,
        "weighted_mean": weighted_mean,
    }


# ── Method 2: L2 minimization (sample-level shifts) ──────────────────────────


def l2_lag(action: np.ndarray, state: np.ndarray, dt: float, max_lag_s: float = 0.3) -> dict:
    """Find L that minimizes mean (state[t+L] - action[t])² across all joints jointly.

    Unlike xcorr, this is sensitive to magnitude — small motions count
    less because squared error is smaller. Reflects how a controller
    that actually USES the lookahead would experience tracking error.
    """
    max_lag = int(max_lag_s / dt)
    n_j = action.shape[1]
    # Per-joint normalization so big-range joints don't dominate
    joint_std = state.std(axis=0)
    joint_std = np.where(joint_std < 1e-9, 1.0, joint_std)

    best_lag = 0
    best_err = np.inf
    per_joint_at_best = np.zeros(n_j)
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            aa = action[: len(action) - k] if k > 0 else action
            ss = state[k:] if k > 0 else state
        else:
            m = -k
            aa = action[m:]
            ss = state[: len(state) - m]
        # Normalized per-joint MSE
        err_per_j = ((ss - aa) / joint_std).var(axis=0)
        err = float(err_per_j.mean())
        if err < best_err:
            best_err = err
            best_lag = k
            per_joint_at_best = err_per_j
    return {
        "lag_s": best_lag * dt,
        "residual_norm_var": best_err,
        "per_joint_residual_at_best": per_joint_at_best,
    }


# ── Method 3: phase-domain ───────────────────────────────────────────────────


def phase_lag(action: np.ndarray, state: np.ndarray, dt: float) -> dict:
    """Estimate effective τ from the phase shift of the dominant motion band.

    Compute the cross-spectrum action* × state, take the phase per
    frequency, weight by spectral magnitude. The weighted-mean unwrapped
    phase divided by 2π·f gives the effective time delay.

    Limits:
      - Phase from a single frequency bin gives -π / (2π·f) = -τ_at_f.
        For broadband motion, the delay is mostly frequency-independent
        (constant-τ system), so the weighted average should match xcorr.
        Disagreement → frequency-dependent latency, which signals the
        motor's response isn't a pure delay.
    """
    n_j = action.shape[1]
    # Window each joint, FFT, accumulate weighted phase
    T = len(action)
    if T < 64:
        return {"lag_s": float("nan"), "per_joint": np.full(n_j, np.nan)}
    # Use windowed signals to reduce spectral leakage
    window = np.hanning(T)
    per_joint = np.full(n_j, np.nan)
    for j in range(n_j):
        a = (action[:, j] - action[:, j].mean()) * window
        s = (state[:, j] - state[:, j].mean()) * window
        A = np.fft.rfft(a)
        S = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(T, d=dt)
        # Cross-spectrum; phase = angle(S/A) but use cross-correlation form
        cross = S * np.conj(A)
        mag = np.abs(cross)
        if mag.sum() < 1e-12:
            continue
        # Only use frequencies with non-trivial energy + below Nyquist/2
        valid = (freqs > 0.05) & (freqs < 0.5 / dt) & (mag > mag.max() * 0.05)
        if not valid.any():
            continue
        # Per-frequency delay: phase / (2π·f). Negative phase = state lags.
        phase = np.angle(cross[valid])
        f_v = freqs[valid]
        # τ = -phase / (2π f) (state lags action when phase is negative)
        tau_per_f = -phase / (2.0 * np.pi * f_v)
        # Weight by spectral magnitude
        weights = mag[valid] / mag[valid].sum()
        per_joint[j] = float(np.sum(tau_per_f * weights))
    # Confident: only joints with finite estimates
    valid_mask = ~np.isnan(per_joint)
    mean = float(np.mean(per_joint[valid_mask])) if valid_mask.any() else np.nan
    return {"lag_s": mean, "per_joint": per_joint}


# ── Backtest driver ──────────────────────────────────────────────────────────


def backtest_episode(action: np.ndarray, state: np.ndarray, dt: float) -> dict:
    """Run all methods on one episode and return their L estimates."""
    xc = xcorr_lag(action, state, dt)
    l2 = l2_lag(action, state, dt)
    ph = phase_lag(action, state, dt)
    return {
        "xcorr_mean_confident": xc["mean_confident"],
        "xcorr_weighted_mean": xc["weighted_mean"],
        "xcorr_per_joint": xc["per_joint"],
        "xcorr_per_joint_corr": xc["per_joint_corr"],
        "xcorr_per_joint_amp": xc["per_joint_amp"],
        "l2": l2["lag_s"],
        "phase": ph["lag_s"],
        "phase_per_joint": ph["per_joint"],
        "n_frames": len(action),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="thewisp/cylinder_ring_assembly")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--min-frames", type=int, default=60, help="Skip episodes shorter than this")
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else None
    print(f"Loading {args.repo_id}...")
    ds = LeRobotDataset(args.repo_id, root=str(root) if root else None)
    dt = 1.0 / ds.fps
    print(f"  fps={ds.fps}  dt={dt * 1000:.2f}ms  n_episodes={ds.meta.total_episodes}")

    n_ep = min(args.max_episodes, ds.meta.total_episodes)
    print(f"\nBacktesting first {n_ep} episodes...")

    results = []
    t0 = time.perf_counter()
    for ep_idx in range(n_ep):
        ep = ds.meta.episodes[ep_idx]
        if ep["length"] < args.min_frames:
            continue
        action, state = load_episode_action_state(ds, ep_idx)
        r = backtest_episode(action, state, dt)
        r["ep_idx"] = ep_idx
        results.append(r)
        if (ep_idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {ep_idx + 1}/{n_ep}  ({elapsed:.1f}s, {elapsed / (ep_idx + 1):.2f}s/ep)")

    # ── Summary ───────────────────────────────────────────────────────────
    xc = np.array([r["xcorr_mean_confident"] for r in results]) * 1000  # ms
    xcw = np.array([r["xcorr_weighted_mean"] for r in results]) * 1000  # amplitude-weighted
    l2 = np.array([r["l2"] for r in results]) * 1000
    ph = np.array([r["phase"] for r in results]) * 1000

    def stats(name: str, arr: np.ndarray):
        v = arr[~np.isnan(arr)]
        if len(v) == 0:
            print(f"  {name:>8}: no valid data")
            return
        p25, p50, p75 = np.percentile(v, [25, 50, 75])
        print(
            f"  {name:>8}: p50={p50:6.1f} ms  p25={p25:6.1f}  p75={p75:6.1f}  "
            f"IQR={p75 - p25:5.1f}  range=[{v.min():.1f}, {v.max():.1f}]  n={len(v)}"
        )

    print(f"\n=== Per-episode optimal lookahead L (n={len(results)} episodes) ===")
    stats("xcorr", xc)
    stats("xcorr-w", xcw)
    stats("L2", l2)
    stats("phase", ph)

    # Agreement between methods (per-episode pairwise)
    valid = ~(np.isnan(xc) | np.isnan(l2) | np.isnan(ph))
    if valid.sum() > 1:
        print("\n=== Method agreement (per-episode pairwise diff, ms) ===")
        diff_xc_l2 = xc[valid] - l2[valid]
        diff_xc_ph = xc[valid] - ph[valid]
        diff_l2_ph = l2[valid] - ph[valid]
        print(
            f"  xcorr - L2:    mean {diff_xc_l2.mean():+5.1f}  σ {diff_xc_l2.std():5.1f}  "
            f"|max| {np.abs(diff_xc_l2).max():.1f}"
        )
        print(
            f"  xcorr - phase: mean {diff_xc_ph.mean():+5.1f}  σ {diff_xc_ph.std():5.1f}  "
            f"|max| {np.abs(diff_xc_ph).max():.1f}"
        )
        print(
            f"  L2    - phase: mean {diff_l2_ph.mean():+5.1f}  σ {diff_l2_ph.std():5.1f}  "
            f"|max| {np.abs(diff_l2_ph).max():.1f}"
        )

    # Per-joint variance — does τ vary across joints?
    print("\n=== Per-joint optimal L from cross-corr (ms, median across episodes) ===")
    n_j = results[0]["xcorr_per_joint"].shape[0]
    joint_names = ds.features["action"].get("names", [f"j{i}" for i in range(n_j)])
    pj = np.stack([r["xcorr_per_joint"] * 1000 for r in results])  # (n_ep, n_j)
    pj_corr = np.stack([r["xcorr_per_joint_corr"] for r in results])
    for j in range(n_j):
        valid_ep = pj_corr[:, j] >= 0.95
        if valid_ep.sum() < 2:
            print(f"  {joint_names[j]:30s} (insufficient confident episodes)")
            continue
        vals = pj[valid_ep, j]
        print(
            f"  {joint_names[j]:30s} p50={np.median(vals):6.1f}  "
            f"IQR={np.percentile(vals, 75) - np.percentile(vals, 25):5.1f}  n={len(vals)}"
        )


if __name__ == "__main__":
    main()
