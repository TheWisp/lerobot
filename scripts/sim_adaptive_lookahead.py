# ruff: noqa: N802, N803, N806
# (Research script: L is the standard math symbol for lookahead; A/S/T are
# frequency-domain transforms; uppercase is intentional.)
"""Simulate the actual adaptive-lookahead loop on recorded data.

The static backtest in `backtest_lookahead.py` measured what
`cross_corr_lag` returns given a full episode. But the prototype's
adaptive controller doesn't run on full episodes — it runs every 2 s
over a rolling 3-second window with an α=0.5 low-pass and a hard
cap. The relevant question is whether THAT loop converges to the
right L, not whether the per-tick estimator is unbiased.

This script replays the loop:
  - tick every 2 s
  - look at last 3 s of (intent, state)
  - compute signed lag (broken / fixed estimator, switchable)
  - update L ← α·(L + lag) + (1 − α)·L
  - clip to [0, max_lookahead_s]

Plot L(t) for several episodes and check:
  1. Does L converge or oscillate?
  2. Where does it settle relative to the ground-truth optimal (133 ms
     from the direct L2 sweep)?
  3. Does the 110 ms cap matter?
  4. Does the amplitude-filter fix change the answer?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset


def cross_corr_lag(
    intent: np.ndarray,
    state: np.ndarray,
    dt: float,
    max_lag_s: float = 0.3,
    require_amplitude: bool = False,
    min_amplitude: float = 1.0,
) -> float | None:
    """Per-joint Pearson r maximization, average of confident joints in seconds.

    If ``require_amplitude=True``, only joints with state.std ≥ min_amplitude
    contribute to the confident set.
    """
    max_lag = int(max_lag_s / dt)
    confident_lags = []
    confident_amps = []
    for j in range(intent.shape[1]):
        a = intent[:, j]
        s = state[:, j]
        if require_amplitude and float(s.std()) < min_amplitude:
            continue
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
        if best_c >= 0.95:
            confident_lags.append(best_k * dt)
            confident_amps.append(float(s.std()))
    if not confident_lags:
        return None
    return float(np.mean(confident_lags))


def _shift_along_axis0(arr: np.ndarray, k: int) -> np.ndarray:
    """Return arr shifted by k samples along axis 0 (positive k = forward in time).

    Edge values are clamped (not wrapped) to keep the time series finite.
    """
    if k == 0:
        return arr.copy()
    out = np.empty_like(arr)
    if k > 0:
        out[:k] = arr[0]
        out[k:] = arr[:-k]
    else:
        out[k:] = arr[-1]
        out[:k] = arr[-k:]
    return out


def simulate(
    intent_t: np.ndarray,  # (T, n_joints) — raw leader positions
    dt: float,
    *,
    tau_truth_s: float,  # actual motor latency observed in this episode
    window_s: float = 3.0,
    update_period_s: float = 2.0,
    alpha: float = 0.5,
    initial_l_s: float = 0.08,
    max_l_s: float = 0.110,
    require_amplitude: bool = False,
) -> list[tuple[float, float, float | None]]:
    """Closed-loop sim: state(t) under lookahead L is intent(t + L - τ_truth).

    The prototype's adaptive update uses cross-corr(raw_intent, state).
    state(t) = action(t-τ) = intent(t-τ+L). So cross-corr should peak
    at k = τ - L, and the update rule converges to L = τ.
    """
    T = len(intent_t)
    win = int(window_s / dt)
    period = int(update_period_s / dt)
    L = initial_l_s
    history: list[tuple[float, float, float | None]] = []
    next_tick = win
    while next_tick < T:
        # Synthesize state under current L: state(t) = intent(t + L - τ).
        # To produce state_sim[i] = intent[i + (L_samples - τ_samples)],
        # apply `_shift_along_axis0(intent, k)` with k = τ_samples - L_samples
        # (positive k = "use arr from k positions ago" = state in the past
        # relative to intent, which is the expected behaviour when L < τ).
        shift_samples = int(round((tau_truth_s - L) / dt))
        state_sim = _shift_along_axis0(intent_t, shift_samples)
        win_intent = intent_t[next_tick - win : next_tick]
        win_state = state_sim[next_tick - win : next_tick]
        lag = cross_corr_lag(win_intent, win_state, dt, require_amplitude=require_amplitude)
        if lag is not None:
            target = L + lag
            L = alpha * target + (1.0 - alpha) * L
            L = max(0.0, min(L, max_l_s))
        history.append((next_tick * dt, L, lag))
        next_tick += period
    return history


def ground_truth_optimal_L(intent: np.ndarray, state: np.ndarray, dt: float, max_lag_s: float = 0.4) -> float:
    """Direct L2 sweep on actively-moving joints (state.std ≥ 1.0)."""
    moving = state.std(axis=0) >= 1.0
    if not moving.any():
        return float("nan")
    a = intent[:, moving]
    s = state[:, moving]
    joint_std = s.std(axis=0)
    joint_std[joint_std < 0.5] = 0.5
    best_L, best_err = 0.0, np.inf
    for k in range(0, int(max_lag_s / dt) + 1):
        if k > 0:
            aa, ss = a[:-k], s[k:]
        else:
            aa, ss = a, s
        err = float(np.mean(((ss - aa) / joint_std) ** 2))
        if err < best_err:
            best_err = err
            best_L = k * dt
    return best_L


def run_episode(ds: LeRobotDataset, ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
    ep = ds.meta.episodes[ep_idx]
    start, end = ep["dataset_from_index"], ep["dataset_to_index"]
    actions = np.stack([ds[i]["action"].cpu().numpy() for i in range(start, end)])
    states = np.stack([ds[i]["observation.state"].cpu().numpy() for i in range(start, end)])
    return actions, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="thewisp/cylinder_ring_assembly")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument(
        "--cap-ms",
        type=float,
        default=300.0,
        help="Cap on adaptive L. Default 300ms (no constraint). Use 110ms to see cap effect.",
    )
    parser.add_argument("--initial-l-ms", type=float, default=50.0)
    args = parser.parse_args()

    root = Path(args.root).expanduser() if args.root else None
    ds = LeRobotDataset(args.repo_id, root=str(root) if root else None)
    dt = 1.0 / ds.fps

    print(
        f"{'ep':>4s}  {'truth':>6s}  {'broken converged':>17s}  {'fixed converged':>16s}  {'broken trace':<40s}  {'fixed trace':<40s}"
    )
    print("-" * 140)

    broken_finals = []
    fixed_finals = []
    truths = []

    for ep_idx in range(args.n_episodes):
        intent, state = run_episode(ds, ep_idx)
        if intent.shape[0] < 200:
            continue
        truth_L = ground_truth_optimal_L(intent, state, dt) * 1000
        # Skip the bimanual problem by using a moving-only mask: pick right
        # arm only (joints 7-13) for both simulations
        intent_r = intent[:, 7:]
        hist_broken = simulate(
            intent_r,
            dt,
            tau_truth_s=truth_L / 1000.0,
            initial_l_s=args.initial_l_ms / 1000.0,
            max_l_s=args.cap_ms / 1000.0,
            require_amplitude=False,
        )
        hist_fixed = simulate(
            intent_r,
            dt,
            tau_truth_s=truth_L / 1000.0,
            initial_l_s=args.initial_l_ms / 1000.0,
            max_l_s=args.cap_ms / 1000.0,
            require_amplitude=True,
        )
        broken_last_L = hist_broken[-1][1] * 1000 if hist_broken else float("nan")
        fixed_last_L = hist_fixed[-1][1] * 1000 if hist_fixed else float("nan")
        # Trace: show every 2nd tick's L in ms
        broken_trace = " → ".join(f"{h[1] * 1000:.0f}" for h in hist_broken[::2])
        fixed_trace = " → ".join(f"{h[1] * 1000:.0f}" for h in hist_fixed[::2])
        print(
            f"{ep_idx:4d}  {truth_L:6.1f}  {broken_last_L:17.1f}  {fixed_last_L:16.1f}  "
            f"{broken_trace[:38]:<40s}  {fixed_trace[:38]:<40s}"
        )
        if not np.isnan(broken_last_L):
            broken_finals.append(broken_last_L)
        if not np.isnan(fixed_last_L):
            fixed_finals.append(fixed_last_L)
        if not np.isnan(truth_L):
            truths.append(truth_L)

    print("\n=== Summary (last-tick L vs L2-optimal truth) ===")
    if broken_finals:
        broken = np.array(broken_finals)
        truth = np.array(truths)
        print(
            f"  broken converged: p50={np.median(broken):5.1f}  IQR={np.percentile(broken, 75) - np.percentile(broken, 25):4.1f}  "
            f"|err vs truth|: p50={np.median(np.abs(broken - truth)):5.1f}  max={np.abs(broken - truth).max():5.1f}"
        )
    if fixed_finals:
        fixed = np.array(fixed_finals)
        print(
            f"  fixed converged:  p50={np.median(fixed):5.1f}  IQR={np.percentile(fixed, 75) - np.percentile(fixed, 25):4.1f}  "
            f"|err vs truth|: p50={np.median(np.abs(fixed - truth)):5.1f}  max={np.abs(fixed - truth).max():5.1f}"
        )
    if truths:
        print(
            f"  truth (L2 sweep): p50={np.median(truths):5.1f}  range=[{min(truths):.0f}, {max(truths):.0f}]"
        )
    print(f"\n  (cap was {args.cap_ms} ms; initial L was {args.initial_l_ms} ms)")


if __name__ == "__main__":
    main()
