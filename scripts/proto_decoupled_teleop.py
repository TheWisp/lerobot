#!/usr/bin/env python
"""Prototype: decoupled control + planning rates with self-correcting lookahead.

Research scaffolding from the latency investigation. Sweep flag combinations
(--velocity-method, --corrector-alpha, --p-coefficient, --max-lookahead-ms)
to A/B prediction algorithms either on a deterministic trajectory replay
(`--leader-mode trajectory_blind` is the unattended option that uses the
trajectory as a synthetic leader and applies the live-style velocity
extrapolation) or on the real BiSO107Leader arm (`--leader-mode live`).

Architecture:

  - Control thread (high rate, e.g. 200 Hz): pure leader→follower loop.
    Calls TrajectoryReplayTeleop.get_action() with the current
    lookahead_s, writes Goal_Position to the follower bus.

  - Planning thread (low rate, e.g. 30 Hz): sync-reads follower state.
    Every 2 s, cross-correlates the last few seconds of (action, state)
    against the trajectory ground truth, subtracts the read-rate bias
    (~16 ms at 30 Hz), and low-passes the result into the shared
    lookahead_s. State estimate uses the actual read timestamps,
    interpolated onto control timestamps — matches the bias-corrected
    analysis we validated in probe_motor_send_rate.py.

  - Single follower bus_lock serializes write and read. Leader is
    trajectory_replay (no bus), so the leader side never blocks.

Honors the production setup: P=16 (configure() default), CW/CCW=0 +
MinStartupForce=0 (now locked in by the SO107Follower configure()
override). No PID overrides.

After the run, prints lag/RMSE/jerk against the trajectory GT and the
lookahead history so we can see whether self-correction converges to
the right value.

Usage:
  uv run python scripts/proto_decoupled_teleop.py \
      --trajectory ~/.config/lerobot/robots/white.trajectory.json \
      --adaptive
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from lerobot.robots.bi_so107_follower import BiSO107FollowerConfig
from lerobot.robots.bi_so107_follower.bi_so107_follower import BiSO107Follower
from lerobot.robots.safe_trajectory import validate_trajectory
from lerobot.teleoperators.bi_so107_leader.bi_so107_leader import BiSO107Leader
from lerobot.teleoperators.bi_so107_leader.config_bi_so107_leader import BiSO107LeaderConfig
from lerobot.teleoperators.trajectory_replay import (
    TrajectoryReplayTeleop,
    TrajectoryReplayTeleopConfig,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


def load_trajectory(path: Path) -> dict:
    traj = json.loads(path.read_text())
    validate_trajectory(traj)
    if not traj["timestamps"]:
        raise ValueError(f"trajectory at {path} is empty")
    return traj


def gt_action_at(elapsed: float, traj: dict, joints: list[str]) -> np.ndarray:
    """Ground-truth trajectory value at exactly `elapsed`, no lookahead."""
    ts = traj["timestamps"]
    pos = traj["positions"]
    traj_joints = traj["joints"]
    if elapsed >= ts[-1]:
        frame = pos[-1]
    elif elapsed <= ts[0]:
        frame = pos[0]
    else:
        idx = 0
        while idx + 1 < len(ts) and ts[idx + 1] <= elapsed:
            idx += 1
        t0, t1 = ts[idx], ts[idx + 1]
        alpha = (elapsed - t0) / (t1 - t0)
        p0, p1 = pos[idx], pos[idx + 1]
        frame = [a + (b - a) * alpha for a, b in zip(p0, p1, strict=True)]
    j_to_idx = {j: i for i, j in enumerate(traj_joints)}
    return np.array([frame[j_to_idx[j]] for j in joints], dtype=np.float64)


def cross_corr_lag(
    action_arr: np.ndarray, state_arr: np.ndarray, dt: float, max_lag_s: float = 0.3
) -> float | None:
    """Per-joint cross-corr, return mean signed lag of confident (corr >= 0.95) joints in seconds.

    Positive = state lags action (motor catching up — need more lookahead).
    Negative = state leads action (overshoot — need less lookahead).

    Negative lags must be reachable for the adaptive update to be a true
    fixed-point: clipping to k >= 0 turns the update into a one-way
    ratchet that can only increase lookahead, never decrease it.
    """
    max_lag = int(max_lag_s / dt)
    confident = []
    for j in range(action_arr.shape[1]):
        a = action_arr[:, j]
        s = state_arr[:, j]
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
            confident.append(best_k * dt)
    if not confident:
        return None
    return float(np.mean(confident))


class DecoupledRunner:
    def __init__(
        self,
        robot: BiSO107Follower,
        teleop: TrajectoryReplayTeleop | BiSO107Leader | None,
        traj: dict | None,
        joints: list[str],
        control_fps: int,
        obs_fps: int,
        adaptive: bool,
        initial_lookahead_ms: float,
        leader_mode: str,
        max_step_deg: float,
        duration_s: float,
        velocity_window_ms: float,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.traj = traj  # only used in trajectory mode
        self.leader_mode = leader_mode  # "trajectory", "live", or "trajectory_blind"
        self.duration = duration_s
        self.joints: list[str] = joints
        # Maximum per-iteration change in commanded position per joint (safety).
        # If extrapolation says "jump 30°", clamp it to this. Defaults to
        # something conservative for teleop (~3° per send at 200Hz = ~600°/s).
        self.max_step_deg = max_step_deg
        # Window over which leader velocity is estimated (live mode only).
        # Too short → noise amplified; too long → phase lag in the estimate.
        # 50 ms is a reasonable default.
        self.velocity_window_s = velocity_window_ms / 1000.0
        # Velocity estimator method — set by caller via attribute.
        self.velocity_method: str = "linear"
        # Predictor-corrector α: 1.0 = pure leader-based prediction (today's
        # default), <1.0 blends in a predictor that extrapolates from a(n-1)
        # forward by velocity·dt. Smaller α → smoother actions, more drift-prone.
        self.corrector_alpha: float = 1.0
        self.control_period = 1.0 / control_fps
        self.obs_period = 1.0 / obs_fps
        self.control_dt = 1.0 / control_fps

        self.adaptive = adaptive
        self._lookahead_s = float(initial_lookahead_ms) / 1000.0
        # Hard upper bound on lookahead. Beyond this, extrapolation overshoot
        # dominates the state signal — measurement gets unreliable, operator
        # feel degrades. Cap is the only thing standing between the
        # adaptive update and runaway when measurements are noisy.
        self.max_lookahead_s = 0.15  # set by caller
        self.lookahead_lock = threading.Lock()
        self.lookahead_history: list[tuple[float, float]] = [(0.0, self._lookahead_s)]

        self.bus_lock = threading.Lock()
        self.shutdown = threading.Event()
        self.start_t = 0.0

        # Logs: each entry is (elapsed_s, np.array of joint values)
        # Large enough to hold the whole run.
        self.action_log: deque[tuple[float, np.ndarray]] = deque(maxlen=20000)
        self.state_log: deque[tuple[float, np.ndarray]] = deque(maxlen=4000)
        # Raw leader position (no lookahead) at each control tick — used as
        # the "intent" signal for cross-corr and final analysis. In
        # trajectory mode this is GT at the elapsed time; in live mode it's
        # the leader's read position before extrapolation.
        self.leader_log: deque[tuple[float, np.ndarray]] = deque(maxlen=20000)
        # Short ring buffer of recent leader positions for velocity estimate
        # (live mode and trajectory_blind). Sized to cover ~velocity_window_s
        # at control rate.
        self._leader_ring: deque[tuple[float, np.ndarray]] = deque(maxlen=64)
        # Ring buffer of recent ACTIONS (post-clamp output values). Used by the
        # predictor-corrector to compute v_action (slope of action history) —
        # smoother than v_leader because actions are already filtered by the
        # prior predictor blends.
        self._action_ring: deque[tuple[float, np.ndarray]] = deque(maxlen=64)
        # Previous action sent — used by both the safety clamp and the
        # predictor step a(n-1) + v_action · dt.
        self._last_action: np.ndarray | None = None
        self.overruns_ctrl = 0
        self.overruns_obs = 0
        self.n_sends = 0
        self.n_reads = 0
        self.n_clamped = 0

    @property
    def lookahead_s(self) -> float:
        with self.lookahead_lock:
            return self._lookahead_s

    def _set_lookahead(self, value: float) -> None:
        with self.lookahead_lock:
            self._lookahead_s = max(0.0, min(value, self.max_lookahead_s))

    def _resolve_action(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (intended_leader_pos, action_to_send) for this tick.

        ``intended_leader_pos``: where the leader IS now (no lookahead applied).
        ``action_to_send``: what we'll write to the follower bus.

        Mode behaviors:
          - trajectory: action = trajectory(elapsed + lookahead).
            Uses GT future — prediction is exact by construction.
          - live: read physical leader, velocity-extrapolate from the ring.
          - trajectory_blind: use trajectory(elapsed) AS THE LEADER but
            apply the same velocity-extrapolation that live uses. Lets
            us evaluate the prediction algorithm unattended.
        """
        if self.leader_mode == "trajectory":
            # Trajectory teleop honors its config.lookahead_s internally.
            self.teleop.config.lookahead_s = self.lookahead_s
            action_dict = self.teleop.get_action()
            # Intent reference for analysis = trajectory at the current
            # elapsed time (no lookahead).
            intent = gt_action_at(elapsed, self.traj, self.joints)
            action = self._dict_to_arr(action_dict)
            return intent, action

        # live or trajectory_blind: read leader, extrapolate via velocity ring.
        if self.leader_mode == "trajectory_blind":
            # Use the trajectory as if it were the leader. No future
            # peeking — only this current sample. Velocity extrapolation
            # is the same as live mode, just on synthetic input.
            leader_arr = gt_action_at(elapsed, self.traj, self.joints)
        else:
            leader_dict = self.teleop.get_action()
            leader_arr = self._dict_to_arr(leader_dict)
        # Push current sample
        self._leader_ring.append((elapsed, leader_arr.copy()))
        # Velocity estimator — selectable for A/B comparison.
        #   2pt:   forward diff between window endpoints only. Cheap, noisy.
        #   linear: LSQ slope over all samples. ~1.65× less noisy + smoother actions.
        #   quad:   LSQ quadratic + acceleration term. Captures direction
        #           reversals on deterministic data, but amplifies sensor noise.
        cutoff = elapsed - self.velocity_window_s
        window = [(t_, p_) for t_, p_ in self._leader_ring if t_ >= cutoff]
        la = self.lookahead_s  # noqa: E741 — la is the natural variable name
        method = self.velocity_method
        velocity = np.zeros_like(leader_arr)
        if len(window) < 2:
            raw_shifted = leader_arr.copy()
        elif method == "2pt" or len(window) < 3:
            ts = np.array([t_ for t_, _ in window], dtype=np.float64)
            ps = np.stack([p_ for _, p_ in window])
            dt_w = ts[-1] - ts[0]
            velocity = (ps[-1] - ps[0]) / dt_w if dt_w > 1e-9 else np.zeros_like(leader_arr)
            raw_shifted = leader_arr + velocity * la
        elif method == "linear":
            ts = np.array([t_ for t_, _ in window], dtype=np.float64)
            ps = np.stack([p_ for _, p_ in window])
            ts_c = ts - ts.mean()
            denom = float((ts_c * ts_c).sum())
            velocity = (ts_c @ ps) / denom if denom > 1e-12 else np.zeros_like(leader_arr)
            raw_shifted = leader_arr + velocity * la
        elif method == "quad":
            ts = np.array([t_ for t_, _ in window], dtype=np.float64)
            ps = np.stack([p_ for _, p_ in window])
            tc = ts - elapsed  # so coefficients are at t=elapsed
            # Solve LSQ for quadratic fit: pos(t) = a + b·tc + c·tc².
            # The quadratic LSQ branch is documented as a known-bad option
            # on noisy live data — kept here for A/B regression testing only.
            xx = np.column_stack([np.ones_like(tc), tc, tc * tc])
            coeffs = np.linalg.lstsq(xx, ps, rcond=None)[0]
            velocity = coeffs[1]
            raw_shifted = leader_arr + coeffs[1] * la + coeffs[2] * (la * la)
        else:
            raw_shifted = leader_arr.copy()

        # Predictor-corrector step:
        #   predictor: a(n) = a(n-1) + v_action · dt
        #              v_action = LSQ slope of the action ring (smoother than
        #              v_leader because the action stream is already filtered
        #              by prior corrector blends). This is the key difference
        #              vs naive "blend with previous action": we propagate
        #              FORWARD by the smoothed velocity, not just hold position.
        #   corrector: blend with the fresh leader-based prediction.
        #              shifted = α · raw_shifted + (1 − α) · predictor
        # α = 1.0 disables the corrector (pure leader-based prediction, today's
        # default). α < 1.0 blends in the predictor, smoothing the action
        # stream. α = 0 would drift since the leader measurement is ignored.
        alpha = self.corrector_alpha
        if alpha < 1.0 and self._last_action is not None and len(self._action_ring) >= 2:
            # v_action from action ring over the same window as v_leader.
            a_cutoff = elapsed - self.velocity_window_s
            a_window = [(t_, a_) for t_, a_ in self._action_ring if t_ >= a_cutoff]
            if len(a_window) >= 2:
                ts_a = np.array([t_ for t_, _ in a_window], dtype=np.float64)
                ps_a = np.stack([a_ for _, a_ in a_window])
                ts_a_c = ts_a - ts_a.mean()
                denom_a = float((ts_a_c * ts_a_c).sum())
                v_action = (ts_a_c @ ps_a) / denom_a if denom_a > 1e-12 else np.zeros_like(leader_arr)
            else:
                v_action = velocity  # fallback to leader-derived
            predictor = self._last_action + v_action * self.control_dt
            shifted = alpha * raw_shifted + (1.0 - alpha) * predictor
        else:
            shifted = raw_shifted
        # Safety clamp: cap per-step delta vs the last action we sent.
        if self._last_action is not None:
            delta = shifted - self._last_action
            cap = self.max_step_deg
            if np.any(np.abs(delta) > cap):
                clamped = np.clip(delta, -cap, cap)
                shifted = self._last_action + clamped
                self.n_clamped += 1
        return leader_arr, shifted

    def _dict_to_arr(self, action_dict: dict[str, float]) -> np.ndarray:
        """Match dict keys to the canonical joint order. Joints in the
        trajectory file are already 'left_{m}.pos' / 'right_{m}.pos';
        leader.get_action() returns the same key shape."""
        return np.array([action_dict[j] for j in self.joints], dtype=np.float64)

    def control_thread(self) -> None:
        next_t = time.perf_counter()
        while not self.shutdown.is_set():
            t_send = time.perf_counter()
            elapsed = t_send - self.start_t

            try:
                intent, action_arr = self._resolve_action(elapsed)
            except Exception as e:
                print(f"resolve_action error: {e}", file=sys.stderr)
                break

            action_dict = {j: float(v) for j, v in zip(self.joints, action_arr, strict=True)}

            try:
                with self.bus_lock:
                    self.robot.send_action(action_dict)
            except Exception as e:
                print(f"send_action error: {e}", file=sys.stderr)
                break

            self.action_log.append((elapsed, action_arr))
            self.leader_log.append((elapsed, intent))
            self._action_ring.append((elapsed, action_arr.copy()))
            self._last_action = action_arr
            self.n_sends += 1

            if self.leader_mode == "trajectory" and self.teleop.is_exhausted:
                self.shutdown.set()
                break
            if self.leader_mode in ("live", "trajectory_blind") and elapsed >= self.duration:
                self.shutdown.set()
                break

            next_t += self.control_period
            now = time.perf_counter()
            sleep_for = next_t - now
            if sleep_for > 0:
                precise_sleep(sleep_for)
            else:
                self.overruns_ctrl += 1
                next_t = now

    def planning_thread(self) -> None:
        next_t = time.perf_counter()
        last_update = next_t
        while not self.shutdown.is_set():
            try:
                with self.bus_lock:
                    # robot.get_observation() includes cameras; we configured
                    # cameras={} so this is motor-only and returns calibrated.
                    obs = self.robot.get_observation()
            except Exception as e:
                print(f"get_observation error: {e}", file=sys.stderr)
                break

            t_read = time.perf_counter()
            elapsed = t_read - self.start_t
            state_arr = np.array([obs[j] for j in self.joints], dtype=np.float64)
            self.state_log.append((elapsed, state_arr))
            self.n_reads += 1

            if self.adaptive and (t_read - last_update) > 2.0:
                self._update_lookahead_from_recent()
                last_update = t_read

            next_t += self.obs_period
            now = time.perf_counter()
            sleep_for = next_t - now
            if sleep_for > 0:
                precise_sleep(sleep_for)
            else:
                self.overruns_obs += 1
                next_t = now

    def _update_lookahead_from_recent(self) -> None:
        """Cross-corr state vs intended leader position over the last few
        seconds, update lookahead with bias correction + low-pass.

        Trajectory mode: intent = trajectory(elapsed) (we logged it).
        Live mode: intent = raw leader read (we logged it).
        """
        if len(self.state_log) < 30 or len(self.action_log) < 200:
            return
        recent_end = self.action_log[-1][0]
        window_start = max(0.0, recent_end - 3.0)

        # Intent (leader_log) sampled at control-rate timestamps
        ctrl_times_and_intent = [(t, p) for t, p in self.leader_log if t >= window_start]
        if len(ctrl_times_and_intent) < 100:
            return
        ctrl_times = np.array([t for t, _ in ctrl_times_and_intent], dtype=np.float64)
        gt = np.stack([p for _, p in ctrl_times_and_intent])

        # Read-time-corrected state at each control timestamp: build a
        # sparse (t_read, state) series and linearly interpolate onto
        # ctrl_times. Removes the stair-casing bias.
        read_pairs = [(t, s) for t, s in self.state_log if t >= window_start - 0.5]
        if len(read_pairs) < 5:
            return
        t_reads = np.array([t for t, _ in read_pairs], dtype=np.float64)
        s_reads = np.stack([s for _, s in read_pairs])  # (n_reads, n_joints)
        state_at_ctrl = np.empty((len(ctrl_times), len(self.joints)), dtype=np.float64)
        for j_idx in range(len(self.joints)):
            state_at_ctrl[:, j_idx] = np.interp(ctrl_times, t_reads, s_reads[:, j_idx])

        # Symmetric cross-corr: returns SIGNED lag.
        # Positive: state still trails intent (need more lookahead).
        # Negative: state leads intent (overshoot — need less lookahead).
        lag = cross_corr_lag(gt, state_at_ctrl, self.control_dt)
        if lag is None:
            return

        # No bias subtraction — we already removed the stair-casing bias
        # via np.interp onto control timestamps. Whatever lag remains is
        # the real residual.
        # Update is a true fixed-point: lookahead converges when lag = 0.
        # Hard cap in _set_lookahead prevents runaway from noise.
        alpha = 0.5
        target = self.lookahead_s + lag
        new_la = alpha * target + (1.0 - alpha) * self.lookahead_s
        self._set_lookahead(new_la)

        elapsed_now = self.action_log[-1][0]
        self.lookahead_history.append((elapsed_now, self.lookahead_s))
        sign = "+" if lag >= 0 else "−"
        print(
            f"[{elapsed_now:5.1f}s] signed_lag={sign}{abs(lag) * 1000:5.1f} ms  "
            f"→ lookahead={self.lookahead_s * 1000:6.1f} ms  (cap {self.max_lookahead_s * 1000:.0f})"
        )

    def run(self) -> None:
        self.start_t = time.perf_counter()
        ctrl_thread = threading.Thread(target=self.control_thread, name="control", daemon=True)
        obs_thread = threading.Thread(target=self.planning_thread, name="planning", daemon=True)
        ctrl_thread.start()
        obs_thread.start()

        # Wait until shutdown or trajectory duration + slack
        ctrl_thread.join(timeout=self.duration + 5.0)
        self.shutdown.set()
        obs_thread.join(timeout=2.0)


def park_to_rest(robot: BiSO107Follower, traj: dict) -> None:
    """Move to trajectory's first frame as a safe rest pose."""
    try:
        from lerobot.robots.rest_position import move_to_rest_position
    except Exception:
        return
    joints = traj["joints"]
    start_pose = {j: float(traj["positions"][0][i]) for i, j in enumerate(joints)}
    try:
        move_to_rest_position(robot, start_pose, duration_s=2.0)
    except Exception as e:
        print(f"park-to-rest failed: {e}", file=sys.stderr)


def analyze(runner: DecoupledRunner) -> None:
    if not runner.action_log or not runner.state_log:
        print("no data logged")
        return

    # Convert to arrays
    t_send = np.array([t for t, _ in runner.action_log], dtype=np.float64)
    action_arr = np.stack([a for _, a in runner.action_log])

    t_read = np.array([t for t, _ in runner.state_log], dtype=np.float64)
    state_raw = np.stack([s for _, s in runner.state_log])

    # Drop the first 1 second of samples from the analysis. Right after
    # start, the follower is wherever it parked, the leader is wherever
    # the operator is holding it, and adaptive hasn't yet picked sensible
    # lookahead. All three pollute the metrics. Lookahead convergence trace
    # stays unfiltered.
    warmup_mask = t_send >= 1.0
    if warmup_mask.sum() >= 100:
        t_send = t_send[warmup_mask]
        action_arr = action_arr[warmup_mask]

    # Interpolate state onto control grid
    state_at_send = np.empty_like(action_arr)
    for j_idx in range(action_arr.shape[1]):
        state_at_send[:, j_idx] = np.interp(t_send, t_read, state_raw[:, j_idx])

    # GT at each control time (intent). Use the logged leader/GT samples
    # so this works identically for trajectory and live modes.
    leader_times = np.array([t for t, _ in runner.leader_log], dtype=np.float64)
    leader_arr = np.stack([p for _, p in runner.leader_log])
    gt_arr = np.empty_like(action_arr)
    for j_idx in range(action_arr.shape[1]):
        gt_arr[:, j_idx] = np.interp(t_send, leader_times, leader_arr[:, j_idx])

    dt = runner.control_dt

    print()
    print("--- end-of-run analysis ---")
    print(
        f"control: {runner.n_sends} sends  ({runner.n_sends / runner.duration:.1f} Hz effective, "
        f"overruns {runner.overruns_ctrl})"
    )
    print(
        f"obs:     {runner.n_reads} reads  ({runner.n_reads / runner.duration:.1f} Hz effective, "
        f"overruns {runner.overruns_obs})"
    )

    if runner.leader_mode == "live":
        print(f"action clamps (per-step delta > {runner.max_step_deg}°): {runner.n_clamped}")

    # Helpers used by both jitter sections.
    win = 20  # 100 ms moving average → −3 dB at ~3 Hz. Residual captures motion + noise above ~3 Hz.
    kernel = np.ones(win) / win

    def _highpass_std(arr: np.ndarray) -> np.ndarray:
        smoothed = np.empty_like(arr)
        for jj in range(arr.shape[1]):
            smoothed[:, jj] = np.convolve(arr[:, jj], kernel, mode="same")
        return (arr - smoothed).std(axis=0)

    def _jerk_per_joint(arr: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(np.diff(arr, n=2, axis=0)), axis=0)

    # ===== 0. TIME-ALIGNED LAG =====
    # Find the integer shift k that best aligns state(t) with leader(t−k).
    # Positive = follower behind leader by k ms (typical motor τ).
    # Negative = follower leads leader (overshoot from lookahead).
    # |~0| = perfect tracking when time-shifted.
    print()
    print("--- time-aligned lag (cross-correlation of state vs leader) ---")
    lag = cross_corr_lag(gt_arr, state_at_send, dt)
    if lag is None:
        print("  (insufficient signal variance — leader didn't move enough)")
    else:
        sign = "+" if lag >= 0 else "−"
        print(f"  state-vs-leader lag: {sign}{abs(lag) * 1000:.1f} ms")

    # ===== 1. FIDELITY =====
    # state-vs-leader at MATCHED times. Smaller = follower is closer to where
    # leader actually is right now. Includes both steady-state offset and
    # oscillation amplitude (RMSE doesn't cancel signs).
    print()
    print("--- fidelity: state[t] vs leader[t] at matched times (smaller = closer tracking) ---")
    fidelity_err = state_at_send - gt_arr  # gt_arr is the leader at t (live) or trajectory(t) (blind)
    fidelity_rmse = np.sqrt(np.mean(fidelity_err**2, axis=0))
    fidelity_p95 = np.percentile(np.abs(fidelity_err), 95, axis=0)
    fidelity_max = np.abs(fidelity_err).max(axis=0)
    print(f"{'joint':<26}{'RMSE':>10}{'p95|err|':>10}{'max|err|':>10}")
    for j_idx, j_name in enumerate(runner.joints):
        print(
            f"  {j_name:<24}"
            f"{fidelity_rmse[j_idx]:>10.4f}"
            f"{fidelity_p95[j_idx]:>10.3f}"
            f"{fidelity_max[j_idx]:>10.3f}"
        )
    overall_rmse = float(np.sqrt(np.mean(fidelity_rmse**2)))
    print(
        f"  {'OVERALL':<24}{overall_rmse:>10.4f}{float(fidelity_p95.mean()):>10.3f}{float(fidelity_max.mean()):>10.3f}"
    )

    # Leader's own jerk — the reference for what "natural motion" looks like
    # this session. Used to compute "excess" jerk below: how much the algorithm
    # / motor added BEYOND what the leader already had. Excess metrics are
    # invariant to how fast or slow the operator was moving — they only
    # capture the algorithm's contribution. This is what should be compared
    # across different teleop sessions.
    leader_jerk_per_joint = _jerk_per_joint(gt_arr)
    leader_hp_per_joint = _highpass_std(gt_arr)
    leader_jerk_overall = float(leader_jerk_per_joint.mean())
    leader_hp_overall = float(leader_hp_per_joint.mean())

    # ===== 2. ACTION JITTER =====
    print()
    print("--- action jitter: command stream high-frequency content ---")
    action_jerk_per_joint = _jerk_per_joint(action_arr)
    action_hp_per_joint = _highpass_std(action_arr)
    excess_action_jerk = action_jerk_per_joint - leader_jerk_per_joint
    excess_action_hp = action_hp_per_joint - leader_hp_per_joint
    print(f"{'joint':<26}{'leader|″|':>11}{'action|″|':>11}{'excess|″|':>11}{'excess HP σ':>13}")
    for j_idx, j_name in enumerate(runner.joints):
        print(
            f"  {j_name:<24}"
            f"{leader_jerk_per_joint[j_idx]:>11.5f}"
            f"{action_jerk_per_joint[j_idx]:>11.5f}"
            f"{excess_action_jerk[j_idx]:>+11.5f}"
            f"{excess_action_hp[j_idx]:>+13.4f}"
        )
    overall_action_jerk = float(action_jerk_per_joint.mean())
    overall_action_hp = float(action_hp_per_joint.mean())
    print(
        f"  {'OVERALL':<24}"
        f"{leader_jerk_overall:>11.5f}"
        f"{overall_action_jerk:>11.5f}"
        f"{overall_action_jerk - leader_jerk_overall:>+11.5f}"
        f"{overall_action_hp - leader_hp_overall:>+13.4f}"
    )
    print("  (excess = action − leader. Comparable across sessions; smaller = algorithm adds less noise.)")

    # ===== 2.5 PLATEAU JITTER (regime-aware) =====
    # Operator motion isn't uniform: big sweeps mixed with idle periods.
    # Averaging jerk across the whole run dilutes the "idle" regime — which
    # is exactly when sensor noise + algorithm noise become visible to the
    # operator. Split by leader velocity and report jitter per-regime.
    print()
    print("--- plateau jitter: jitter ONLY when leader is quiet (idle / stationary) ---")
    leader_vel = np.gradient(gt_arr, dt, axis=0)
    print(f"{'joint':<26}{'leader|″|q':>11}{'action|″|q':>11}{'state|″|q':>11}{'state HP σ q':>14}")
    for j_idx, j_name in enumerate(runner.joints):
        v_abs = np.abs(leader_vel[:, j_idx])
        thresh = float(np.percentile(v_abs, 30))
        quiet = v_abs <= thresh
        # Drop the first/last sample for the 2nd diff so masks align
        quiet_diff = quiet[1:-1]
        if quiet_diff.sum() < 10:
            print(f"  {j_name:<24}{'—':>11}{'—':>11}{'—':>11}{'—':>14}")
            continue
        lj = float(np.mean(np.abs(np.diff(gt_arr[:, j_idx], n=2))[quiet_diff]))
        aj = float(np.mean(np.abs(np.diff(action_arr[:, j_idx], n=2))[quiet_diff]))
        sj = float(np.mean(np.abs(np.diff(state_at_send[:, j_idx], n=2))[quiet_diff]))
        # HP σ of state on quiet rows
        smoothed_s = np.convolve(state_at_send[:, j_idx], kernel, mode="same")
        s_hp = float((state_at_send[:, j_idx] - smoothed_s)[quiet].std())
        print(f"  {j_name:<24}{lj:>11.5f}{aj:>11.5f}{sj:>11.5f}{s_hp:>14.4f}")
    print("  (q = quiet only. state|″|q is the jitter you feel when supposedly holding still.)")

    # ===== 3. STATE JITTER =====
    print()
    print("--- state jitter: motor output high-frequency content (what operator feels) ---")
    state_jerk_per_joint = _jerk_per_joint(state_at_send)
    state_hp_per_joint = _highpass_std(state_at_send)
    excess_state_jerk = state_jerk_per_joint - leader_jerk_per_joint
    excess_state_hp = state_hp_per_joint - leader_hp_per_joint
    print(f"{'joint':<26}{'leader|″|':>11}{'state|″|':>11}{'excess|″|':>11}{'excess HP σ':>13}")
    for j_idx, j_name in enumerate(runner.joints):
        print(
            f"  {j_name:<24}"
            f"{leader_jerk_per_joint[j_idx]:>11.5f}"
            f"{state_jerk_per_joint[j_idx]:>11.5f}"
            f"{excess_state_jerk[j_idx]:>+11.5f}"
            f"{excess_state_hp[j_idx]:>+13.4f}"
        )
    overall_state_jerk = float(state_jerk_per_joint.mean())
    overall_state_hp = float(state_hp_per_joint.mean())
    print(
        f"  {'OVERALL':<24}"
        f"{leader_jerk_overall:>11.5f}"
        f"{overall_state_jerk:>11.5f}"
        f"{overall_state_jerk - leader_jerk_overall:>+11.5f}"
        f"{overall_state_hp - leader_hp_overall:>+13.4f}"
    )
    print("  (excess = state − leader. Negative = motor smoother than leader (good lowpass).)")

    # Lookahead convergence
    print()
    print("--- lookahead history (adaptive) ---")
    if not runner.lookahead_history:
        print("  (none)")
    else:
        for t, la in runner.lookahead_history:
            print(f"  t={t:5.2f}s  lookahead={la * 1000:.1f} ms")


def main(args: argparse.Namespace) -> int:
    init_logging()

    cfg = BiSO107FollowerConfig(
        id="white",
        left_arm_port=args.left_port,
        right_arm_port=args.right_port,
        left_arm_disable_torque_on_disconnect=False,
        right_arm_disable_torque_on_disconnect=False,
        left_arm_use_degrees=False,
        right_arm_use_degrees=False,
        cameras={},  # no cameras — keep observation cheap
    )
    robot = BiSO107Follower(cfg)
    robot.connect(calibrate=False)

    # PID override (after connect, before the loops start). configure() in
    # SOFollower writes P=16, D=32. Bump P=48 to shorten motor τ and test
    # whether smooth 200 Hz commands + lookahead make the prior shake
    # (observed at P=48 with coarse 30 Hz commands) tolerable.
    if args.p_coefficient is not None:
        for arm in (robot.left_arm, robot.right_arm):
            for motor in arm.bus.motors:
                arm.bus.write("P_Coefficient", motor, args.p_coefficient)
        print(f"overrode P_Coefficient = {args.p_coefficient} on all motors")
    if args.d_coefficient is not None:
        for arm in (robot.left_arm, robot.right_arm):
            for motor in arm.bus.motors:
                arm.bus.write("D_Coefficient", motor, args.d_coefficient)
        print(f"overrode D_Coefficient = {args.d_coefficient} on all motors")

    traj: dict | None = None
    teleop: TrajectoryReplayTeleop | BiSO107Leader | None = None
    if args.leader_mode == "trajectory":
        traj = load_trajectory(Path(args.trajectory).expanduser())
        teleop_cfg = TrajectoryReplayTeleopConfig(
            id="proto",
            trajectory_path=str(Path(args.trajectory).expanduser()),
            lookahead_s=args.initial_lookahead_ms / 1000.0,
        )
        teleop = TrajectoryReplayTeleop(teleop_cfg)
        teleop.connect(calibrate=False)
        joints = list(traj["joints"])
        duration_s = float(traj["timestamps"][-1])
    elif args.leader_mode == "trajectory_blind":
        # Synthetic leader: trajectory file, no teleop instance. Control
        # thread reads trajectory(elapsed) each tick and applies velocity
        # extrapolation — identical algorithm to live mode.
        traj = load_trajectory(Path(args.trajectory).expanduser())
        joints = list(traj["joints"])
        duration_s = float(traj["timestamps"][-1])
    else:
        leader_cfg = BiSO107LeaderConfig(
            id="blue",
            left_arm_port=args.leader_left_port,
            right_arm_port=args.leader_right_port,
        )
        teleop = BiSO107Leader(leader_cfg)
        teleop.connect(calibrate=False)
        joints = list(teleop.action_features.keys())
        duration_s = float(args.duration_s)

    runner = DecoupledRunner(
        robot=robot,
        teleop=teleop,
        traj=traj,
        joints=joints,
        control_fps=args.control_fps,
        obs_fps=args.obs_fps,
        adaptive=args.adaptive,
        initial_lookahead_ms=args.initial_lookahead_ms,
        leader_mode=args.leader_mode,
        max_step_deg=args.max_step_deg,
        duration_s=duration_s,
        velocity_window_ms=args.velocity_window_ms,
    )
    runner.max_lookahead_s = args.max_lookahead_ms / 1000.0
    runner.velocity_method = args.velocity_method
    runner.corrector_alpha = args.corrector_alpha

    print(
        f"running: mode={args.leader_mode}, control={args.control_fps} Hz, obs={args.obs_fps} Hz, "
        f"adaptive={args.adaptive}, initial_lookahead={args.initial_lookahead_ms} ms, "
        f"duration={duration_s:.1f} s"
    )
    if args.leader_mode == "live":
        print(
            f"live leader: velocity_window={args.velocity_window_ms} ms, "
            f"max_step={args.max_step_deg}° per send.  Ctrl-C or wait {duration_s:.0f}s to stop."
        )

    try:
        runner.run()
    finally:
        try:
            if args.leader_mode in ("trajectory", "trajectory_blind") and traj is not None:
                park_to_rest(robot, traj)
        finally:
            if teleop is not None:
                teleop.disconnect()
            robot.disconnect()

    analyze(runner)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--leader-mode",
        choices=["trajectory", "live", "trajectory_blind"],
        default="trajectory",
        help="trajectory: replay --trajectory with GT-future lookahead. "
        "live: read from a real bi_so107_leader, extrapolate via velocity "
        "ring buffer. "
        "trajectory_blind: use --trajectory as a SYNTHETIC leader (one "
        "current-time sample per tick, no future peek), then apply the "
        "same velocity-extrapolation as live mode. Lets us evaluate the "
        "prediction algorithm unattended.",
    )
    p.add_argument(
        "--trajectory",
        default="",
        help="Required for --leader-mode=trajectory. Path to .trajectory.json.",
    )
    p.add_argument("--left-port", default="/dev/ttyACM0", help="Follower left port")
    p.add_argument("--right-port", default="/dev/ttyACM2", help="Follower right port")
    p.add_argument("--leader-left-port", default="/dev/ttyACM3", help="Leader left port (live mode)")
    p.add_argument("--leader-right-port", default="/dev/ttyACM1", help="Leader right port (live mode)")
    p.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="How long to run in live mode (trajectory mode uses the file's duration).",
    )
    p.add_argument("--control-fps", type=int, default=200)
    p.add_argument("--obs-fps", type=int, default=30)
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable self-correcting lookahead. Otherwise lookahead stays at "
        "--initial-lookahead-ms throughout the run.",
    )
    p.add_argument(
        "--initial-lookahead-ms",
        type=float,
        default=0.0,
        help="Starting lookahead value. With --adaptive it converges from this; "
        "without --adaptive it stays fixed for the whole run.",
    )
    p.add_argument(
        "--max-lookahead-ms",
        type=float,
        default=130.0,
        help="Hard upper bound on lookahead (safety cap for adaptive). Beyond "
        "this, extrapolation overshoot dominates and operator feel degrades.",
    )
    p.add_argument(
        "--velocity-method",
        choices=["2pt", "linear", "quad"],
        default="linear",
        help="Velocity estimator: 2pt (forward diff endpoints), linear (LSQ "
        "slope over all samples), quad (LSQ with acceleration term).",
    )
    p.add_argument(
        "--corrector-alpha",
        type=float,
        default=1.0,
        help="Predictor-corrector weight: 1.0 = pure leader-based prediction "
        "(default), <1.0 blends in a(n-1) + velocity·dt to smooth the action "
        "stream. e.g., 0.3 = 30%% fresh prediction + 70%% extrapolation from "
        "previous action. α=0 would drift (no leader correction).",
    )
    p.add_argument(
        "--p-coefficient",
        type=int,
        default=None,
        help="Override follower P_Coefficient after connect. Default None = "
        "leave at configure() value (P=16). P=48 shortens motor τ by ~35 ms "
        "but historically caused visible shake with coarse 30 Hz commands.",
    )
    p.add_argument(
        "--d-coefficient",
        type=int,
        default=None,
        help="Override follower D_Coefficient after connect. Default None = "
        "leave at configure() value (D=32).",
    )
    p.add_argument(
        "--velocity-window-ms",
        type=float,
        default=50.0,
        help="Live mode: span of the velocity-estimate window (ring buffer "
        "lookback). Smaller = more responsive, more noise.",
    )
    p.add_argument(
        "--max-step-deg",
        type=float,
        default=5.0,
        help="Live mode safety clamp: maximum |action[t] − action[t-1]| per "
        "joint per send. Caps extrapolation overshoot from sudden leader "
        "jerks.",
    )
    sys.exit(main(p.parse_args()))
