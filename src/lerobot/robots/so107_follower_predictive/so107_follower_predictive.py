#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SO-107 follower with the predictive-lookahead controller built in.

This is treated as a SEPARATE EMBODIMENT from the plain ``so107_follower``
even though the hardware is identical, because the trained-policy contract
is materially different:

  * ``so107_follower``: ``state(t) ≈ leader(t − τ)`` — motor lag visible in
    every observation. Policy implicitly compensates τ via action chunks.

  * ``so107_follower_predictive``: ``state(t) ≈ intent(t)`` — controller
    transparently compensates motor τ. Dataset has aligned action/state
    pairs at matching timestamps. Policy never observes τ.

Mixing datasets across the two regimes in training is a real hazard:
training-time state-action alignment would be inconsistent and the policy
would output actions that lurch when deployed. The robot_type
separation makes this prevention automatic — recording / training /
inference all key off the same ``robot_type`` string.

Architecture: controller logic lives in this module only. Caller-facing
interface (``Robot.send_action``, ``Robot.get_observation``) is unchanged.
A 200 Hz background thread does the actual motor writes; the caller's
``send_action(intent)`` is non-blocking and just publishes the latest
intent to the controller.

Lifted from scripts/proto_decoupled_teleop.py with the amplitude-gated
cross-correlation patch from commit 7a1e92c61.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.types import ActionChunk, RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.robot_utils import precise_sleep

from ..robot import Robot
from ..so_follower.so_follower import SO107Follower
from .config_so107_follower_predictive import SO107FollowerPredictiveRobotConfig

logger = logging.getLogger(__name__)


class SO107FollowerPredictive(SO107Follower):
    """SO-107 follower with predictive-lookahead controller always on.

    Treated as a distinct embodiment (own ``name`` / ``config_class``) so
    the dataset / policy contract is unambiguous.
    """

    config_class = SO107FollowerPredictiveRobotConfig
    name = "so107_follower_predictive"

    def __init__(self, config: SO107FollowerPredictiveRobotConfig):
        # Bypass SOFollower.__init__ (6-motor map) and SO107Follower.__init__
        # (which only differs by calling Robot.__init__ then setting up the
        # 7-motor map) to install our own motor map plus the controller-side
        # bookkeeping. Same reason SO107Follower itself bypasses
        # SOFollower.__init__: the motor count and IDs differ.
        Robot.__init__(self, config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "forearm_roll": Motor(4, "sts3215", norm_mode_body),
                "wrist_flex": Motor(5, "sts3215", norm_mode_body),
                "wrist_roll": Motor(6, "sts3215", norm_mode_body),
                "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._cached_motor_positions: dict[str, float] = {}

        # FeetechMotorsBus has no internal locking. The controller's 200 Hz
        # writer thread + the main-loop's 30 Hz reader share this lock.
        self._bus_lock = threading.Lock()
        # Controller instance is created here; its background thread is
        # started in ``connect()`` after the bus is open.
        self._controller = _PredictiveLookaheadController(self)

    # ── send_action / get_observation: controller-aware overrides ─────────

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Same connect flow as SO107Follower (which inherits from
        # SOFollower); we re-implement instead of calling super().connect()
        # because we need a precise insertion point for the controller
        # thread (between configure() and the "connected" log).
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info("Calibration mismatch or missing; running calibration")
            self.calibrate()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        # Start the controller AFTER configure() so motor PIDs are set and
        # the bus is fully initialised, but BEFORE we log "connected" so
        # the user sees a single coherent startup message.
        self._controller.start()
        logger.info(
            "%s connected (predictive controller: L=%.0fms, α=%.2f, %.0fHz, adaptive=%s)",
            self,
            self.config.lookahead_ms,
            self.config.corrector_alpha,
            self.config.control_rate_hz,
            self.config.adaptive,
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        with self._bus_lock:
            obs_dict = self._sync_read_with_motor_fallback("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Feed state to the controller's adaptive cross-correlation update.
        self._controller.observe_state(time.perf_counter(), obs_dict)

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def attach_teleop(self, teleop) -> None:
        """Bind a teleop the controller polls at control rate.

        With a teleop attached, the controller's 200 Hz tick calls
        ``teleop.get_action()`` directly to get the latest intent
        sample, bypassing ``_latest_intent`` set by ``send_action``.
        This is the path that benefits from a high-rate leader
        (``SO107LeaderHighRate``) whose ``get_action`` is a fast
        cached read — the controller's velocity LSQ then sees one
        distinct sample per tick (eliminating the under-shoot bias
        of stair-stepped 30 Hz intents).

        ``send_action`` continues to work as before; whichever path
        wrote most recently wins each tick. The loop driver typically
        keeps calling ``send_action`` for dataset recording. When a
        teleop is attached, the recorded action is what the loop
        driver passes in — usually ``teleop.get_action()`` — so the
        dataset still gets the operator's intent.

        Pass ``teleop=None`` to detach.
        """
        self._controller.set_teleop(teleop)
        if teleop is None:
            logger.info("%s: teleop detached; controller falls back to send_action push path", self)
        else:
            logger.info(
                "%s: bound to teleop %r — controller polls teleop.get_action() at %.0f Hz",
                self,
                type(teleop).__name__,
                self.config.control_rate_hz,
            )

    @check_if_not_connected
    def send_action(self, action: RobotAction | ActionChunk) -> RobotAction:
        """Publish intent to the controller (non-blocking).

        Two payload shapes:
          * ``RobotAction`` (a dict) — single intent at "now". The
            controller extrapolates ``now + L`` via velocity LSQ over
            the recent intent stream. Used by leader-arm teleop.
          * :class:`ActionChunk` — fixed-cadence horizon of intent
            samples starting at "now". The controller picks the
            lookahead target at index ``L * fps`` by interpolation;
            falls back to chunk-tail velocity extrapolation past the
            last frame. Used by trajectory replay + chunked policy
            inference.

        The recorded ``action`` is the operator's raw intent — ``frames[0]``
        when a chunk was passed, the dict itself otherwise. State
        observed via ``get_observation`` will track that intent with
        residual ≈ τ − L (≈ 0 at adaptive convergence). The actual
        motor command (the L-shifted target) is computed by the
        controller's 200 Hz thread and never visible to the caller.
        """
        t = time.perf_counter()
        self._controller.set_intent(t, action)
        return dict(action.frames[0]) if isinstance(action, ActionChunk) else action

    @check_if_not_connected
    def disconnect(self):
        # Best-effort teardown: each phase is wrapped so an earlier
        # failure does not leak resources from a later phase. The
        # controller MUST stop first — its 200 Hz writer would crash
        # writing into a closed bus.
        errors: list[BaseException] = []
        try:
            self._controller.stop()
        except BaseException as e:
            logger.exception("predictive controller stop failed")
            errors.append(e)
        try:
            self.bus.disconnect(self.config.disable_torque_on_disconnect)
        except BaseException as e:
            logger.exception("bus disconnect failed")
            errors.append(e)
        for cam_key, cam in self.cameras.items():
            try:
                cam.disconnect()
            except BaseException as e:
                logger.exception("camera %s disconnect failed", cam_key)
                errors.append(e)
        logger.info(f"{self} disconnected.")
        # Surface the first failure (Python lets caller see the rest via
        # __context__ if they care). Re-raising is important so the
        # operator hears about torque-disable failures etc.
        if errors:
            raise errors[0]


# ============================================================================
# Predictive-lookahead controller — private to this module.
# ============================================================================
#
# Lifted from scripts/proto_decoupled_teleop.py with the amplitude-gated
# cross-correlation patch (commit 7a1e92c61). Adapted to live inside the
# robot:
#
#   - Intent source: prototype reads from a ``teleop`` object every control
#     tick. Here the caller pushes intent via ``robot.send_action(intent)``,
#     which calls ``set_intent``. The control thread polls the latest
#     pushed intent on its own schedule.
#   - State source: prototype's planning thread does its own sync_read.
#     Here state is fed via ``robot.get_observation()`` → ``observe_state``,
#     so only one reader touches the bus.
#   - Bus lock: prototype owns ``bus_lock``. Here we share the robot's
#     ``_bus_lock`` so this controller and the main-thread read coordinate.
#
# What's deliberately rough (prototype, not production):
#   - ``max_relative_target`` safety clamp from SOFollower is not applied
#     here. Only the smaller ``max_step_deg`` per-tick clamp is enforced.
#     The bigger safety bound should be wired in before this hits production.
#   - No latency-monitor integration yet. The controller's L history,
#     residual lag, and adaptive update logs aren't published to the GUI
#     dashboard.
#   - No tests. The hardware-side validation lives in the prototype script
#     (scripts/proto_decoupled_teleop.py); the algorithmic regression
#     coverage lives in scripts/backtest_lookahead.py and
#     scripts/sim_adaptive_lookahead.py.


class _PredictiveLookaheadController:
    """Background thread + state for the predictive-lookahead controller.

    Lifecycle:
      ``__init__(robot)`` → ``start()`` (called from robot.connect()) →
      caller drives main loop (sending intents via ``robot.send_action`` →
      ``set_intent``, reading state via ``robot.get_observation`` →
      ``observe_state``) → ``stop()`` (called from robot.disconnect()).

    The control thread runs at ``config.control_rate_hz``, polls the
    latest intent, extrapolates by ``L · v_leader``, applies the
    predictor-corrector smoothing, clamps the per-step delta, and writes
    ``Goal_Position`` to the bus under ``robot._bus_lock``. The adaptive
    cross-correlation update runs from the same thread every 2 s.
    """

    # Cross-corr scan radius (seconds).
    _MAX_LAG_S: float = 0.3
    # Confidence floor for cross-corr peak.
    _CORR_FLOOR: float = 0.95
    # State-amplitude floor (motor units). Filters out stationary joints
    # whose encoder noise can produce spurious 0.95 correlations.
    _AMP_FLOOR: float = 1.0
    # Adaptive update cadence and rolling window.
    _UPDATE_PERIOD_S: float = 2.0
    _WINDOW_S: float = 3.0
    _ADAPTIVE_ALPHA: float = 0.5

    def __init__(self, robot: SO107FollowerPredictive):
        cfg = robot.config
        self._robot = robot
        self._bus = robot.bus
        self._bus_lock = robot._bus_lock
        # Lock-down: motor key ordering is captured once so we don't
        # rebuild the dict-to-array mapping every tick.
        self._motor_keys: list[str] = [f"{m}.pos" for m in self._bus.motors]

        # Parameters (immutable after __init__)
        self._max_lookahead_s = cfg.max_lookahead_ms / 1000.0
        self._velocity_window_s = cfg.velocity_window_ms / 1000.0
        self._corrector_alpha = cfg.corrector_alpha
        self._control_dt = 1.0 / cfg.control_rate_hz
        self._max_step = cfg.max_step_deg
        self._adaptive = cfg.adaptive
        self._velocity_estimator: str = cfg.velocity_estimator

        # Mutable state (control thread)
        self._lookahead_s = cfg.lookahead_ms / 1000.0
        self._intent_ring: deque[tuple[float, np.ndarray]] = deque(maxlen=64)
        self._action_ring: deque[tuple[float, np.ndarray]] = deque(maxlen=64)
        self._last_action: np.ndarray | None = None
        self._last_adaptive_t: float = 0.0

        # State log fed from get_observation() — read by control thread.
        # deque append + iteration is atomic on CPython under the GIL.
        self._state_log: deque[tuple[float, np.ndarray]] = deque(maxlen=200)
        # Intent log fed from set_intent() — also read by control thread.
        self._intent_log: deque[tuple[float, np.ndarray]] = deque(maxlen=2000)

        # Cross-thread variables for the latest intent (control thread reads
        # this each tick to know where to aim).
        self._target_lock = threading.Lock()
        self._latest_intent: np.ndarray | None = None
        # Chunk record (one per send_action call that passed an ActionChunk).
        # Layout: (received_at, fps, frames_arr) where frames_arr is
        # shape (N, n_motors). ``None`` means the latest send_action was a
        # single dict → controller takes the velocity-extrapolation path.
        # Latest-chunk-wins: a fresh chunk fully replaces any previous one;
        # a single-dict send replaces with None. Tracked under _target_lock
        # alongside _latest_intent so a tick sees a consistent snapshot.
        self._latest_chunk: tuple[float, float, np.ndarray] | None = None

        # Optional direct teleop binding. When set, ``_tick`` polls
        # ``teleop.get_action()`` each iteration instead of reading
        # ``_latest_intent`` (which is updated at the caller's
        # send_action cadence — typically 30 Hz). Polling a high-rate
        # teleop (SO107LeaderHighRate) at the controller's 200 Hz tick
        # gives the velocity LSQ ~14 distinct samples in a 70 ms
        # window — enough for an unbiased velocity estimate. Setting
        # this to None reverts to the send_action-driven path.
        self._teleop = None

        # Thread control
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._control_loop,
            name=f"{self._robot}_predictive_lookahead",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("%s controller thread did not stop within 2s", self._robot)
        self._thread = None

    # ── Caller-facing hooks (invoked from main thread) ────────────────────

    def set_intent(self, t: float, action: RobotAction | ActionChunk) -> None:
        """Receive the latest intent from the caller. Non-blocking.

        For an ``ActionChunk``, ``frames[0]`` is the "current intent" used
        as the fallback when the chunk is consumed. The rest of the chunk
        is held under ``_latest_chunk`` for the control thread's
        exact-lookup path.

        For a single ``RobotAction`` dict, ``_latest_chunk`` is cleared so
        subsequent ticks fall back to velocity extrapolation.

        NB: ``_intent_log`` is NOT populated here. It's populated from
        ``_tick`` instead, at the control rate, so the adaptive xcorr can
        use ``control_dt`` as its time unit cleanly (matches the
        prototype in scripts/proto_decoupled_teleop.py where ``leader_log``
        is appended inside the control loop). Logging here would have
        sample spacing = caller rate (typically 30 Hz), and the xcorr
        would mis-convert index shifts to time by a factor of
        ``caller_rate / control_rate``.
        """
        if isinstance(action, ActionChunk):
            frames_arr = self._frames_to_array(action.frames)
            current_intent = frames_arr[0]
            with self._target_lock:
                self._latest_intent = current_intent
                self._latest_chunk = (t, action.fps, frames_arr)
        else:
            current_intent = self._action_to_array(action)
            with self._target_lock:
                self._latest_intent = current_intent
                self._latest_chunk = None

    def observe_state(self, t: float, obs: RobotObservation) -> None:
        """Receive a state sample from get_observation(). Non-blocking."""
        state_arr = self._observation_to_array(obs)
        if state_arr is not None:
            self._state_log.append((t, state_arr))

    def set_teleop(self, teleop) -> None:
        """Bind an intent source the control thread polls each tick.

        When set, ``_tick`` calls ``teleop.get_action()`` to get the
        latest intent dict. This bypasses the ``send_action`` →
        ``set_intent`` → ``_latest_intent`` push path. Use with a
        high-rate teleop (e.g. SO107LeaderHighRate) whose
        ``get_action()`` is a fast cached read; the controller's
        velocity LSQ then sees one fresh sample per tick.

        Pass ``None`` to detach. The teleop is expected to be
        thread-safe with respect to concurrent ``get_action`` calls
        (SO107LeaderHighRate guarantees this via its background-thread
        cache).
        """
        self._teleop = teleop

    # ── Background control loop ───────────────────────────────────────────

    def _control_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                # Catch-all keeps the bus alive even if one tick blows up
                # (e.g. a transient TxRxResult). Log + continue.
                logger.exception("%s predictive-lookahead tick failed", self._robot)
            next_tick += self._control_dt
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                precise_sleep(sleep_for)
            else:
                # Fell behind — reset target so we don't spin trying to
                # catch up over many ticks.
                next_tick = time.perf_counter()

    def _tick(self) -> None:
        # Pull-path: a bound teleop is the authoritative intent source.
        # Polling here at control rate gives the velocity LSQ one fresh
        # sample per tick (vs. the push-path's stair-stepped 30 Hz).
        # The polled intent is treated as a dict (no chunk semantics
        # for the leader case — chunks come through send_action from
        # sources that have a real future like trajectory_replay).
        teleop = self._teleop
        if teleop is not None:
            try:
                action = teleop.get_action()
            except Exception:
                logger.exception("%s teleop.get_action() failed", self._robot)
                return
            if not action:
                return  # teleop returned empty — wait for next tick
            intent = self._action_to_array(action)
            chunk = None  # leader teleop never publishes a chunk
        else:
            with self._target_lock:
                intent = self._latest_intent
                chunk = self._latest_chunk
            if intent is None:
                return  # no intent received yet; nothing to send

        now = time.perf_counter()

        # Log "intent at now" (no lookahead) for the adaptive xcorr.
        # Done from the control thread so the sample cadence matches
        # ``control_dt`` — the xcorr then converts index shifts to time
        # cleanly. Logging from set_intent() instead would sample at the
        # caller's rate (typically 30 Hz) while the xcorr's ``dt`` would
        # still be the control rate (200 Hz), under-reporting lag by a
        # factor of caller_rate / control_rate.
        # Chunk path: interpolate the chunk at "now" so the log captures
        # the source's smooth intent, not the stair-stepped latest_intent.
        # Dict path: use the latest_intent as-is — that's the truth (the
        # caller didn't publish a smoother signal).
        if chunk is not None:
            intent_at_now = self._lookup_in_chunk(chunk, now, lookahead_s_override=0.0)
        else:
            intent_at_now = intent
        self._intent_log.append((now, intent_at_now))

        # 1. Compute the lookahead target.
        # Two paths:
        #   * Chunk available → exact lookup at "now + L" by index =
        #     (elapsed + L) * fps. Past the chunk's last frame, fall back
        #     to chunk-tail velocity extrapolation (which is exactly
        #     what a real chunked policy faces at boundaries).
        #   * No chunk → velocity LSQ over the recent intent stream,
        #     same as the leader-arm teleop case.
        if chunk is not None:
            raw_shifted = self._lookup_in_chunk(chunk, now)
        else:
            self._intent_ring.append((now, intent.copy()))
            cutoff = now - self._velocity_window_s
            win = [(t, p) for t, p in self._intent_ring if t >= cutoff]
            if len(win) < 2:
                raw_shifted = intent.copy()
            else:
                ts = np.array([t for t, _ in win], dtype=np.float64)
                ps = np.stack([p for _, p in win])
                v_leader = self._estimate_velocity(ts, ps)
                raw_shifted = intent + v_leader * self._lookahead_s if v_leader is not None else intent.copy()

        # 2. Predictor-corrector smoothing
        alpha = self._corrector_alpha
        if alpha < 1.0 and self._last_action is not None and len(self._action_ring) >= 2:
            a_cutoff = now - self._velocity_window_s
            a_win = [(t, a) for t, a in self._action_ring if t >= a_cutoff]
            if len(a_win) >= 2:
                ts_a = np.array([t for t, _ in a_win], dtype=np.float64)
                ps_a = np.stack([a for _, a in a_win])
                ts_a_c = ts_a - ts_a.mean()
                denom_a = float((ts_a_c * ts_a_c).sum())
                v_action = (ts_a_c @ ps_a) / denom_a if denom_a > 1e-12 else np.zeros_like(intent)
            else:
                v_action = np.zeros_like(intent)
            predictor = self._last_action + v_action * self._control_dt
            shifted = alpha * raw_shifted + (1.0 - alpha) * predictor
        else:
            shifted = raw_shifted

        # 3. Per-step safety clamp
        if self._last_action is not None:
            delta = shifted - self._last_action
            if np.any(np.abs(delta) > self._max_step):
                shifted = self._last_action + np.clip(delta, -self._max_step, self._max_step)

        # 4. Write to motors under the bus lock
        goal_dict = {self._motor_keys[i].removesuffix(".pos"): float(shifted[i]) for i in range(len(shifted))}
        with self._bus_lock:
            self._bus.sync_write("Goal_Position", goal_dict)
        self._last_action = shifted
        self._action_ring.append((now, shifted.copy()))

        # 5. Periodic adaptive update
        if self._adaptive and (now - self._last_adaptive_t) >= self._UPDATE_PERIOD_S:
            self._last_adaptive_t = now
            self._maybe_update_lookahead(now)

    # ── Adaptive update ──────────────────────────────────────────────────

    def _maybe_update_lookahead(self, now: float) -> None:
        """Amplitude-gated cross-correlation update of self._lookahead_s.

        Same logic as cross_corr_lag in scripts/proto_decoupled_teleop.py
        (post the 7a1e92c61 patch): symmetric scan, corr ≥ 0.95 AND
        amplitude ≥ _AMP_FLOOR, amplitude-weighted aggregate, α=0.5
        low-pass, hard cap.
        """
        win_start = now - self._WINDOW_S
        # Snapshot the cross-thread deques into lists before filtering.
        # set_intent / observe_state can mutate _intent_log / _state_log
        # concurrently from the caller thread; a comprehension iterating
        # the deque directly can raise "deque mutated during iteration".
        # list(deque) is atomic under the GIL and gives a stable view.
        intent_snapshot = list(self._intent_log)
        state_snapshot = list(self._state_log)
        intent_samples = [(t, p) for t, p in intent_snapshot if t >= win_start]
        state_samples = [(t, s) for t, s in state_snapshot if t >= win_start - 0.5]
        # Minimum intent-sample floor scales with the expected count
        # (window_s × control_rate). 30 % of full is the same fraction the
        # prototype uses (100 of 600 at 3 s / 200 Hz) and stays sensible
        # across control_rate_hz / WINDOW_S changes — hardcoding 30
        # samples would be a different fraction at every control rate.
        expected_intent = self._WINDOW_S / self._control_dt
        min_intent_samples = max(10, int(expected_intent * 0.3))
        if len(intent_samples) < min_intent_samples or len(state_samples) < 5:
            return

        ts_intent = np.array([t for t, _ in intent_samples], dtype=np.float64)
        gt = np.stack([p for _, p in intent_samples])
        ts_state = np.array([t for t, _ in state_samples], dtype=np.float64)
        s_reads = np.stack([s for _, s in state_samples])
        state_at_intent = np.empty_like(gt)
        for j in range(gt.shape[1]):
            state_at_intent[:, j] = np.interp(ts_intent, ts_state, s_reads[:, j])

        dt = self._control_dt
        max_lag = int(self._MAX_LAG_S / dt)
        confident_lags: list[float] = []
        confident_amps: list[float] = []
        for j in range(gt.shape[1]):
            a = gt[:, j]
            s = state_at_intent[:, j]
            state_amp = float(s.std())
            if state_amp < self._AMP_FLOOR:
                continue
            best_k, best_c = 0, -np.inf
            for k in range(-max_lag, max_lag + 1):
                if k >= 0:
                    aa, ss = (a[: len(a) - k], s[k:]) if k > 0 else (a, s)
                else:
                    m = -k
                    aa, ss = a[m:], s[: len(s) - m]
                aa_c = aa - aa.mean()
                ss_c = ss - ss.mean()
                na, ns = float(np.linalg.norm(aa_c)), float(np.linalg.norm(ss_c))
                if na < 1e-12 or ns < 1e-12:
                    continue
                c = float(np.dot(aa_c, ss_c) / (na * ns))
                if c > best_c:
                    best_c, best_k = c, k
            if best_c >= self._CORR_FLOOR:
                confident_lags.append(best_k * dt)
                confident_amps.append(state_amp)
        if not confident_lags:
            return

        weights = np.asarray(confident_amps, dtype=np.float64)
        lag = float(np.sum(np.asarray(confident_lags) * weights) / weights.sum())
        target = self._lookahead_s + lag
        new_l = self._ADAPTIVE_ALPHA * target + (1.0 - self._ADAPTIVE_ALPHA) * self._lookahead_s
        self._lookahead_s = max(0.0, min(new_l, self._max_lookahead_s))
        logger.debug(
            "%s adaptive L update: lag=%+.1fms → L=%.1fms (cap %.0fms)",
            self._robot,
            lag * 1000,
            self._lookahead_s * 1000,
            self._max_lookahead_s * 1000,
        )

    # ── Velocity estimators ──────────────────────────────────────────────
    #
    # Three implementations selectable by ``config.velocity_estimator``:
    #
    #   * ``"quad"``         — LSQ quadratic fit, slope evaluated at the
    #                          most-recent sample's time. Unbiased v(now)
    #                          for any motion with bounded acceleration
    #                          over the window. Backtest winner on real
    #                          teleop motion AND 1-3 Hz sinusoids.
    #   * ``"linear"``       — LSQ linear slope (centered velocity).
    #                          Original prototype behaviour. Biased by
    #                          ~a·w/2 under acceleration → systematically
    #                          inflates measured apparent τ by 50-80 ms in
    #                          the leader-teleop case (vs. chunk-path
    #                          ground truth ~95 ms).
    #   * ``"forward_diff"`` — Two-sample causal difference. Lowest bias,
    #                          highest noise. Tied with quad on the
    #                          recorded backtest, slightly noisier in
    #                          theory.
    #
    # All three return ``np.ndarray`` of shape (n_motors,) for the
    # estimated velocity at "now", or ``None`` when the window is
    # degenerate (single sample, all-same timestamps).
    # Backtest comparison: scripts/backtest_velocity_estimators.py.

    def _estimate_velocity(self, ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
        """Dispatch to the configured estimator."""
        if self._velocity_estimator == "quad":
            return self._velocity_lsq_quad_end(ts, ps)
        if self._velocity_estimator == "linear":
            return self._velocity_lsq_linear(ts, ps)
        if self._velocity_estimator == "forward_diff":
            return self._velocity_forward_diff(ts, ps)
        # Unknown — fall back to linear with a one-time warning. Should not
        # be reachable because the field is Literal-typed at construction,
        # but defensive in case someone instantiates the controller
        # bypassing the dataclass init.
        logger.warning(
            "Unknown velocity_estimator=%r; falling back to 'linear'",
            self._velocity_estimator,
        )
        return self._velocity_lsq_linear(ts, ps)

    @staticmethod
    def _velocity_lsq_linear(ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
        """Original prototype behaviour. Slope = centered velocity over the
        window. Returns slope for each joint. Biased under acceleration."""
        ts_c = ts - ts.mean()
        denom = float((ts_c * ts_c).sum())
        if denom < 1e-12:
            return None
        return (ts_c @ ps) / denom

    @staticmethod
    def _velocity_lsq_quad_end(ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
        """Fit p(t) = a + b·t + c·t² with t expressed relative to the
        most-recent sample. The slope at the most-recent sample is the
        coefficient b. Same column-rank check as linear; falls back to
        linear on rank deficiency (e.g. only 2 samples available)."""
        if ts.shape[0] < 3:
            # Quadratic fit needs at least 3 points to be defined.
            return _PredictiveLookaheadController._velocity_lsq_linear(ts, ps)
        t_rel = ts - ts[-1]  # ≤ 0; slope at t_rel=0 is the "now" slope
        # Design matrix for [a, b, c] in p = a + b·t + c·t²
        design = np.stack([np.ones_like(t_rel), t_rel, t_rel * t_rel], axis=1)
        # lstsq returns coefficients shape (3,) or (3, n_motors) depending
        # on ps shape; ps is (n, n_motors), result is (3, n_motors). The
        # slope at t_rel=0 is coef[1].
        try:
            coef, *_ = np.linalg.lstsq(design, ps, rcond=None)
        except np.linalg.LinAlgError:
            return _PredictiveLookaheadController._velocity_lsq_linear(ts, ps)
        return coef[1]

    @staticmethod
    def _velocity_forward_diff(ts: np.ndarray, ps: np.ndarray) -> np.ndarray | None:
        """v(now) ≈ (p[-1] − p[-2]) / dt. Two-sample causal difference at
        the tail of the window. Unbiased, but noise-sensitive at small
        dt — fine when the source's intent log is smooth, marginal when
        samples carry measurement noise."""
        if ts.shape[0] < 2:
            return None
        dt = float(ts[-1] - ts[-2])
        if abs(dt) < 1e-9:
            return None
        return (ps[-1] - ps[-2]) / dt

    # ── Chunk lookup ─────────────────────────────────────────────────────

    def _lookup_in_chunk(
        self,
        chunk: tuple[float, float, np.ndarray],
        now: float,
        lookahead_s_override: float | None = None,
    ) -> np.ndarray:
        """Exact-lookup target at ``now + L`` using a received chunk.

        Within the chunk: linear interpolation between adjacent frames.
        Past the chunk's last frame: linear extrapolation from the
        chunk's tail velocity. Sub-frame interpolation matches what
        TrajectoryReplayTeleop already does for its own playback head;
        keeping the math symmetric avoids a subtle bias between "what
        was recorded" and "what was replayed".

        ``lookahead_s_override``: when not None, use this in place of
        ``self._lookahead_s``. Used by the adaptive log path to compute
        "intent at now" (L=0) regardless of the controller's current
        lookahead.
        """
        lookahead_s = self._lookahead_s if lookahead_s_override is None else lookahead_s_override
        received_at, fps, frames_arr = chunk
        n_frames = frames_arr.shape[0]
        elapsed = now - received_at
        target_idx_f = (elapsed + lookahead_s) * fps
        if target_idx_f <= n_frames - 1:
            lo = max(0, int(target_idx_f))
            hi = min(n_frames - 1, lo + 1)
            alpha = target_idx_f - lo
            return frames_arr[lo] * (1.0 - alpha) + frames_arr[hi] * alpha
        # Past chunk end — extrapolate from the tail velocity. Using the
        # last two frames keeps the extrapolation responsive to the most
        # recent dynamics of the source rather than averaging the whole
        # chunk, which would over-smooth at sharp transitions (e.g. an
        # end-of-trajectory deceleration).
        tail_v = (frames_arr[-1] - frames_arr[-2]) * fps if n_frames >= 2 else np.zeros_like(frames_arr[-1])
        excess_s = (target_idx_f - (n_frames - 1)) / fps
        return frames_arr[-1] + tail_v * excess_s

    # ── Helpers ──────────────────────────────────────────────────────────

    def _frames_to_array(self, frames: tuple[dict[str, float], ...]) -> np.ndarray:
        """Stack ``ActionChunk.frames`` into shape ``(N, n_motors)``.

        Each frame goes through ``_action_to_array`` so the strict-key
        check applies — a missing motor in any frame raises immediately
        instead of producing a malformed chunk that the lookup path
        would silently propagate.
        """
        return np.stack([self._action_to_array(f) for f in frames])

    def _action_to_array(self, action: RobotAction) -> np.ndarray:
        # Strict: every motor key must be present. A short dict here would
        # produce a partial sync_write (the missing motors silently hold
        # their last goal), which is unsafe — fail fast instead.
        missing = [k for k in self._motor_keys if k not in action]
        if missing:
            raise ValueError(f"action missing keys for motors: {missing}")
        return np.array([float(action[k]) for k in self._motor_keys], dtype=np.float64)

    def _observation_to_array(self, obs: RobotObservation) -> np.ndarray | None:
        # State samples drive the adaptive update only. A missing motor key
        # here is unexpected (get_observation builds the dict from the same
        # bus) so log + drop the sample rather than fail the whole tick —
        # the controller can still extrapolate from the last good state.
        missing = [k for k in self._motor_keys if k not in obs]
        if missing:
            logger.warning("state observation missing keys %s; dropping sample", missing)
            return None
        return np.array([float(obs[k]) for k in self._motor_keys], dtype=np.float64)
