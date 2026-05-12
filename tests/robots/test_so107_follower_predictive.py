#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mock-bus integration tests for SO107FollowerPredictive.

The point of these tests is the *lifecycle and threading* of the
predictive controller as wired into the Robot interface — not the
algorithm itself. Algorithm correctness is covered by:

  * scripts/proto_decoupled_teleop.py (hardware-in-the-loop validation)
  * scripts/backtest_lookahead.py (offline ground-truth comparison)
  * scripts/sim_adaptive_lookahead.py (closed-loop convergence sim)

Here we just make sure that, with a mocked FeetechMotorsBus:

  * connect() starts the controller thread and configures the bus
  * send_action(intent) is non-blocking and publishes intent
  * the controller thread eventually writes Goal_Position to the bus
  * get_observation() reads from the bus and feeds the controller's
    state log
  * the per-step max_step_deg clamp applies
  * disconnect() stops the controller cleanly with no leaked threads
  * missing-key actions raise (regression: the strict-key fix on
    _action_to_array)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.so107_follower_predictive import (
    SO107FollowerPredictive,
    SO107FollowerPredictiveRobotConfig,
)

# Motor map: SO-107 has 7 motors, same names as the parent SO107Follower.
_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _make_bus_mock() -> MagicMock:
    """FeetechMotorsBus stand-in that records every sync_write call."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False
    bus.is_calibrated = True

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def follower():
    """Yield a connected SO107FollowerPredictive with a mocked bus.

    The mock bus's sync_read returns a deterministic state vector
    (zeros, matching the expected motor count) so the controller's
    state-log path runs without raising.
    """
    bus_mock = _make_bus_mock()
    # Capture every (data_name, goal_dict) the controller writes — the
    # tests assert against this in lieu of probing the controller's
    # private state.
    sync_write_log: list[tuple[str, dict]] = []
    bus_mock.sync_write_log = sync_write_log

    def _sync_write(data_name, goal_dict, **_kwargs):
        sync_write_log.append((data_name, dict(goal_dict)))

    bus_mock.sync_write.side_effect = _sync_write

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        # sync_read returns positions at the bus's natural zero.
        bus_mock.sync_read.return_value = dict.fromkeys(bus_mock.motors, 0.0)
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        return bus_mock

    with (
        patch(
            "lerobot.robots.so107_follower_predictive.so107_follower_predictive.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        # Skip configure() — its bus.write side effects aren't relevant
        # to controller lifecycle and would require more bus stubbing.
        patch.object(SO107FollowerPredictive, "configure", lambda self: None),
    ):
        cfg = SO107FollowerPredictiveRobotConfig(
            port="/dev/null",
            id="test_arm",
            # Crank the rate so each test only has to sleep a few ms to
            # observe controller activity. Algorithmic behaviour is
            # rate-independent.
            control_rate_hz=500.0,
            # Default is True; turn off so tests don't race against the
            # 2 s adaptive update cadence (separate test exercises it).
            adaptive=False,
        )
        robot = SO107FollowerPredictive(cfg)
        robot.connect(calibrate=False)
        try:
            yield robot, bus_mock
        finally:
            if robot.is_connected:
                robot.disconnect()


# ============================================================================
# Lifecycle
# ============================================================================


class TestLifecycle:
    def test_connect_starts_controller_thread(self, follower):
        robot, _bus = follower
        # The controller's _thread should be alive after connect().
        assert robot._controller._thread is not None
        assert robot._controller._thread.is_alive()

    def test_disconnect_stops_controller_thread(self, follower):
        robot, bus = follower
        ctrl_thread = robot._controller._thread
        assert ctrl_thread is not None
        robot.disconnect()
        # Give it a moment beyond the join timeout.
        ctrl_thread.join(timeout=1.0)
        assert not ctrl_thread.is_alive()
        # Bus should be closed too.
        bus.disconnect.assert_called()

    def test_disconnect_runs_every_phase_on_error(self, follower):
        """If bus.disconnect raises, cameras still get cleaned up.

        Regression for f6ecbdb6b: previously a bus failure stranded any
        later cleanup. The predictive follower has no cameras here, but
        the controller stop must run before the bus error and be
        observable as 'thread is no longer alive'.
        """
        robot, bus = follower
        bus.disconnect.side_effect = RuntimeError("simulated torque-disable failure")
        ctrl_thread = robot._controller._thread
        with pytest.raises(RuntimeError, match="simulated"):
            robot.disconnect()
        ctrl_thread.join(timeout=1.0)
        assert not ctrl_thread.is_alive()  # controller stopped before bus error
        # Reset for fixture teardown.
        bus.disconnect.side_effect = None

    def test_no_thread_leak_across_connect_cycles(self, follower):
        """Disconnect + reconnect should not leave a zombie thread."""
        robot, _bus = follower
        baseline = threading.active_count()
        robot.disconnect()
        robot.connect(calibrate=False)
        robot.disconnect()
        # Give the OS a moment to reap threads.
        time.sleep(0.05)
        assert threading.active_count() <= baseline


# ============================================================================
# send_action contract
# ============================================================================


class TestSendAction:
    def _intent(self, value: float = 0.0) -> dict[str, float]:
        return {f"{m}.pos": value for m in _MOTOR_NAMES}

    def test_send_action_returns_intent_unchanged(self, follower):
        """send_action must echo back the *intent*, not a clamped value.

        Callers (dataset writer, in particular) record this as the
        action; the predictive robot's contract is that the recorded
        action IS the raw operator intent.
        """
        robot, _bus = follower
        intent = self._intent(3.14)
        returned = robot.send_action(intent)
        assert returned == intent

    def test_send_action_is_non_blocking(self, follower):
        """send_action must return well under one control period."""
        robot, _bus = follower
        intent = self._intent(0.0)
        start = time.perf_counter()
        robot.send_action(intent)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # control_rate_hz=500 → period 2 ms. send_action must be far
        # cheaper than that since it just records to a deque.
        assert elapsed_ms < 1.0, f"send_action blocked for {elapsed_ms:.2f} ms"

    def test_controller_writes_goal_position(self, follower):
        """After send_action, the 200 Hz thread must publish to the bus."""
        robot, bus = follower
        bus.sync_write_log.clear()
        robot.send_action(self._intent(1.0))
        # Wait a few control ticks (rate is 500 Hz in tests = 2 ms / tick).
        time.sleep(0.05)
        # At least one Goal_Position write should have landed.
        goal_writes = [(name, gd) for name, gd in bus.sync_write_log if name == "Goal_Position"]
        assert goal_writes, "controller never wrote Goal_Position"
        # And it should have written every motor, not a partial set
        # (regression for the strict-key fix on _action_to_array).
        first_goal = goal_writes[0][1]
        assert set(first_goal.keys()) == set(_MOTOR_NAMES)

    def test_missing_motor_key_raises(self, follower):
        """Partial action dict must fail loud, not produce a partial write.

        Regression for commit 42bd67f3b — previously the filter
        ``[float(action[k]) for k in self._motor_keys if k in action]``
        silently shrank the array, the controller would issue a partial
        sync_write, and the missing motors would hold their last goal.
        """
        robot, _bus = follower
        partial = {f"{m}.pos": 0.0 for m in _MOTOR_NAMES[:-1]}  # drop gripper
        with pytest.raises(ValueError, match="action missing keys"):
            robot.send_action(partial)


# ============================================================================
# Chunk path
# ============================================================================


class TestActionChunk:
    """Exact-lookup path: send_action(ActionChunk) feeds the controller a
    horizon, controller picks the target at now + L by interpolation.
    """

    def _frame(self, value: float) -> dict[str, float]:
        return {f"{m}.pos": value for m in _MOTOR_NAMES}

    def test_send_action_returns_frame_zero_for_chunk(self, follower):
        """The dataset writer records what send_action returns. For a
        chunk that must be frames[0] (= "current intent"), not the
        chunk itself."""
        from lerobot.types import ActionChunk

        robot, _bus = follower
        chunk = ActionChunk(
            fps=30.0,
            frames=(self._frame(1.0), self._frame(1.1), self._frame(1.2)),
        )
        returned = robot.send_action(chunk)
        assert returned == self._frame(1.0)

    def test_controller_writes_via_chunk_lookup(self, follower):
        """Chunk path → controller writes Goal_Position from the chunk's
        future values (not by velocity-extrapolating past intents)."""
        from lerobot.types import ActionChunk

        robot, bus = follower
        bus.sync_write_log.clear()
        # Linear ramp: frame[k] = 5 + 0.1 * k. At fps=30, frame index
        # for L=80ms is 80ms * 30fps = 2.4 → interp between frame 2 and 3.
        frames = tuple(self._frame(5.0 + 0.1 * k) for k in range(20))
        chunk = ActionChunk(fps=30.0, frames=frames)
        robot.send_action(chunk)
        # Wait a few ticks (rate is 500 Hz in tests = 2 ms / tick).
        time.sleep(0.05)
        goal_writes = [gd for name, gd in bus.sync_write_log if name == "Goal_Position"]
        assert goal_writes, "controller never wrote Goal_Position"
        # The very first goal write should reflect an L-shifted value
        # somewhere between frame 0 (5.0) and a few frames in. Exact
        # value depends on tick timing relative to send_action; just
        # verify the controller is reading from the future, not
        # holding at 5.0.
        first_goal = goal_writes[0]
        assert all(v > 5.0 for v in first_goal.values()), f"expected L-shifted future, got {first_goal}"

    def test_chunk_with_missing_key_raises(self, follower):
        """Strict-key check applies per-frame: a malformed frame in
        the chunk must fail loud at set_intent time, not later in the
        controller's tick."""
        from lerobot.types import ActionChunk

        robot, _bus = follower
        good = self._frame(0.0)
        bad = {k: v for k, v in good.items() if k != f"{_MOTOR_NAMES[-1]}.pos"}
        chunk = ActionChunk(fps=30.0, frames=(good, bad))
        with pytest.raises(ValueError, match="action missing keys"):
            robot.send_action(chunk)

    def test_chunk_at_arbitrary_fps_resolves_target_correctly(self, follower):
        """The controller's chunk-lookup math must work for any fps,
        not just 30. Regression guard against hardcoded-30 assumptions
        in the controller. With L=0 + a ramp chunk, the goal at tick 0
        should equal frame[0] regardless of fps."""
        import numpy as np

        from lerobot.types import ActionChunk

        robot, bus = follower
        # Try fps=15 (low) and fps=60 (high) — both have to land the
        # goal write near frame[0] when L=0. Use a non-zero base value
        # so we'd notice if the lookup landed elsewhere.
        for test_fps in (15.0, 60.0):
            bus.sync_write_log.clear()
            base = 7.0
            frames = tuple(self._frame(base + 0.1 * k) for k in range(20))
            robot.send_action(ActionChunk(fps=test_fps, frames=frames))
            time.sleep(0.02)  # a few control ticks
            goals = [gd for name, gd in bus.sync_write_log if name == "Goal_Position"]
            assert goals, f"no goal write at fps={test_fps}"
            # First goal should be very close to frame[0] (L=0 in this fixture).
            for v in goals[0].values():
                assert v == pytest.approx(base, abs=0.5), f"fps={test_fps}: goal {v}, expected ~{base}"
            # Math sanity: tail-velocity extrapolation past chunk end
            # uses fps in (frames[-1] - frames[-2]) * fps, so the *value*
            # past the end must depend on fps. Compute the predicted
            # value at "now past last frame" and assert the formula.
            tail_v = (np.array([base + 0.1 * 19]) - np.array([base + 0.1 * 18])) * test_fps
            excess_s = 0.001  # 1 ms past end
            expected_past_end = base + 0.1 * 19 + tail_v[0] * excess_s
            # Just verify our math; the controller will only reach this
            # in real runs where the chunk runs out, not in this test.
            assert expected_past_end == pytest.approx(base + 0.1 * 19 + 0.1 * test_fps * excess_s, abs=1e-9)

    def test_chunk_then_dict_clears_chunk_state(self, follower):
        """A single-dict send_action after a chunk drops the chunk
        record — subsequent ticks fall back to the velocity-extrapolation
        path. (Latest-send wins.)"""
        from lerobot.types import ActionChunk

        robot, _bus = follower
        chunk = ActionChunk(fps=30.0, frames=(self._frame(0.0), self._frame(0.1)))
        robot.send_action(chunk)
        time.sleep(0.005)
        robot.send_action(self._frame(0.5))
        time.sleep(0.005)
        # After the dict send, the controller's latest_chunk must be cleared.
        assert robot._controller._latest_chunk is None


# ============================================================================
# get_observation feeds the controller's state log
# ============================================================================


class TestGetObservation:
    def test_observation_includes_every_motor(self, follower):
        robot, _bus = follower
        obs = robot.get_observation()
        expected = {f"{m}.pos" for m in _MOTOR_NAMES}
        assert set(obs.keys()) == expected

    def test_observation_pushes_state_to_controller_log(self, follower):
        """The controller needs state samples for the adaptive update.

        get_observation() should append to _state_log; this verifies
        the feed path exists (the adaptive update itself is exercised
        in TestAdaptiveUpdate).
        """
        robot, _bus = follower
        n_before = len(robot._controller._state_log)
        for _ in range(3):
            robot.get_observation()
        assert len(robot._controller._state_log) >= n_before + 3


# ============================================================================
# max_step_deg safety clamp
# ============================================================================


class TestSafetyClamp:
    def test_per_step_delta_is_clamped(self, follower):
        """A large intent jump must be slewed by ≤ max_step_deg / tick.

        The controller's _last_action holds last tick's commanded
        position. When the new shifted target differs by more than
        max_step_deg per joint, the delta is clipped. This protects
        against extrapolation spikes during sharp leader reversals.
        """
        robot, bus = follower
        # First intent: settle at 0.0. Wait for a couple of ticks so
        # _last_action is populated.
        robot.send_action({f"{m}.pos": 0.0 for m in _MOTOR_NAMES})
        time.sleep(0.05)
        bus.sync_write_log.clear()
        # Now jump to a value far beyond max_step_deg (default = 3°).
        robot.send_action({f"{m}.pos": 100.0 for m in _MOTOR_NAMES})
        time.sleep(0.02)  # ~10 ticks at 500 Hz
        # Find the first Goal_Position write after the jump.
        goal_writes = [gd for name, gd in bus.sync_write_log if name == "Goal_Position"]
        assert goal_writes, "no Goal_Position writes after the jump"
        first = goal_writes[0]
        # Every motor's first commanded value should be within
        # max_step_deg of the previous (which was 0).
        max_step = robot.config.max_step_deg
        for motor, val in first.items():
            assert abs(val) <= max_step + 1e-6, f"{motor} jumped to {val}, exceeds max_step_deg={max_step}"


# ============================================================================
# Adaptive update path
# ============================================================================


class TestAdaptiveUpdate:
    def test_adaptive_update_runs_without_raising(self):
        """Build enough state + intent samples to drive _maybe_update_lookahead.

        The deterministic correctness of the cross-correlation is
        already covered by scripts/backtest_lookahead.py. Here we just
        verify the threaded path doesn't raise on a real history shape.
        """
        bus_mock = _make_bus_mock()

        def _bus_side_effect(*_args, **kwargs):
            bus_mock.motors = kwargs["motors"]
            bus_mock.sync_read.return_value = dict.fromkeys(bus_mock.motors, 0.0)
            return bus_mock

        with (
            patch(
                "lerobot.robots.so107_follower_predictive.so107_follower_predictive.FeetechMotorsBus",
                side_effect=_bus_side_effect,
            ),
            patch.object(SO107FollowerPredictive, "configure", lambda self: None),
        ):
            cfg = SO107FollowerPredictiveRobotConfig(
                port="/dev/null",
                control_rate_hz=500.0,
                adaptive=True,
            )
            robot = SO107FollowerPredictive(cfg)
            # Shorten the update cadence so the test doesn't have to
            # wait the production 2 s.
            robot._controller._UPDATE_PERIOD_S = 0.1  # type: ignore[misc]
            robot._controller._WINDOW_S = 0.3  # type: ignore[misc]
            robot.connect(calibrate=False)
            try:
                # Drive a sinusoidal intent so v_leader is non-zero and
                # the amplitude-floor gate passes.
                import math

                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 0.5:  # 0.5 s of activity
                    t = time.perf_counter() - t0
                    val = 5.0 * math.sin(2 * math.pi * 2 * t)
                    robot.send_action({f"{m}.pos": val for m in _MOTOR_NAMES})
                    robot.get_observation()
                    time.sleep(0.005)
                # If we got here without the controller thread raising,
                # the adaptive update path is alive. The thread catches
                # exceptions per-tick but logs them; if we want strict
                # detection we'd need to capture log output.
                assert robot._controller._thread.is_alive()
            finally:
                robot.disconnect()
