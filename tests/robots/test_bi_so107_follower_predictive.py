#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mock-bus integration smoke test for BiSO107FollowerPredictive.

The single-arm tests in test_so107_follower_predictive.py exercise the
algorithmic and threading surface. This file's job is just to verify
the bimanual composer wires up correctly: two independent buses, two
independent controller threads, shared controller settings, joint-name
prefixing, and clean shutdown.

If you find yourself wanting to add per-tick algorithmic assertions
here, prefer adding them to the single-arm file — the bimanual just
inherits via composition.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_so107_follower_predictive import (
    BiSO107FollowerPredictive,
    BiSO107FollowerPredictiveConfig,
)

_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _make_bus_mock(name: str) -> MagicMock:
    bus = MagicMock(name=name)
    bus.is_connected = False
    bus.is_calibrated = True
    bus.sync_write_log: list[tuple[str, dict]] = []

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    def _sync_write(data_name, goal_dict, **_kwargs):
        bus.sync_write_log.append((data_name, dict(goal_dict)))

    @contextmanager
    def _dummy_cm():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.sync_write.side_effect = _sync_write
    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def bi_follower():
    """Yield a connected BiSO107FollowerPredictive with two mock buses.

    Each call to FeetechMotorsBus(...) returns a distinct bus mock so
    we can verify that left and right writes go to *different* buses.
    """
    left_bus = _make_bus_mock("LeftBusMock")
    right_bus = _make_bus_mock("RightBusMock")
    buses_in_order = [left_bus, right_bus]
    bus_iter = iter(buses_in_order)

    def _bus_side_effect(*_args, **kwargs):
        bus = next(bus_iter)
        bus.motors = kwargs["motors"]
        bus.sync_read.return_value = dict.fromkeys(bus.motors, 0.0)
        bus.write.return_value = None
        return bus

    # Patch the import site used by SO107FollowerPredictive (each arm
    # is a predictive instance, so this is the only patch needed —
    # BiSO107FollowerPredictive doesn't construct buses directly).
    from lerobot.robots.so107_follower_predictive import (
        SO107FollowerPredictive,
    )

    with (
        patch(
            "lerobot.robots.so107_follower_predictive.so107_follower_predictive.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO107FollowerPredictive, "configure", lambda self: None),
    ):
        cfg = BiSO107FollowerPredictiveConfig(
            left_arm_port="/dev/null_left",
            right_arm_port="/dev/null_right",
            left_arm_use_degrees=False,
            right_arm_use_degrees=False,
            id="white_test",
            control_rate_hz=500.0,
            adaptive=False,
        )
        robot = BiSO107FollowerPredictive(cfg)
        robot.connect(calibrate=False)
        try:
            yield robot, left_bus, right_bus
        finally:
            if robot.is_connected:
                robot.disconnect()


def test_two_independent_controller_threads(bi_follower):
    """Each arm must have its own running controller thread."""
    robot, _l, _r = bi_follower
    left_thread = robot.left_arm._controller._thread
    right_thread = robot.right_arm._controller._thread
    assert left_thread is not None and right_thread is not None
    assert left_thread is not right_thread
    assert left_thread.is_alive()
    assert right_thread.is_alive()


def test_controller_settings_shared_across_arms(bi_follower):
    """The bimanual config's controller knobs propagate to both arms."""
    robot, _l, _r = bi_follower
    for arm in (robot.left_arm, robot.right_arm):
        assert arm.config.lookahead_ms == 80.0
        assert arm.config.max_lookahead_ms == 110.0
        assert arm.config.corrector_alpha == 1.0
        assert arm.config.velocity_window_ms == 70.0
        assert arm.config.control_rate_hz == 500.0
        assert arm.config.adaptive is False
        assert arm.config.max_step_deg == 3.0


def test_send_action_routes_prefixed_keys_to_correct_arm(bi_follower):
    """left_*.pos must land on left bus, right_*.pos on right bus."""
    robot, left_bus, right_bus = bi_follower
    left_bus.sync_write_log.clear()
    right_bus.sync_write_log.clear()

    action = {}
    for m in _MOTOR_NAMES:
        action[f"left_{m}.pos"] = 1.0
        action[f"right_{m}.pos"] = 2.0
    robot.send_action(action)
    # Wait for both control threads to land at least one tick.
    time.sleep(0.05)

    left_goals = [gd for name, gd in left_bus.sync_write_log if name == "Goal_Position"]
    right_goals = [gd for name, gd in right_bus.sync_write_log if name == "Goal_Position"]
    assert left_goals, "left arm controller never wrote Goal_Position"
    assert right_goals, "right arm controller never wrote Goal_Position"

    # Each bus saw values that match its arm's intent (1.0 vs 2.0), and
    # no cross-talk — left bus should never see a 2.0 goal in this test.
    # Tolerance is loose because the controller's velocity LSQ over a
    # near-constant intent stream produces a tiny non-zero slope, so
    # `intent + v * L` drifts ~1e-5 from the raw intent value. The
    # test's purpose is routing (left ≠ right), not bit-equality.
    for gd in left_goals:
        for v in gd.values():
            assert abs(v - 1.0) < 0.01, f"left bus got {v}, expected ~1.0"
    for gd in right_goals:
        for v in gd.values():
            assert abs(v - 2.0) < 0.01, f"right bus got {v}, expected ~2.0"


def test_disconnect_stops_both_threads(bi_follower):
    robot, _l, _r = bi_follower
    left_thread = robot.left_arm._controller._thread
    right_thread = robot.right_arm._controller._thread
    robot.disconnect()
    left_thread.join(timeout=1.0)
    right_thread.join(timeout=1.0)
    assert not left_thread.is_alive()
    assert not right_thread.is_alive()


def test_get_observation_returns_prefixed_keys(bi_follower):
    robot, _l, _r = bi_follower
    obs = robot.get_observation()
    expected = {f"left_{m}.pos" for m in _MOTOR_NAMES} | {f"right_{m}.pos" for m in _MOTOR_NAMES}
    # observation_features may include cameras; the motor subset must be
    # exactly the expected prefixed names.
    motor_keys = {k for k in obs if k.endswith(".pos")}
    assert motor_keys == expected


def test_attach_teleop_routes_per_arm(bi_follower):
    """A bimanual leader teleop with left_arm / right_arm sub-attributes
    must be split per-arm — each follower controller polls its own
    arm's teleop, not the bimanual wrapper."""
    robot, _l, _r = bi_follower

    bi_teleop = MagicMock()
    bi_teleop.left_arm = MagicMock()
    bi_teleop.right_arm = MagicMock()

    robot.attach_teleop(bi_teleop)

    assert robot.left_arm._controller._teleop is bi_teleop.left_arm
    assert robot.right_arm._controller._teleop is bi_teleop.right_arm

    robot.attach_teleop(None)
    assert robot.left_arm._controller._teleop is None
    assert robot.right_arm._controller._teleop is None


def test_attach_teleop_skips_non_bimanual_teleop(bi_follower):
    """A non-bimanual teleop (e.g. TrajectoryReplayTeleop has no
    left_arm/right_arm attrs but DOES support the chunk path via
    send_action) shouldn't trigger an error from attach_teleop —
    the bimanual robot just no-ops the attach and lets send_action
    route the chunk. Binding the same single-arm teleop to both
    arms would have mixed up left/right poses on the buses, so we
    skip rather than do that. Test asserts: no exception, and
    neither arm's controller has _teleop bound."""
    robot, _l, _r = bi_follower
    single_arm_teleop = MagicMock(spec=["get_action"])  # no left_arm / right_arm
    # Should NOT raise.
    robot.attach_teleop(single_arm_teleop)
    assert robot.left_arm._controller._teleop is None
    assert robot.right_arm._controller._teleop is None
