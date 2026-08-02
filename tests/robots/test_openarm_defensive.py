# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fail-fast guards for two OpenArm states that are otherwise silent.

Both were found by review rather than by a failure, which is the point: neither
announces itself at runtime. A gravity-feedforward model with the wrong dof
layout keeps driving the arm with torque computed for other joints, and one
with no declared actuator force range clips every torque to zero while logging
that the feature is enabled. A reconnect that inherits the previous session's
last command rate-limits toward a pose the arm left when torque was released.

The validator is a pure function precisely so these can be tested without
MuJoCo or the OpenArm model installed — the guards are worth nothing if they
only run on the one machine that has the hardware stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.robots.openarm_follower.gravity_ff import DOFS, validate_model_layout

REAL_MODEL_NV = 18  # OpenArm 2.0 bimanual: 7 joints + gripper, per arm
# Measured on the rig for side="right": torque_frac 0.5 of the model's ranges.
REAL_LIMITS = np.array([20.0, 20.0, 13.5, 13.5, 3.5, 3.5, 3.5])


class TestGravityFeedforwardModelGuard:
    @pytest.mark.parametrize("side", sorted(DOFS))
    def test_the_shipped_model_layout_is_accepted(self, side):
        """The guard must not reject the model the robot actually runs."""
        validate_model_layout(side, REAL_MODEL_NV, REAL_LIMITS)

    @pytest.mark.parametrize("side", sorted(DOFS))
    def test_a_model_with_too_few_dofs_is_rejected(self, side):
        """A single-arm or differently-ordered MJCF must not silently proceed."""
        too_small = max(DOFS[side])  # one short of the highest index read

        with pytest.raises(ValueError, match="dofs"):
            validate_model_layout(side, too_small, REAL_LIMITS)

    @pytest.mark.parametrize("side", sorted(DOFS))
    def test_a_model_without_actuator_force_ranges_is_rejected(self, side):
        """All-zero limits mean the feature does nothing while claiming otherwise."""
        with pytest.raises(ValueError, match="silently do nothing"):
            validate_model_layout(side, REAL_MODEL_NV, np.zeros(len(DOFS[side])))

    def test_the_rejection_names_the_config_field_to_change(self):
        """An error an operator cannot act on is barely better than silence."""
        with pytest.raises(ValueError, match="gravity_ff_xml"):
            validate_model_layout("right", REAL_MODEL_NV, np.zeros(7))

    def test_partially_limited_models_are_accepted(self):
        """Only a fully unlimited model is the silent-no-op case."""
        partial = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        validate_model_layout("right", REAL_MODEL_NV, partial)  # must not raise


class TestDisconnectClearsSessionState:
    """A reconnect must ramp from the measured pose, not last session's command."""

    def _follower(self):
        from lerobot.robots.openarm_follower.openarm_follower import OpenArmFollower

        robot = OpenArmFollower.__new__(OpenArmFollower)
        robot.id = "test-arm"  # disconnect() logs f"{self} disconnected."
        robot.cameras = {}
        robot._gravity_ff = None
        robot._last_cmd_deg = {"joint_1": -120.0}
        robot._last_send_time = 1234.5
        robot._last_jump_log = 99.0

        class _Bus:
            is_connected = True  # disconnect() is guarded by @check_if_connected

            def disconnect(self, disable_torque):
                self.disconnected_with = disable_torque

        robot.bus = _Bus()

        class _Config:
            disable_torque_on_disconnect = True

        robot.config = _Config()
        return robot

    def test_last_command_does_not_survive_disconnect(self):
        robot = self._follower()

        robot.disconnect()

        assert robot._last_cmd_deg == {}, (
            "a surviving command makes the next connect ramp toward a pose the arm "
            "left when torque was released"
        )
        assert robot._last_send_time is None
        assert robot._last_jump_log == 0.0

    def test_jump_log_timestamp_stays_a_float(self):
        """Guards the rate-limiter arithmetic: `now - self._last_jump_log`."""
        robot = self._follower()

        robot.disconnect()

        assert isinstance(robot._last_jump_log, float)

    def test_gravity_feedforward_fade_is_reset(self):
        """Otherwise the first post-reconnect command applies full gain, unramped."""
        robot = self._follower()
        reset_calls = []

        class _FF:
            def reset(self):
                reset_calls.append(True)

        robot._gravity_ff = _FF()

        robot.disconnect()

        assert reset_calls == [True]
