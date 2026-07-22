# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The port-identify wiggle probe must derive its motor from the selected robot
profile — bus protocol, motor id and model all come from the robot's own
definition, never from an assumed servo (a hardcoded sts3215 previously broke
any non-Feetech robot). No hardware is touched: the spec is derived from a
constructed-but-unconnected robot.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from lerobot.gui.api.robot import _probe_motor_spec


def test_spec_derived_from_bi_so107_profile():
    """Bi-arm SO107: first motor of the left arm's Feetech bus, ports unassigned."""
    from lerobot.motors.feetech import FeetechMotorsBus

    # No port fields at all — mid-setup profiles have them cleared; the probe
    # must dummy-fill them rather than fail draccus decoding.
    bus_cls, motor_name, motor = _probe_motor_spec({"type": "bi_so107_follower", "fields": {}})

    assert bus_cls is FeetechMotorsBus
    assert motor_name == "shoulder_pan"
    assert motor.model == "sts3215"
    assert motor.id == 1


def test_spec_derived_from_single_arm_profile():
    """Single-arm robots expose .bus directly; same derivation path."""
    bus_cls, motor_name, motor = _probe_motor_spec({"type": "so107_follower", "fields": {}})

    assert motor_name == "shoulder_pan"
    assert motor.id == 1


def test_unknown_robot_type_raises():
    with pytest.raises(ValueError, match="Unknown robot type"):
        _probe_motor_spec({"type": "not_a_robot", "fields": {}})
