#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest

from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfig
from lerobot.teleoperators.bi_openarm_leader.bi_openarm_leader import BiOpenArmLeader
from lerobot.teleoperators.bi_openarm_leader.config_bi_openarm_leader import BiOpenArmLeaderConfig
from lerobot.teleoperators.openarm_leader import openarm_leader as leader_module
from lerobot.teleoperators.openarm_leader.config_openarm_leader import (
    OpenArmLeaderConfig,
    OpenArmLeaderConfigBase,
)

MOTORS = [f"joint_{index}" for index in range(1, 8)] + ["gripper"]


class _StateBus:
    def __init__(self, port, motors, **kwargs):
        self.port = port
        self.motors = motors
        self.is_connected = True
        self.is_calibrated = True

    def sync_read_all_states(self):
        return {
            motor: {
                "status": 0,
                "position": float(index),
                "velocity": float(index) / 10.0,
                "torque": -float(index) / 20.0,
                "temp_mos": 30.0 + index,
                "temp_rotor": 29.0 + index,
            }
            for index, motor in enumerate(self.motors, start=1)
        }

    def disconnect(self, disable_torque=True):
        self.is_connected = False


class _LifecycleBus(_StateBus):
    def __init__(self, port, motors, **kwargs):
        super().__init__(port, motors, **kwargs)
        self.is_connected = False
        self.zero_calls = 0

    def connect(self):
        self.is_connected = True

    def disable_torque(self):
        pass

    def configure_motors(self):
        pass

    def set_zero_position(self):
        self.zero_calls += 1


def _position_keys(prefix=""):
    return {f"{prefix}{motor}.pos" for motor in MOTORS}


def test_openarm_leader_emits_position_only_and_retains_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(leader_module, "DamiaoMotorsBus", _StateBus)
    leader = leader_module.OpenArmLeader(
        OpenArmLeaderConfig(id="leader", calibration_dir=tmp_path, port="can0")
    )

    action = leader.get_action()

    follower_config = OpenArmFollowerConfig(port="can1")
    follower_action_keys = {f"{motor}.pos" for motor in follower_config.motor_config}
    assert set(leader.action_features) == follower_action_keys
    assert set(action) == follower_action_keys
    assert action["joint_3.pos"] == 3.0
    assert leader.last_motor_states["joint_3"]["velocity"] == 0.3
    assert leader.last_motor_states["joint_3"]["torque"] == -0.15
    assert leader.last_motor_states["joint_3"]["temp_mos"] == 33.0

    diagnostics = leader.last_motor_states
    diagnostics["joint_3"]["velocity"] = 999.0
    assert leader.last_motor_states["joint_3"]["velocity"] == 0.3


def test_bimanual_leader_keys_match_bimanual_follower_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(leader_module, "DamiaoMotorsBus", _StateBus)
    leader = BiOpenArmLeader(
        BiOpenArmLeaderConfig(
            id="bi_leader",
            calibration_dir=tmp_path,
            left_arm_config=OpenArmLeaderConfigBase(port="can0"),
            right_arm_config=OpenArmLeaderConfigBase(port="can1"),
        )
    )

    action = leader.get_action()
    follower_config = OpenArmFollowerConfig(port="can2")
    expected = {f"{side}_{motor}.pos" for side in ("left", "right") for motor in follower_config.motor_config}

    assert set(leader.action_features) == expected
    assert set(action) == expected
    expected_diagnostic_keys = {f"left_{motor}" for motor in MOTORS} | {f"right_{motor}" for motor in MOTORS}
    assert set(leader.last_motor_states) == expected_diagnostic_keys
    assert all(key.endswith(".pos") for key in action)


@pytest.mark.parametrize(("set_zero_on_connect", "expected_calls"), [(False, 0), (True, 1)])
def test_leader_zero_write_requires_explicit_opt_in(
    tmp_path, monkeypatch, set_zero_on_connect, expected_calls
):
    monkeypatch.setattr(leader_module, "DamiaoMotorsBus", _LifecycleBus)
    leader = leader_module.OpenArmLeader(
        OpenArmLeaderConfig(
            id="lifecycle",
            calibration_dir=tmp_path,
            port="can0",
            **({"set_zero_on_connect": True} if set_zero_on_connect else {}),
        )
    )

    leader.connect(calibrate=False)

    assert leader.bus.zero_calls == expected_calls
