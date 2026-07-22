#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math

import pytest

from lerobot.robots.openarm_follower import openarm_follower as follower_module
from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfig

MOTORS = [f"joint_{index}" for index in range(1, 8)] + ["gripper"]


class StubBus:
    def __init__(self, port, motors, **kwargs):
        self.port = port
        self.motors = motors
        self.is_connected = False
        self.is_calibrated = True
        self.connect_handshake = None
        self.configure_calls = 0
        self.enable_calls = 0
        self.enable_motor_calls = []
        self.disable_calls = 0
        self.zero_calls = 0
        self.disconnect_calls = []
        self.sent = []
        self.posforce_sent = []
        self.positions = dict.fromkeys(motors, 0.0)
        self.velocities = dict.fromkeys(motors, 0.0)
        self.temperatures = dict.fromkeys(motors, 25.0)
        self.statuses = dict.fromkeys(motors, 0)
        self.events = []
        self.fault_latched = False
        self.fault_reason = None
        self.mode_queries = []
        self.modes = {motor: (4 if motor == "gripper" else 1) for motor in motors}

    def connect(self, handshake=True):
        self.connect_handshake = handshake
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        self.disconnect_calls.append(disable_torque)
        self.is_connected = False

    def configure_motors(self):
        self.configure_calls += 1

    def query_control_mode(self, motor):
        self.mode_queries.append(motor)
        return self.modes[motor]

    def write_control_mode(self, motor, mode):
        self.modes[motor] = mode
        return mode

    def enable_torque(self, motors=None):
        self.enable_calls += 1
        target = list(self.motors) if motors is None else list(motors)
        self.enable_motor_calls.append(target)
        self.events.append(("enable", target))
        for motor in target:
            self.statuses[motor] = 1

    def disable_torque(self, motors=None):
        self.disable_calls += 1
        target = list(self.motors) if motors is None else list(motors)
        self.events.append(("disable", target))
        for motor in target:
            self.statuses[motor] = 0

    def set_zero_position(self):
        self.zero_calls += 1

    def sync_read(self, data_name):
        assert data_name == "Present_Position"
        return dict(self.positions)

    def sync_read_all_states(self):
        self.events.append(("read", None))
        return {
            motor: {
                "status": self.statuses[motor],
                "position": position,
                "velocity": self.velocities[motor],
                "torque": 0.0,
                "temp_mos": self.temperatures[motor],
                "temp_rotor": self.temperatures[motor],
            }
            for motor, position in self.positions.items()
        }

    def mit_control_batch(self, commands):
        self.sent.append(commands)
        self.events.append(("mit", list(commands)))
        states = self.sync_read_all_states()
        return {motor: states[motor] for motor in commands}

    def posforce_control(self, motor, *, position_rad, speed_rad_s, current_pu):
        self.posforce_sent.append((motor, position_rad, speed_rad_s, current_pu))
        self.events.append(("posforce", motor))
        return self.sync_read_all_states()[motor]

    def posforce_command(self, motor, *, position_rad, speed_rad_s, current_pu):
        self.posforce_sent.append((motor, position_rad, speed_rad_s, current_pu))
        self.events.append(("posforce_command", motor))


class StubCamera:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.is_connected = False

    def connect(self):
        if self.fail:
            raise RuntimeError("camera failed")
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False


def make_follower(tmp_path, monkeypatch, **overrides):
    cameras = overrides.pop("stub_cameras", {})
    monkeypatch.setattr(follower_module, "DamiaoMotorsBus", StubBus)
    monkeypatch.setattr(follower_module, "make_cameras_from_configs", lambda configs: cameras)
    config = OpenArmFollowerConfig(
        id="test_openarm",
        calibration_dir=tmp_path,
        port="can0",
        **overrides,
    )
    return follower_module.OpenArmFollower(config)


def full_action(value=0.0):
    return {f"{motor}.pos": float(value) for motor in MOTORS}


def connect_and_enable(follower):
    follower.connect()
    follower.enable_torque()
    # Most action tests inspect only writes made by the action under test, not
    # the mandatory enable-time hold frames.
    follower.bus.sent.clear()
    follower.bus.posforce_sent.clear()


def test_action_schema_is_position_only(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    assert set(follower.action_features) == {f"{motor}.pos" for motor in MOTORS}
    assert "joint_1.vel" in follower.observation_features
    assert "joint_1.torque" in follower.observation_features


def test_connect_is_transport_only_by_default(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    follower.connect()

    assert follower.bus.connect_handshake is False
    assert follower.bus.configure_calls == 0
    assert follower.bus.enable_calls == 0
    assert follower.bus.zero_calls == 0


def test_connect_side_effects_require_explicit_config(tmp_path, monkeypatch):
    follower = make_follower(
        tmp_path,
        monkeypatch,
        handshake_on_connect=True,
        configure_on_connect=True,
        enable_torque_on_connect=True,
    )
    follower.connect()

    assert follower.bus.connect_handshake is True
    assert follower.bus.configure_calls == 0
    assert len(follower.bus.mode_queries) == 16
    assert follower.bus.enable_calls == 2
    assert follower.bus.zero_calls == 0


def test_connect_rolls_back_bus_when_camera_fails(tmp_path, monkeypatch):
    follower = make_follower(
        tmp_path,
        monkeypatch,
        stub_cameras={"bad": StubCamera(fail=True)},
    )

    with pytest.raises(RuntimeError, match="camera failed"):
        follower.connect()

    assert not follower.bus.is_connected
    assert follower.bus.disconnect_calls == [True]


def test_control_mode_mismatch_blocks_torque_enable(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    follower.connect()
    follower.bus.modes["gripper"] = 1

    with pytest.raises(RuntimeError, match="control-mode mismatch"):
        follower.enable_torque()

    assert follower.bus.enable_calls == 0


def test_gripper_mode_change_is_explicit_and_read_back(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    follower.connect()
    follower.bus.modes["gripper"] = 1

    assert follower.configure_gripper_control_mode() == 4
    assert follower.bus.modes["gripper"] == 4
    assert follower._control_modes_validated
    assert follower.bus.enable_calls == 0


def test_arming_reads_two_disabled_snapshots_then_holds_current_pose(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, side="right", arming_sample_interval_s=0.001)
    follower.connect()
    follower.bus.positions["joint_4"] = -0.8

    follower.enable_torque()

    first_enable = next(index for index, event in enumerate(follower.bus.events) if event[0] == "enable")
    assert sum(event[0] == "read" for event in follower.bus.events[:first_enable]) == 2
    assert follower.bus.enable_motor_calls == [MOTORS[:7], ["gripper"]]
    assert follower.bus.sent[0]["joint_4"][2] == pytest.approx(-0.8)
    assert follower.bus.posforce_sent[0][1] == pytest.approx(0.0)
    assert follower.bus.posforce_sent[0][3] == 0.0
    assert follower._torque_enabled
    assert follower._gripper_torque_enabled


def test_arming_rejects_motion_before_any_enable(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, arming_sample_interval_s=0.001)
    follower.connect()
    follower.bus.velocities["joint_2"] = 3.0

    with pytest.raises(ConnectionError, match="velocity"):
        follower.enable_torque()

    assert follower.bus.enable_calls == 0
    assert not follower._torque_enabled


def test_arming_rejects_overtemperature_before_any_enable(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, arming_sample_interval_s=0.001)
    follower.connect()
    follower.bus.temperatures["joint_6"] = 71.0

    with pytest.raises(ConnectionError, match="temperatures"):
        follower.enable_torque()

    assert follower.bus.enable_calls == 0


def test_arming_deadline_failure_disables_every_motor(tmp_path, monkeypatch):
    follower = make_follower(
        tmp_path,
        monkeypatch,
        arming_sample_interval_s=0.001,
        arming_hold_timeout_s=1e-9,
    )
    follower.connect()

    with pytest.raises(TimeoutError, match="arming deadline"):
        follower.enable_torque()

    assert follower.bus.disable_calls == 1
    assert set(follower.bus.statuses.values()) == {0}
    assert not follower._torque_enabled


def test_gripper_can_be_armed_as_a_separate_zero_current_stage(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, arming_sample_interval_s=0.001)
    follower.connect()

    follower.enable_torque(include_gripper=False)

    assert follower.bus.enable_motor_calls == [MOTORS[:7]]
    assert follower._torque_enabled
    assert not follower._gripper_torque_enabled
    with pytest.raises(RuntimeError, match="gripper torque"):
        follower.send_action(full_action())

    follower.enable_gripper_torque()

    assert follower.bus.enable_motor_calls[-1] == ["gripper"]
    assert follower.bus.posforce_sent[-1][3] == 0.0
    assert follower._gripper_torque_enabled


def test_arming_hold_feedback_failure_disables_every_motor(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, arming_sample_interval_s=0.001)
    follower.connect()
    original_batch = follower.bus.mit_control_batch

    def disabled_feedback(commands):
        states = original_batch(commands)
        states["joint_3"]["status"] = 0
        return states

    follower.bus.mit_control_batch = disabled_feedback

    with pytest.raises(ConnectionError, match="hold feedback"):
        follower.enable_torque()

    assert follower.bus.disable_calls == 1
    assert set(follower.bus.statuses.values()) == {0}
    assert not follower._torque_enabled
    assert not follower._gripper_torque_enabled


def test_explicit_joint_limits_are_not_overwritten_by_side(tmp_path, monkeypatch):
    limits = dict.fromkeys(MOTORS, (-2.0, 2.0))
    follower = make_follower(tmp_path, monkeypatch, side="left", joint_limits=limits)
    assert follower.config.joint_limits == limits


def test_integer_relative_limit_is_normalized(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, max_relative_target=1)
    assert follower.config.max_relative_target == 1.0


@pytest.mark.parametrize(
    "action",
    [
        {},
        {"joint_1.pos": 0.0},
        {**full_action(), "unknown.pos": 0.0},
        {**full_action(), "joint_1.pos": math.nan},
        {**full_action(), "joint_1.pos": math.inf},
    ],
)
def test_invalid_action_is_rejected_before_can_write(tmp_path, monkeypatch, action):
    follower = make_follower(tmp_path, monkeypatch)
    connect_and_enable(follower)

    with pytest.raises(ValueError):
        follower.send_action(action)

    assert follower.bus.sent == []


def test_default_relative_limit_clamps_every_motor(tmp_path, monkeypatch):
    follower = make_follower(
        tmp_path,
        monkeypatch,
        joint_limits=dict.fromkeys(MOTORS, (-100.0, 100.0)),
    )
    connect_and_enable(follower)
    follower.bus.positions = dict.fromkeys(MOTORS, 10.0)

    sent = follower.send_action(full_action(20.0))

    assert sent["joint_1.pos"] == pytest.approx(10.2)
    assert follower.bus.sent[-1]["joint_1"][2] == pytest.approx(10.2)


def test_joint4_tolerance_corridor_holds_then_recovers_monotonically(tmp_path, monkeypatch):
    follower = make_follower(
        tmp_path,
        monkeypatch,
        side="right",
        arming_sample_interval_s=0.001,
    )
    follower.connect()
    follower.bus.positions["joint_4"] = -0.8
    follower.enable_torque()
    assert follower.bus.sent[0]["joint_4"][2] == pytest.approx(-0.8)
    follower.bus.sent.clear()
    follower.bus.posforce_sent.clear()

    follower.send_action(full_action(-10.0))

    recovered = follower.bus.sent[-1]["joint_4"][2]
    assert recovered == pytest.approx(-0.6)
    assert -0.8 <= recovered <= 0.0


def test_action_status_preflight_disables_whole_arm_before_control(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, arming_sample_interval_s=0.001)
    connect_and_enable(follower)
    follower.bus.statuses["joint_5"] = 0
    disable_calls = follower.bus.disable_calls

    with pytest.raises(ConnectionError, match="status"):
        follower.send_action(full_action())

    assert follower.bus.sent == []
    assert follower.bus.posforce_sent == []
    assert follower.bus.disable_calls == disable_calls + 1
    assert set(follower.bus.statuses.values()) == {0}


def test_gravity_feedforward_is_zero_by_default(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    connect_and_enable(follower)

    follower.send_action(full_action())

    assert all(command[4] == 0.0 for command in follower.bus.sent[-1].values())


def test_standard_gripper_uses_posforce_frame_path(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    connect_and_enable(follower)

    follower.send_action(full_action())

    assert "gripper" not in follower.bus.sent[-1]
    motor, position_rad, speed_rad_s, current_pu = follower.bus.posforce_sent[-1]
    assert motor == "gripper"
    assert position_rad == pytest.approx(0.0)
    assert speed_rad_s == pytest.approx(5.0)
    assert current_pu == pytest.approx(0.1)


def test_action_rejects_current_position_outside_absolute_limits(tmp_path, monkeypatch):
    follower = make_follower(
        tmp_path,
        monkeypatch,
        joint_limits=dict.fromkeys(MOTORS, (-100.0, 100.0)),
    )
    connect_and_enable(follower)
    follower.bus.positions["joint_1"] = -720.0

    with pytest.raises(ConnectionError, match="outside_limits"):
        follower.send_action(full_action())

    assert follower.bus.sent == []


def test_observation_rejects_missing_or_nonfinite_state(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    follower.connect()
    del follower.bus.positions["joint_7"]
    with pytest.raises(ConnectionError, match="joint_7"):
        follower.get_observation()

    follower.bus.positions["joint_7"] = math.nan
    with pytest.raises(ConnectionError, match="joint_7"):
        follower.get_observation()


def test_observation_failure_disables_an_armed_whole_arm(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, arming_sample_interval_s=0.001)
    connect_and_enable(follower)
    del follower.bus.positions["joint_7"]
    disable_calls = follower.bus.disable_calls

    with pytest.raises(ConnectionError, match="joint_7"):
        follower.get_observation()

    assert follower.bus.disable_calls == disable_calls + 1
    assert set(follower.bus.statuses.values()) == {0}
    assert not follower._torque_enabled
    assert not follower._gripper_torque_enabled
