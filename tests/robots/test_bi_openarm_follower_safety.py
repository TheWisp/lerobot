#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math
import threading

import pytest

from lerobot.robots.bi_openarm_follower import bi_openarm_follower as bi_module
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase

MOTORS = [f"joint_{index}" for index in range(1, 8)] + ["gripper"]


class CameraConfigStub:
    width = 640
    height = 480
    fps = 30


class FakeArm:
    instances = []
    fail_port = None

    def __init__(self, config):
        self.config = config
        self.is_connected = False
        self.is_calibrated = True
        self.cameras = config.cameras
        self.connect_calls = []
        self.disconnect_calls = 0
        self.enable_calls = 0
        self.disable_calls = 0
        self.sent = []
        FakeArm.instances.append(self)

    @property
    def _motors_ft(self):
        return {
            **{f"{motor}.pos": float for motor in MOTORS},
            **{f"{motor}.vel": float for motor in MOTORS},
            **{f"{motor}.torque": float for motor in MOTORS},
        }

    @property
    def action_features(self):
        return {f"{motor}.pos": float for motor in MOTORS}

    @property
    def _cameras_ft(self):
        return {}

    def connect(self, calibrate=False):
        self.connect_calls.append(calibrate)
        if self.config.port == self.fail_port:
            raise RuntimeError("arm connect failed")
        self.is_connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self.is_connected = False

    def enable_torque(self):
        self.enable_calls += 1

    def disable_torque(self):
        self.disable_calls += 1

    def configure(self):
        pass

    def calibrate(self):
        pass

    def get_observation(self):
        return {f"{motor}.pos": 0.0 for motor in MOTORS}

    def send_action(self, action, custom_kp=None, custom_kd=None):
        self.sent.append(action)
        return action

    def _ensure_ready_for_action(self):
        pass

    def _prepare_action(self, action, custom_kp=None, custom_kd=None):
        for value in action.values():
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError("invalid action")
        return action

    def _execute_prepared_action(self, prepared):
        return self.send_action(prepared)


@pytest.fixture
def robot(tmp_path, monkeypatch):
    FakeArm.instances = []
    FakeArm.fail_port = None
    monkeypatch.setattr(bi_module, "OpenArmFollower", FakeArm)
    config = BiOpenArmFollowerConfig(
        calibration_dir=tmp_path,
        left_arm_config=OpenArmFollowerConfigBase(port="can0", side="left"),
        right_arm_config=OpenArmFollowerConfigBase(port="can1", side="right"),
    )
    return bi_module.BiOpenArmFollower(config)


def full_action(value=0.0):
    return {f"{side}_{motor}.pos": float(value) for side in ("left", "right") for motor in MOTORS}


def test_bimanual_action_schema_is_position_only(robot):
    assert set(robot.action_features) == set(full_action())
    assert "left_joint_1.vel" in robot.observation_features
    assert "right_joint_1.torque" in robot.observation_features


def test_right_connect_failure_rolls_back_left(robot):
    FakeArm.fail_port = "can1"

    with pytest.raises(RuntimeError, match="arm connect failed"):
        robot.connect()

    assert robot.left_arm.disconnect_calls == 1
    assert not robot.left_arm.is_connected


def test_bimanual_action_must_be_complete(robot):
    robot.connect()

    with pytest.raises(ValueError, match="missing"):
        robot.send_action({"left_joint_1.pos": 0.0})

    assert robot.left_arm.sent == []
    assert robot.right_arm.sent == []


def test_bimanual_action_routes_both_complete_arms(robot):
    robot.connect()
    action = full_action(2.0)

    sent = robot.send_action(action)

    assert sent == action
    assert set(robot.left_arm.sent[-1]) == {f"{motor}.pos" for motor in MOTORS}
    assert set(robot.right_arm.sent[-1]) == {f"{motor}.pos" for motor in MOTORS}


def test_bimanual_actions_execute_concurrently(robot, monkeypatch):
    robot.connect()
    rendezvous = threading.Barrier(2)

    def execute(arm, prepared):
        rendezvous.wait(timeout=1.0)
        arm.sent.append(prepared)
        return prepared

    monkeypatch.setattr(
        robot.left_arm,
        "_execute_prepared_action",
        lambda prepared: execute(robot.left_arm, prepared),
    )
    monkeypatch.setattr(
        robot.right_arm,
        "_execute_prepared_action",
        lambda prepared: execute(robot.right_arm, prepared),
    )

    assert robot.send_action(full_action(2.0)) == full_action(2.0)
    assert len(robot.left_arm.sent) == 1
    assert len(robot.right_arm.sent) == 1


def test_invalid_right_action_is_rejected_before_left_send(robot):
    robot.connect()
    action = full_action()
    action["right_joint_4.pos"] = math.nan

    with pytest.raises(ValueError, match="invalid action"):
        robot.send_action(action)

    assert robot.left_arm.sent == []
    assert robot.right_arm.sent == []


def test_duplicate_camera_names_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(bi_module, "OpenArmFollower", FakeArm)
    config = BiOpenArmFollowerConfig(
        calibration_dir=tmp_path,
        left_arm_config=OpenArmFollowerConfigBase(port="can0", cameras={"wrist": CameraConfigStub()}),
        right_arm_config=OpenArmFollowerConfigBase(port="can1", cameras={"wrist": CameraConfigStub()}),
    )

    with pytest.raises(ValueError, match="camera names must be unique"):
        bi_module.BiOpenArmFollower(config)


def test_camera_motor_feature_collision_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(bi_module, "OpenArmFollower", FakeArm)
    config = BiOpenArmFollowerConfig(
        calibration_dir=tmp_path,
        left_arm_config=OpenArmFollowerConfigBase(port="can0"),
        right_arm_config=OpenArmFollowerConfigBase(port="can1"),
        cameras={"left_joint_1.pos": CameraConfigStub()},
    )

    with pytest.raises(ValueError, match="camera names must not collide"):
        bi_module.BiOpenArmFollower(config)
