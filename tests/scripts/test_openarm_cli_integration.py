#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import draccus

from lerobot.robots import (
    bi_openarm_follower as bi_openarm_follower_package,
    openarm_follower as openarm_follower_package,
)
from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower.config_openarm_follower import (
    OpenArmFollowerConfig,
    OpenArmFollowerConfigBase,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.teleoperators import (
    bi_openarm_leader as bi_openarm_leader_package,
    openarm_leader as openarm_leader_package,
    quest_vr as quest_vr_package,
)
from lerobot.teleoperators.bi_openarm_leader.config_bi_openarm_leader import BiOpenArmLeaderConfig
from lerobot.teleoperators.openarm_leader.config_openarm_leader import (
    OpenArmLeaderConfig,
    OpenArmLeaderConfigBase,
)
from lerobot.teleoperators.quest_vr.configuration_quest_vr import QuestVRTeleopConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config


class _ConstructedDevice:
    def __init__(self, config):
        self.config = config


def test_bimanual_openarm_cli_config_and_factories(monkeypatch):
    cfg = draccus.parse(
        TeleoperateConfig,
        args=[
            "--robot.type=bi_openarm_follower",
            "--robot.left_arm_config.port=can0",
            "--robot.left_arm_config.side=left",
            "--robot.left_arm_config.gripper_control_mode=mit",
            "--robot.right_arm_config.port=can1",
            "--robot.right_arm_config.side=right",
            "--robot.right_arm_config.gripper_control_mode=mit",
            "--teleop.type=quest_vr",
        ],
    )

    assert isinstance(cfg.robot, BiOpenArmFollowerConfig)
    assert isinstance(cfg.robot.left_arm_config, OpenArmFollowerConfigBase)
    assert cfg.robot.left_arm_config.port == "can0"
    assert cfg.robot.left_arm_config.side == "left"
    assert cfg.robot.left_arm_config.gripper_control_mode == "mit"
    assert cfg.robot.right_arm_config.port == "can1"
    assert cfg.robot.right_arm_config.side == "right"
    assert isinstance(cfg.teleop, QuestVRTeleopConfig)

    monkeypatch.setattr(bi_openarm_follower_package, "BiOpenArmFollower", _ConstructedDevice)
    monkeypatch.setattr(quest_vr_package, "QuestVRTeleop", _ConstructedDevice)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    assert isinstance(robot, _ConstructedDevice)
    assert robot.config is cfg.robot
    assert isinstance(teleop, _ConstructedDevice)
    assert teleop.config is cfg.teleop


def test_openarm_single_and_bimanual_factory_dispatch(monkeypatch):
    monkeypatch.setattr(openarm_follower_package, "OpenArmFollower", _ConstructedDevice)
    monkeypatch.setattr(bi_openarm_follower_package, "BiOpenArmFollower", _ConstructedDevice)
    monkeypatch.setattr(openarm_leader_package, "OpenArmLeader", _ConstructedDevice)
    monkeypatch.setattr(bi_openarm_leader_package, "BiOpenArmLeader", _ConstructedDevice)

    robot_configs = [
        OpenArmFollowerConfig(port="can0"),
        BiOpenArmFollowerConfig(
            left_arm_config=OpenArmFollowerConfigBase(port="can0"),
            right_arm_config=OpenArmFollowerConfigBase(port="can1"),
        ),
    ]
    teleop_configs = [
        OpenArmLeaderConfig(port="can2"),
        BiOpenArmLeaderConfig(
            left_arm_config=OpenArmLeaderConfigBase(port="can2"),
            right_arm_config=OpenArmLeaderConfigBase(port="can3"),
        ),
    ]

    robots = [make_robot_from_config(config) for config in robot_configs]
    teleops = [make_teleoperator_from_config(config) for config in teleop_configs]

    assert [device.config for device in robots] == robot_configs
    assert [device.config for device in teleops] == teleop_configs
