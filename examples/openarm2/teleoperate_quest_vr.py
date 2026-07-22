#!/usr/bin/env python

"""Run the validated OpenArm 2 standard-edition Quest teleop configuration."""

from lerobot.configs.openarm2_standard import (
    make_openarm2_quest_config,
    make_openarm2_standard_robot_config,
)
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate

CONTROL_FPS = 60


def build_config() -> TeleoperateConfig:
    return TeleoperateConfig(
        robot=make_openarm2_standard_robot_config(
            control_fps=CONTROL_FPS,
            enable_torque_on_connect=True,
        ),
        teleop=make_openarm2_quest_config(),
        fps=CONTROL_FPS,
        display_data=False,
        latency_monitor=True,
    )


if __name__ == "__main__":
    teleoperate(build_config())
