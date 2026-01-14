#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from ..so_leader_base import SOLeaderBase
from .config_so107_leader import SO107LeaderConfig

logger = logging.getLogger(__name__)


class SO107Leader(SOLeaderBase):
    """
    SO-107 Leader Arm designed by TheRobotStudio and Hugging Face.
    7-axis arm with forearm_roll joint.
    """

    config_class = SO107LeaderConfig
    name = "so107_leader"

    def __init__(self, config: SO107LeaderConfig):
        # Call Teleoperator.__init__ directly to skip SOLeaderBase.__init__
        # We need to define our own motor configuration
        super(SOLeaderBase, self).__init__(config)
        self.config = config

        # SO-107 specific: 7 motors including forearm_roll
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "forearm_roll": Motor(4, "sts3215", norm_mode_body),  # Extra motor for SO-107
                "wrist_flex": Motor(5, "sts3215", norm_mode_body),
                "wrist_roll": Motor(6, "sts3215", norm_mode_body),
                "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    def connect(self, calibrate: bool = True) -> None:
        """Connect and optionally setup gripper bounce if configured."""
        super().connect(calibrate)

        # Setup gripper bounce if configured
        if self.config.gripper_bounce:
            logger.info("Setting up gripper bounce to neutral position (50% open)...")
            self.set_gripper_bounce()

    def set_gripper_bounce(self) -> None:
        """Set gripper to bounce back to 50% open position with weak torque."""
        weak_torque = 100
        neutral_position = 50.0  # 50% open

        logger.info(f"Applying weak torque limit: {weak_torque}/1000 ({weak_torque/10:.0f}%)")
        logger.info(f"Setting gripper neutral position to: {neutral_position}%")

        # Set weak torque limit for gentle return
        self.bus.write("Torque_Limit", "gripper", weak_torque, normalize=False)

        # Command to neutral position with weak torque
        self.bus.write("Goal_Position", "gripper", neutral_position, normalize=True)
