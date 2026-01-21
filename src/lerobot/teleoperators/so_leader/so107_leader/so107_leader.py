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
from typing import Any

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from ..so_leader_base import SOLeaderBase
from ...utils import TeleopEvents
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

        # Intervention state (latched - once triggered, stays active until reset)
        self._intervention_active = False
        self._keyboard_listener = None

    def connect(self, calibrate: bool = True) -> None:
        """Connect and optionally setup gripper bounce if configured."""
        super().connect(calibrate)

        # Setup gripper bounce if configured
        if self.config.gripper_bounce:
            logger.info("Setting up gripper bounce to neutral position (50% open)...")
            self.set_gripper_bounce()

        # Start keyboard listener if intervention enabled
        if self.config.intervention_enabled:
            self._start_keyboard_listener()

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

    @property
    def feedback_features(self) -> dict[str, type]:
        """Features that can be sent as feedback (motor positions)."""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Command leader motors to given positions (for inverse-follow).

        Args:
            feedback: Dict mapping keys like "shoulder_pan.pos" to target positions.
        """
        if not feedback:
            return

        goal_positions = {}
        for motor_name in self.bus.motors:
            pos_key = f"{motor_name}.pos"
            if pos_key in feedback:
                goal_positions[motor_name] = feedback[pos_key]

        if goal_positions:
            self.bus.sync_write("Goal_Position", goal_positions)

    def enable_torque(self, num_retry: int = 5) -> None:
        """Enable torque on leader motors (for inverse-follow mode)."""
        self.bus.enable_torque(num_retry=num_retry)

    def disable_torque(self) -> None:
        """Disable torque on leader motors (for human control)."""
        self.bus.disable_torque()

    def _start_keyboard_listener(self) -> None:
        """Start keyboard listener for intervention detection (space key)."""
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.space:
                self._intervention_active = not self._intervention_active
                if self._intervention_active:
                    logger.info("INTERVENTION ON - Switched to teleop mode")
                else:
                    logger.info("INTERVENTION OFF - Returning to policy mode")

        self._keyboard_listener = keyboard.Listener(on_press=on_press)
        self._keyboard_listener.start()
        logger.info("Intervention enabled: Press SPACE to toggle between policy and teleop")

    def get_teleop_events(self) -> dict[str, Any]:
        """Return intervention status.

        Returns:
            Dict with TeleopEvents keys indicating current intervention state.
        """
        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_active if self.config.intervention_enabled else False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def reset_intervention(self) -> None:
        """Reset intervention state for new episode."""
        self._intervention_active = False

    def disconnect(self) -> None:
        """Disconnect and clean up keyboard listener."""
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        super().disconnect()
