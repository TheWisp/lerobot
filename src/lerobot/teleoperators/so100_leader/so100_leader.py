#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_so100_leader import SO100LeaderConfig

logger = logging.getLogger(__name__)


class SO100Leader(Teleoperator):
    """
    [SO-100 Leader Arm](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = SO100LeaderConfig
    name = "so100_leader"

    def __init__(self, config: SO100LeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.shoulder_pan_neutral_position = None
        self.wrist_roll_neutral_position = None
        self.gripper_neutral_position = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def set_shoulder_pan_neutral(self) -> None:
        """Set the current shoulder_pan position as neutral with weak return force.

        Torque_Limit range: 0-1000 (where 1000 = 100% of motor's stall torque)
        """
        weak_torque = 150

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read current max torque limit to inform user
        max_torque = self.bus.read("Max_Torque_Limit", "shoulder_pan", normalize=False)
        logger.info(f"Current Max_Torque_Limit for shoulder_pan: {max_torque}/1000")

        input("Press ENTER to set current shoulder_pan position as neutral position...")

        # Read current position
        current_pos = self.bus.read("Present_Position", "shoulder_pan", normalize=True)
        self.shoulder_pan_neutral_position = current_pos

        logger.info(f"Shoulder pan neutral position set to: {current_pos:.2f}")
        logger.info(f"Applying weak torque limit: {weak_torque}/1000 ({weak_torque/10:.1f}%)")

        # Set weak torque limit for gentle return
        self.bus.write("Torque_Limit", "shoulder_pan", weak_torque, normalize=False)

        # Command to neutral position with weak torque
        self.bus.write("Goal_Position", "shoulder_pan", self.shoulder_pan_neutral_position, normalize=True)

    def set_wrist_roll_neutral(self) -> None:
        """Set the current wrist_roll position as neutral with weak return force.

        Torque_Limit range: 0-1000 (where 1000 = 100% of motor's stall torque)
        """
        weak_torque = 250

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read current max torque limit to inform user
        max_torque = self.bus.read("Max_Torque_Limit", "wrist_roll", normalize=False)
        logger.info(f"Current Max_Torque_Limit for wrist_roll: {max_torque}/1000")

        input("Press ENTER to set current wrist_roll position as neutral position...")

        # Read current position
        current_pos = self.bus.read("Present_Position", "wrist_roll", normalize=True)
        self.wrist_roll_neutral_position = current_pos

        logger.info(f"Wrist roll neutral position set to: {current_pos:.2f}")
        logger.info(f"Applying weak torque limit: {weak_torque}/1000 ({weak_torque/10:.1f}%)")

        # Set weak torque limit for gentle return
        self.bus.write("Torque_Limit", "wrist_roll", weak_torque, normalize=False)

        # Command to neutral position with weak torque
        self.bus.write("Goal_Position", "wrist_roll", self.wrist_roll_neutral_position, normalize=True)

    def set_gripper_neutral(self) -> None:
        """Set the current gripper position as neutral with weak return force.

        Torque_Limit range: 0-1000 (where 1000 = 100% of motor's stall torque)
        """
        weak_torque = 125

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read current max torque limit to inform user
        max_torque = self.bus.read("Max_Torque_Limit", "gripper", normalize=False)
        logger.info(f"Current Max_Torque_Limit for gripper: {max_torque}/1000")

        input("Press ENTER to set current gripper position as neutral position...")

        # Read current position
        current_pos = self.bus.read("Present_Position", "gripper", normalize=True)
        self.gripper_neutral_position = current_pos

        logger.info(f"Gripper neutral position set to: {current_pos:.2f}")
        logger.info(f"Applying weak torque limit: {weak_torque}/1000 ({weak_torque/10:.1f}%)")

        # Set weak torque limit for gentle return
        self.bus.write("Torque_Limit", "gripper", weak_torque, normalize=False)

        # Command to neutral position with weak torque
        self.bus.write("Goal_Position", "gripper", self.gripper_neutral_position, normalize=True)

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
