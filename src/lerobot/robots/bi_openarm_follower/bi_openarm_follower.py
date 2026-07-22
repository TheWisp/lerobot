#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from ..robot import Robot
from .config_bi_openarm_follower import BiOpenArmFollowerConfig

logger = logging.getLogger(__name__)


class BiOpenArmFollower(Robot):
    """
    Bimanual OpenArm Follower Arms
    """

    config_class = BiOpenArmFollowerConfig
    name = "bi_openarm_follower"

    def __init__(self, config: BiOpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        # Top-level cameras are distributed evenly: each arm's OpenArmFollower
        # will only open the cameras assigned to it. Per-arm cameras are used
        # as fallback when top-level cameras are empty.
        if config.cameras:
            left_cameras = config.cameras
            right_cameras = {}
        else:
            left_cameras = config.left_arm_config.cameras
            right_cameras = config.right_arm_config.cameras

        duplicate_camera_names = set(left_cameras) & set(right_cameras)
        if duplicate_camera_names:
            raise ValueError(
                "Bimanual OpenArm camera names must be unique across both arms; "
                f"duplicates={sorted(duplicate_camera_names)}"
            )

        left_arm_config = OpenArmFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            handshake_on_connect=config.left_arm_config.handshake_on_connect,
            configure_on_connect=config.left_arm_config.configure_on_connect,
            enable_torque_on_connect=config.left_arm_config.enable_torque_on_connect,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.left_arm_config.max_relative_target,
            present_position_tolerance_deg=config.left_arm_config.present_position_tolerance_deg,
            arming_sample_interval_s=config.left_arm_config.arming_sample_interval_s,
            arming_max_position_delta_deg=config.left_arm_config.arming_max_position_delta_deg,
            arming_max_velocity_deg_s=config.left_arm_config.arming_max_velocity_deg_s,
            arming_max_temperature_c=config.left_arm_config.arming_max_temperature_c,
            arming_hold_timeout_s=config.left_arm_config.arming_hold_timeout_s,
            gravity_ff_gain=config.left_arm_config.gravity_ff_gain,
            gravity_ff_xml=config.left_arm_config.gravity_ff_xml,
            gripper_control_mode=config.left_arm_config.gripper_control_mode,
            gripper_speed_rad_s=config.left_arm_config.gripper_speed_rad_s,
            gripper_torque_pu=config.left_arm_config.gripper_torque_pu,
            cameras=left_cameras,
            side=config.left_arm_config.side,
            can_interface=config.left_arm_config.can_interface,
            use_can_fd=config.left_arm_config.use_can_fd,
            can_bitrate=config.left_arm_config.can_bitrate,
            can_data_bitrate=config.left_arm_config.can_data_bitrate,
            motor_config=config.left_arm_config.motor_config,
            position_kd=config.left_arm_config.position_kd,
            position_kp=config.left_arm_config.position_kp,
            joint_limits=config.left_arm_config.joint_limits,
        )

        right_arm_config = OpenArmFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            handshake_on_connect=config.right_arm_config.handshake_on_connect,
            configure_on_connect=config.right_arm_config.configure_on_connect,
            enable_torque_on_connect=config.right_arm_config.enable_torque_on_connect,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.right_arm_config.max_relative_target,
            present_position_tolerance_deg=config.right_arm_config.present_position_tolerance_deg,
            arming_sample_interval_s=config.right_arm_config.arming_sample_interval_s,
            arming_max_position_delta_deg=config.right_arm_config.arming_max_position_delta_deg,
            arming_max_velocity_deg_s=config.right_arm_config.arming_max_velocity_deg_s,
            arming_max_temperature_c=config.right_arm_config.arming_max_temperature_c,
            arming_hold_timeout_s=config.right_arm_config.arming_hold_timeout_s,
            gravity_ff_gain=config.right_arm_config.gravity_ff_gain,
            gravity_ff_xml=config.right_arm_config.gravity_ff_xml,
            gripper_control_mode=config.right_arm_config.gripper_control_mode,
            gripper_speed_rad_s=config.right_arm_config.gripper_speed_rad_s,
            gripper_torque_pu=config.right_arm_config.gripper_torque_pu,
            cameras=right_cameras,
            side=config.right_arm_config.side,
            can_interface=config.right_arm_config.can_interface,
            use_can_fd=config.right_arm_config.use_can_fd,
            can_bitrate=config.right_arm_config.can_bitrate,
            can_data_bitrate=config.right_arm_config.can_data_bitrate,
            motor_config=config.right_arm_config.motor_config,
            position_kd=config.right_arm_config.position_kd,
            position_kp=config.right_arm_config.position_kp,
            joint_limits=config.right_arm_config.joint_limits,
        )

        self.left_arm = OpenArmFollower(left_arm_config)
        self.right_arm = OpenArmFollower(right_arm_config)
        self._ik_kinematics: dict[str, Any] | None = None
        self._attached_cartesian_teleop: Any | None = None

        reserved_motor_features = {
            **{f"left_{key}": value for key, value in self.left_arm._motors_ft.items()},
            **{f"right_{key}": value for key, value in self.right_arm._motors_ft.items()},
        }
        camera_motor_collisions = (set(self.left_arm.cameras) | set(self.right_arm.cameras)) & set(
            reserved_motor_features
        )
        if camera_motor_collisions:
            raise ValueError(
                "Bimanual OpenArm camera names must not collide with motor feature names; "
                f"collisions={sorted(camera_motor_collisions)}"
            )

        # Only for compatibility with other parts of the codebase that expect a `robot.cameras` attribute
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_arm_motors_ft = self.left_arm._motors_ft
        right_arm_motors_ft = self.right_arm._motors_ft

        # Right first, then left — matches the teleoperator (OpenArmMini) ordering
        # and the dataset feature names recorded during data collection.
        return {
            **{f"right_{k}": v for k, v in right_arm_motors_ft.items()},
            **{f"left_{k}": v for k, v in left_arm_motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # Cameras already have unique user-chosen names (e.g. "left_wrist", "base",
        # "right_wrist"), so we merge them directly — unlike motors which need the
        # left_/right_ prefix to disambiguate identical per-arm joint names.
        return {**self.left_arm._cameras_ft, **self.right_arm._cameras_ft}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"right_{key}": value for key, value in self.right_arm.action_features.items()},
            **{f"left_{key}": value for key, value in self.left_arm.action_features.items()},
        }

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        left_connected = False
        try:
            self.left_arm.connect(calibrate)
            left_connected = True
            self.right_arm.connect(calibrate)
        except Exception:
            if left_connected:
                try:
                    self.left_arm.disconnect()
                except Exception:
                    logger.exception("Failed to roll back left arm connection")
            raise

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        try:
            self.left_arm.configure()
            self.right_arm.configure()
        except Exception:
            try:
                self.disable_torque()
            except Exception:
                logger.exception("Failed to disable both arms after configuration failure")
            raise

    def enable_torque(self) -> None:
        left_enabled = False
        try:
            self.left_arm.enable_torque()
            left_enabled = True
            self.right_arm.enable_torque()
        except Exception:
            if left_enabled:
                try:
                    self.disable_torque()
                except Exception:
                    logger.exception("Failed to roll back bimanual torque enable")
            raise

    def disable_torque(self) -> None:
        first_error: Exception | None = None
        for arm in (self.left_arm, self.right_arm):
            try:
                arm.disable_torque()
            except Exception as exc:
                first_error = first_error or exc
                logger.exception("Failed to disable arm torque")
        if first_error is not None:
            raise first_error

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    def _mjcf_xml_for_cartesian_ik(self) -> str | None:
        configured = [
            value
            for value in (
                self.left_arm.config.gravity_ff_xml,
                self.right_arm.config.gravity_ff_xml,
            )
            if value is not None
        ]
        if len(configured) == 2 and Path(configured[0]).expanduser().resolve() != Path(
            configured[1]
        ).expanduser().resolve():
            raise ValueError("Both OpenArm sides must use the same bimanual MJCF for Cartesian IK")
        return configured[0] if configured else None

    def attach_teleop(self, teleop: Any) -> None:
        """Install or clear the MJCF Cartesian transform used by Quest VR."""
        from ..openarm_description import (
            MJCFArmKinematics,
            build_openarm_bimanual_mjcf_ik_transform,
            is_openarm_bimanual_cartesian_teleop,
        )

        previous = self._attached_cartesian_teleop
        if previous is not None:
            previous.set_action_transform(None)
            self._attached_cartesian_teleop = None

        if teleop is None:
            return
        if not is_openarm_bimanual_cartesian_teleop(teleop):
            return
        if not self.is_connected:
            raise RuntimeError("Attach the Cartesian teleop only after both OpenArm sides are connected")
        if not hasattr(teleop, "set_action_transform"):
            raise TypeError("A Cartesian teleop must expose set_action_transform()")

        if self._ik_kinematics is None:
            xml = self._mjcf_xml_for_cartesian_ik()
            self._ik_kinematics = {
                side: MJCFArmKinematics(
                    side,
                    xml=xml,
                    max_iterations=self.config.ik_max_iterations,
                    damping=self.config.ik_damping,
                )
                for side in ("left", "right")
            }

        transform = build_openarm_bimanual_mjcf_ik_transform(
            self._ik_kinematics,
            self.left_arm,
            self.right_arm,
        )
        teleop.set_action_transform(transform)
        self._attached_cartesian_teleop = teleop
        logger.info("%s: installed MJCF Cartesian transform into %s", self.name, type(teleop).__name__)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict = {}

        # Camera keys that should NOT get the arm prefix (they already have unique names)
        left_cam_keys = set(self.left_arm.cameras.keys())
        right_cam_keys = set(self.right_arm.cameras.keys())

        # Right first, then left — matches the teleoperator (OpenArmMini) ordering
        # and the dataset feature names recorded during data collection.
        right_obs = self.right_arm.get_observation()
        for key, value in right_obs.items():
            obs_dict[key if key in right_cam_keys else f"right_{key}"] = value

        left_obs = self.left_arm.get_observation()
        for key, value in left_obs.items():
            obs_dict[key if key in left_cam_keys else f"left_{key}"] = value

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        expected = set(self.action_features)
        received = set(action)
        if received != expected:
            raise ValueError(
                "Bimanual OpenArm actions must contain every left/right motor position; "
                f"missing={sorted(expected - received)}, unknown={sorted(received - expected)}"
            )

        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        try:
            # Complete all read-only validation for both arms before the first
            # control frame. This prevents a bad right-side input/state from
            # allowing a partial left-side action.
            self.left_arm._ensure_ready_for_action()
            self.right_arm._ensure_ready_for_action()
            prepared_left = self.left_arm._prepare_action(left_action, custom_kp, custom_kd)
            prepared_right = self.right_arm._prepare_action(right_action, custom_kp, custom_kd)
            # can0 and can1 are independent interfaces. Release both worker
            # calls from the same event so their first control frames are not
            # separated by a complete arm batch/response round trip.
            start = threading.Barrier(2)

            def execute(arm: OpenArmFollower, prepared: Any) -> RobotAction:
                start.wait(timeout=1.0)
                return arm._execute_prepared_action(prepared)

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="openarm-send") as executor:
                left_future = executor.submit(execute, self.left_arm, prepared_left)
                right_future = executor.submit(execute, self.right_arm, prepared_right)
                sent_action_left = left_future.result()
                sent_action_right = right_future.result()
        except Exception:
            try:
                self.disable_torque()
            except Exception:
                logger.exception("Failed to disable both arms after bimanual command failure")
            raise

        # Add prefixes back
        prefixed_sent_action_left = {f"left_{key}": value for key, value in sent_action_left.items()}
        prefixed_sent_action_right = {f"right_{key}": value for key, value in sent_action_right.items()}

        return {**prefixed_sent_action_right, **prefixed_sent_action_left}

    def disconnect(self) -> None:
        self.attach_teleop(None)
        first_error: Exception | None = None
        for arm in (self.left_arm, self.right_arm):
            if not arm.is_connected:
                continue
            try:
                arm.disconnect()
            except Exception as exc:
                first_error = first_error or exc
                logger.exception("Failed to disconnect arm")
        if first_error is not None:
            raise first_error
