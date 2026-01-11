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
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

from ..robot import Robot
from .config_bi_so107_follower import BiSO107FollowerConfig

logger = logging.getLogger(__name__)


class BiSO107Follower(Robot):
    """
    Bimanual SO-107 Follower Arms designed by TheRobotStudio and Hugging Face.
    7-axis arms with forearm_roll joint for advanced manipulation.
    """

    config_class = BiSO107FollowerConfig
    name = "bi_so107_follower"

    def __init__(self, config: BiSO107FollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO107FollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )

        right_arm_config = SO107FollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        self.left_arm = SO107Follower(left_arm_config)
        self.right_arm = SO107Follower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features = {}
        for cam_key in self.cameras:
            # Only RGB frames are stored in dataset
            # Depth frames (if enabled) are used by processors for visualization but not stored
            features[cam_key] = (self.config.cameras[cam_key].height, self.config.cameras[cam_key].width, 3)

        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()

            # For RealSense cameras with depth enabled, read aligned color+depth together
            from lerobot.cameras.realsense import RealSenseCamera

            if isinstance(cam, RealSenseCamera) and cam.use_depth:
                try:
                    color_frame, depth_frame = cam.read_color_and_aligned_depth()
                    obs_dict[cam_key] = color_frame
                    obs_dict[f"{cam_key}_depth"] = depth_frame
                    dt_ms = (time.perf_counter() - start) * 1e3
                    logger.debug(f"{self} read {cam_key} + aligned depth: {dt_ms:.1f}ms")
                except Exception as e:
                    logger.warning(f"{self} failed to read aligned frames for {cam_key}: {e}")
                    # Fallback: try reading color only
                    obs_dict[cam_key] = cam.async_read()
            else:
                # For non-RealSense or depth-disabled cameras, use async_read
                obs_dict[cam_key] = cam.async_read()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

    def get_observation_processor_steps(self) -> list:
        """Return custom observation processor steps for this robot."""
        from lerobot.cameras.realsense import RealSenseCamera
        from lerobot.processor import DepthEdgeOverlayProcessorStep

        steps = []

        # Add depth edge detection for RealSense cameras with depth enabled
        for cam_key, cam in self.cameras.items():
            if isinstance(cam, RealSenseCamera) and cam.use_depth:
                steps.append(
                    DepthEdgeOverlayProcessorStep(
                        camera_key=cam_key,
                        threshold_percentile=90,  # Edge sensitivity (85-95, higher = fewer edges)
                        blur_kernel=3,  # Noise reduction (1, 3, 5, 7)
                        dilation_kernel=2,  # Edge thickness (0-5)
                        alpha=0.7,  # Edge opacity (0.0-1.0)
                        min_depth=0.2,  # Min depth in meters (20cm)
                        max_depth=0.6,  # Max depth in meters (60cm)
                    )
                )

        return steps
