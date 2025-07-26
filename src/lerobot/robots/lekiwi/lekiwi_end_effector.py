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
from typing import Any
import numpy as np

from dataclasses import dataclass, field
from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from . import LeKiwiConfig
from .lekiwi import LeKiwi
from ..config import RobotConfig

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("lekiwi_end_effector")
@dataclass
class LeKiwiEndEffectorConfig(LeKiwiConfig):
    # Path to URDF file for kinematics
    urdf_path: str | None = None

    # End-effector frame name in the URDF
    # https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/URDF/LeKiwi.urdf
    target_frame_name: str = "STS3215_03a-v1-4"

    # Duration of the application
    connection_time_s: int = 300

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    # Same as so100/101
    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )


class LeKiwiEndEffector(LeKiwi):
    """
    Lekiwi robot with end-effector space control.

    This robot inherits from Lekiwi but transforms actions from
    end-effector space to joint space before sending them to the motors.
    """

    config_class = LeKiwiEndEffectorConfig
    name = "lekiwi_end_effector"

    def __init__(self, config: LeKiwiEndEffectorConfig):
        super().__init__(config)

        # TODO self.bus

        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.config = config

        # Initialize the kinematics module for the lekiwi robot
        if self.config.urdf_path is None:
            raise ValueError(
                "urdf_path must be provided in the configuration for end-effector control. "
                "Please set urdf_path in your LeKiwiEndEffectorConfig."
            )
        
        # The first 6 motors are the arm motors
        self.arm_bus_motors = dict(list(self.bus.motors.items())[:6])

        self.kinematics = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
            # This is to ensure IK doesn't use the wheels. We should refactor the code later for simplicity.
            joint_names=[
                'STS3215_03a-v1_Revolute-45', 
                'STS3215_03a-v1-1_Revolute-49', 
                'STS3215_03a-v1-2_Revolute-51', 
                'STS3215_03a-v1-3_Revolute-53', 
                'STS3215_03a_Wrist_Roll-v1_Revolute-55', 
                'STS3215_03a-v1-4_Revolute-57', 
            ]
        )


        # Store the bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        self.current_ee_pos = None
        self.current_joint_pos = None
        print (f"Using URDF: {self.config.urdf_path}")

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "arm_gripper": 3},
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        try:
            if not self.is_connected:
                raise DeviceNotConnectedError(f"{self} is not connected.")

            # Convert action to numpy array if not already
            if isinstance(action, dict):
                if all(k in action for k in ["delta_x", "delta_y", "delta_z"]):
                    delta_ee = np.array(
                        [
                            action["delta_x"] * self.config.end_effector_step_sizes["x"],
                            action["delta_y"] * self.config.end_effector_step_sizes["y"],
                            action["delta_z"] * self.config.end_effector_step_sizes["z"],
                        ],
                        dtype=np.float32,
                    )
                    if "arm_gripper" not in action:
                        action["arm_gripper"] = [1.0]
                    action = np.append(delta_ee, action["arm_gripper"])
                else:
                    logger.warning(
                        f"Expected action keys 'delta_x', 'delta_y', 'delta_z', got {list(action.keys())}"
                    )
                    action = np.zeros(4, dtype=np.float32)

            if self.current_joint_pos is None:
                # Read current joint positions
                current_joint_pos = self.bus.sync_read("Present_Position")
                self.current_joint_pos = np.array(
                    [current_joint_pos[name] for name in self.arm_bus_motors]
                )

            # Calculate current end-effector position using forward kinematics
            if self.current_ee_pos is None:
                self.current_ee_pos = self.kinematics.forward_kinematics(
                    self.current_joint_pos
                )

            # Set desired end-effector position by adding delta
            desired_ee_pos = np.eye(4)
            desired_ee_pos[:3, :3] = self.current_ee_pos[:3, :3]  # Keep orientation

            # Add delta to position and clip to bounds
            desired_ee_pos[:3, 3] = self.current_ee_pos[:3, 3] + action[:3]
            if self.end_effector_bounds is not None:
                desired_ee_pos[:3, 3] = np.clip(
                    desired_ee_pos[:3, 3],
                    self.end_effector_bounds["min"],
                    self.end_effector_bounds["max"],
                )

            # Compute inverse kinematics to get joint positions
            target_joint_values_in_degrees = self.kinematics.inverse_kinematics(
                self.current_joint_pos, desired_ee_pos
            )

            # Create joint space action dictionary
            joint_action = {
                f"{key}.pos": target_joint_values_in_degrees[i]
                for i, key in enumerate(self.arm_bus_motors.keys())
            }

            # Handle gripper separately if included in action
            # Gripper delta action is in the range 0 - 2,
            # We need to shift the action to the range -1, 1 so that we can expand it to -Max_gripper_pos, Max_gripper_pos
            joint_action["arm_gripper.pos"] = np.clip(
                self.current_joint_pos[-1] + (action[-1] - 1) * self.config.max_gripper_pos,
                5,
                self.config.max_gripper_pos,
            )

            self.current_ee_pos = desired_ee_pos.copy()
            self.current_joint_pos = target_joint_values_in_degrees.copy()
            self.current_joint_pos[-1] = joint_action["arm_gripper.pos"]

            # Copy over wheel movements
            joint_action["x.vel"] = action["x.vel"] if "x.vel" in action else 0.0
            joint_action["y.vel"] = action["y.vel"] if "y.vel" in action else 0.0
            joint_action["theta.vel"] = action["theta.vel"] if "theta.vel" in action else 0.0

            # Log before sending
            logger.warning(f"Sending joint action: {joint_action}")

            # Send joint space action to parent class
            return super().send_action(joint_action)
        except Exception:
            import traceback
            # print the full original traceback (file, line, call stack, message)
            traceback.print_exc()
            # re‑raise the exact same exception so the caller still sees it
            raise
