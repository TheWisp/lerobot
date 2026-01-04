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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOFollowerConfigBase(RobotConfig):
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # Motor sensitivity parameters to improve responsiveness to small movements
    # P_Coefficient: Proportional gain for position control (default 16, motor default is 32)
    # Higher values = more responsive but potentially shakier
    p_coefficient: int = 16

    # Dead zones: Range around target position where motor doesn't move
    # Set to 0 to allow all movements, even very small ones
    cw_dead_zone: int = 0
    ccw_dead_zone: int = 0

    # Minimum force threshold - motor won't move if computed torque is below this
    # Set to 0 to allow movement even with very small position errors
    minimum_startup_force: int = 0
