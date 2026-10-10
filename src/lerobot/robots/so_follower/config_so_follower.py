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
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    #: Feetech servos over a serial bus; none of it is a core dependency.
    required_extras = ("feetech", "pyserial-dep", "deepdiff-dep")

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
    use_degrees: bool = True

    # Onboard servo loop gains written at connect. The defaults are what this
    # follower has always written: P halved from Feetech's factory value to keep
    # leader teleop from chattering, no integral term. A loaded joint at these
    # gains settles short of its goal; raise P (and use `gravity_ff_alpha`) for
    # scripted position control that has to hold a height across the workspace.
    p_coefficient: int = 16
    i_coefficient: int = 0
    d_coefficient: int = 32

    # Gravity feed-forward: shift every goal by the droop the load will cause
    # (`alpha` = servo compliance in degrees per N*m, measured at the gains
    # above; it scales with 1/P). 0 disables it. Needs a robot description with
    # masses and the `pin` extra; only the SO-107 provides one today.
    gravity_ff_alpha: float = 0.0
    # Which arm's motor->URDF alignment the feed-forward uses ("left" | "right").
    gravity_ff_arm: str = "left"


@RobotConfig.register_subclass("so107_follower")
@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass


SO100FollowerConfig = SOFollowerRobotConfig
SO101FollowerConfig = SOFollowerRobotConfig
SO107FollowerConfig = SOFollowerRobotConfig
