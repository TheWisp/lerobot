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

from dataclasses import dataclass, field

from ..config import RobotConfig
from ..so_follower.config_so_follower import SOFollowerConfig

ROBOT_TYPE = "so107_follower_gravity"


@RobotConfig.register_subclass(ROBOT_TYPE)
@dataclass
class SO107FollowerGravityRobotConfig(RobotConfig, SOFollowerConfig):
    """The SO-107 follower held on its commanded pose against gravity, for control that commands joints from the
    calibrated kinematics without watching the fingertip. The Show-and-Servo act engine accepts only this type.

    Its own robot type so that nothing records, replays or trains on it by mistake: the plain ``so107_follower``
    writes LeRobot's P 16 with no feed-forward, and the two settle differently under load. Holding a 60 mm circle at
    constant height, P 16 fell 33 mm on average and stalled its lift; P 32 with this feed-forward held -0.1 mm mean,
    3.9 mm max (``so107_description/docs/gravity_sag.md``).
    """

    # Which arm this is: the feed-forward and the kinematics take that arm's motor->URDF alignment.
    arm: str = "left"
    # Servo loop gains written at connect: Feetech's factory P, no integral term (it winds up, gravity_sag.md).
    p_coefficient: int = 32
    i_coefficient: int = 0
    d_coefficient: int = 32
    # Every goal moves by alpha * tau_g(goal): the servos' compliance in degrees per N*m, measured at P 32 (3.1-3.4
    # across the loaded joints, gravity_sag.md). It scales with 1/P.
    gravity_ff_alpha: float = 3.2
    # Working bounds measured on this assembly, motor degrees: joint -> {"min": ..., "max": ...}, either side
    # optional. They only narrow the calibrated range (a cable that stops a joint early); the jog holds goals in them.
    joint_limits_deg: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {self.arm!r}")
        if self.gravity_ff_alpha < 0:
            raise ValueError("gravity_ff_alpha is a compliance, never negative")
