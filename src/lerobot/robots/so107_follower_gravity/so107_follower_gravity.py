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

import dataclasses

from lerobot.motors.feetech import OperatingMode
from lerobot.types import ActionChunk, RobotAction, action_first_frame
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.decorators import check_if_not_connected

from ..so_follower.so_follower import SO107Follower
from .config_so107_follower_gravity import ROBOT_TYPE, SO107FollowerGravityRobotConfig


class SO107FollowerGravity(SO107Follower):
    """The SO-107 follower with its own servo gains, and every goal moved by the droop gravity will put on it.

    A loaded joint of a position servo settles short of its goal by its compliance times the gravity torque; the
    goal sent is ``q + alpha * tau_g(q)`` for the pose ``q`` wanted (the "virtual displacement" gravity schemes on
    these servos use), applied to the wanted pose so it holds a trajectory rather than floating. The offset is
    computed in motor degrees and converted to action units through the calibration, then the follower's
    ``max_relative_target`` clamp bounds what reaches the bus.
    """

    config_class = SO107FollowerGravityRobotConfig
    name = ROBOT_TYPE

    def __init__(self, config: SO107FollowerGravityRobotConfig):
        if (
            config.calibration_dir is None
        ):  # the same servos as the plain follower: one calibration, not a copy
            config = dataclasses.replace(
                config, calibration_dir=HF_LEROBOT_CALIBRATION / ROBOTS / SO107Follower.name
            )
        super().__init__(config)
        from ..so107_description.gravity import GravityFeedForward
        from ..so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, RIGHT_ARM_ALIGNMENT

        alignment = LEFT_ARM_ALIGNMENT if config.arm == "left" else RIGHT_ARM_ALIGNMENT
        self.gravity = GravityFeedForward(config.gravity_ff_alpha, alignment)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, int(self.config.p_coefficient))
                self.bus.write("I_Coefficient", motor, int(self.config.i_coefficient))
                self.bus.write("D_Coefficient", motor, int(self.config.d_coefficient))
                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def units_per_motor_degree(self, motor: str) -> float:
        """Action units per motor degree: 1 in degrees mode, calibration-range based otherwise."""
        if self.config.use_degrees:
            return 1.0
        cal = self.calibration[motor]
        span_deg = (cal.range_max - cal.range_min) * 360.0 / 4095.0
        assert span_deg > 0, f"{motor}: empty calibration range"
        return 200.0 / span_deg

    @check_if_not_connected
    def send_action(self, action: RobotAction | ActionChunk) -> RobotAction:
        """As the follower's, with every goal moved by the gravity offset at that goal. An action that does not name
        every motor (a gripper-only command) has no pose to compensate at and goes through as it is."""
        action = action_first_frame(action)
        goal = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        if goal and all(m in goal for m in self.bus.motors):
            q_deg = {m: goal[m] / self.units_per_motor_degree(m) for m in goal}
            offsets = self.gravity.offset_deg(q_deg)
            action = {
                **action,
                **{f"{m}.pos": goal[m] + offsets.get(m, 0.0) * self.units_per_motor_degree(m) for m in goal},
            }
        return super().send_action(action)
