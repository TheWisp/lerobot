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

"""Configuration for the Quest VR teleoperator."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("quest_vr")
@dataclass
class QuestVRTeleopConfig(TeleoperatorConfig):
    """Quest 3 WebXR teleop.

    The teleop spins up an HTTPS server in a daemon thread. The Quest's
    built-in browser opens the served page, enters an immersive XR session,
    and streams controller poses over a WebSocket. Each frame's pose is
    converted to an EE-delta action and cached; ``get_action()`` returns
    the latest cached value (matches the HighRateLeaderMixin pattern —
    fast lock-free reads, sample-rate fully decoupled from poll-rate).

    Output action keys (compatible with EEReferenceAndDelta upstream):
        enabled, target_x, target_y, target_z, target_wx, target_wy,
        target_wz, gripper_vel

    Attributes:
        port: HTTPS / WebSocket port. Self-signed cert is auto-generated
            into the package's cache directory on first run.
        clutch_button_index: Index of the controller button used as clutch
            (default: 1 = grip/squeeze side button on Quest 3 right hand).
        gripper_button_index: Index of the controller button used as gripper
            (default: 0 = analog index trigger).
        position_scale: How user's hand motion in m maps to robot EE motion.
            ``1.0`` = 1:1 (push hand 5cm -> EE moves 5cm). Pair with
            EEReferenceAndDelta's ``end_effector_step_sizes`` to scale further.
        max_rot_step_rad_per_tick: Cap rotation update per emit (radians).
            Bounds wild controller jumps; only affects within-engagement smoothness.
        controller_hand: Which controller to use as the IK source ("right" or "left").
    """

    port: int = 8443
    clutch_button_index: int = 1
    gripper_button_index: int = 0
    position_scale: float = 1.0
    max_rot_step_rad_per_tick: float = 0.5  # 0.5 rad ~= 28 deg
    controller_hand: str = "right"
