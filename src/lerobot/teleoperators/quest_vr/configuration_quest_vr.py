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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig

# Reusable descriptions shared by the unimanual and bimanual configs.
_PORT_DESC = (
    "HTTPS / WebSocket port the WebXR page is served on. The Quest 3 browser "
    "opens https://<your-LAN-IP>:<port>/ to enter the immersive session. A "
    "self-signed cert is auto-generated into the package directory on first "
    "run; the Quest will show a warning once per device, then remember it. "
    "Default 8443 is the standard HTTPS-alternate port."
)
_CLUTCH_DESC = (
    "Which Quest controller button toggles arm tracking (rising-edge engages, "
    "release disengages). Default 1 = grip / squeeze side button on Quest 3. "
    "While released, the arm freezes its commanded pose; the gripper is "
    "still operable so you can release objects without engaging the arm."
)
_GRIPPER_DESC = (
    "Which Quest controller button drives the gripper. Default 0 = analog "
    "index trigger. The trigger position is differentiated each frame and "
    "sent as gripper_vel (negative = close, positive = open); the downstream "
    "GripperVelocityToJoint step integrates it to a position target."
)
_POSITION_SCALE_DESC = (
    "Hand-motion to robot-EE motion ratio. 1.0 = 1:1 (push hand 5 cm -> EE "
    "moves 5 cm). Use values <1 for finer control of small tasks, >1 if you "
    "need to cover a workspace larger than your reach. End-effector axis "
    "scaling can also be applied downstream via EEReferenceAndDelta's "
    "end_effector_step_sizes."
)
_MAX_ROT_DESC = (
    "Caps the magnitude (in radians) of the rotation delta emitted per "
    "frame, measured from the engage snapshot. Bounds wild controller "
    "jumps from tracking dropouts or fast wrist flicks; does not limit "
    "smooth steady rotation since the cap acts on the delta-from-engage, "
    "not the per-frame increment. Default 0.5 rad ~= 28 deg; lower for "
    "stiffer feel, higher (up to pi) for full rotation freedom."
)


@TeleoperatorConfig.register_subclass("quest_vr")
@dataclass
class QuestVRTeleopConfig(TeleoperatorConfig):
    """Quest 3 WebXR teleop — drives one robot arm with one controller.

    The teleop spins up an HTTPS server in a daemon thread. The Quest's
    built-in browser opens the served page, enters an immersive XR session,
    and streams controller poses over a WebSocket. Each frame's pose is
    converted to an EE-delta action and cached; ``get_action()`` returns
    the latest cached value (matches the HighRateLeaderMixin pattern —
    fast lock-free reads, sample-rate fully decoupled from poll-rate).

    Output action keys (compatible with EEReferenceAndDelta upstream):
        enabled, target_x, target_y, target_z, target_wx, target_wy,
        target_wz, gripper_vel

    For bimanual robots use :class:`BimanualQuestVRTeleopConfig` instead.
    """

    port: int = field(default=8443, metadata={"description": _PORT_DESC})
    clutch_button_index: int = field(default=1, metadata={"description": _CLUTCH_DESC})
    gripper_button_index: int = field(default=0, metadata={"description": _GRIPPER_DESC})
    position_scale: float = field(default=1.0, metadata={"description": _POSITION_SCALE_DESC})
    max_rot_step_rad_per_tick: float = field(default=0.5, metadata={"description": _MAX_ROT_DESC})
    controller_hand: str = field(
        default="right",
        metadata={
            "description": (
                "Which physical Quest 3 controller to read (the Quest has two; "
                "this picks the active one for unimanual teleop). Does NOT "
                'select a robot arm — the output keys are unprefixed ("target_x", '
                'etc.). Set to "left" if you hold the active controller in your '
                "left hand. Bimanual setups should use bimanual_quest_vr instead, "
                "which reads both controllers and prefixes its action keys."
            ),
        },
    )


@TeleoperatorConfig.register_subclass("bimanual_quest_vr")
@dataclass
class BimanualQuestVRTeleopConfig(TeleoperatorConfig):
    """Bimanual Quest 3 WebXR teleop: both controllers drive both arms.

    Same single Quest WebXR session as the unimanual variant — Quest streams
    poses for both controllers in each frame, so one server callback dispatches
    to two per-arm controllers. Action keys are prefixed: ``left_*`` for the
    left controller, ``right_*`` for the right.

    Output action keys (compatible with the bimanual Cartesian IK pipeline):
        left_enabled, left_target_x/y/z, left_target_wx/wy/wz, left_gripper_vel,
        right_enabled, right_target_x/y/z, right_target_wx/wy/wz, right_gripper_vel

    Per-arm clutch and gripper buttons are independent; clutch one controller
    to drive that arm without affecting the other.
    """

    port: int = field(default=8443, metadata={"description": _PORT_DESC})
    clutch_button_index: int = field(
        default=1,
        metadata={
            "description": (
                _CLUTCH_DESC + " Applies to BOTH controllers (the Quest 3's left "
                "and right have the same button mapping)."
            ),
        },
    )
    gripper_button_index: int = field(
        default=0,
        metadata={
            "description": _GRIPPER_DESC + " Applies to BOTH controllers.",
        },
    )
    position_scale: float = field(default=1.0, metadata={"description": _POSITION_SCALE_DESC})
    max_rot_step_rad_per_tick: float = field(default=0.5, metadata={"description": _MAX_ROT_DESC})
