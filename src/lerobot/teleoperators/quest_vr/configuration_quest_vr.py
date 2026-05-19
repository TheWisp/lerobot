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

import math
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
    "index trigger. The trigger value (0..1) is mapped absolutely to motor "
    "position: trigger=0 -> gripper_open_motor, trigger=1 -> "
    "gripper_closed_motor. No integration / velocity tracking, so the "
    "trigger feels like a continuous open/close knob with no latching."
)
_GRIPPER_OPEN_DESC = (
    "Motor-space position commanded when the trigger is fully released "
    "(grip=0). Calibrate by physically opening the gripper to where you "
    "want trigger-released to land and reading the motor position."
)
_GRIPPER_CLOSED_DESC = (
    "Motor-space position commanded when the trigger is fully pulled "
    "(grip=1). Motor direction differs across calibrations — closing the "
    "gripper might mean increasing OR decreasing motor.pos depending on "
    "how zero was set during calibration. So this value can be either "
    "larger or smaller than gripper_open_motor; pick whatever matches "
    "the physically-closed pose. The bimanual SO-107 default has "
    "closed=90 on the left arm (opens-as-decreasing) and closed=10 on "
    "the right (opens-as-increasing)."
)
_POSITION_SCALE_DESC = (
    "Hand-motion to robot-EE motion ratio. 1.0 = 1:1 (push hand 5 cm -> EE "
    "moves 5 cm). Use values <1 for finer control of small tasks, >1 if you "
    "need to cover a workspace larger than your reach. End-effector axis "
    "scaling can also be applied downstream via EEReferenceAndDelta's "
    "end_effector_step_sizes."
)
_MAX_ROT_DESC = (
    "Caps the magnitude (in radians) of the rotation delta from the engage "
    "snapshot. Despite the name this is an absolute cap on |delta_rot|, not "
    "a per-tick rate limit. Default 3.14 rad (pi) = effectively uncapped — "
    "the wrist tracks the controller through any reachable angle. Lower it "
    "(e.g. 0.5 rad = 28 deg) only if you want a stiffer 'small rotations "
    "only' feel; the prior 0.5 default felt limited because you'd hit the "
    "cap on any meaningful wrist motion."
)
_MAX_POS_VEL_DESC = (
    "Per-WebXR-frame cap on the Quest controller's POSITION delta (meters). "
    "Suppresses tracking glitches: a 380mm jump in 33ms is 11 m/s, clearly "
    "not real hand motion. The Quest streams at ~90 Hz while the teleop loop "
    "polls at 30 Hz, so multiple WebXR frames overwrite the cached action "
    "between loop ticks — this cap applies to each WebXR frame individually. "
    "Default 0.04 m/frame allows ~3.6 m/s at 90 Hz (faster than typical "
    "teleop, slower than tracking-teleport glitches). Set 0.0 to disable."
)


@TeleoperatorConfig.register_subclass("quest_vr")
@dataclass
class QuestVRTeleopConfig(TeleoperatorConfig):
    """Quest 3 WebXR teleop — drives one robot arm with one controller.

    The teleop spins up an HTTPS server in a daemon thread. The Quest's
    built-in browser opens the served page, enters an immersive XR session,
    and streams controller poses over a WebSocket. Each frame's pose is
    converted to an EE-delta action and cached under a lock; ``get_action()``
    takes the same lock briefly to copy it out. Sample-rate (Quest WebXR,
    ~90 Hz) is fully decoupled from poll-rate (loop driver, 30 Hz).

    Output action keys (compatible with EEReferenceAndDelta upstream):
        enabled, target_x, target_y, target_z, target_wx, target_wy,
        target_wz, gripper_pos

    For bimanual robots use :class:`BimanualQuestVRTeleopConfig` instead.
    """

    port: int = field(default=8443, metadata={"description": _PORT_DESC})
    clutch_button_index: int = field(default=1, metadata={"description": _CLUTCH_DESC})
    gripper_button_index: int = field(default=0, metadata={"description": _GRIPPER_DESC})
    position_scale: float = field(default=1.0, metadata={"description": _POSITION_SCALE_DESC})
    max_rot_step_rad_per_tick: float = field(default=math.pi, metadata={"description": _MAX_ROT_DESC})
    max_pos_step_m_per_tick: float = field(default=0.04, metadata={"description": _MAX_POS_VEL_DESC})
    # Default rest position is half-open (motor 50 in RANGE_0_100 norm
    # mode) so a freshly-connected teleop doesn't slam the gripper to a
    # mechanical end. Trigger-pulled lands at 90 (mostly closed without
    # pushing the motor against its stop). Override per robot.
    gripper_open_motor: float = field(default=50.0, metadata={"description": _GRIPPER_OPEN_DESC})
    gripper_closed_motor: float = field(default=90.0, metadata={"description": _GRIPPER_CLOSED_DESC})
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
        left_enabled, left_target_x/y/z, left_target_wx/wy/wz, left_gripper_pos,
        right_enabled, right_target_x/y/z, right_target_wx/wy/wz, right_gripper_pos

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
    max_rot_step_rad_per_tick: float = field(default=math.pi, metadata={"description": _MAX_ROT_DESC})
    max_pos_step_m_per_tick: float = field(default=0.04, metadata={"description": _MAX_POS_VEL_DESC})
    # Per-arm gripper motor mapping. The bimanual SO-107 has OPPOSITE
    # motor-direction conventions between left and right (verified
    # empirically: with both at open=0/closed=80 the left arm felt
    # "rest open, trigger closes" while the right felt "rest closed,
    # trigger opens"). The defaults below give "rest half-open, trigger
    # closes" on both arms — adjust if your calibration differs.
    #
    # LEFT  arm motor: 0 = open, 100 = closed. Rest at 50, close toward 90.
    # RIGHT arm motor: 0 = closed, 100 = open. Rest at 50, close toward 10.
    left_gripper_open_motor: float = field(default=50.0, metadata={"description": _GRIPPER_OPEN_DESC})
    left_gripper_closed_motor: float = field(default=90.0, metadata={"description": _GRIPPER_CLOSED_DESC})
    right_gripper_open_motor: float = field(default=50.0, metadata={"description": _GRIPPER_OPEN_DESC})
    right_gripper_closed_motor: float = field(default=10.0, metadata={"description": _GRIPPER_CLOSED_DESC})
