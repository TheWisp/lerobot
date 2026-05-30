# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Config for the click-to-goto bimanual EE teleop.

The teleop emits the bimanual Cartesian-VR action surface
(``left_target_x/y/z/wx/wy/wz, left_gripper_pos`` + right keys, plus
the absolute-world keys ``use_world_target, target_world_x/y/z,
world_target_top_down``) so the predictive follower's ``attach_teleop``
detects it as a Cartesian source and installs the IK adapter. The
absolute-world keys are populated by clicks read from the file-based
mailbox at ``mailbox_path``.

Calibration (file at ``extrinsics_path``) is built up via the GUI's
modal — each captured pair is one Kabsch input row; on finalize the
fit is saved here and goto is unlocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("click_target_bimanual_ee")
@dataclass
class ClickTargetBimanualEETeleopConfig(TeleoperatorConfig):
    top_camera_key: str = field(
        default="top",
        metadata={
            "description": (
                "Robot's camera key for the top-mounted depth camera. The "
                "teleop reads color + aligned depth from this camera at "
                "click time and uses its intrinsics to unproject."
            )
        },
    )
    arm: str = field(
        default="right",
        metadata={
            "description": "Which arm responds to clicks: 'left' or 'right'.",
            "enum": ["left", "right"],
        },
    )
    extrinsics_path: str = field(
        # Plain string default so a profile JSON storing ``null`` overrides
        # cleanly back to this canonical path on launch. The teleop applies
        # ``Path(...).expanduser()`` at use time.
        default="~/.config/lerobot/click_target_extrinsics.json",
        metadata={
            "description": (
                "JSON file storing the calibrated T_base_camera (4x4 SE(3)) "
                "and RMSE. Goto mode requires this file to exist; if missing "
                "the GUI must run the calibration modal first."
            )
        },
    )
    mailbox_path: str = field(
        # nosec B108: single-host IPC, the path is a fixed app-specific
        # filename in the shared tmpdir — no symlink attack surface
        # because both writer and reader rewrite the file atomically
        # via rename, and an attacker on the same host can already
        # affect the running teleop directly.
        default="/tmp/lerobot_click_target_mailbox.json",  # nosec B108
        metadata={
            "description": (
                "File-based IPC: GUI writes capture/goto/clear requests; "
                "teleop polls and writes responses. Single-host only."
            )
        },
    )
    gripper_value: float = field(
        default=10.0,
        metadata={
            "description": (
                "Constant gripper motor position emitted on every tick "
                "(motor-space degrees; open ~10, closed ~0). The click "
                "teleop does not toggle gripper from clicks — change here."
            )
        },
    )
    mailbox_poll_hz: float = field(
        default=20.0,
        metadata={
            "description": (
                "Background polling rate for the mailbox file. 20 Hz keeps "
                "click latency under ~50 ms without thrashing the FS."
            )
        },
    )
    world_target_top_down: bool = field(
        # Default False: the IK keeps the boot orientation; only the EE
        # position is driven by clicks. Most boot poses are at some
        # arbitrary orientation, and snapping to top-down (-Z world) on
        # tick 1 means a 60–90 deg wrist swing that trips the IK's
        # 20 deg/tick implausible-jump backstop — the arm then holds
        # forever. Calibration only needs the gripper TIP at known XYZ;
        # orientation does not enter Kabsch. Set True only if you've
        # pre-oriented the arm top-down and want clicks to preserve that.
        default=False,
        metadata={
            "description": (
                "If True, force gripper top-down (Z -> -Z world) on every "
                "goto. If False (default), keep the latched boot orientation."
            )
        },
    )
