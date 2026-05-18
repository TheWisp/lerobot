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

"""Per-arm Quest controller state + frame-to-action conversion.

A single instance of :class:`QuestArmController` owns the clutch-engage
snapshot and last gripper value for one controller (one robot arm). The
single-arm :class:`QuestVRTeleop` holds one of these; the bimanual variant
holds two (one per hand) and merges their action dicts under ``left_``/
``right_`` prefixes.

Keeping this state in its own class means the two teleop classes share
the conversion logic — only the wiring (one server callback dispatching
to one or two controllers) differs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .server import quest_delta_to_robot, quest_rot_to_robot


class QuestArmController:
    """State machine for one Quest controller -> one robot arm's EE deltas.

    Preconditions for :meth:`process_pose`:
        * The caller has already filtered the controller pose by hand
          (``"left"`` or ``"right"``).
        * ``pose["pos"]`` is a 3-vector in Quest stage space.
        * ``pose["rot"]`` is a length-4 quaternion ``[x, y, z, w]``.

    Postconditions:
        * Returns an action dict with the eight EE-delta keys
          (``enabled, target_x/y/z, target_wx/wy/wz, gripper_vel``), each
          optionally prefixed by ``key_prefix``.
        * On rising-edge of the clutch button, the engage pose is latched
          internally; subsequent calls produce deltas relative to that pose.
        * While the clutch is released, ``enabled=0`` and target_* are zero;
          ``gripper_vel`` is still derived from the trigger delta so the
          gripper can be operated while not driving the arm.
    """

    def __init__(
        self,
        clutch_button_index: int,
        gripper_button_index: int,
        position_scale: float,
        max_rot_step_rad_per_tick: float,
        key_prefix: str = "",
    ) -> None:
        self.clutch_button_index = int(clutch_button_index)
        self.gripper_button_index = int(gripper_button_index)
        self.position_scale = float(position_scale)
        self.max_rot_step_rad_per_tick = float(max_rot_step_rad_per_tick)
        self.key_prefix = key_prefix
        self._engaged: bool = False
        self._quest_pos_at_engage: np.ndarray | None = None
        self._quest_rot_at_engage = None  # scipy Rotation
        self._gripper_last_value: float = 0.0

    def idle_action(self) -> dict[str, float]:
        """Action dict for a disengaged frame (also the pre-connect default)."""
        p = self.key_prefix
        return {
            f"{p}enabled": 0.0,
            f"{p}target_x": 0.0,
            f"{p}target_y": 0.0,
            f"{p}target_z": 0.0,
            f"{p}target_wx": 0.0,
            f"{p}target_wy": 0.0,
            f"{p}target_wz": 0.0,
            f"{p}gripper_vel": 0.0,
        }

    def reset(self) -> None:
        """Forget the engage snapshot and last gripper value."""
        self._engaged = False
        self._quest_pos_at_engage = None
        self._quest_rot_at_engage = None
        self._gripper_last_value = 0.0

    def process_pose(self, pose: dict[str, Any]) -> dict[str, float]:
        """Convert one controller's pose into a (possibly-prefixed) action dict.

        Returns idle action when the clutch is released. Always returns a
        full action dict (never None) so the caller can merge it into a
        bimanual action without guarding on Nones.
        """
        quest_pos = np.asarray(pose["pos"], dtype=float)
        quest_quat = pose.get("rot", [0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        buttons = pose.get("buttons") or []
        clutch = float(buttons[self.clutch_button_index]) if len(buttons) > self.clutch_button_index else 0.0
        grip = float(buttons[self.gripper_button_index]) if len(buttons) > self.gripper_button_index else 0.0
        engaged = clutch > 0.5

        # Gripper as a velocity command derived from trigger delta.
        # CONVENTION: trigger pulled (grip increasing) = CLOSE = gripper_vel
        # POSITIVE = downstream GripperVelocityToJoint INCREASES motor.pos.
        # This matches the user's bi_so107 calibration where motor.pos LOW
        # = open and motor.pos HIGH = closed. If a future SO-107 arm is
        # calibrated with the opposite gripper direction, expose a sign
        # flip here (or per-arm via the joint_map's gripper.sign).
        gripper_vel = grip - self._gripper_last_value
        self._gripper_last_value = grip

        if engaged and not self._engaged:
            self._quest_pos_at_engage = quest_pos.copy()
            self._quest_rot_at_engage = quest_rot_to_robot(quest_quat)
        self._engaged = engaged

        p = self.key_prefix
        if not engaged:
            action = self.idle_action()
            action[f"{p}gripper_vel"] = float(gripper_vel)
            return action

        # Position delta in robot frame.
        assert self._quest_pos_at_engage is not None
        assert self._quest_rot_at_engage is not None
        dquest = quest_pos - self._quest_pos_at_engage
        drobot = quest_delta_to_robot(dquest) * self.position_scale

        # Rotation delta in robot frame (as rotvec).
        quest_rot_now = quest_rot_to_robot(quest_quat)
        delta_rot = quest_rot_now * self._quest_rot_at_engage.inv()
        rotvec = delta_rot.as_rotvec()
        mag = float(np.linalg.norm(rotvec))
        cap = self.max_rot_step_rad_per_tick
        if mag > cap > 0.0:
            rotvec = rotvec * (cap / mag)

        return {
            f"{p}enabled": 1.0,
            f"{p}target_x": float(drobot[0]),
            f"{p}target_y": float(drobot[1]),
            f"{p}target_z": float(drobot[2]),
            f"{p}target_wx": float(rotvec[0]),
            f"{p}target_wy": float(rotvec[1]),
            f"{p}target_wz": float(rotvec[2]),
            f"{p}gripper_vel": float(gripper_vel),
        }
