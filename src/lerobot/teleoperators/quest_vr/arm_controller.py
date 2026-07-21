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

import logging
import time
from typing import Any

import numpy as np

from .server import QUEST_TO_ROBOT_M, quest_delta_to_robot, quest_rot_to_robot

logger = logging.getLogger(__name__)

# After a tracking-dropout recovery, keep the clutch released this long even
# if the grip is physically held: rediscovered controllers often report OK
# while their pose is still settling, and the settle-snap would otherwise
# reach the arm. Ported from the validated dora OpenArm VR stack
# (_SETTLE_SECS in quest_receiver.py).
_SETTLE_SECS = 0.25

# Target jump guard (diagnostic): warn when the emitted EE target steps more
# than this in one WebXR tick. The per-frame glitch clamps below bound the
# step, but a step near the cap is still suspicious — legit hand motion at
# 90 Hz rarely exceeds these. Ports the dora stack's target-step warning;
# rate-limited per controller so a fast-but-legit stretch doesn't spam.
_TARGET_STEP_WARN_POS_M = 0.02
_TARGET_STEP_WARN_ROT_RAD = 0.1
_TARGET_STEP_WARN_MIN_INTERVAL_S = 1.0


class QuestArmController:
    """State machine for one Quest controller -> one robot arm's EE deltas.

    Preconditions for :meth:`process_pose`:
        * The caller has already filtered the controller pose by hand
          (``"left"`` or ``"right"``).
        * ``pose["pos"]`` is a 3-vector in Quest stage space.
        * ``pose["rot"]`` is a length-4 quaternion ``[x, y, z, w]``.

    Postconditions:
        * Returns an action dict with the eight EE-delta keys
          (``enabled, target_x/y/z, target_wx/wy/wz, gripper_pos``), each
          optionally prefixed by ``key_prefix``.
        * On rising-edge of the clutch button, the engage pose is latched
          internally; subsequent calls produce deltas relative to that pose.
        * While the clutch is released, ``enabled=0`` and target_* are zero;
          ``gripper_pos`` is still driven by the trigger so the gripper can
          be operated while the arm itself isn't tracking.
    """

    def __init__(
        self,
        clutch_button_index: int,
        gripper_button_index: int,
        position_scale: float,
        max_rot_step_rad_per_tick: float,
        gripper_open_motor: float,
        gripper_closed_motor: float,
        max_pos_step_m_per_tick: float = 0.10,
        key_prefix: str = "",
        reset_button_index: int = 4,
        quest_to_robot_m: np.ndarray | None = None,
        settle_secs: float = _SETTLE_SECS,
    ) -> None:
        self.clutch_button_index = int(clutch_button_index)
        self.gripper_button_index = int(gripper_button_index)
        # Rising edge sets the ``reset`` action key, which the IK controller
        # interprets as "snap back to the attach-time seed pose". Default 4
        # = the upper face button (A on right Quest controller / X on left
        # — the WebXR gamepad mapping puts both at index 4).
        self.reset_button_index = int(reset_button_index)
        self.position_scale = float(position_scale)
        self.max_rot_step_rad_per_tick = float(max_rot_step_rad_per_tick)
        self.max_pos_step_m_per_tick = float(max_pos_step_m_per_tick)
        self.gripper_open_motor = float(gripper_open_motor)
        self.gripper_closed_motor = float(gripper_closed_motor)
        self.key_prefix = key_prefix
        # Quest stage -> robot base rotation. None = the server module's
        # SO-107 default; QuestVRTeleop passes one built from the config's
        # robot_forward_in_urdf / robot_up_in_urdf.
        self._q2r = (
            QUEST_TO_ROBOT_M if quest_to_robot_m is None else np.asarray(quest_to_robot_m, dtype=float)
        )
        assert self._q2r.shape == (3, 3), "quest_to_robot_m must be 3x3"
        # Post-recovery settle window (see _SETTLE_SECS).
        self.settle_secs = float(settle_secs)
        self._recover_t: float | None = None
        self._engaged: bool = False
        self._quest_pos_at_engage: np.ndarray | None = None
        self._quest_rot_at_engage = None  # scipy Rotation
        # Previous frame's raw Quest position, in Quest stage space. Used
        # to detect tracking glitches (huge teleport-style jumps between
        # consecutive samples). Reset to None on disconnect / clutch loss.
        self._quest_pos_prev: np.ndarray | None = None
        # Previous ENGAGED emission's (target_xyz, target_wxyz rotvec), for
        # the jump-guard diagnostic log. Cleared on disengage so a fresh
        # engage (new anchor) never compares across anchors.
        self._prev_emitted: tuple[np.ndarray, np.ndarray] | None = None
        self._last_step_warn_t: float = 0.0
        # Latched gripper command while disengaged. Init to the open value
        # so a freshly-connected teleop with no clutch press doesn't move
        # the gripper at all.
        self._last_gripper_pos: float = float(gripper_open_motor)
        # Human-readable arm label for log lines.
        self._label: str = key_prefix.rstrip("_") or "controller"
        # Whether this controller streamed a pose on the most recent frame.
        # Starts True so a controller that is asleep at session start emits
        # a "not tracked" log line on the first frame rather than being
        # silently dead.
        self._tracked: bool = True

    def _gripper_pos_from_trigger(self, grip: float) -> float:
        """Absolute trigger->motor mapping. Same shape as the experimental
        sim_receiver: trigger=0 -> open value, trigger=1 -> closed value."""
        return self.gripper_open_motor + grip * (self.gripper_closed_motor - self.gripper_open_motor)

    def idle_action(self) -> dict[str, float]:
        """Action dict for a disengaged frame (also the pre-connect default).

        gripper_pos defaults to fully OPEN so a fresh teleop session doesn't
        spuriously squeeze the gripper before the user touches the trigger.
        """
        p = self.key_prefix
        return {
            f"{p}enabled": 0.0,
            f"{p}reset": 0.0,
            f"{p}target_x": 0.0,
            f"{p}target_y": 0.0,
            f"{p}target_z": 0.0,
            f"{p}target_wx": 0.0,
            f"{p}target_wy": 0.0,
            f"{p}target_wz": 0.0,
            f"{p}gripper_pos": float(self.gripper_open_motor),
        }

    def reset(self) -> None:
        """Forget the engage snapshot."""
        self._engaged = False
        self._quest_pos_at_engage = None
        self._quest_rot_at_engage = None
        self._quest_pos_prev = None
        self._prev_emitted = None
        self._recover_t = None
        self._last_gripper_pos = self.gripper_open_motor
        self._tracked = True

    def on_tracking_lost(self) -> dict[str, float]:
        """Handle a frame that carried no pose for this controller.

        A Quest controller drops out of tracking (asleep, occluded, past
        the FOV edge) and can be *moved* while untracked. On the
        tracked->untracked edge, drop the engage state: the next tracked
        frame then re-anchors the engage snapshot from scratch, so motion
        accumulated during the dropout can never be counted as a teleop
        delta (which would slew the arm to a wrong pose).

        Returns a disengaged action; the gripper holds its last command.
        """
        if self._tracked:
            logger.warning("Quest %s controller not tracked — check it is awake and in view", self._label)
            self._tracked = False
            self._engaged = False
            self._quest_pos_at_engage = None
            self._quest_rot_at_engage = None
            self._quest_pos_prev = None
        p = self.key_prefix
        return {
            f"{p}enabled": 0.0,
            f"{p}reset": 0.0,
            f"{p}target_x": 0.0,
            f"{p}target_y": 0.0,
            f"{p}target_z": 0.0,
            f"{p}target_wx": 0.0,
            f"{p}target_wy": 0.0,
            f"{p}target_wz": 0.0,
            f"{p}gripper_pos": float(self._last_gripper_pos),
        }

    def process_pose(self, pose: dict[str, Any]) -> dict[str, float]:
        """Convert one controller's pose into a (possibly-prefixed) action dict.

        Returns idle action when the clutch is released. Always returns a
        full action dict (never None) so the caller can merge it into a
        bimanual action without guarding on Nones.
        """
        if not self._tracked:
            # Re-acquired after a dropout. on_tracking_lost() already cleared
            # the engage state, so a still-held clutch re-anchors below as a
            # fresh rising edge — no stale-snapshot delta. Arm the settle
            # window: for ``settle_secs`` after recovery the clutch is
            # treated as released even while physically held, so the
            # rediscovered controller's settle-snap never reaches the arm.
            logger.info(
                "Quest %s controller tracking re-acquired — settling %.2fs",
                self._label,
                self.settle_secs,
            )
            self._tracked = True
            self._recover_t = time.monotonic()

        now = time.monotonic()

        quest_pos = np.asarray(pose["pos"], dtype=float)
        quest_quat = pose.get("rot", [0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        buttons = pose.get("buttons") or []
        clutch = float(buttons[self.clutch_button_index]) if len(buttons) > self.clutch_button_index else 0.0
        grip = float(buttons[self.gripper_button_index]) if len(buttons) > self.gripper_button_index else 0.0
        reset = (
            1.0
            if (len(buttons) > self.reset_button_index and buttons[self.reset_button_index] > 0.5)
            else 0.0
        )
        engaged = clutch > 0.5

        # Post-recovery settle window: a controller that just came back from
        # a tracking dropout reports OK while its pose is still settling.
        # Treat the clutch as released until the window expires — the pose
        # baselines below keep following the (settling) pose, so when the
        # window lapses the next engaged frame re-anchors seamlessly and
        # none of the settle-snap becomes a teleop delta.
        if engaged and self._recover_t is not None:
            if now - self._recover_t < self.settle_secs:
                engaged = False
            else:
                self._recover_t = None

        # Tracking-glitch suppression: if the raw Quest position jumped
        # more than max_pos_step_m_per_tick between consecutive frames,
        # it's a controller-tracking dropout (occlusion, FOV edge, lost
        # then re-acquired). Clamp the step toward the previous pose so
        # the bogus delta doesn't propagate into target_xyz. A real hand
        # moves at most ~3 m/s; at 90 Hz WebXR rate the default cap
        # (0.04 m / frame, see configuration_quest_vr) allows ~3.6 m/s.
        cap_m = self.max_pos_step_m_per_tick
        if self._quest_pos_prev is not None and cap_m > 0.0:
            step = quest_pos - self._quest_pos_prev
            step_mag = float(np.linalg.norm(step))
            if step_mag > cap_m:
                quest_pos = self._quest_pos_prev + step * (cap_m / step_mag)
        self._quest_pos_prev = quest_pos.copy()

        # Absolute trigger->motor mapping. Gated on clutch: a controller
        # resting on the table with a stray trigger touch should NOT slam
        # the gripper. While disengaged, the gripper holds its last
        # commanded value (or the open default before the user has touched
        # the trigger — ``__init__`` seeds ``_last_gripper_pos`` to that).
        if engaged:
            gripper_pos = self._gripper_pos_from_trigger(grip)
            self._last_gripper_pos = gripper_pos
        else:
            gripper_pos = self._last_gripper_pos

        if engaged and not self._engaged:
            self._quest_pos_at_engage = quest_pos.copy()
            self._quest_rot_at_engage = quest_rot_to_robot(quest_quat, self._q2r)
            # Reset the glitch-cap baseline to the current pose so the next
            # WebXR frame's step is measured from "where we just engaged",
            # not from a possibly-stale value left over from before engage.
            # Without this, a long disengaged period followed by re-engage
            # could let target_xyz accumulate from a stale baseline as
            # later frames within the same loop tick get incorporated.
            self._quest_pos_prev = quest_pos.copy()
        elif engaged and reset > 0.5:
            # Reset pressed WHILE the clutch is held. The IK side is
            # ramping the arm toward q_init and ignoring the Cartesian
            # delta, so the engage snapshot is sliding out of sync with
            # the now-reset arm pose. If we leave the original snapshot
            # in place, the moment the user releases reset (clutch still
            # held), the next non-reset call applies
            # ``delta_p = (quest_now - quest_at_original_engage) * scale`` —
            # easily 20-30 cm of accumulated hand motion — to the reset
            # arm pose, and the arm jumps. Re-anchoring to the current
            # pose each frame keeps delta_p at zero across the reset →
            # release transition.
            self._quest_pos_at_engage = quest_pos.copy()
            self._quest_rot_at_engage = quest_rot_to_robot(quest_quat, self._q2r)
            self._quest_pos_prev = quest_pos.copy()
        self._engaged = engaged

        p = self.key_prefix
        if not engaged:
            # New anchor on next engage: don't compare emitted targets
            # across anchors in the jump-guard log.
            self._prev_emitted = None
            action = self.idle_action()
            action[f"{p}gripper_pos"] = float(gripper_pos)
            action[f"{p}reset"] = reset
            return action

        # Position delta in robot frame.
        assert self._quest_pos_at_engage is not None
        assert self._quest_rot_at_engage is not None
        dquest = quest_pos - self._quest_pos_at_engage
        drobot = quest_delta_to_robot(dquest, self._q2r) * self.position_scale

        # Rotation delta in robot frame (as rotvec).
        quest_rot_now = quest_rot_to_robot(quest_quat, self._q2r)
        delta_rot = quest_rot_now * self._quest_rot_at_engage.inv()
        rotvec = delta_rot.as_rotvec()
        mag = float(np.linalg.norm(rotvec))
        cap = self.max_rot_step_rad_per_tick
        if mag > cap > 0.0:
            rotvec = rotvec * (cap / mag)

        # Target jump guard (diagnostic only): the per-frame glitch clamps
        # above bound the emitted step, but a step near those bounds means a
        # snap likely slipped through — say so in the run log. Rate-limited
        # per controller; compared only across consecutive ENGAGED frames
        # (same anchor), cleared on disengage.
        emitted = (drobot.copy(), rotvec.copy())
        if self._prev_emitted is not None:
            dp = float(np.linalg.norm(emitted[0] - self._prev_emitted[0]))
            da = float(np.linalg.norm(emitted[1] - self._prev_emitted[1]))
            if (dp > _TARGET_STEP_WARN_POS_M or da > _TARGET_STEP_WARN_ROT_RAD) and (
                now - self._last_step_warn_t > _TARGET_STEP_WARN_MIN_INTERVAL_S
            ):
                self._last_step_warn_t = now
                logger.warning(
                    "Quest %s target step %.0f mm / %.1f deg in one tick",
                    self._label,
                    dp * 1000,
                    float(np.degrees(da)),
                )
        self._prev_emitted = emitted

        return {
            f"{p}enabled": 1.0,
            f"{p}reset": reset,
            f"{p}target_x": float(drobot[0]),
            f"{p}target_y": float(drobot[1]),
            f"{p}target_z": float(drobot[2]),
            f"{p}target_wx": float(rotvec[0]),
            f"{p}target_wy": float(rotvec[1]),
            f"{p}target_wz": float(rotvec[2]),
            f"{p}gripper_pos": float(gripper_pos),
        }
