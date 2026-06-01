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

"""Quest 3 WebXR teleoperator.

One Quest session, two controllers, two robot arms. Quest is always
bimanual (two controllers in one immersive session) — there is no
unimanual variant. Action keys are prefixed ``left_`` / ``right_``
to route per-arm. A robot turns them into joint commands by installing
a per-arm Cartesian-IK transform via :meth:`set_action_transform` from
its ``attach_teleop`` (see ``BiSO107Follower.attach_teleop``).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _pin_pink_available

from ..teleoperator import Teleoperator
from .arm_controller import QuestArmController
from .configuration_quest_vr import QuestVRTeleopConfig
from .server import QuestServer

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
HTML_PATH = HERE / "webxr_teleop.html"
CERT_DIR = HERE / "_cert"  # gitignored; reuses the cert from the unimanual variant.

# Output action keys: every single-arm key, doubled with left_ / right_ prefixes.
_PER_ARM_KEYS = (
    "enabled",
    "reset",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_pos",
)
ACTION_KEYS = tuple(f"{p}{k}" for p in ("left_", "right_") for k in _PER_ARM_KEYS)


class QuestVRTeleop(Teleoperator):
    """LeRobot Teleoperator that streams 6-DOF EE deltas for both arms.

    Same plumbing as the unimanual :class:`QuestVRTeleop` — one server
    daemon thread, one cached action under a lock, lock-free reads —
    but each WebXR frame is dispatched to TWO :class:`QuestArmController`
    instances (one per hand) and their action dicts are merged.
    """

    config_class = QuestVRTeleopConfig
    name = "quest_vr"

    def __init__(self, config: QuestVRTeleopConfig):
        if not _pin_pink_available:
            logger.warning(
                "pin-pink not installed; QuestVRTeleop output needs a non-pink "
                "IK ProcessorStep downstream. Install with `uv pip install pin-pink "
                "qpsolvers[open_source_solvers]` for the recommended path."
            )
        super().__init__(config)
        self.config: QuestVRTeleopConfig = config
        self._cache_lock = threading.Lock()
        self._cached_action: dict[str, float] | None = None
        self._left = QuestArmController(
            clutch_button_index=config.clutch_button_index,
            gripper_button_index=config.gripper_button_index,
            reset_button_index=config.reset_button_index,
            position_scale=config.position_scale,
            max_rot_step_rad_per_tick=config.max_rot_step_rad_per_tick,
            max_pos_step_m_per_tick=config.max_pos_step_m_per_tick,
            gripper_open_motor=config.left_gripper_open_motor,
            gripper_closed_motor=config.left_gripper_closed_motor,
            key_prefix="left_",
        )
        self._right = QuestArmController(
            clutch_button_index=config.clutch_button_index,
            gripper_button_index=config.gripper_button_index,
            reset_button_index=config.reset_button_index,
            position_scale=config.position_scale,
            max_rot_step_rad_per_tick=config.max_rot_step_rad_per_tick,
            max_pos_step_m_per_tick=config.max_pos_step_m_per_tick,
            gripper_open_motor=config.right_gripper_open_motor,
            gripper_closed_motor=config.right_gripper_closed_motor,
            key_prefix="right_",
        )
        self._server: QuestServer | None = None
        # Optional per-tick action transform, installed by a robot's
        # ``attach_teleop`` via :meth:`set_action_transform`. None = the
        # teleop emits its native EE deltas.
        self._action_transform: Callable[[dict], dict] | None = None
        # Optional control channel handoff from the orchestrator —
        # installed via :meth:`set_control_channel`. When present,
        # rising edges on configured Quest face buttons emit channel
        # actions (e.g. right B → ``rlt_success`` for HVLA RLT terminal
        # signalling). Hardcoded mapping below; the hotkeys.json layer
        # (P2 in CONTROL_CHANNEL.md) will eventually own this.
        self._control_channel = None
        self._prev_left_buttons: list[float] = []
        self._prev_right_buttons: list[float] = []

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(ACTION_KEYS),),
            "names": {k: i for i, k in enumerate(ACTION_KEYS)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._server is not None and self._server.is_running

    @property
    def is_calibrated(self) -> bool:
        return True

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        self._server = QuestServer(
            html_path=HTML_PATH,
            port=self.config.port,
            cert_dir=CERT_DIR,
            on_frame=self._on_frame,
            # ``{{KEY}}`` markers in the served HTML get these values
            # substituted at request time. Used for the haptic-feedback
            # path: the page fires a pulse on the rising/falling edge of
            # the clutch button, but it doesn't otherwise know which
            # button index that is.
            page_vars={"CLUTCH_BUTTON_INDEX": str(self.config.clutch_button_index)},
            get_hold_state=self._read_ik_hold_state,
        )
        self._server.start()
        logger.info(
            f"QuestVRTeleop connected. Open {self._server.url} in the Quest 3 "
            f"browser (accept the self-signed cert warning), then Connect + Enter VR."
        )

    def disconnect(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
        with self._cache_lock:
            self._cached_action = None
            self._left.reset()
            self._right.reset()

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not implemented.

    def get_action_raw(self) -> dict[str, float]:
        """Return the raw EE-delta action, bypassing any installed transform.

        Lets a downstream consumer that runs its own IK (e.g. the
        :class:`BimanualCartesianIKAdapter` polling at WebXR rate) access
        the Cartesian-delta dict without going through whatever transform
        ``set_action_transform`` installed on top — otherwise the adapter
        would feed its own cached joint dict back into itself.
        """
        with self._cache_lock:
            return self._idle_action() if self._cached_action is None else dict(self._cached_action)

    @check_if_not_connected
    def get_action(self):
        action = self.get_action_raw()
        if self._action_transform is not None:
            action = self._action_transform(action)
        return action

    def set_action_transform(self, transform: Callable[[dict], dict] | None) -> None:
        """Install (or clear, with ``None``) a per-tick action transform.

        A robot's ``attach_teleop`` installs a Cartesian-IK transform here so
        ``get_action()`` returns motor-space joint commands instead of EE
        deltas. This keeps the IK robot-owned and leaves the upstream
        teleop / record / replay loops untouched — they just call
        ``get_action()`` and receive whatever the robot wired up.
        """
        self._action_transform = transform

    def _read_ik_hold_state(self) -> tuple[bool, bool]:
        """Probe the installed transform for per-arm IK-hold state.

        Returns ``(left_holding, right_holding)``. Called from the server's
        asyncio thread (after each received WebXR frame) — a cheap
        attribute read with no lock; the tuple is constructed fresh each
        call and the underlying booleans are GIL-atomic.

        Returns ``(False, False)`` when no transform with ``hold_per_arm``
        is installed (e.g. a non-Cartesian downstream, or before
        ``attach_teleop`` runs).
        """
        t = self._action_transform
        hold = getattr(t, "hold_per_arm", None)
        if hold is None:
            return (False, False)
        return (bool(hold[0]), bool(hold[1]))

    # ── Server-thread callback ────────────────────────────────────────────

    def _idle_action(self) -> dict[str, float]:
        """Both arms idle (also the pre-first-frame default)."""
        return {**self._left.idle_action(), **self._right.idle_action()}

    def _on_frame(self, frame: dict[str, Any]) -> None:
        """Called from the server's asyncio thread per WebXR frame.

        Dispatches each hand's pose to its controller and merges the two
        action dicts. A hand missing from the frame (the Quest dropped that
        controller's tracking) is routed to ``on_tracking_lost``, which
        disengages that arm and re-anchors it on the next tracked frame —
        so a dropout cannot leak a stale-snapshot delta.
        """
        poses = frame.get("poses") or []
        left_pose = next((p for p in poses if p.get("hand") == "left"), None)
        right_pose = next((p for p in poses if p.get("hand") == "right"), None)

        with self._cache_lock:
            base = dict(self._cached_action) if self._cached_action is not None else self._idle_action()
            base.update(
                self._left.process_pose(left_pose) if left_pose is not None else self._left.on_tracking_lost()
            )
            base.update(
                self._right.process_pose(right_pose)
                if right_pose is not None
                else self._right.on_tracking_lost()
            )
            self._cached_action = base

        # Flow-control: edge-detect face buttons against the installed
        # channel. Done OUTSIDE the cache lock so a slow channel emit
        # never stalls the per-frame action update.
        if self._control_channel is not None:
            self._dispatch_button_edges(left_pose, right_pose)

    # ── Flow-control via the channel ──────────────────────────────────────

    # Quest Touch Plus button index → channel action name.
    # WebXR gamepad indexing on Quest 3:
    #   0: trigger (analog) — owned by gripper, not flow-control
    #   1: grip (analog) — owned by clutch, not flow-control
    #   3: thumbstick click — bindable
    #   4: A (right) / X (left) — owned by reset ramp, not flow-control
    #   5: B (right) / Y (left) — bindable
    # Hardcoded for first cut. P2 (hotkeys.json) replaces this.
    _LEFT_BUTTON_BINDINGS: dict[int, str] = {
        5: "rlt_abort",  # Y → abort terminal (negative reward, save)
        3: "rlt_toggle_engage",  # left joystick click → toggle RL actor
    }
    _RIGHT_BUTTON_BINDINGS: dict[int, str] = {
        5: "rlt_success",  # B → success terminal (+1 reward)
        3: "rlt_ignore",  # right joystick click → discard episode
    }

    def _dispatch_button_edges(self, left_pose, right_pose) -> None:
        """Emit channel actions on rising edges of bound Quest buttons.

        Only emits if (a) a channel was handed in via
        :meth:`set_control_channel` and (b) the action name was
        registered on that channel by the orchestrator. Unregistered
        actions are silently dropped (logged by the channel itself) so
        the Quest VR teleop works in record / pure-teleop sessions
        that don't know about RLT.
        """
        left_btns = list(left_pose.get("buttons", [])) if left_pose else []
        right_btns = list(right_pose.get("buttons", [])) if right_pose else []

        for idx, action in self._LEFT_BUTTON_BINDINGS.items():
            if _is_rising_edge(
                self._prev_left_buttons, left_btns, idx
            ) and self._control_channel.is_registered(action):
                self._control_channel.emit(action, source=f"quest_left_{idx}")

        for idx, action in self._RIGHT_BUTTON_BINDINGS.items():
            if _is_rising_edge(
                self._prev_right_buttons, right_btns, idx
            ) and self._control_channel.is_registered(action):
                self._control_channel.emit(action, source=f"quest_right_{idx}")

        self._prev_left_buttons = left_btns
        self._prev_right_buttons = right_btns

    def set_control_channel(self, channel) -> None:
        """Install (or clear, with ``None``) the orchestrator's channel.

        After install, rising edges on configured Quest face buttons
        (B / Y / joystick clicks) call ``channel.emit(...)`` with the
        bound action name. Continuous-state buttons (grip / trigger /
        A-X — owned by the teleop's clutch / gripper / reset semantics)
        are untouched.
        """
        self._control_channel = channel
        # Reset edge baselines so the first frame after install
        # doesn't fire spurious edges from buttons held at install time.
        self._prev_left_buttons = []
        self._prev_right_buttons = []

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def url(self) -> str | None:
        return self._server.url if self._server is not None else None

    @property
    def last_rtt_ms(self) -> float | None:
        return self._server.last_rtt_ms if self._server is not None else None


def _is_rising_edge(prev: list[float], curr: list[float], idx: int, threshold: float = 0.5) -> bool:
    """True iff button ``idx`` went from below to above threshold between frames."""
    prev_pressed = idx < len(prev) and prev[idx] > threshold
    curr_pressed = idx < len(curr) and curr[idx] > threshold
    return curr_pressed and not prev_pressed
