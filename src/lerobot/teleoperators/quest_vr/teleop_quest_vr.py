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

Conforms to LeRobot's Teleoperator interface. Internally runs an HTTPS +
WebSocket server in a daemon thread (see ``server.py``); the Quest 3's
browser opens the served page, enters an immersive XR session, and streams
controller poses at 90 Hz. Each frame is converted to an EE-delta action:

    enabled, target_x, target_y, target_z, target_wx, target_wy, target_wz,
    gripper_pos

These are end-effector deltas, not joint commands. A Cartesian-IK robot
turns them into joints by installing a transform via
:meth:`set_action_transform` from its ``attach_teleop`` (see
``lerobot.robots.so107_description.cartesian_ik.CartesianIKController``).
``gripper_pos`` is an absolute motor-space target (trigger fully released
maps to the configured "open" value; fully pulled maps to "closed").

``get_action()`` takes the cache lock briefly and returns a copy of
whatever the server thread last cached. The caller can poll at any
rate; the Quest streams independently at its native frame rate.
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
CERT_DIR = HERE / "_cert"  # gitignored; holds auto-generated self-signed cert

# Output action keys; CartesianIKController consumes exactly these.
ACTION_KEYS = (
    "enabled",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_pos",
)


class QuestVRTeleop(Teleoperator):
    """LeRobot Teleoperator that streams 6-DOF EE deltas from a Quest 3.

    Plumbing is identical to the HighRateLeaderMixin pattern: server thread
    updates a single cached action under a lock; get_action() is lock-free.
    """

    config_class = QuestVRTeleopConfig
    name = "quest_vr"

    def __init__(self, config: QuestVRTeleopConfig):
        if not _pin_pink_available:
            # Hard requirement only if you use the pink IK ProcessorStep
            # downstream; QuestVR itself doesn't import pink, but the
            # typical pipeline that consumes its output does. Warning, not
            # an error, so non-pink pipelines remain possible.
            logger.warning(
                "pin-pink not installed; QuestVRTeleop output will need a non-pink "
                "IK ProcessorStep downstream. Install with `uv pip install pin-pink "
                "qpsolvers[open_source_solvers]` for the recommended path."
            )
        super().__init__(config)
        self.config: QuestVRTeleopConfig = config
        self._cache_lock = threading.Lock()
        self._cached_action: dict[str, float] | None = None
        self._arm = QuestArmController(
            clutch_button_index=config.clutch_button_index,
            gripper_button_index=config.gripper_button_index,
            position_scale=config.position_scale,
            max_rot_step_rad_per_tick=config.max_rot_step_rad_per_tick,
            max_pos_step_m_per_tick=config.max_pos_step_m_per_tick,
            gripper_open_motor=config.gripper_open_motor,
            gripper_closed_motor=config.gripper_closed_motor,
            key_prefix="",
        )
        self._server: QuestServer | None = None
        # Optional per-tick action transform, installed by a robot's
        # ``attach_teleop`` via :meth:`set_action_transform`. None = the
        # teleop emits its native EE deltas.
        self._action_transform: Callable[[dict], dict] | None = None

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
        return {}  # Quest 3 controllers support haptic, not wired here yet.

    @property
    def is_connected(self) -> bool:
        return self._server is not None and self._server.is_running

    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration step; clutch re-anchors per engagement.

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        self._server = QuestServer(
            html_path=HTML_PATH,
            port=self.config.port,
            cert_dir=CERT_DIR,
            on_frame=self._on_frame,
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
            self._arm.reset()

    def calibrate(self) -> None:
        pass  # No calibration step.

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not implemented.

    @check_if_not_connected
    def get_action(self):
        with self._cache_lock:
            action = self._idle_action() if self._cached_action is None else dict(self._cached_action)
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

    # ── Server-thread callback ────────────────────────────────────────────

    def _idle_action(self) -> dict[str, float]:
        """Sent when no frame has arrived yet, or while disengaged."""
        return self._arm.idle_action()

    def _on_frame(self, frame: dict[str, Any]) -> None:
        """Called from the server's asyncio thread per WebXR frame.

        A frame without this controller's pose (tracking dropout) is routed
        to ``on_tracking_lost`` — the arm disengages and re-anchors on the
        next tracked frame, instead of acting on a stale engage snapshot.
        """
        poses = frame.get("poses") or []
        hand = self.config.controller_hand
        pose = next((p for p in poses if p.get("hand") == hand), None)
        action = self._arm.process_pose(pose) if pose is not None else self._arm.on_tracking_lost()
        with self._cache_lock:
            self._cached_action = action

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def url(self) -> str | None:
        return self._server.url if self._server is not None else None

    @property
    def last_rtt_ms(self) -> float | None:
        return self._server.last_rtt_ms if self._server is not None else None
