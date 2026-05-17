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
controller poses at 90 Hz. Each frame is converted to an EE-delta action
in the format consumed upstream by ``EEReferenceAndDelta``:

    enabled, target_x, target_y, target_z, target_wx, target_wy, target_wz,
    gripper_vel

This output is what ``EEReferenceAndDelta`` -> ``GripperVelocityToJoint``
-> ``PinkInverseKinematicsEEToJoints`` consume to produce joint commands.

``get_action()`` is lock-free and returns whatever the server thread last
cached. Caller can poll at any rate; the Quest streams independently at
its native frame rate.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _pin_pink_available

from ..teleoperator import Teleoperator
from .configuration_quest_vr import QuestVRTeleopConfig
from .server import QuestServer, quest_delta_to_robot, quest_rot_to_robot

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
HTML_PATH = HERE / "webxr_teleop.html"
CERT_DIR = HERE / "_cert"  # gitignored; holds auto-generated self-signed cert

# Output action keys must match what EEReferenceAndDelta upstream consumes.
ACTION_KEYS = (
    "enabled",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_vel",
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
        # Per-engage snapshot kept on the server thread (the only writer).
        self._engaged: bool = False
        self._quest_pos_at_engage: np.ndarray | None = None
        self._quest_rot_at_engage = None  # scipy Rotation
        self._gripper_last_value: float = 0.0
        self._server: QuestServer | None = None

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
        return (
            self._server is not None and self._server._thread is not None and self._server._thread.is_alive()
        )

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
            self._engaged = False
            self._quest_pos_at_engage = None
            self._quest_rot_at_engage = None

    def calibrate(self) -> None:
        pass  # No calibration step.

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not implemented.

    @check_if_not_connected
    def get_action(self):
        with self._cache_lock:
            if self._cached_action is None:
                return self._idle_action()
            return dict(self._cached_action)

    # ── Server-thread callback ────────────────────────────────────────────

    def _idle_action(self) -> dict[str, float]:
        """Sent when no frame has arrived yet, or while disengaged."""
        return {
            "enabled": 0.0,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper_vel": 0.0,
        }

    def _on_frame(self, frame: dict[str, Any]) -> None:
        """Called from the server's asyncio thread per WebXR frame."""
        poses = frame.get("poses") or []
        hand = self.config.controller_hand
        right = next((p for p in poses if p.get("hand") == hand), None)
        if right is None:
            return

        quest_pos = np.asarray(right["pos"], dtype=float)
        quest_quat = right.get("rot", [0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
        buttons = right.get("buttons") or []
        clutch = (
            float(buttons[self.config.clutch_button_index])
            if len(buttons) > self.config.clutch_button_index
            else 0.0
        )
        grip = (
            float(buttons[self.config.gripper_button_index])
            if len(buttons) > self.config.gripper_button_index
            else 0.0
        )
        engaged = clutch > 0.5

        # Gripper is a velocity command: re-derive from trigger position delta.
        # Simple model: trigger pulled = close (negative velocity), released = open.
        # This matches the GripperVelocityToJoint upstream consumer.
        gripper_vel = self._gripper_last_value - grip  # in [-1, +1] roughly
        self._gripper_last_value = grip

        if engaged and not self._engaged:
            # Rising edge: snapshot pose for clutch reference.
            self._quest_pos_at_engage = quest_pos.copy()
            self._quest_rot_at_engage = quest_rot_to_robot(quest_quat)
        self._engaged = engaged

        if not engaged:
            with self._cache_lock:
                self._cached_action = {**self._idle_action(), "gripper_vel": gripper_vel}
            return

        # Position delta in robot frame.
        dquest = quest_pos - self._quest_pos_at_engage
        drobot = quest_delta_to_robot(dquest) * float(self.config.position_scale)

        # Rotation delta in robot frame (as rotvec).
        quest_rot_now = quest_rot_to_robot(quest_quat)
        delta_rot = quest_rot_now * self._quest_rot_at_engage.inv()
        # Cap rotation step magnitude per emit to bound wild jumps.
        rotvec = delta_rot.as_rotvec()
        mag = float(np.linalg.norm(rotvec))
        cap = float(self.config.max_rot_step_rad_per_tick)
        if mag > cap > 0.0:
            rotvec = rotvec * (cap / mag)

        action = {
            "enabled": 1.0,
            "target_x": float(drobot[0]),
            "target_y": float(drobot[1]),
            "target_z": float(drobot[2]),
            "target_wx": float(rotvec[0]),
            "target_wy": float(rotvec[1]),
            "target_wz": float(rotvec[2]),
            "gripper_vel": float(gripper_vel),
        }
        with self._cache_lock:
            self._cached_action = action

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def url(self) -> str | None:
        return self._server.url if self._server is not None else None

    @property
    def last_rtt_ms(self) -> float | None:
        return self._server.last_rtt_ms if self._server is not None else None
