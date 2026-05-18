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

"""Bimanual Quest 3 WebXR teleoperator.

One Quest session, two controllers, two robot arms. Emits the same
EE-delta keys as the single-arm variant but prefixed with ``left_`` /
``right_``, matching the bimanual Cartesian IK pipeline composed by
:func:`lerobot.processor.cartesian_ik_pipeline.make_cartesian_ik_pipeline`
when the registered robot config has two arms.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _pin_pink_available

from ..teleoperator import Teleoperator
from .arm_controller import QuestArmController
from .configuration_quest_vr import BimanualQuestVRTeleopConfig
from .server import QuestServer

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
HTML_PATH = HERE / "webxr_teleop.html"
CERT_DIR = HERE / "_cert"  # gitignored; reuses the cert from the unimanual variant.

# Output action keys: every single-arm key, doubled with left_ / right_ prefixes.
_PER_ARM_KEYS = (
    "enabled",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_vel",
)
ACTION_KEYS = tuple(f"{p}{k}" for p in ("left_", "right_") for k in _PER_ARM_KEYS)


class BimanualQuestVRTeleop(Teleoperator):
    """LeRobot Teleoperator that streams 6-DOF EE deltas for both arms.

    Same plumbing as the unimanual :class:`QuestVRTeleop` — one server
    daemon thread, one cached action under a lock, lock-free reads —
    but each WebXR frame is dispatched to TWO :class:`QuestArmController`
    instances (one per hand) and their action dicts are merged.
    """

    config_class = BimanualQuestVRTeleopConfig
    name = "bimanual_quest_vr"

    def __init__(self, config: BimanualQuestVRTeleopConfig):
        if not _pin_pink_available:
            logger.warning(
                "pin-pink not installed; BimanualQuestVRTeleop output needs a non-pink "
                "IK ProcessorStep downstream. Install with `uv pip install pin-pink "
                "qpsolvers[open_source_solvers]` for the recommended path."
            )
        super().__init__(config)
        self.config: BimanualQuestVRTeleopConfig = config
        self._cache_lock = threading.Lock()
        self._cached_action: dict[str, float] | None = None
        self._left = QuestArmController(
            clutch_button_index=config.clutch_button_index,
            gripper_button_index=config.gripper_button_index,
            position_scale=config.position_scale,
            max_rot_step_rad_per_tick=config.max_rot_step_rad_per_tick,
            key_prefix="left_",
        )
        self._right = QuestArmController(
            clutch_button_index=config.clutch_button_index,
            gripper_button_index=config.gripper_button_index,
            position_scale=config.position_scale,
            max_rot_step_rad_per_tick=config.max_rot_step_rad_per_tick,
            key_prefix="right_",
        )
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
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self._server is not None and self._server._thread is not None and self._server._thread.is_alive()
        )

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
        )
        self._server.start()
        logger.info(
            f"BimanualQuestVRTeleop connected. Open {self._server.url} in the Quest 3 "
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

    @check_if_not_connected
    def get_action(self):
        with self._cache_lock:
            if self._cached_action is None:
                return self._idle_action()
            return dict(self._cached_action)

    # ── Server-thread callback ────────────────────────────────────────────

    def _idle_action(self) -> dict[str, float]:
        """Both arms idle (also the pre-first-frame default)."""
        return {**self._left.idle_action(), **self._right.idle_action()}

    def _on_frame(self, frame: dict[str, Any]) -> None:
        """Called from the server's asyncio thread per WebXR frame.

        Dispatches each hand's pose to the corresponding controller; merges
        the two action dicts. If one hand is missing from the frame (Quest
        sometimes drops a controller when it's out of tracking range), that
        arm's last cached values are kept.
        """
        poses = frame.get("poses") or []
        left_pose = next((p for p in poses if p.get("hand") == "left"), None)
        right_pose = next((p for p in poses if p.get("hand") == "right"), None)

        with self._cache_lock:
            base = dict(self._cached_action) if self._cached_action is not None else self._idle_action()
            if left_pose is not None:
                base.update(self._left.process_pose(left_pose))
            if right_pose is not None:
                base.update(self._right.process_pose(right_pose))
            self._cached_action = base

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def url(self) -> str | None:
        return self._server.url if self._server is not None else None

    @property
    def last_rtt_ms(self) -> float | None:
        return self._server.last_rtt_ms if self._server is not None else None
