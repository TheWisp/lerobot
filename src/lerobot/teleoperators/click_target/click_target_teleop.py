# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Click-to-goto bimanual EE teleoperator (experimental).

A thin Cartesian-VR-shape teleop: it emits the bimanual EE-delta action
keys that the predictive follower's ``attach_teleop`` recognises (so it
installs the IK adapter), plus the absolute-world keys the extended
:class:`CartesianIKController` consumes when ``use_world_target`` is set.

Calibration capture and mailbox polling do NOT live here — those are
owned by :class:`ClickCalibrationService` on the robot, so the user can
calibrate using any teleop. This class only owns the goto target state:
:meth:`set_world_target` is the public setter the service calls when a
goto mailbox command arrives, and :meth:`get_action_raw` emits the
absolute-world action dict for the active arm.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from .configuration_click_target import ClickTargetBimanualEETeleopConfig

logger = logging.getLogger(__name__)

# Per-arm keys: original Cartesian-VR shape, plus the four absolute-world keys.
# Matches the shape ``CartesianIKController.__call__`` now consumes.
_PER_ARM_KEYS: tuple[str, ...] = (
    "enabled",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_pos",
    "use_world_target",
    "target_world_x",
    "target_world_y",
    "target_world_z",
    "world_target_top_down",
)
_ACTION_KEYS: tuple[str, ...] = tuple(f"{p}{k}" for p in ("left_", "right_") for k in _PER_ARM_KEYS)


class ClickTargetBimanualEETeleop(Teleoperator):
    """Cartesian-VR-shape teleop driven by an externally-set world target.

    Preconditions for goto:
        * The robot has a :class:`ClickCalibrationService` running and the
          predictive follower wired this teleop's :meth:`set_world_target`
          into it via ``set_goto_target_callback`` — done automatically in
          ``_attach_cartesian_teleop``.

    Postconditions:
        * :meth:`get_action_raw` returns a fully-populated bimanual dict
          over :data:`_ACTION_KEYS`. With no active target, all fields
          are zero except ``gripper_pos`` and ``enabled`` — the IK then
          holds at its latched reference. When a world target is set,
          ``{arm}_use_world_target`` becomes 1 and the world XYZ is
          carried in ``target_world_x/y/z``.
    """

    config_class = ClickTargetBimanualEETeleopConfig
    name = "click_target_bimanual_ee"

    def __init__(self, config: ClickTargetBimanualEETeleopConfig):
        super().__init__(config)
        self.config: ClickTargetBimanualEETeleopConfig = config
        self._lock = threading.Lock()
        self._connected = False
        self._action_transform: Callable[[dict], dict] | None = None
        self._target_world: np.ndarray | None = None

    # ── Teleoperator properties ──────────────────────────────────────────

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(_ACTION_KEYS),),
            "names": {k: i for i, k in enumerate(_ACTION_KEYS)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_exhausted(self) -> bool:
        return False

    # ── External target setter (called by ClickCalibrationService) ───────

    def set_world_target(self, world_xyz: np.ndarray | None) -> None:
        """Set the absolute-world goto target for the active arm.

        ``None`` clears the target, so the IK reverts to holding at the
        latched reference. Thread-safe — invoked from the service's poll
        thread while the action emission runs on the script thread.
        """
        with self._lock:
            self._target_world = None if world_xyz is None else np.asarray(world_xyz, dtype=float).copy()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        self._connected = True
        logger.info(
            "ClickTargetBimanualEETeleop connected (arm=%s, top_down=%s)",
            self.config.arm,
            self.config.world_target_top_down,
        )

    def disconnect(self) -> None:
        self._connected = False
        self._action_transform = None
        with self._lock:
            self._target_world = None

    def send_feedback(self, feedback: dict[str, Any]) -> None:  # noqa: ARG002
        pass

    # ── Action transform installation (for the IK adapter) ───────────────

    def set_action_transform(self, transform: Callable[[dict], dict] | None) -> None:
        """Install the IK transform. Same contract as ScriptedBimanualEETeleop."""
        self._action_transform = transform

    # ── Action emission ──────────────────────────────────────────────────

    def _per_arm_dict(self, *, target: np.ndarray | None) -> dict[str, float]:
        grip = float(self.config.gripper_value)
        top_down_flag = 1.0 if self.config.world_target_top_down else 0.0
        if target is None:
            return {
                "enabled": 1.0,
                "target_x": 0.0,
                "target_y": 0.0,
                "target_z": 0.0,
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper_pos": grip,
                "use_world_target": 0.0,
                "target_world_x": 0.0,
                "target_world_y": 0.0,
                "target_world_z": 0.0,
                "world_target_top_down": top_down_flag,
            }
        return {
            "enabled": 1.0,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper_pos": grip,
            "use_world_target": 1.0,
            "target_world_x": float(target[0]),
            "target_world_y": float(target[1]),
            "target_world_z": float(target[2]),
            "world_target_top_down": top_down_flag,
        }

    def get_action_raw(self) -> dict[str, float]:
        with self._lock:
            target = None if self._target_world is None else self._target_world.copy()
        active_arm = self.config.arm
        out: dict[str, float] = {}
        for arm_name in ("left", "right"):
            arm_target = target if arm_name == active_arm else None
            for k, v in self._per_arm_dict(target=arm_target).items():
                out[f"{arm_name}_{k}"] = v
        return out

    def get_action(self) -> dict[str, Any]:
        action = self.get_action_raw()
        if self._action_transform is not None:
            return self._action_transform(action)
        return action
