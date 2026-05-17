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

"""Helpers to auto-compose the Cartesian-IK ProcessorStep chain for the
common case of a Cartesian teleop (Quest VR, keyboard_ee, phone, ...)
driving a follower that expects joint commands.

The default ``lerobot-teleoperate`` pipeline is identity; Cartesian teleops
emit ``target_x/y/z/wx/wy/wz/gripper_vel`` and the robot expects ``<motor>.pos``
keys, so we need to inject:

    EEReferenceAndDelta  ->  EEBoundsAndSafety  ->  GripperVelocityToJoint
    ->  PinkInverseKinematicsEEToJoints

This module detects whether the teleop is Cartesian-style and, when the
robot has a registered config, returns a fully-composed pipeline. The
script then uses it in place of the identity ``robot_action_processor``.

Robot-specific config (URDF path, EE frame, motor names, workspace bounds)
lives in a registry keyed by ``robot.name``. New robots add an entry in
their package; nothing in upstream needs to change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot
    from lerobot.teleoperators.teleoperator import Teleoperator

    from .pipeline import RobotProcessorPipeline

logger = logging.getLogger(__name__)


# Action keys that mark a teleop as "Cartesian-style" (consumed by EEReferenceAndDelta).
CARTESIAN_TELEOP_KEYS = ("target_x", "target_y", "target_z")


@dataclass(kw_only=True)
class CartesianIKRobotConfig:
    """Per-robot info needed to build a Cartesian IK pipeline.

    The robot package owns this. Default values match the common case
    (single 7-DOF arm with the standard SO-family motor naming).
    """

    urdf_path: str
    ee_frame_name: str
    motor_names: list[str]
    joint_names: list[str] | None = None
    # Workspace box for EEBoundsAndSafety (in robot base frame, meters).
    workspace_min: tuple[float, float, float] = (-0.30, -0.40, +0.02)
    workspace_max: tuple[float, float, float] = (+0.30, +0.10, +0.40)
    # Per-axis EE step scaling for EEReferenceAndDelta.
    end_effector_step_sizes: dict[str, float] | None = None
    # Max EE jump per tick (m) for EEBoundsAndSafety.
    max_ee_step_m: float = 0.10
    # Gripper velocity -> position integration speed.
    gripper_speed_factor: float = 20.0


# Registry: robot.name -> CartesianIKRobotConfig.
# Robot packages register themselves at import time by calling
# ``register_cartesian_ik_robot(name, cfg)``. The auto-discovery in
# ``gui/api/robot.py:_ensure_configs_loaded`` and the noqa-imports in
# ``lerobot-teleoperate`` both trigger those registrations.
_REGISTRY: dict[str, CartesianIKRobotConfig] = {}


def register_cartesian_ik_robot(name: str, cfg: CartesianIKRobotConfig) -> None:
    """Register a robot's Cartesian-IK config so auto-composition can find it."""
    if name in _REGISTRY:
        logger.warning(f"CartesianIK config for {name!r} re-registered; overwriting")
    _REGISTRY[name] = cfg


def get_cartesian_ik_robot_config(name: str) -> CartesianIKRobotConfig | None:
    return _REGISTRY.get(name)


def is_cartesian_teleop(teleop: Teleoperator) -> bool:
    """Return True if the teleop emits Cartesian EE-delta actions."""
    try:
        names = teleop.action_features.get("names", {})
    except (AttributeError, KeyError):
        return False
    return all(k in names for k in CARTESIAN_TELEOP_KEYS)


def make_cartesian_ik_pipeline(robot: Robot) -> RobotProcessorPipeline | None:
    """Build the Cartesian-IK pipeline for ``robot`` if a config is registered.

    Returns None when the robot has no registered Cartesian config; the caller
    should fall back to identity in that case (and log).
    """
    cfg = _REGISTRY.get(getattr(robot, "name", ""))
    if cfg is None:
        return None

    from lerobot.model import RobotKinematics
    from lerobot.processor.converters import (
        robot_action_observation_to_transition,
        transition_to_robot_action,
    )
    from lerobot.processor.pipeline import RobotProcessorPipeline
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        EEBoundsAndSafety,
        EEReferenceAndDelta,
        GripperVelocityToJoint,
    )

    # FK for EEReferenceAndDelta (uses placo; FK is unaffected by the
    # missing-PostureTask issue we hit with placo's IK).
    fk_kin = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name=cfg.ee_frame_name,
        joint_names=cfg.joint_names,
    )

    # IK uses pink (with posture regularization).
    try:
        from lerobot.model.pink_kinematics import PinkKinematics
        from lerobot.robots.so_follower.pink_kinematic_processor import (
            PinkInverseKinematicsEEToJoints,
        )

        pink_kin = PinkKinematics(
            urdf_path=cfg.urdf_path,
            target_frame_name=cfg.ee_frame_name,
            joint_names=cfg.joint_names,
        )
        ik_step: Any = PinkInverseKinematicsEEToJoints(kinematics=pink_kin, motor_names=cfg.motor_names)
    except ImportError:
        from lerobot.robots.so_follower.robot_kinematic_processor import (
            InverseKinematicsEEToJoints,
        )

        logger.warning(
            "pink-pink not installed; falling back to placo-based IK. Install "
            "pin-pink + qpsolvers[open_source_solvers] for posture-regularized IK."
        )
        ik_step = InverseKinematicsEEToJoints(kinematics=fk_kin, motor_names=cfg.motor_names)

    step_sizes = cfg.end_effector_step_sizes or {"x": 1.0, "y": 1.0, "z": 1.0}
    steps = [
        EEReferenceAndDelta(
            kinematics=fk_kin,
            end_effector_step_sizes=step_sizes,
            motor_names=cfg.motor_names,
            use_latched_reference=True,
        ),
        EEBoundsAndSafety(
            end_effector_bounds={
                "min": list(cfg.workspace_min),
                "max": list(cfg.workspace_max),
            },
            max_ee_step_m=cfg.max_ee_step_m,
        ),
        GripperVelocityToJoint(speed_factor=cfg.gripper_speed_factor),
        ik_step,
    ]
    return RobotProcessorPipeline(
        steps=steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
