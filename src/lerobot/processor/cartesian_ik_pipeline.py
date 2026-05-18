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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.robots.robot import Robot
    from lerobot.teleoperators.teleoperator import Teleoperator

    from .pipeline import RobotProcessorPipeline

logger = logging.getLogger(__name__)


# Action keys that mark a teleop as "Cartesian-style" (consumed by EEReferenceAndDelta).
# We detect either unimanual (target_x) or bimanual (left_target_x + right_target_x).
UNIMANUAL_CARTESIAN_KEYS = ("target_x", "target_y", "target_z")
BIMANUAL_CARTESIAN_KEYS = (
    "left_target_x",
    "left_target_y",
    "left_target_z",
    "right_target_x",
    "right_target_y",
    "right_target_z",
)


@dataclass(kw_only=True)
class CartesianIKArmConfig:
    """One arm's parameters for a Cartesian-IK chain.

    For unimanual robots, instantiate one of these with ``key_prefix=""`` (the
    default). For bimanual, instantiate two with ``key_prefix="left_"`` and
    ``key_prefix="right_"``; the prefix is prepended to every action /
    observation key the chain reads or writes, so the two arms' chains stay
    namespace-separated when they share a single transition.
    """

    urdf_path: str
    ee_frame_name: str
    motor_names: list[str]
    joint_names: list[str] | None = None
    key_prefix: str = ""
    # Mounting orientation of this physical arm relative to the teleop's user-
    # reference frame, expressed as a yaw (rotation around URDF +Z) in degrees.
    # 0.0 = arm faces the same direction as the teleop's reference (the unimanual
    # default and the parallel-bimanual case). 180.0 = mirrored mounting (two
    # arms facing each other across the workspace). Other angles work for
    # arbitrary mountings (e.g. arms at 90 deg on adjacent table edges). The
    # pipeline rotates the incoming target_x/y/z/wx/wy/wz by this yaw before
    # composing with the reference pose, so the user's "push controller forward"
    # consistently means "push EE in this arm's URDF -Y" regardless of mounting.
    world_yaw_deg: float = 0.0
    # Workspace box for EEBoundsAndSafety (in robot base frame, meters).
    workspace_min: tuple[float, float, float] = (-0.30, -0.40, +0.02)
    workspace_max: tuple[float, float, float] = (+0.30, +0.10, +0.40)
    # Per-axis EE step scaling for EEReferenceAndDelta.
    end_effector_step_sizes: dict[str, float] | None = None
    # Max EE jump per tick (m) for EEBoundsAndSafety.
    max_ee_step_m: float = 0.10
    # Gripper velocity -> position integration speed.
    gripper_speed_factor: float = 20.0


@dataclass(kw_only=True)
class CartesianIKRobotConfig:
    """Per-robot info needed to build a Cartesian IK pipeline.

    One arm = unimanual; two arms = bimanual. Either pass ``arms=[...]``
    directly, or pass the flat single-arm fields (``urdf_path`` etc.) and a
    single-arm config is built for you. The flat-field form is for the common
    unimanual case; the explicit ``arms`` form is used for bimanual setups
    that need per-arm key prefixes and workspace boxes.
    """

    arms: list[CartesianIKArmConfig] = field(default_factory=list)
    # Single-arm shortcut: any subset of these promote into a one-arm `arms`
    # list with key_prefix="". Mutually exclusive with passing `arms` directly.
    urdf_path: str | None = None
    ee_frame_name: str | None = None
    motor_names: list[str] | None = None
    joint_names: list[str] | None = None
    workspace_min: tuple[float, float, float] = (-0.30, -0.40, +0.02)
    workspace_max: tuple[float, float, float] = (+0.30, +0.10, +0.40)
    end_effector_step_sizes: dict[str, float] | None = None
    max_ee_step_m: float = 0.10
    gripper_speed_factor: float = 20.0

    def __post_init__(self) -> None:
        if not self.arms:
            assert self.urdf_path is not None, (
                "CartesianIKRobotConfig: pass either `arms=[...]` or the flat single-arm fields"
            )
            assert self.ee_frame_name is not None
            assert self.motor_names is not None
            self.arms = [
                CartesianIKArmConfig(
                    urdf_path=self.urdf_path,
                    ee_frame_name=self.ee_frame_name,
                    motor_names=self.motor_names,
                    joint_names=self.joint_names,
                    key_prefix="",
                    workspace_min=self.workspace_min,
                    workspace_max=self.workspace_max,
                    end_effector_step_sizes=self.end_effector_step_sizes,
                    max_ee_step_m=self.max_ee_step_m,
                    gripper_speed_factor=self.gripper_speed_factor,
                )
            ]
        assert 1 <= len(self.arms) <= 2, f"CartesianIKRobotConfig: expected 1 or 2 arms, got {len(self.arms)}"

    @property
    def is_bimanual(self) -> bool:
        return len(self.arms) == 2


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
    """Return True if the teleop emits Cartesian EE-delta actions (unimanual or bimanual)."""
    try:
        names = teleop.action_features.get("names", {})
    except (AttributeError, KeyError):
        return False
    return all(k in names for k in UNIMANUAL_CARTESIAN_KEYS) or all(
        k in names for k in BIMANUAL_CARTESIAN_KEYS
    )


def _build_arm_steps(arm: CartesianIKArmConfig) -> list[Any]:
    """Build the four-step Cartesian-IK chain for one arm (prefix-aware)."""
    from lerobot.model import RobotKinematics
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        EEBoundsAndSafety,
        EEReferenceAndDelta,
        GripperVelocityToJoint,
    )

    # FK for EEReferenceAndDelta (uses placo; FK is unaffected by the
    # missing-PostureTask issue we hit with placo's IK).
    fk_kin = RobotKinematics(
        urdf_path=arm.urdf_path,
        target_frame_name=arm.ee_frame_name,
        joint_names=arm.joint_names,
    )

    # IK uses pink (with posture regularization).
    try:
        from lerobot.model.pink_kinematics import PinkKinematics
        from lerobot.robots.so_follower.pink_kinematic_processor import (
            PinkInverseKinematicsEEToJoints,
        )

        pink_kin = PinkKinematics(
            urdf_path=arm.urdf_path,
            target_frame_name=arm.ee_frame_name,
            joint_names=arm.joint_names,
        )
        ik_step: Any = PinkInverseKinematicsEEToJoints(
            kinematics=pink_kin,
            motor_names=arm.motor_names,
            key_prefix=arm.key_prefix,
        )
    except ImportError:
        from lerobot.robots.so_follower.robot_kinematic_processor import (
            InverseKinematicsEEToJoints,
        )

        logger.warning(
            "pink-pink not installed; falling back to placo-based IK. Install "
            "pin-pink + qpsolvers[open_source_solvers] for posture-regularized IK."
        )
        ik_step = InverseKinematicsEEToJoints(
            kinematics=fk_kin,
            motor_names=arm.motor_names,
            key_prefix=arm.key_prefix,
        )

    step_sizes = arm.end_effector_step_sizes or {"x": 1.0, "y": 1.0, "z": 1.0}
    return [
        EEReferenceAndDelta(
            kinematics=fk_kin,
            end_effector_step_sizes=step_sizes,
            motor_names=arm.motor_names,
            use_latched_reference=True,
            key_prefix=arm.key_prefix,
            world_yaw_deg=arm.world_yaw_deg,
        ),
        EEBoundsAndSafety(
            end_effector_bounds={
                "min": list(arm.workspace_min),
                "max": list(arm.workspace_max),
            },
            max_ee_step_m=arm.max_ee_step_m,
            key_prefix=arm.key_prefix,
        ),
        GripperVelocityToJoint(
            speed_factor=arm.gripper_speed_factor,
            key_prefix=arm.key_prefix,
        ),
        ik_step,
    ]


def make_cartesian_ik_pipeline(robot: Robot) -> RobotProcessorPipeline | None:
    """Build the Cartesian-IK pipeline for ``robot`` if a config is registered.

    For unimanual robots, returns a single 4-step chain. For bimanual, returns
    a chain that runs the per-arm steps back-to-back; each per-arm step is
    prefixed (left_/right_) so they read disjoint keys from the same transition.

    Returns None when the robot has no registered Cartesian config; the caller
    should fall back to identity in that case (and log).
    """
    cfg = _REGISTRY.get(getattr(robot, "name", ""))
    if cfg is None:
        return None

    from lerobot.processor.converters import (
        robot_action_observation_to_transition,
        transition_to_robot_action,
    )
    from lerobot.processor.pipeline import RobotProcessorPipeline

    steps: list[Any] = []
    for arm in cfg.arms:
        steps.extend(_build_arm_steps(arm))

    return RobotProcessorPipeline(
        steps=steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
