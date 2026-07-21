#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Cartesian end-effector teleop -> joint commands for one OpenArm 2.0 arm.

This is the OpenArm equivalent of
:mod:`lerobot.robots.so107_description.cartesian_ik`; the stateful
controller, the bimanual split/merge, and the hold-state wrapper are the
same generic machinery and are imported from there. What is OpenArm-
specific here:

* motor / URDF joint naming (``joint_1..joint_7`` +
  ``openarm_{side}_joint1..7``, see :mod:`openarm_description`);
* the IK target frame (``openarm_{side}_ee_base_link`` — the gripper
  flange, same control point as the ``*_ee_control_point`` sites in the
  upstream MuJoCo model). No tip offset: the validated dora OpenArm VR
  stack controls exactly this point, and its EE zero
  (``(0, ±0.1225, -0.436)`` m, identity orientation at q=0) matches this
  URDF's FK at the all-zero pose;
* NO motor<->URDF alignment wrapper: the OpenArm 2.0 motor factory zero
  (arm hanging straight down) coincides with the URDF joint zero, and the
  per-side left/right mirror is baked into the URDF joint axes — so
  motor degrees ARE URDF joint degrees, verified by FK at q=0 (see the
  package README). :class:`PinkKinematics` is used directly;
* the workspace clip box below.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lerobot.robots.so107_description.cartesian_ik import (
    CartesianIKController,
    make_bimanual_ik_transform,
)

# Motor order matches the OpenArmFollower motor list; the gripper is a
# passthrough (not IK-tracked) — the controller overwrites it post-IK.
MOTOR_NAMES: tuple[str, ...] = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "gripper",
)


def urdf_joint_names(side: str) -> list[str]:
    """URDF joint names in motor order for one arm (``side`` = left/right)."""
    return [f"openarm_{side}_joint{i}" for i in range(1, 8)]


def urdf_tip_frame(side: str) -> str:
    """IK target frame: the wrist / gripper-flange link (downstream of joint7)."""
    return f"openarm_{side}_ee_base_link"


# Reachable EE box for one OpenArm 2.0 arm, robot base frame, meters.
# Hand-tuned estimates, NOT measured (same caveat as the SO-107 box):
# the arm hangs 0.436 m below the base at zero and reaches ~0.55 m, so
# these bounds keep an out-of-reach Quest command from driving the solver
# into garbage while covering the whole natural workspace. Forward is +X
# (see the package README for the frame convention).
OPENARM_WORKSPACE_MIN: tuple[float, float, float] = (-0.20, -0.45, -0.55)
OPENARM_WORKSPACE_MAX: tuple[float, float, float] = (+0.55, +0.45, +0.25)


def make_openarm_arm_kinematics(
    side: str,
    *,
    posture_cost: float = 0.05,
    max_iters: int = 50,
):
    """Build the kinematics for one OpenArm 2.0 arm.

    This is the slow half of the Cartesian-IK setup: ``PinkKinematics``
    parses the URDF into a pinocchio model (~1-2 s, CPU-bound, holds the
    GIL). Call it once, eagerly — e.g. in a robot's ``__init__``, before
    ``connect()`` starts any camera read thread.

    Args:
        side: ``"left"`` or ``"right"`` — selects the per-arm URDF. The
            two arms are mirror builds baked into the URDF joint axes, so
            no per-arm alignment is needed beyond picking the file.
        posture_cost: PostureTask weight relative to the FrameTask (1.0).
            Default 0.05 makes posture a null-space tiebreaker. Raise
            (e.g. 0.3) for stronger "stay near previous pose" — tighter
            joint continuity near singularities, at the cost of small EE
            tracking lag.
        max_iters: QP iteration budget per ``inverse_kinematics`` call.
            Pink defaults to 10; 50 closes per-call lag on a moving teleop
            target to mm-scale at typical teleop speeds. Negligible CPU
            impact at 30 Hz.

    Requires the optional ``pin-pink`` dependency (raises ``ImportError``
    otherwise).
    """
    from lerobot.model.pink_kinematics import PinkKinematics

    from . import get_urdf_path

    return PinkKinematics(
        urdf_path=str(get_urdf_path(side)),
        target_frame_name=urdf_tip_frame(side),
        joint_names=urdf_joint_names(side),
        posture_cost=posture_cost,
        max_iters=max_iters,
    )


def make_openarm_arm_ik_controller(
    kinematics,
    q_init: np.ndarray,
    workspace_min: tuple[float, float, float] = OPENARM_WORKSPACE_MIN,
    workspace_max: tuple[float, float, float] = OPENARM_WORKSPACE_MAX,
    *,
    label: str = "",
) -> CartesianIKController:
    """Build a Cartesian-IK controller for one OpenArm 2.0 arm.

    Fast: wires a pre-built kinematics, the seed configuration, and the
    OpenArm workspace box into a controller. Build ``kinematics`` ahead of
    time with :func:`make_openarm_arm_kinematics`.

    Args:
        kinematics: This arm's kinematics from :func:`make_openarm_arm_kinematics`.
        q_init: The arm's current joint configuration, motor-space degrees,
            in :data:`MOTOR_NAMES` order (7 arm joints + gripper).
        workspace_min, workspace_max: EE-position clip box, robot base frame.
    """
    return CartesianIKController(
        kinematics=kinematics,
        motor_names=list(MOTOR_NAMES),
        q_init=q_init,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        label=label,
    )


def is_openarm_bimanual_cartesian_teleop(teleop: Any) -> bool:
    """True iff ``teleop`` looks like a bimanual Cartesian source.

    Cheap structural check: ``action_features.names`` contains
    ``left_target_x`` and ``right_target_x``. Use to gate the Cartesian
    branch in ``BiOpenArmFollower.attach_teleop`` before pulling in
    pin-pink-dependent IK setup. (Same contract as the SO-107 detector —
    the Quest VR action features are robot-agnostic.)

    Does NOT verify the teleop has ``set_action_transform`` /
    ``get_action_raw``; callers should assert those separately if
    they require them.
    """
    try:
        names = teleop.action_features.get("names", {})
    except (AttributeError, TypeError):
        return False
    return "left_target_x" in names and "right_target_x" in names


class BimanualOpenArmIKTransform:
    """Callable Cartesian-action → joint-action transform for a bimanual
    OpenArm follower, with per-arm IK-hold state readable by the teleop.

    Wraps :func:`make_bimanual_ik_transform` so callers that only need the
    dict→dict behavior (``set_action_transform``) keep working unchanged,
    and adds :attr:`hold_per_arm` for callers (the Quest VR teleop) that
    want to surface "the IK is holding the last command" to the operator
    as a tactile signal.
    """

    def __init__(self, left: CartesianIKController, right: CartesianIKController) -> None:
        self.left = left
        self.right = right
        self._inner = make_bimanual_ik_transform(left, right)

    def __call__(self, action: dict) -> dict:
        return self._inner(action)

    @property
    def hold_per_arm(self) -> tuple[bool, bool]:
        """``(left_holding, right_holding)`` from the last ``__call__``.

        True iff that arm's IK ended up returning the held command
        (NoSolutionFound, implausible-jump backstop). The teleop signals
        each rising / falling edge to the operator via haptic pulse.
        """
        return (self.left.is_holding, self.right.is_holding)


def build_openarm_bimanual_ik_transform(
    ik_kinematics: dict[str, Any],
    left_arm: Any,
    right_arm: Any,
    workspace_min: tuple[float, float, float] = OPENARM_WORKSPACE_MIN,
    workspace_max: tuple[float, float, float] = OPENARM_WORKSPACE_MAX,
) -> BimanualOpenArmIKTransform:
    """Build a Cartesian-action → joint-action transform for a bimanual
    OpenArm follower.

    Seeds per-arm IK controllers from each arm's current observation
    (both arms must be connected — ``arm.get_observation()`` is called
    once at build time to read the latch reference), then composes them
    through the bimanual prefix split/merge.

    The returned object is callable (the shape ``teleop.set_action_transform``
    expects: takes a bimanual Cartesian action dict, returns a
    motor-joint dict prefixed with ``left_`` / ``right_``) AND exposes
    :attr:`BimanualOpenArmIKTransform.hold_per_arm` for callers that want
    per-arm hold state.
    """

    def _seed(arm: Any) -> np.ndarray:
        obs = arm.get_observation()
        return np.array([float(obs[f"{m}.pos"]) for m in MOTOR_NAMES], dtype=float)

    left_ik = make_openarm_arm_ik_controller(
        ik_kinematics["left"], _seed(left_arm), workspace_min, workspace_max, label="left"
    )
    right_ik = make_openarm_arm_ik_controller(
        ik_kinematics["right"], _seed(right_arm), workspace_min, workspace_max, label="right"
    )
    return BimanualOpenArmIKTransform(left_ik, right_ik)
