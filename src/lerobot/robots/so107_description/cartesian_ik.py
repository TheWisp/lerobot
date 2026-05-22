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

"""Cartesian end-effector teleop -> joint commands for one SO-107 arm.

A :class:`CartesianIKController` turns the eight EE-delta keys a Quest VR
controller emits (``enabled, target_x/y/z, target_wx/wy/wz, gripper_pos``)
into a ``{motor}.pos`` joint dict, by latching a reference pose on clutch
engage and running IK on the composed target.

It is deliberately a plain stateful callable, not a ``ProcessorStep``: a
robot installs it into a Cartesian teleop via ``attach_teleop`` so that
``teleop.get_action()`` returns joint commands, and the upstream teleop /
record / replay loops stay untouched. Because of that it must run with no
per-tick robot observation — so the reference pose is the forward
kinematics of the *last commanded* joints and the IK seed is likewise the
last command. Both are tracked internally; the only external input is the
one-time ``q_init`` seed handed in at construction (the arm's joint
configuration when the teleop is attached).

``JointMappedKinematics`` adapts a CAD-exported URDF whose joint-zero does
not match the motor calibration zero (see :mod:`joint_alignment`); it lets
the controller work entirely in motor-degree space while pinocchio runs in
URDF space. The upstream kinematic ProcessorSteps are not used or modified.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from lerobot.utils.rotation import Rotation

from .joint_alignment import MOTOR_NAMES, URDF_JOINT_NAMES, URDF_TIP_FRAME, JointAlignment

# Reachable EE box for the SO-107, robot base frame, meters. Hand-tuned
# estimates from teleop trials, NOT measured — IK targets are clipped here
# so an out-of-reach Quest command can't drive the solver into garbage.
# TODO(so107): replace with a measured/calibrated workspace once the
# guided-calibration tool lands (see gui/TODO.md).
SO107_WORKSPACE_MIN: tuple[float, float, float] = (-0.20, -0.35, +0.03)
SO107_WORKSPACE_MAX: tuple[float, float, float] = (+0.25, +0.05, +0.36)

# Backstop cap on per-tick EE position change (m). The Quest teleop already
# bounds and de-glitches its own deltas; this only catches anything past that.
_MAX_EE_STEP_M: float = 0.10


class _Kinematics(Protocol):
    """Minimal FK/IK surface the controller needs (degrees in, degrees out)."""

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray: ...

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray: ...


class JointMappedKinematics:
    """Wrap a URDF-space kinematics so callers can work in motor-degree space.

    The SO-107 URDF (CAD-exported) has per-joint ``(sign, offset_deg)``
    relative to the motors' calibrated zero — see :mod:`joint_alignment`.
    This wrapper applies ``urdf = sign * motor + offset`` on the way into
    the inner solver and the inverse on the way out, so the upstream IK
    code never needs a ``joint_map`` parameter and the upstream kinematic
    processors stay unmodified.
    """

    def __init__(
        self,
        inner: _Kinematics,
        motor_names: list[str],
        alignment: dict[str, JointAlignment],
    ) -> None:
        assert all(m in alignment for m in motor_names), "alignment missing a motor"
        self._inner = inner
        self._sign = np.array([alignment[m].sign for m in motor_names], dtype=float)
        self._offset = np.array([alignment[m].offset_deg for m in motor_names], dtype=float)
        assert np.all(np.abs(self._sign) == 1.0), "joint-alignment sign must be +/-1"

    def _motor_to_urdf(self, q_motor: np.ndarray) -> np.ndarray:
        return self._sign * np.asarray(q_motor, dtype=float) + self._offset

    def _urdf_to_motor(self, q_urdf: np.ndarray) -> np.ndarray:
        return (np.asarray(q_urdf, dtype=float) - self._offset) / self._sign

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Return the 4x4 SE(3) base->EE transform for motor-space joints (degrees)."""
        return self._inner.forward_kinematics(self._motor_to_urdf(joint_pos_deg))

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Solve IK for a 4x4 SE(3) ``target``; seed and result are motor-space degrees."""
        q_urdf = self._inner.inverse_kinematics(self._motor_to_urdf(seed_deg), target)
        return self._urdf_to_motor(q_urdf)


class CartesianIKController:
    """Stateful EE-delta -> motor-joint transform for a single arm.

    Call once per tick with the eight EE-delta keys a Quest controller
    emits; returns a ``{motor}.pos`` dict in motor space.

    Preconditions:
        * ``q_init`` has one entry per ``motor_names`` entry, motor-space
          degrees, and ``motor_names`` contains ``"gripper"``.
        * Each call's ``action`` has keys ``enabled``, ``target_x/y/z``,
          ``target_wx/wy/wz``, ``gripper_pos`` (unprefixed).

    Postconditions:
        * Returns ``{name}.pos`` for every name in ``motor_names``.
        * While ``enabled`` is false the arm joints hold their last
          commanded values; ``gripper.pos`` still follows ``gripper_pos``
          (the trigger works while the clutch is released).
        * On the rising edge of ``enabled`` the reference pose is latched
          to ``FK(last commanded joints)``; targets are composed onto it.
        * The commanded EE position is clipped to the workspace box and
          its per-tick change capped, so IK never sees a wild target.
    """

    def __init__(
        self,
        *,
        kinematics: _Kinematics,
        motor_names: list[str],
        q_init: np.ndarray,
        workspace_min: tuple[float, float, float],
        workspace_max: tuple[float, float, float],
        max_ee_step_m: float = _MAX_EE_STEP_M,
    ) -> None:
        assert "gripper" in motor_names, "motor_names must contain 'gripper'"
        assert len(q_init) == len(motor_names), "q_init length must match motor_names"
        self._kin = kinematics
        self._motor_names = list(motor_names)
        self._gripper_idx = self._motor_names.index("gripper")
        self._q_last = np.asarray(q_init, dtype=float).copy()
        self._ws_min = np.asarray(workspace_min, dtype=float)
        self._ws_max = np.asarray(workspace_max, dtype=float)
        self._max_step = float(max_ee_step_m)
        self._prev_enabled = False
        self._ref: np.ndarray | None = None  # latched reference pose (4x4)
        self._last_pos: np.ndarray | None = None  # last commanded EE position

    def __call__(self, action: dict[str, Any]) -> dict[str, float]:
        enabled = bool(action["enabled"])
        gripper_pos = float(action["gripper_pos"])
        # The gripper is not IK-tracked: the teleop emits an absolute
        # motor-space target and it works whether or not the clutch is
        # engaged. Keep it in q_last so the seed stays a faithful command.
        self._q_last[self._gripper_idx] = gripper_pos

        if enabled:
            if not self._prev_enabled or self._ref is None:
                self._ref = self._kin.forward_kinematics(self._q_last)
                self._last_pos = self._ref[:3, 3].copy()

            delta_p = np.array(
                [float(action["target_x"]), float(action["target_y"]), float(action["target_z"])],
                dtype=float,
            )
            r_delta = Rotation.from_rotvec(
                [float(action["target_wx"]), float(action["target_wy"]), float(action["target_wz"])]
            ).as_matrix()

            desired = np.eye(4, dtype=float)
            desired[:3, :3] = self._ref[:3, :3] @ r_delta
            pos = self._ref[:3, 3] + delta_p
            pos = np.clip(pos, self._ws_min, self._ws_max)

            assert self._last_pos is not None
            step = pos - self._last_pos
            n = float(np.linalg.norm(step))
            if n > self._max_step > 0.0:
                pos = self._last_pos + step * (self._max_step / n)
            desired[:3, 3] = pos
            self._last_pos = pos.copy()

            q_new = self._kin.inverse_kinematics(self._q_last, desired)
            q_new[self._gripper_idx] = gripper_pos
            self._q_last = np.asarray(q_new, dtype=float)

        self._prev_enabled = enabled
        return {f"{name}.pos": float(self._q_last[i]) for i, name in enumerate(self._motor_names)}


def make_bimanual_ik_transform(
    left: Callable[[dict], dict],
    right: Callable[[dict], dict],
) -> Callable[[dict], dict]:
    """Build a ``left_``/``right_`` split-merge transform over two arm controllers.

    Returns a ``dict -> dict`` callable: it takes a bimanual EE-delta action
    (every key prefixed ``left_`` or ``right_``), routes each arm's
    deprefixed slice to that arm's controller, and merges the two
    ``{motor}.pos`` results back under the prefixes — the joint dict a
    bimanual follower's ``send_action`` consumes. A robot installs the
    result into a Cartesian teleop via ``set_action_transform``.
    """

    def _transform(ee_action: dict) -> dict:
        left_in = {k.removeprefix("left_"): v for k, v in ee_action.items() if k.startswith("left_")}
        right_in = {k.removeprefix("right_"): v for k, v in ee_action.items() if k.startswith("right_")}
        return {f"left_{k}": v for k, v in left(left_in).items()} | {
            f"right_{k}": v for k, v in right(right_in).items()
        }

    return _transform


def make_so107_arm_ik_controller(
    alignment: dict[str, JointAlignment],
    q_init: np.ndarray,
) -> CartesianIKController:
    """Build a Cartesian-IK controller for one SO-107 arm.

    Args:
        alignment: This arm's motor->URDF alignment (``LEFT_ARM_ALIGNMENT``
            or ``RIGHT_ARM_ALIGNMENT`` from :mod:`joint_alignment`).
        q_init: The arm's current joint configuration, motor-space degrees,
            in :data:`joint_alignment.MOTOR_NAMES` order.

    Returns:
        A controller seeded at ``q_init``. Building it parses the SO-107
        URDF into a pinocchio model (~1-2 s); do this once at teleop
        attach time, not per tick. Requires the optional ``pin-pink``
        dependency (raises ``ImportError`` otherwise).
    """
    from lerobot.model.pink_kinematics import PinkKinematics

    from . import get_urdf_path

    inner = PinkKinematics(
        urdf_path=str(get_urdf_path()),
        target_frame_name=URDF_TIP_FRAME,
        joint_names=list(URDF_JOINT_NAMES),
    )
    kinematics = JointMappedKinematics(inner, list(MOTOR_NAMES), alignment)
    return CartesianIKController(
        kinematics=kinematics,
        motor_names=list(MOTOR_NAMES),
        q_init=q_init,
        workspace_min=SO107_WORKSPACE_MIN,
        workspace_max=SO107_WORKSPACE_MAX,
    )
