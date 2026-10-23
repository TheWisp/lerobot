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

"""Gravity feed-forward for the SO-107 in position mode.

The STS3215 has no torque input: its onboard loop takes a goal position and a
steady load leaves a steady error, ``droop = -alpha * tau_g``, where ``alpha``
is the servo's compliance at the gains in use. The compensation is the
"virtual displacement" used by every position-mode gravity scheme on these
arms: shift the goal by the droop the load will cause, so the joint settles on
the desired angle instead of below it.

``tau_g`` comes from pinocchio's generalized-gravity term on the vendored URDF
(masses set by link role; see the note at the top of ``SO107.urdf``). ``alpha``
is measured, not modelled — one number for all joints, since every joint is
the same servo at the same gain. It scales with 1/P.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .joint_alignment import MOTOR_NAMES, URDF_JOINT_NAMES, JointAlignment


class GravityFeedForward:
    """Goal offsets that pre-empt gravity droop, in motor degrees.

    Pre: ``alignment`` maps every motor in :data:`MOTOR_NAMES` to its
    ``(sign, offset_deg)``; ``alpha_deg_per_nm`` is the compliance measured at
    the servo gains the arm will run with.
    Post: :meth:`offset_deg` returns one offset per motor in motor-degree
    space, zero for the gripper, and scales linearly with ``alpha``.

    Requires the optional ``pin`` dependency (raises ``ImportError`` otherwise).
    """

    def __init__(self, alpha_deg_per_nm: float, alignment: Mapping[str, JointAlignment]) -> None:
        import pinocchio as pin

        from . import get_urdf_path

        assert alpha_deg_per_nm >= 0.0, "alpha is a compliance, never negative"
        missing = [m for m in MOTOR_NAMES if m not in alignment]
        assert not missing, f"alignment lacks {missing}"
        self.alpha = float(alpha_deg_per_nm)
        self._model = pin.buildModelFromUrdf(str(get_urdf_path()))
        self._data = self._model.createData()
        names = [self._model.names[i] for i in range(1, self._model.njoints)]
        assert tuple(names) == tuple(URDF_JOINT_NAMES), f"URDF joint order {names} != {URDF_JOINT_NAMES}"
        self._signs = np.array([alignment[m].sign for m in MOTOR_NAMES], dtype=float)
        self._offsets = np.array([alignment[m].offset_deg for m in MOTOR_NAMES], dtype=float)
        self._gripper = MOTOR_NAMES.index("gripper")
        self._pin = pin

    def gravity_torque_nm(self, q_motor_deg: np.ndarray) -> np.ndarray:
        """Generalized gravity about each motor axis, N*m, in :data:`MOTOR_NAMES` order."""
        q = np.asarray(q_motor_deg, dtype=float)
        assert q.shape == (len(MOTOR_NAMES),), q.shape
        q_urdf = np.deg2rad(self._signs * q + self._offsets)
        g = self._pin.computeGeneralizedGravity(self._model, self._data, q_urdf)
        # The alignment sign flips the axis, so a torque about the URDF joint
        # points the other way about the motor's own axis when sign is -1.
        return self._signs * np.asarray(g, dtype=float)

    def offset_deg(self, q_motor_deg: Mapping[str, float]) -> dict[str, float]:
        """Goal offsets for a desired pose. Post: keys are exactly the arm motors, gripper 0."""
        q = np.array([float(q_motor_deg[m]) for m in MOTOR_NAMES])
        out = self.alpha * self.gravity_torque_nm(q)
        out[self._gripper] = 0.0
        assert np.all(np.isfinite(out)), "non-finite gravity offset"
        return {m: float(out[i]) for i, m in enumerate(MOTOR_NAMES)}
