"""Adapt full robot kinematics to the joints a spatial controller commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np


class FullKinematics(Protocol):
    def forward_kinematics(self, joints_deg: np.ndarray) -> np.ndarray: ...

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray: ...


class SelectedJointKinematics:
    """Expose selected controlled joints while holding downstream passive joints fixed.

    For the SO-107 virtual EE on L6_1, gripper S7 is downstream of the EE,
    so omitting it from arm IK is geometrically exact. The caller must make
    the same independence argument for any other excluded joint.
    """

    def __init__(
        self,
        inner: FullKinematics,
        full_joint_names: tuple[str, ...],
        controlled_joint_names: tuple[str, ...],
        fixed_joint_deg: Mapping[str, float],
    ) -> None:
        if len(set(full_joint_names)) != len(full_joint_names) or not full_joint_names:
            raise ValueError("full joint names must be unique and nonempty")
        if len(set(controlled_joint_names)) != len(controlled_joint_names) or not controlled_joint_names:
            raise ValueError("controlled joint names must be unique and nonempty")
        if not set(controlled_joint_names) <= set(full_joint_names):
            raise ValueError("controlled joints must be part of the full model")
        if set(fixed_joint_deg) != set(full_joint_names) - set(controlled_joint_names):
            raise ValueError("each excluded joint needs exactly one fixed value")
        if any(not np.isfinite(value) for value in fixed_joint_deg.values()):
            raise ValueError("fixed joint values must be finite")
        self.inner = inner
        self._full_names = full_joint_names
        self._controlled_names = controlled_joint_names
        self._controlled_indices = np.array(
            [full_joint_names.index(name) for name in controlled_joint_names], dtype=int
        )
        self._fixed = np.array([float(fixed_joint_deg.get(name, 0.0)) for name in full_joint_names])

    def _full(self, controlled_deg: np.ndarray) -> np.ndarray:
        values = np.asarray(controlled_deg, dtype=float)
        if values.shape != (len(self._controlled_indices),) or not np.isfinite(values).all():
            raise ValueError("invalid controlled joint vector")
        full = self._fixed.copy()
        full[self._controlled_indices] = values
        return full

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        return self.inner.forward_kinematics(self._full(joint_pos_deg))

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray:
        solution = np.asarray(self.inner.inverse_kinematics(self._full(seed_deg), target), dtype=float)
        if solution.shape != self._fixed.shape or not np.isfinite(solution).all():
            raise ValueError("full IK returned an invalid joint vector")
        return solution[self._controlled_indices]
