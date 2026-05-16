"""
ikpy-based forward/inverse kinematics for SO-107.

ikpy uses scipy.optimize.minimize (SLSQP) under the hood — iterative numerical
IK that handles redundancy and singularities much better than placo's single-step
solve. Used here for teleop where we want each tick's IK call to produce motor
commands that trace a straight line in EE space (not a curved one as the
table-lookup approach did).

Same interface as So107Kinematics so it can drop in:
    fk_from_motors(motor_pos)        -> 4x4 SE(3)
    ik_to_motors(current_pos, target_T, ...) -> {motor_name: deg}
"""

from __future__ import annotations

import warnings

import numpy as np

# ikpy loads URDFs that have <inertia> elements with potential edge cases.
# Suppress the noisy warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ikpy.chain import Chain

import contextlib

from . import get_urdf_path
from .kinematics import (
    URDF_JOINT_NAMES,
    JointMap,
    motor_pos_to_urdf_q,
    urdf_q_to_motor_pos,
)

URDF_TIP_FRAME = "L7_1"


class So107IkpyKinematics:
    """Forward/inverse kinematics via ikpy. Drop-in for So107Kinematics."""

    def __init__(self, joint_map: dict[str, JointMap]):
        self.joint_map = joint_map
        # ikpy expects an active_links_mask matching its parsed chain.
        # When loading our URDF starting from BASE_1, ikpy creates 1 "Base link"
        # + 7 joints (S1..S7). The base is fixed, others are active.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.chain = Chain.from_urdf_file(
                str(get_urdf_path()),
                base_elements=["BASE_1"],
                active_links_mask=[False] + [True] * 7,
            )
        # Position of S1..S7 in chain.links (chain order).
        self._joint_link_indices = [
            i for i, link in enumerate(self.chain.links) if link.name in URDF_JOINT_NAMES
        ]
        assert len(self._joint_link_indices) == 7, (
            f"expected 7 active joints in chain, got {len(self._joint_link_indices)}"
        )
        # Expand parsed URDF joint limits to ±2π. The URDF's ±π is a placeholder
        # from the CAD exporter and doesn't reflect actual mechanical range; with
        # the user's bridge offsets, calibrated motor positions can map to URDF
        # angles past ±π, causing ikpy to reject the initial guess. Expanding to
        # ±2π is safe because the real safety bound is the calibrated joint
        # range, enforced separately in the teleop slew clamp.
        WIDE = (-2 * np.pi, 2 * np.pi)
        for idx in self._joint_link_indices:
            link = self.chain.links[idx]
            with contextlib.suppress(AttributeError):
                link.bounds = WIDE

    def _q_full(self, q7: np.ndarray) -> np.ndarray:
        """Embed 7-vector of S1..S7 angles (rad) into the full chain vector."""
        q = np.zeros(len(self.chain.links))
        for i, idx in enumerate(self._joint_link_indices):
            q[idx] = q7[i]
        return q

    def _q7_from_full(self, q_full: np.ndarray) -> np.ndarray:
        return np.array([q_full[idx] for idx in self._joint_link_indices])

    def fk_from_motors(self, motor_pos: dict[str, float]) -> np.ndarray:
        """Forward kinematics: motor degrees -> 4x4 SE(3) of tip in base frame."""
        q7 = motor_pos_to_urdf_q(motor_pos, self.joint_map)
        T = self.chain.forward_kinematics(self._q_full(q7))
        return T

    def ik_to_motors(
        self,
        current_motor_pos: dict[str, float],
        target_pose: np.ndarray,
        orientation_mode: str | None = None,
    ) -> tuple[dict[str, float], float]:
        """Inverse kinematics. Returns (motor_pos_dict_in_deg, position_error_mm).

        orientation_mode:
            None     -> ignore orientation, fit position only (recommended for teleop)
            "all"    -> match position + full 3D orientation
            "X"/"Y"/"Z" -> match position + one axis orientation
        """
        q_init_7 = motor_pos_to_urdf_q(current_motor_pos, self.joint_map)
        q_init = self._q_full(q_init_7)

        if orientation_mode is None:
            # Position-only IK.
            q_full = self.chain.inverse_kinematics(
                target_position=target_pose[:3, 3],
                initial_position=q_init,
            )
        else:
            # Full frame IK.
            q_full = self.chain.inverse_kinematics_frame(
                target_pose,
                initial_position=q_init,
                orientation_mode=orientation_mode,
            )

        q7 = self._q7_from_full(q_full)
        motor_out = urdf_q_to_motor_pos(q7, self.joint_map)

        # Position error after solve, for the caller to monitor.
        T_achieved = self.chain.forward_kinematics(q_full)
        err_mm = float(np.linalg.norm(T_achieved[:3, 3] - target_pose[:3, 3])) * 1000
        return motor_out, err_mm
