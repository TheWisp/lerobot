"""Build SO-107 motor-space kinematics from a measured rig calibration."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .cartesian_ik import make_so107_arm_kinematics
from .joint_alignment import LEFT_ARM_ALIGNMENT, RIGHT_ARM_ALIGNMENT, JointAlignment


def make_calibrated_so107_kinematics(calibration: Mapping, *, arm: str):
    """Use the same joint-zero and anchor-to-tool transform as the recorded demo.

    Calibration is explicit rig data, never inferred from a task or object. A
    missing tool-point calibration is an error: the nominal URDF tip can be
    more than a grasp width from the actual contact point.
    """
    if arm not in ("left", "right"):
        raise ValueError("SO-107 arm must be left or right")
    tool = calibration.get("tool_point")
    if not isinstance(tool, Mapping) or "offset_m" not in tool:
        raise ValueError("SO-107 FLASH execution needs a measured tool-point offset")
    offset = np.asarray(tool["offset_m"], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("invalid calibrated tool-point offset")
    zero = calibration.get("joint_zero_deg", {})
    if not isinstance(zero, Mapping):
        raise ValueError("joint-zero calibration must be a mapping")
    original = LEFT_ARM_ALIGNMENT if arm == "left" else RIGHT_ARM_ALIGNMENT
    if set(zero) - set(original) or any(not np.isfinite(float(value)) for value in zero.values()):
        raise ValueError("invalid joint-zero calibration")
    corrected = {
        name: JointAlignment(
            sign=joint.sign,
            offset_deg=joint.offset_deg + joint.sign * float(zero.get(name, 0.0)),
        )
        for name, joint in original.items()
    }
    tip = np.eye(4)
    tip[:3, 3] = offset
    return make_so107_arm_kinematics(corrected, tip_offset=tip)
