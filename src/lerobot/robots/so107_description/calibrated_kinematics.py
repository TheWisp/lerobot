"""Build SO-107 motor-space kinematics from a measured rig calibration."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from lerobot.model.pink_kinematics import PinkKinematics

from . import get_urdf_path
from .cartesian_ik import JointMappedKinematics, TipOffsetKinematics
from .joint_alignment import (
    LEFT_ARM_ALIGNMENT,
    MOTOR_NAMES,
    RIGHT_ARM_ALIGNMENT,
    URDF_ANCHOR_FRAME,
    URDF_JOINT_NAMES,
    JointAlignment,
)


def calibrated_motor_limits(calibration) -> dict[str, tuple[float, float]]:
    """Body-joint travel in motor degrees, from the connected arm's motor calibration."""
    limits = {}
    for name in MOTOR_NAMES[:-1]:
        joint = calibration[name]
        half_span = (joint.range_max - joint.range_min) * 360 / 4095 / 2
        if not np.isfinite(half_span) or half_span <= 0:
            raise ValueError(f"Invalid motor calibration range: {name}")
        limits[name] = (-half_span, half_span)
    return limits


class _CalibratedPinkKinematics(PinkKinematics):
    """Measured FK and explicitly supplied planning bounds on one private model.

    The CAD export's +/-pi coordinates are not measured motor travel limits.
    Unspecified joints remain unconstrained for offline geometric IK. Hardware
    callers must supply every controlled joint's calibrated motor bounds.
    """

    def __init__(self, solver_limits):
        super().__init__(str(get_urdf_path()), URDF_ANCHOR_FRAME, list(URDF_JOINT_NAMES), max_iters=50)
        model = self.robot.model
        model.lowerPositionLimit[:] = -np.inf
        model.upperPositionLimit[:] = np.inf
        for name, bounds in solver_limits.items():
            index = model.joints[model.getJointId(name)].idx_q
            model.lowerPositionLimit[index], model.upperPositionLimit[index] = np.deg2rad(bounds)
        self.q_lo = model.lowerPositionLimit.copy()
        self.q_hi = model.upperPositionLimit.copy()

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        import pinocchio as pin

        q = np.deg2rad(joint_pos_deg[: len(self.joint_names)])
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        fid = self.robot.model.getFrameId(self.target_frame_name)
        return self.robot.data.oMf[fid].homogeneous.copy()


def make_calibrated_so107_kinematics(
    calibration: Mapping, *, arm: str, joint_limits_deg: dict[str, tuple[float, float]] | None = None
):
    """Use the same joint-zero and anchor-to-tool transform as the recorded demo.

    Calibration is explicit rig data, never inferred from a task or object. A
    missing tool-point calibration is an error: the nominal URDF tip can be
    more than a grasp width from the actual contact point.

    ``joint_limits_deg`` constrains planning, never measured FK. Omitted bounds
    are for offline geometric use; hardware callers supply finite bounds for
    every controlled joint from their motor calibration.
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
    if set(joint_limits_deg or {}) - set(MOTOR_NAMES):
        raise ValueError("Unknown motor in joint limits")
    for motor, bounds in (joint_limits_deg or {}).items():
        if np.shape(bounds) != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid motor joint limits: {motor}")
    solver_limits = {
        urdf: tuple(sorted(corrected[motor].sign * value + corrected[motor].offset_deg for value in bounds))
        for motor, urdf in zip(MOTOR_NAMES, URDF_JOINT_NAMES, strict=True)
        if (bounds := (joint_limits_deg or {}).get(motor)) is not None
    }
    inner = _CalibratedPinkKinematics(solver_limits)
    return JointMappedKinematics(TipOffsetKinematics(inner, tip), list(MOTOR_NAMES), corrected)
