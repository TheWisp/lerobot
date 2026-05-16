"""
Bridge between SO-107 motor space (calibrated degrees per servo) and URDF joint
space (radians, ordered S1..S7), plus a thin kinematics wrapper.

Motor calibration (raw count -> calibrated degrees) lives on the FeetechMotorsBus
and is unchanged. This module handles the additional mapping needed because the
URDF's notion of "joint angle 0" doesn't match the motor's calibrated 0 in general.

Per-joint config (sign, offset_deg):
    urdf_angle_deg = sign * motor_angle_deg + offset_deg
    motor_angle_deg = (urdf_angle_deg - offset_deg) / sign

The starting values below are PLACEHOLDERS (identity). Replace them after running
the discovery helper (see motor_to_viewer.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.model.kinematics import RobotKinematics

from . import get_urdf_path

# Order matches the SO-107 follower motor list AND the URDF joint order S1..S7.
# If you ever change one, change the other.
MOTOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
URDF_JOINT_NAMES: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
URDF_TIP_FRAME: str = "L7_1"  # placeholder until a proper gripper_frame_link is added


@dataclass(frozen=True)
class JointMap:
    """Per-joint (sign, offset_deg). urdf_deg = sign * motor_deg + offset_deg."""

    sign: float
    offset_deg: float


# --- Per-arm mappings ---------------------------------------------------------
# Values are filled in empirically via `motor_to_viewer.py`.
RIGHT_ARM_MAP: dict[str, JointMap] = {
    "shoulder_pan": JointMap(sign=-1.0, offset_deg=+0.00),
    "shoulder_lift": JointMap(sign=+1.0, offset_deg=-90.00),
    "elbow_flex": JointMap(sign=+1.0, offset_deg=+78.87),
    "forearm_roll": JointMap(sign=-1.0, offset_deg=-3.71),
    "wrist_flex": JointMap(sign=-1.0, offset_deg=+0.00),
    "wrist_roll": JointMap(sign=-1.0, offset_deg=+90.00),
    "gripper": JointMap(sign=-1.0, offset_deg=+0.00),
}
LEFT_ARM_MAP: dict[str, JointMap] = {name: JointMap(sign=+1.0, offset_deg=0.0) for name in MOTOR_NAMES}


# --- Conversions --------------------------------------------------------------
def motor_pos_to_urdf_q(motor_pos: dict[str, float], joint_map: dict[str, JointMap]) -> np.ndarray:
    """Convert {motor_name: degrees} -> URDF joint vector in radians, ordered S1..S7."""
    q_deg = np.zeros(len(MOTOR_NAMES))
    for i, name in enumerate(MOTOR_NAMES):
        jm = joint_map[name]
        q_deg[i] = jm.sign * motor_pos[name] + jm.offset_deg
    return np.deg2rad(q_deg)


def urdf_q_to_motor_pos(q_rad: np.ndarray, joint_map: dict[str, JointMap]) -> dict[str, float]:
    """Convert URDF joint vector (radians) -> {motor_name: degrees}."""
    assert q_rad.shape == (len(MOTOR_NAMES),), f"expected shape ({len(MOTOR_NAMES)},), got {q_rad.shape}"
    q_deg = np.rad2deg(q_rad)
    out: dict[str, float] = {}
    for i, name in enumerate(MOTOR_NAMES):
        jm = joint_map[name]
        out[name] = (q_deg[i] - jm.offset_deg) / jm.sign
    return out


# --- Kinematics wrapper -------------------------------------------------------
class So107Kinematics:
    """Thin wrapper around the placo-based RobotKinematics for SO-107."""

    def __init__(self, joint_map: dict[str, JointMap] = RIGHT_ARM_MAP, tip_frame: str = URDF_TIP_FRAME):
        self.joint_map = joint_map
        self.kin = RobotKinematics(
            urdf_path=str(get_urdf_path()),
            target_frame_name=tip_frame,
            joint_names=list(URDF_JOINT_NAMES),
        )

    def fk_from_motors(self, motor_pos: dict[str, float]) -> np.ndarray:
        """Forward kinematics from motor degrees -> 4x4 tip pose in base frame."""
        q_rad = motor_pos_to_urdf_q(motor_pos, self.joint_map)
        # RobotKinematics expects degrees per its own API.
        return self.kin.forward_kinematics(np.rad2deg(q_rad))

    def ik_to_motors(
        self,
        current_motor_pos: dict[str, float],
        target_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        max_iters: int = 20,
        tol_deg: float = 0.01,
    ) -> dict[str, float]:
        """Inverse kinematics. Returns new motor targets (degrees) for the desired tip pose.

        Iterates placo's solve until joint changes drop below tol_deg, since a single
        placo `solve()` call is one Gauss-Newton step, not a full convergence.

        Default weights give equal priority to position and orientation. Lower
        orientation_weight (e.g. 0.01) frees the arm to spin around the position
        axis but tends to cause jitter / rotation under tight loops.
        """
        q_curr_rad = motor_pos_to_urdf_q(current_motor_pos, self.joint_map)
        q_iter_deg = np.rad2deg(q_curr_rad)
        for _ in range(max_iters):
            q_prev = q_iter_deg.copy()
            q_iter_deg = self.kin.inverse_kinematics(
                q_iter_deg,
                target_pose,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
            if np.max(np.abs(q_iter_deg - q_prev)) < tol_deg:
                break
        q_target_rad = np.deg2rad(q_iter_deg)
        return urdf_q_to_motor_pos(q_target_rad, self.joint_map)
