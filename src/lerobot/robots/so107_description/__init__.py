"""SO-107 robot description (URDF + meshes) for kinematics and visualization."""

from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def get_urdf_path() -> Path:
    """Absolute path to the SO-107 URDF file."""
    p = _PKG_DIR / "urdf" / "SO107.urdf"
    assert p.exists(), f"SO-107 URDF missing at {p}"
    return p


def get_meshes_dir() -> Path:
    """Absolute path to the directory containing the SO-107 link meshes."""
    p = _PKG_DIR / "meshes"
    assert p.is_dir(), f"SO-107 meshes dir missing at {p}"
    return p


# Register Cartesian IK config for the so107_follower so that
# lerobot-teleoperate can auto-compose the Cartesian -> joints pipeline
# when a Cartesian teleop (quest_vr, keyboard_ee, phone) drives this robot.
# Done at import time so the GUI's auto-discovery
# (`gui/api/robot.py:_ensure_configs_loaded`) picks it up automatically.
_SO107_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _register_cartesian_ik() -> None:
    try:
        from lerobot.processor.cartesian_ik_pipeline import (
            CartesianIKArmConfig,
            CartesianIKRobotConfig,
            register_cartesian_ik_robot,
        )

        from .kinematics import LEFT_ARM_MAP, RIGHT_ARM_MAP
    except ImportError:
        return
    urdf = str(get_urdf_path())
    joint_names = [f"S{i}" for i in range(1, 8)]
    # Workspace derived from the training data extent.
    # Both arms are mounted PARALLEL (same forward direction) and share one
    # URDF, so in each arm's own URDF frame the reachable box is identical.
    # Don't mirror — mirroring forces the EE-bounds clipper to slam the
    # commanded pose into the opposite hemisphere on engage, which is the
    # most likely cause of the "90 deg off" symptom we hit on first hardware
    # test. If you ever physically rotate one arm 180 deg, supply a flipped
    # URDF for that arm instead of trying to mirror the workspace here.
    workspace_min = (-0.20, -0.35, +0.03)
    workspace_max = (+0.25, +0.05, +0.36)

    # Unimanual (single-arm) variants share one config (URDF, motor layout,
    # workspace are all identical; the predictive controller only changes the
    # control rate, not the kinematic / action-key contract).
    # joint_map=RIGHT_ARM_MAP: the existing unimanual SO-107 setup is "the right
    # arm" (its calibration is what RIGHT_ARM_MAP was originally discovered for).
    uni_cfg = CartesianIKRobotConfig(
        urdf_path=urdf,
        ee_frame_name="L7_1",
        motor_names=_SO107_MOTOR_NAMES,
        joint_names=joint_names,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
        max_ee_step_m=0.10,
        joint_map=RIGHT_ARM_MAP,
    )
    register_cartesian_ik_robot("so107_follower", uni_cfg)
    register_cartesian_ik_robot("so107_follower_predictive", uni_cfg)

    # Bimanual variants: parallel mounting (both arms facing the same direction
    # as the user's teleop reference) -> identical per-arm workspaces and yaw=0
    # for both. If you ever mount the arms mirrored (180 deg around Z), bump
    # the left arm's world_yaw_deg to 180.0 (or build a different bi_cfg).
    # Each arm has its own joint_map (motor->URDF sign+offset) because
    # the physical mounting / calibration zero differs between sides.
    def _arm(prefix: str, joint_map, world_yaw_deg: float = 0.0) -> CartesianIKArmConfig:
        return CartesianIKArmConfig(
            urdf_path=urdf,
            ee_frame_name="L7_1",
            motor_names=_SO107_MOTOR_NAMES,
            joint_names=joint_names,
            key_prefix=prefix,
            world_yaw_deg=world_yaw_deg,
            joint_map=joint_map,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
            max_ee_step_m=0.10,
        )

    bi_cfg = CartesianIKRobotConfig(
        arms=[
            _arm("left_", LEFT_ARM_MAP, world_yaw_deg=0.0),
            _arm("right_", RIGHT_ARM_MAP, world_yaw_deg=0.0),
        ],
    )
    register_cartesian_ik_robot("bi_so107_follower", bi_cfg)
    register_cartesian_ik_robot("bi_so107_follower_predictive", bi_cfg)


_register_cartesian_ik()


__all__ = ["get_urdf_path", "get_meshes_dir"]
