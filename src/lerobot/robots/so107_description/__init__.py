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
    except ImportError:
        return
    urdf = str(get_urdf_path())
    joint_names = [f"S{i}" for i in range(1, 8)]
    # Workspace derived from the training data extent for the right arm.
    right_workspace_min = (-0.20, -0.35, +0.03)
    right_workspace_max = (+0.25, +0.05, +0.36)
    # The left arm mirrors the right across the robot's sagittal plane (the
    # URDF X axis): swap and negate Y bounds, keep X and Z. The bimanual
    # follower uses the same URDF for both arms so the kinematics are correct
    # without modification; only the reachable workspace differs.
    left_workspace_min = (right_workspace_min[0], -right_workspace_max[1], right_workspace_min[2])
    left_workspace_max = (right_workspace_max[0], -right_workspace_min[1], right_workspace_max[2])

    # Unimanual (single-arm) variants share one config (URDF, motor layout,
    # workspace are all identical; the predictive controller only changes the
    # control rate, not the kinematic / action-key contract).
    uni_cfg = CartesianIKRobotConfig(
        urdf_path=urdf,
        ee_frame_name="L7_1",
        motor_names=_SO107_MOTOR_NAMES,
        joint_names=joint_names,
        workspace_min=right_workspace_min,
        workspace_max=right_workspace_max,
        end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
        max_ee_step_m=0.10,
        gripper_speed_factor=20.0,
    )
    register_cartesian_ik_robot("so107_follower", uni_cfg)
    register_cartesian_ik_robot("so107_follower_predictive", uni_cfg)

    # Bimanual variants: same per-arm config, but two arms with left_/right_
    # prefixes and mirrored workspace boxes.
    def _arm(prefix: str, wmin: tuple, wmax: tuple) -> CartesianIKArmConfig:
        return CartesianIKArmConfig(
            urdf_path=urdf,
            ee_frame_name="L7_1",
            motor_names=_SO107_MOTOR_NAMES,
            joint_names=joint_names,
            key_prefix=prefix,
            workspace_min=wmin,
            workspace_max=wmax,
            end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
            max_ee_step_m=0.10,
            gripper_speed_factor=20.0,
        )

    bi_cfg = CartesianIKRobotConfig(
        arms=[
            _arm("left_", left_workspace_min, left_workspace_max),
            _arm("right_", right_workspace_min, right_workspace_max),
        ]
    )
    register_cartesian_ik_robot("bi_so107_follower", bi_cfg)
    register_cartesian_ik_robot("bi_so107_follower_predictive", bi_cfg)


_register_cartesian_ik()


__all__ = ["get_urdf_path", "get_meshes_dir"]
