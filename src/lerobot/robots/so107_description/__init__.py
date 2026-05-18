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
            CartesianIKRobotConfig,
            register_cartesian_ik_robot,
        )
    except ImportError:
        return
    cfg = CartesianIKRobotConfig(
        urdf_path=str(get_urdf_path()),
        ee_frame_name="L7_1",
        motor_names=_SO107_MOTOR_NAMES,
        joint_names=[f"S{i}" for i in range(1, 8)],
        # Workspace derived from the training data extent for the right arm.
        workspace_min=(-0.20, -0.35, +0.03),
        workspace_max=(+0.25, +0.05, +0.36),
        end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
        max_ee_step_m=0.10,
        gripper_speed_factor=20.0,
    )
    # Same hardware + URDF + motor layout for the predictive variant; the
    # predictive controller wraps send_action with the same <motor>.pos contract.
    register_cartesian_ik_robot("so107_follower", cfg)
    register_cartesian_ik_robot("so107_follower_predictive", cfg)


_register_cartesian_ik()


__all__ = ["get_urdf_path", "get_meshes_dir"]
