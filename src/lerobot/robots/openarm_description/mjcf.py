#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Optional OpenArm MJCF helpers.

The physical controller does not depend on this module unless gravity
feed-forward is explicitly enabled. Cartesian IK intentionally has no URDF
fallback: it needs a verified OpenArm MJCF model, frame names and coordinate
convention before it can safely produce hardware commands.
"""

import importlib
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lerobot.robots.so107_description.cartesian_ik import (
    CartesianIKController,
    _NoSolutionFound,
)
from lerobot.utils.rotation import Rotation

OPENARM_JOINT_NAMES = {
    side: tuple(f"openarm_{side}_joint{index}" for index in range(1, 8)) for side in ("left", "right")
}
OPENARM_EE_SITES = {
    "left": "left_ee_control_point",
    "right": "right_ee_control_point",
}
OPENARM_MOTOR_NAMES = tuple(f"joint_{index}" for index in range(1, 8)) + ("gripper",)
OPENARM_CARTESIAN_ACTION_KEYS = (
    "enabled",
    "reset",
    "target_x",
    "target_y",
    "target_z",
    "target_wx",
    "target_wy",
    "target_wz",
    "gripper_pos",
)

# Conservative Cartesian bounds in the OpenArm base frame. They contain the
# natural table-top workspace while keeping an erroneous target away from
# obviously unreachable poses. The motor-side absolute and relative limits
# remain the final safety boundary.
OPENARM_WORKSPACE_MIN = (-0.20, -0.45, -0.55)
OPENARM_WORKSPACE_MAX = (+0.55, +0.45, +0.25)


class MJCFCartesianSolver(Protocol):
    """Interface for a future, model-verified Cartesian solver.

    Implementations consume and return joint angles in radians. Keeping this
    interface independent of a particular IK package lets the hardware layer
    remain on MJCF without silently falling back to an unrelated URDF model.
    """

    def solve(self, current_q: Sequence[float], target_pose: Sequence[float]) -> np.ndarray: ...


class MJCFArmKinematics:
    """Motor-degree FK/IK for one arm in the verified bimanual MJCF.

    The full model is loaded so the arm base transform and end-effector site
    exactly match ``openarm_bimanual.xml``. Only the selected arm's seven
    joints participate in IK; the gripper is carried through unchanged by the
    Cartesian controller.
    """

    def __init__(
        self,
        side: str,
        *,
        xml: str | Path | None = None,
        max_iterations: int = 80,
        damping: float = 0.05,
        max_joint_step_rad: float = 0.15,
        position_tolerance_m: float = 2e-4,
        orientation_tolerance_rad: float = 2e-3,
    ) -> None:
        if side not in OPENARM_JOINT_NAMES:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        positive = {
            "damping": damping,
            "max_joint_step_rad": max_joint_step_rad,
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": orientation_tolerance_rad,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive.values()):
            raise ValueError(f"IK parameters must be finite and positive: {positive}")

        try:
            self._mujoco = importlib.import_module("mujoco")
        except ImportError as exc:
            raise ImportError("OpenArm MJCF Cartesian IK requires the mujoco package.") from exc

        self.side = side
        self.model_path = resolve_openarm_bimanual_mjcf(xml)
        self._model = self._mujoco.MjModel.from_xml_path(str(self.model_path))
        self._data = self._mujoco.MjData(self._model)
        self.max_iterations = max_iterations
        self.damping = float(damping)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)

        self._joint_ids = tuple(
            int(self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_JOINT, name))
            for name in OPENARM_JOINT_NAMES[side]
        )
        missing = [
            name
            for name, joint_id in zip(OPENARM_JOINT_NAMES[side], self._joint_ids, strict=True)
            if joint_id < 0
        ]
        if missing:
            raise ValueError(f"MJCF is missing OpenArm joints: {missing}")
        self._site_id = int(
            self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_SITE, OPENARM_EE_SITES[side])
        )
        if self._site_id < 0:
            raise ValueError(f"MJCF is missing OpenArm end-effector site: {OPENARM_EE_SITES[side]}")
        self._dofs = np.asarray([int(self._model.jnt_dofadr[joint_id]) for joint_id in self._joint_ids])
        self._qpos_addresses = np.asarray(
            [int(self._model.jnt_qposadr[joint_id]) for joint_id in self._joint_ids]
        )
        ranges = np.asarray([self._model.jnt_range[joint_id] for joint_id in self._joint_ids], dtype=float)
        self._lower = ranges[:, 0]
        self._upper = ranges[:, 1]

    @staticmethod
    def _validated_motor_degrees(joint_pos_deg: Sequence[float]) -> np.ndarray:
        q = np.asarray(joint_pos_deg, dtype=float)
        if q.shape not in ((7,), (8,)) or not np.all(np.isfinite(q)):
            raise ValueError("OpenArm joints must contain seven finite arm values and an optional gripper")
        return q.copy()

    @staticmethod
    def _validated_target(target: np.ndarray) -> np.ndarray:
        pose = np.asarray(target, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("Cartesian target must be a finite 4x4 transform")
        if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError("Cartesian target must be a homogeneous transform")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.0:
            raise ValueError("Cartesian target rotation must be orthonormal and right-handed")
        return pose

    def _set_arm_qpos(self, q_rad: np.ndarray) -> None:
        self._data.qpos[:] = 0.0
        self._data.qvel[:] = 0.0
        self._data.qacc[:] = 0.0
        self._data.qpos[self._qpos_addresses] = q_rad
        self._mujoco.mj_forward(self._model, self._data)

    def _site_pose(self) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.asarray(self._data.site_xpos[self._site_id], dtype=float)
        pose[:3, :3] = np.asarray(self._data.site_xmat[self._site_id], dtype=float).reshape(3, 3)
        return pose

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        q_deg = self._validated_motor_degrees(joint_pos_deg)
        self._set_arm_qpos(np.deg2rad(q_deg[:7]))
        pose = self._site_pose()
        if not np.all(np.isfinite(pose)):
            raise RuntimeError("MJCF forward kinematics produced a non-finite pose")
        return pose

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray:
        seed = self._validated_motor_degrees(seed_deg)
        target_pose = self._validated_target(target)
        q_rad = np.clip(np.deg2rad(seed[:7]), self._lower, self._upper)
        jacp = np.zeros((3, self._model.nv), dtype=float)
        jacr = np.zeros((3, self._model.nv), dtype=float)
        damping_eye = (self.damping * self.damping) * np.eye(6, dtype=float)

        for _ in range(self.max_iterations):
            self._set_arm_qpos(q_rad)
            current = self._site_pose()
            position_error = target_pose[:3, 3] - current[:3, 3]
            rotation_error = Rotation.from_matrix(target_pose[:3, :3] @ current[:3, :3].T).as_rotvec()
            if not np.all(np.isfinite(position_error)) or not np.all(np.isfinite(rotation_error)):
                raise _NoSolutionFound("OpenArm MJCF IK produced a non-finite Cartesian error")
            if (
                float(np.linalg.norm(position_error)) <= self.position_tolerance_m
                and float(np.linalg.norm(rotation_error)) <= self.orientation_tolerance_rad
            ):
                result = seed.copy()
                result[:7] = np.rad2deg(q_rad)
                return result

            jacp.fill(0.0)
            jacr.fill(0.0)
            self._mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site_id)
            jacobian = np.vstack((jacp[:, self._dofs], jacr[:, self._dofs]))
            error = np.concatenate((position_error, rotation_error))
            try:
                dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping_eye, error)
            except np.linalg.LinAlgError as exc:
                raise _NoSolutionFound("OpenArm MJCF IK Jacobian solve failed") from exc
            if not np.all(np.isfinite(dq)):
                raise _NoSolutionFound("OpenArm MJCF IK produced a non-finite joint step")
            largest = float(np.max(np.abs(dq)))
            if largest > self.max_joint_step_rad:
                dq *= self.max_joint_step_rad / largest
            q_rad = np.clip(q_rad + dq, self._lower, self._upper)

        raise _NoSolutionFound("OpenArm MJCF IK did not converge")


def is_openarm_bimanual_cartesian_teleop(teleop: Any) -> bool:
    """Return whether a teleop advertises the bimanual Cartesian contract."""
    try:
        names = teleop.action_features.get("names", {})
    except (AttributeError, TypeError):
        return False
    return all(f"{side}_{key}" in names for side in ("left", "right") for key in OPENARM_CARTESIAN_ACTION_KEYS)


class BimanualOpenArmMJCFIKTransform:
    """Strict Quest Cartesian action to 16 OpenArm motor positions."""

    def __init__(self, left: CartesianIKController, right: CartesianIKController) -> None:
        self.left = left
        self.right = right
        self._expected_input = {
            f"{side}_{key}" for side in ("left", "right") for key in OPENARM_CARTESIAN_ACTION_KEYS
        }
        self._expected_output = {
            f"{side}_{motor}.pos" for side in ("left", "right") for motor in OPENARM_MOTOR_NAMES
        }

    def __call__(self, action: Mapping[str, Any]) -> dict[str, float]:
        received = set(action)
        if received != self._expected_input:
            raise ValueError(
                "OpenArm Cartesian action keys mismatch: "
                f"missing={sorted(self._expected_input - received)}, "
                f"unknown={sorted(received - self._expected_input)}"
            )
        values = {key: float(value) for key, value in action.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("OpenArm Cartesian action values must all be finite")

        left_in = {key.removeprefix("left_"): value for key, value in values.items() if key.startswith("left_")}
        right_in = {
            key.removeprefix("right_"): value for key, value in values.items() if key.startswith("right_")
        }
        output = {f"left_{key}": value for key, value in self.left(left_in).items()} | {
            f"right_{key}": value for key, value in self.right(right_in).items()
        }
        if set(output) != self._expected_output or not all(math.isfinite(float(value)) for value in output.values()):
            raise RuntimeError("OpenArm MJCF IK did not produce exactly 16 finite motor positions")
        return output

    @property
    def hold_per_arm(self) -> tuple[bool, bool]:
        return self.left.is_holding, self.right.is_holding


def build_openarm_bimanual_mjcf_ik_transform(
    kinematics: Mapping[str, Any],
    left_arm: Any,
    right_arm: Any,
) -> BimanualOpenArmMJCFIKTransform:
    """Seed and build the MJCF transform from fresh motor observations."""

    def seed(arm: Any) -> np.ndarray:
        observation = arm.get_observation()
        values = np.asarray([float(observation[f"{motor}.pos"]) for motor in OPENARM_MOTOR_NAMES])
        if values.shape != (8,) or not np.all(np.isfinite(values)):
            raise ValueError("OpenArm IK seed must contain eight finite motor positions")
        return values

    left = CartesianIKController(
        kinematics=kinematics["left"],
        motor_names=list(OPENARM_MOTOR_NAMES),
        q_init=seed(left_arm),
        workspace_min=OPENARM_WORKSPACE_MIN,
        workspace_max=OPENARM_WORKSPACE_MAX,
        label="left",
    )
    right = CartesianIKController(
        kinematics=kinematics["right"],
        motor_names=list(OPENARM_MOTOR_NAMES),
        q_init=seed(right_arm),
        workspace_min=OPENARM_WORKSPACE_MIN,
        workspace_max=OPENARM_WORKSPACE_MAX,
        label="right",
    )
    return BimanualOpenArmMJCFIKTransform(left, right)


def resolve_openarm_bimanual_mjcf(xml: str | Path | None = None) -> Path:
    if xml is not None:
        path = Path(xml).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OpenArm MJCF model does not exist: {path}")
        return path

    try:
        model_module = importlib.import_module("openarm_mujoco.v2")
    except ImportError as exc:
        raise ImportError(
            "MJCF support requires the openarm-mujoco package or an explicit gravity_ff_xml path."
        ) from exc

    # A's locked VR dataflow pins openarm_bimanual.xml explicitly. Joint names
    # are still resolved rather than relying on fixed scene-wide DOF offsets.
    model_path = Path(model_module.openarm_bimanual_xml()).resolve()
    # ``pip --target`` is useful for an isolated hardware-test dependency
    # directory, but openarm-mujoco resolves data through the interpreter's
    # sysconfig prefix. Recover the matching target-local share directory
    # without searching outside the imported distribution.
    if not model_path.is_file() and getattr(model_module, "__file__", None):
        distribution_root = Path(model_module.__file__).resolve().parents[2]
        target_local = distribution_root / "share" / "openarm_mujoco" / "v2" / "openarm_bimanual.xml"
        if target_local.is_file():
            model_path = target_local
    if not model_path.is_file():
        raise FileNotFoundError(f"openarm-mujoco returned a missing MJCF model: {model_path}")
    return model_path


class MJCFGravityCompensator:
    """Conservative gravity torque computed from the OpenArm bimanual MJCF."""

    def __init__(
        self,
        side: str,
        *,
        xml: str | Path | None = None,
        gain: float = 0.0,
        torque_fraction: float = 0.25,
        fade_seconds: float = 2.0,
        low_pass_hz: float = 10.0,
    ) -> None:
        if side not in OPENARM_JOINT_NAMES:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if not 0.0 <= gain <= 1.0:
            raise ValueError(f"gain must be in [0, 1], got {gain}")
        if not 0.0 < torque_fraction <= 0.5:
            raise ValueError(f"torque_fraction must be in (0, 0.5], got {torque_fraction}")
        if fade_seconds < 0.0 or low_pass_hz <= 0.0:
            raise ValueError("fade_seconds must be non-negative and low_pass_hz must be positive")

        try:
            self._mujoco = importlib.import_module("mujoco")
        except ImportError as exc:
            raise ImportError("MJCF gravity feed-forward requires the mujoco package.") from exc

        self.side = side
        self.gain = gain
        self.fade_seconds = fade_seconds
        self.low_pass_hz = low_pass_hz
        self.model_path = resolve_openarm_bimanual_mjcf(xml)
        self._model = self._mujoco.MjModel.from_xml_path(str(self.model_path))
        self._data = self._mujoco.MjData(self._model)
        joint_ids = tuple(
            int(self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_JOINT, name))
            for name in OPENARM_JOINT_NAMES[side]
        )
        missing_names = [
            name for name, joint_id in zip(OPENARM_JOINT_NAMES[side], joint_ids, strict=True) if joint_id < 0
        ]
        if missing_names:
            raise ValueError(f"MJCF is missing OpenArm joints: {missing_names}")
        self._dofs = tuple(int(self._model.jnt_dofadr[joint_id]) for joint_id in joint_ids)
        self._qpos_addresses = tuple(int(self._model.jnt_qposadr[joint_id]) for joint_id in joint_ids)
        self._limits = torque_fraction * np.asarray(
            [abs(float(self._model.jnt_actfrcrange[joint_id][1])) for joint_id in joint_ids],
            dtype=np.float64,
        )
        self._filtered: np.ndarray | None = None
        self._enabled_at: float | None = None
        self._updated_at: float | None = None

    def raw_torque(self, joint_positions_rad: Sequence[float]) -> np.ndarray:
        q = np.asarray(joint_positions_rad, dtype=np.float64)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("joint_positions_rad must contain seven finite values")

        self._data.qpos[:] = 0.0
        self._data.qvel[:] = 0.0
        self._data.qacc[:] = 0.0
        for qpos_address, value in zip(self._qpos_addresses, q, strict=True):
            self._data.qpos[qpos_address] = value
        self._mujoco.mj_forward(self._model, self._data)
        torque = np.asarray([self._data.qfrc_bias[dof] for dof in self._dofs], dtype=np.float64)
        if not np.all(np.isfinite(torque)):
            raise RuntimeError("MJCF produced non-finite gravity torque")
        return torque

    def torque(self, joint_positions_rad: Sequence[float], *, now: float | None = None) -> np.ndarray:
        if self.gain == 0.0:
            return np.zeros(7, dtype=np.float64)

        now = time.monotonic() if now is None else now
        raw = self.raw_torque(joint_positions_rad)
        if self._enabled_at is None:
            self._enabled_at = now
            self._updated_at = now

        if self._filtered is None:
            self._filtered = raw
        else:
            dt = min(0.5, max(0.001, now - (self._updated_at or now)))
            rc = 1.0 / (2.0 * math.pi * self.low_pass_hz)
            alpha = dt / (dt + rc)
            self._filtered += alpha * (raw - self._filtered)
        self._updated_at = now

        fade = 1.0
        if self.fade_seconds > 0.0:
            fade = min(1.0, max(0.0, (now - self._enabled_at) / self.fade_seconds))
        return np.clip(self.gain * fade * self._filtered, -self._limits, self._limits)
