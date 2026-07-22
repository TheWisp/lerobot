#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Official OpenArm MJCF/Mink adapter for Quest Cartesian actions.

The public boundary remains LeRobot's bimanual dictionary contract. Internally
one ``openarm_control.Kinematics`` instance solves both arms in the shared
bimanual MJCF, matching the pinned dora control stack.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.rotation import Rotation

from .cartesian_ik import MOTOR_NAMES, OPENARM_WORKSPACE_MAX, OPENARM_WORKSPACE_MIN

_ACTION_KEYS = (
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
_MAX_EE_STEP_M = 0.10
_MAX_JOINT_STEP_DEG = 20.0
_RESET_RATE_DEG_S = 30.0
_RESET_DT_CAP_S = 0.1


def _pose7_to_matrix(pose: Sequence[float]) -> np.ndarray:
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("OpenArm FK pose must contain seven finite values")
    result = np.eye(4, dtype=float)
    result[:3, 3] = value[:3]
    # openarm_control: xyz + quaternion wxyz; Rotation: quaternion xyzw.
    result[:3, :3] = Rotation.from_quat([value[4], value[5], value[6], value[3]]).as_matrix()
    return result


def _matrix_to_pose7(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("OpenArm IK target must be a finite 4x4 transform")
    quat_xyzw = Rotation.from_matrix(value[:3, :3]).as_quat()
    return np.asarray(
        [*value[:3, 3], quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )


def _seed(arm: Any) -> np.ndarray:
    observation = arm.get_observation()
    values = np.asarray([float(observation[f"{motor}.pos"]) for motor in MOTOR_NAMES])
    if values.shape != (8,) or not np.all(np.isfinite(values)):
        raise ValueError("OpenArm IK seed must contain eight finite motor positions")
    return values


class BimanualOpenArmMinkIKTransform:
    """Convert Quest EE deltas to 16 motor positions using one shared Mink QP."""

    def __init__(
        self,
        kinematics: Any,
        setup: Any,
        *,
        left_seed_deg: Sequence[float],
        right_seed_deg: Sequence[float],
        workspace_min: Sequence[float] = OPENARM_WORKSPACE_MIN,
        workspace_max: Sequence[float] = OPENARM_WORKSPACE_MAX,
    ) -> None:
        self._kin = kinematics
        self._setup = setup
        self._q_init = {
            "left": np.asarray(left_seed_deg, dtype=float).copy(),
            "right": np.asarray(right_seed_deg, dtype=float).copy(),
        }
        if any(q.shape != (8,) or not np.all(np.isfinite(q)) for q in self._q_init.values()):
            raise ValueError("OpenArm Mink seeds must contain eight finite values per arm")
        self._q_last = {side: q.copy() for side, q in self._q_init.items()}
        self._ws_min = np.asarray(workspace_min, dtype=float)
        self._ws_max = np.asarray(workspace_max, dtype=float)
        self._ref: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._last_pos: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._previous_enabled = {"left": False, "right": False}
        self._reset_previous_t: dict[str, float | None] = {"left": None, "right": None}
        self._hold = (False, False)
        self._joint_bounds_deg = self._resolve_joint_bounds_deg()
        self._expected_input = {f"{side}_{key}" for side in ("left", "right") for key in _ACTION_KEYS}

    def _resolve_joint_bounds_deg(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        import mujoco

        bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in ("left", "right"):
            ranges = []
            for index in range(1, 8):
                name = f"openarm_{side}_joint{index}"
                joint_id = mujoco.mj_name2id(self._setup.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0:
                    raise ValueError(f"OpenArm MJCF is missing joint {name}")
                ranges.append(self._setup.model.jnt_range[joint_id])
            limits = np.degrees(np.asarray(ranges, dtype=float))
            bounds[side] = (limits[:, 0], limits[:, 1])
        return bounds

    def _fk(self) -> dict[str, np.ndarray]:
        right, left = self._kin.fk_bimanual(
            np.deg2rad(self._q_last["right"]),
            np.deg2rad(self._q_last["left"]),
        )
        return {"right": _pose7_to_matrix(right), "left": _pose7_to_matrix(left)}

    def _target_for_side(
        self,
        side: str,
        values: Mapping[str, float],
        current_pose: np.ndarray,
    ) -> np.ndarray:
        enabled = values[f"{side}_enabled"] > 0.5
        self._q_last[side][7] = values[f"{side}_gripper_pos"]

        if not enabled:
            self._previous_enabled[side] = False
            return current_pose

        if not self._previous_enabled[side] or self._ref[side] is None:
            self._ref[side] = current_pose.copy()
            self._last_pos[side] = current_pose[:3, 3].copy()

        reference = self._ref[side]
        assert reference is not None
        desired = np.eye(4, dtype=float)
        rotation_delta = Rotation.from_rotvec(
            [
                values[f"{side}_target_wx"],
                values[f"{side}_target_wy"],
                values[f"{side}_target_wz"],
            ]
        ).as_matrix()
        desired[:3, :3] = rotation_delta @ reference[:3, :3]
        position = reference[:3, 3] + np.asarray(
            [
                values[f"{side}_target_x"],
                values[f"{side}_target_y"],
                values[f"{side}_target_z"],
            ]
        )
        position = np.clip(position, self._ws_min, self._ws_max)
        previous_position = self._last_pos[side]
        assert previous_position is not None
        delta = position - previous_position
        magnitude = float(np.linalg.norm(delta))
        if magnitude > _MAX_EE_STEP_M:
            position = previous_position + delta * (_MAX_EE_STEP_M / magnitude)
        desired[:3, 3] = position
        self._previous_enabled[side] = True
        return desired

    def __call__(self, action: Mapping[str, Any]) -> dict[str, float]:
        if set(action) != self._expected_input:
            raise ValueError("OpenArm Cartesian action keys do not match the bimanual contract")
        values = {key: float(value) for key, value in action.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("OpenArm Cartesian action values must all be finite")

        # Reset is an explicit joint-space ramp. Freeze the other arm for
        # this tick instead of asking the Cartesian QP to undo the ramp.
        reset_requested = False
        now = time.perf_counter()
        for side in ("left", "right"):
            self._q_last[side][7] = values[f"{side}_gripper_pos"]
            if values[f"{side}_reset"] > 0.5:
                reset_requested = True
                previous = self._reset_previous_t[side]
                dt = 0.0 if previous is None else min(now - previous, _RESET_DT_CAP_S)
                self._reset_previous_t[side] = now
                step = _RESET_RATE_DEG_S * dt
                difference = self._q_init[side][:7] - self._q_last[side][:7]
                self._q_last[side][:7] += np.clip(difference, -step, step)
                self._ref[side] = None
                self._last_pos[side] = None
                self._previous_enabled[side] = False
            else:
                self._reset_previous_t[side] = None
        if reset_requested:
            self._hold = (False, False)
            return self._output()
        if not any(values[f"{side}_enabled"] > 0.5 for side in ("left", "right")):
            self._hold = (False, False)
            return self._output()

        current = self._fk()
        targets = {
            side: self._target_for_side(side, values, current[side]) for side in ("left", "right")
        }
        state_rad = np.concatenate(
            [np.deg2rad(self._q_last["right"]), np.deg2rad(self._q_last["left"])]
        )
        self._kin.sync(state_rad)
        for side in ("right", "left"):
            self._kin.set_target(side, _matrix_to_pose7(targets[side]))
            self._kin.set_gripper(side, math.radians(self._q_last[side][7]))
        solved = self._kin.solve()
        if solved is None:
            self._hold = (True, True)
            return self._output()

        solved_deg = np.degrees(np.asarray(solved, dtype=float))
        if solved_deg.shape != (16,) or not np.all(np.isfinite(solved_deg)):
            self._hold = (True, True)
            return self._output()
        candidates = {"right": solved_deg[:8], "left": solved_deg[8:]}
        for side in ("left", "right"):
            # A clutch-released arm is a strict joint hold. The shared QP's
            # posture task may otherwise move redundant joints while keeping
            # the EE pose unchanged.
            if not self._previous_enabled[side]:
                candidates[side][:7] = self._q_last[side][:7]
            candidates[side][7] = values[f"{side}_gripper_pos"]
            lower, upper = self._joint_bounds_deg[side]
            if np.any(candidates[side][:7] < lower - 1e-6) or np.any(
                candidates[side][:7] > upper + 1e-6
            ):
                self._hold = (True, True)
                return self._output()
            if float(np.max(np.abs(candidates[side][:7] - self._q_last[side][:7]))) > _MAX_JOINT_STEP_DEG:
                self._hold = (True, True)
                return self._output()

        self._q_last = {side: candidates[side].copy() for side in ("left", "right")}
        for side in ("left", "right"):
            if self._previous_enabled[side]:
                self._last_pos[side] = targets[side][:3, 3].copy()
        self._hold = (False, False)
        return self._output()

    def _output(self) -> dict[str, float]:
        return {
            f"{side}_{motor}.pos": float(self._q_last[side][index])
            for side in ("right", "left")
            for index, motor in enumerate(MOTOR_NAMES)
        }

    @property
    def hold_per_arm(self) -> tuple[bool, bool]:
        return self._hold


def make_openarm_mink_kinematics(
    *,
    xml: str | Path,
    max_iters: int = 10,
    damping: float = 0.1,
    posture_cost: float = 0.01,
) -> tuple[Any, Any]:
    """Build the official shared bimanual OpenArm solver using D's tuned values."""
    from openarm_control import ArmSetup, IKParams, Kinematics

    setup = ArmSetup.from_args(
        xml=str(Path(xml).expanduser().resolve()),
        mode="bimanual",
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe=None,
    )
    params = IKParams(
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.01,
        damping=damping,
        posture_cost=posture_cost,
        dt=0.1,
        max_iters=max_iters,
    )
    return Kinematics(setup, params), setup


def build_openarm_mink_transform(
    kinematics: Any,
    setup: Any,
    left_arm: Any,
    right_arm: Any,
) -> BimanualOpenArmMinkIKTransform:
    return BimanualOpenArmMinkIKTransform(
        kinematics,
        setup,
        left_seed_deg=_seed(left_arm),
        right_seed_deg=_seed(right_arm),
    )
