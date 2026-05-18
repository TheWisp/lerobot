#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PinkInverseKinematicsEEToJoints ProcessorStep."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.processor import TransitionKey
from lerobot.utils.import_utils import _pin_pink_available

pytestmark = pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")


SO107_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@pytest.fixture
def step():
    from lerobot.model.pink_kinematics import PinkKinematics
    from lerobot.robots.so107_description import get_urdf_path
    from lerobot.robots.so_follower.pink_kinematic_processor import (
        PinkInverseKinematicsEEToJoints,
    )

    pk = PinkKinematics(urdf_path=str(get_urdf_path()), target_frame_name="L7_1")
    return PinkInverseKinematicsEEToJoints(kinematics=pk, motor_names=SO107_MOTOR_NAMES)


def _seed_observation() -> dict:
    return {
        f"{name}.pos": float(v)
        for name, v in zip(SO107_MOTOR_NAMES, [0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 20.0], strict=False)
    }


def _ee_action_for_target(target_ee_4x4, gripper_pos: float = 42.0) -> dict:
    """Build an action dict in the format the step expects."""
    from lerobot.utils.rotation import Rotation

    pos = target_ee_4x4[:3, 3]
    rotvec = Rotation.from_matrix(target_ee_4x4[:3, :3]).as_rotvec()
    return {
        "ee.x": float(pos[0]),
        "ee.y": float(pos[1]),
        "ee.z": float(pos[2]),
        "ee.wx": float(rotvec[0]),
        "ee.wy": float(rotvec[1]),
        "ee.wz": float(rotvec[2]),
        "ee.gripper_pos": gripper_pos,
    }


def _run_step(step, observation: dict, action: dict) -> dict:
    """Invoke the step through its __call__ (which sets _current_transition properly)."""
    transition = {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
    }
    out_transition = step(transition)
    return out_transition[TransitionKey.ACTION]


def test_step_is_registered():
    """Importing the module triggers the @ProcessorStepRegistry.register decorator."""
    import lerobot.robots.so_follower.pink_kinematic_processor  # noqa: F401
    from lerobot.processor import ProcessorStepRegistry

    cls = ProcessorStepRegistry.get("pink_inverse_kinematics_ee_to_joints")
    assert cls.__name__ == "PinkInverseKinematicsEEToJoints"


def test_step_outputs_motor_pos_for_each_motor_name(step):
    """Action output should contain a <motor>.pos entry for every configured motor."""
    obs = _seed_observation()
    q_seed = np.array([float(obs[f"{n}.pos"]) for n in SO107_MOTOR_NAMES])
    target = step.kinematics.forward_kinematics(q_seed)
    out = _run_step(step, obs, _ee_action_for_target(target))

    # ee.* input fields should be consumed.
    for k in ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"):
        assert k not in out, f"{k} should have been consumed"
    # All motor outputs present.
    for name in SO107_MOTOR_NAMES:
        assert f"{name}.pos" in out


def test_step_passes_gripper_through_from_ee_gripper_pos(step):
    obs = _seed_observation()
    q_seed = np.array([float(obs[f"{n}.pos"]) for n in SO107_MOTOR_NAMES])
    target = step.kinematics.forward_kinematics(q_seed)
    out = _run_step(step, obs, _ee_action_for_target(target, gripper_pos=73.5))

    assert out["gripper.pos"] == 73.5


def test_step_identity_when_target_equals_fk_of_seed(step):
    """If target = FK(seed), IK should return seed joints (pink + posture)."""
    obs = _seed_observation()
    q_seed = np.array([float(obs[f"{n}.pos"]) for n in SO107_MOTOR_NAMES])
    target = step.kinematics.forward_kinematics(q_seed)
    out = _run_step(step, obs, _ee_action_for_target(target))

    for i, name in enumerate(SO107_MOTOR_NAMES):
        if name == "gripper":
            continue
        np.testing.assert_allclose(out[f"{name}.pos"], q_seed[i], atol=1e-2)


def test_step_reaches_perturbed_target(step):
    obs = _seed_observation()
    q_seed = np.array([float(obs[f"{n}.pos"]) for n in SO107_MOTOR_NAMES])
    target = step.kinematics.forward_kinematics(q_seed)
    target[0, 3] += 0.01  # +1cm in URDF x
    out = _run_step(step, obs, _ee_action_for_target(target))

    # Compute FK of the output joints; should match the perturbed target.
    q_out = np.array([out[f"{n}.pos"] if n != "gripper" else 0.0 for n in SO107_MOTOR_NAMES])
    achieved = step.kinematics.forward_kinematics(q_out)
    pos_err_mm = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3])) * 1000
    assert pos_err_mm < 1.0


def test_transform_features_swaps_ee_for_motor_pos(step):
    from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature

    features = {
        PipelineFeatureType.ACTION: {
            f"ee.{f}": PolicyFeature(type=FeatureType.ACTION, shape=(1,))
            for f in ("x", "y", "z", "wx", "wy", "wz", "gripper_pos")
        }
    }
    out = step.transform_features(features)

    # ee.* removed
    for f in ("x", "y", "z", "wx", "wy", "wz", "gripper_pos"):
        assert f"ee.{f}" not in out[PipelineFeatureType.ACTION]
    # motor names added
    for name in SO107_MOTOR_NAMES:
        assert f"{name}.pos" in out[PipelineFeatureType.ACTION]


def test_reset_clears_q_curr(step):
    obs = _seed_observation()
    q_seed = np.array([float(obs[f"{n}.pos"]) for n in SO107_MOTOR_NAMES])
    target = step.kinematics.forward_kinematics(q_seed)
    _run_step(step, obs, _ee_action_for_target(target))

    assert step.q_curr is not None
    step.reset()
    assert step.q_curr is None


# ── Prefix-aware behavior (bimanual support) ─────────────────────────────


def _prefixed_obs(prefix: str) -> dict:
    """Seed observation with a per-arm prefix on every motor name."""
    return {
        f"{prefix}{name}.pos": float(v)
        for name, v in zip(SO107_MOTOR_NAMES, [0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 20.0], strict=False)
    }


def _prefixed_ee_action(target_ee_4x4, prefix: str, gripper_pos: float = 42.0) -> dict:
    from lerobot.utils.rotation import Rotation

    pos = target_ee_4x4[:3, 3]
    rotvec = Rotation.from_matrix(target_ee_4x4[:3, :3]).as_rotvec()
    return {
        f"{prefix}ee.x": float(pos[0]),
        f"{prefix}ee.y": float(pos[1]),
        f"{prefix}ee.z": float(pos[2]),
        f"{prefix}ee.wx": float(rotvec[0]),
        f"{prefix}ee.wy": float(rotvec[1]),
        f"{prefix}ee.wz": float(rotvec[2]),
        f"{prefix}ee.gripper_pos": gripper_pos,
    }


@pytest.fixture
def left_step():
    from lerobot.model.pink_kinematics import PinkKinematics
    from lerobot.robots.so107_description import get_urdf_path
    from lerobot.robots.so_follower.pink_kinematic_processor import (
        PinkInverseKinematicsEEToJoints,
    )

    pk = PinkKinematics(urdf_path=str(get_urdf_path()), target_frame_name="L7_1")
    return PinkInverseKinematicsEEToJoints(kinematics=pk, motor_names=SO107_MOTOR_NAMES, key_prefix="left_")


def test_prefixed_step_reads_and_writes_prefixed_keys(left_step):
    """With key_prefix='left_', input ee.* and output <motor>.pos are all prefixed."""
    obs = _prefixed_obs("left_")
    q_seed = np.array([float(obs[f"left_{n}.pos"]) for n in SO107_MOTOR_NAMES])
    target = left_step.kinematics.forward_kinematics(q_seed)
    out = _run_step(left_step, obs, _prefixed_ee_action(target, "left_"))

    # All input keys consumed (under the prefix).
    for k in ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"):
        assert f"left_{k}" not in out
    # Outputs prefixed.
    for name in SO107_MOTOR_NAMES:
        assert f"left_{name}.pos" in out
    # Unprefixed motor.pos keys must NOT appear.
    for name in SO107_MOTOR_NAMES:
        assert f"{name}.pos" not in out


def test_motor_to_urdf_round_trip_with_so107_right_arm_map():
    """For any motor vector, urdf_to_motor(motor_to_urdf(v)) == v under RIGHT_ARM_MAP."""
    from lerobot.robots.so107_description.kinematics import RIGHT_ARM_MAP
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        _motor_to_urdf_deg,
        _urdf_to_motor_deg,
    )

    rng = np.random.default_rng(42)
    motor_in = rng.uniform(-90.0, 90.0, size=7)
    urdf = _motor_to_urdf_deg(motor_in, list(SO107_MOTOR_NAMES), RIGHT_ARM_MAP)
    motor_out = _urdf_to_motor_deg(urdf, list(SO107_MOTOR_NAMES), RIGHT_ARM_MAP)
    np.testing.assert_allclose(motor_out, motor_in, atol=1e-9)


def test_motor_to_urdf_identity_when_map_is_none():
    """With joint_map=None, motor degrees pass through unchanged."""
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        _motor_to_urdf_deg,
        _urdf_to_motor_deg,
    )

    motor_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(_motor_to_urdf_deg(motor_in, list(SO107_MOTOR_NAMES), None), motor_in)
    np.testing.assert_array_equal(_urdf_to_motor_deg(motor_in, list(SO107_MOTOR_NAMES), None), motor_in)


def test_pink_step_applies_joint_map_to_seed_and_output():
    """With joint_map=RIGHT_ARM_MAP, the pink IK step should:
    - Convert motor observation -> URDF before seeding pink.
    - Convert pink's URDF output back to motor space in action.

    Sanity check: at FK(seed), the IK should return motor values that round-trip
    back through the map to the same seed (no spurious sign flips / offsets).
    """
    from lerobot.model.pink_kinematics import PinkKinematics
    from lerobot.robots.so107_description import get_urdf_path
    from lerobot.robots.so107_description.kinematics import RIGHT_ARM_MAP
    from lerobot.robots.so_follower.pink_kinematic_processor import (
        PinkInverseKinematicsEEToJoints,
    )
    from lerobot.robots.so_follower.robot_kinematic_processor import _motor_to_urdf_deg

    pk = PinkKinematics(urdf_path=str(get_urdf_path()), target_frame_name="L7_1")
    step = PinkInverseKinematicsEEToJoints(
        kinematics=pk, motor_names=list(SO107_MOTOR_NAMES), joint_map=RIGHT_ARM_MAP
    )

    motor_obs = {
        f"{name}.pos": float(v)
        for name, v in zip(SO107_MOTOR_NAMES, [0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 20.0], strict=False)
    }
    # Compute target = FK(seed-in-urdf-space). IK should return motor values
    # close to motor_obs.
    motor_vec = np.array([motor_obs[f"{n}.pos"] for n in SO107_MOTOR_NAMES])
    urdf_seed_deg = _motor_to_urdf_deg(motor_vec, list(SO107_MOTOR_NAMES), RIGHT_ARM_MAP)
    target = pk.forward_kinematics(urdf_seed_deg)

    out = _run_step(step, motor_obs, _ee_action_for_target(target))

    # All arm motors in output should round-trip near the input observation.
    for i, name in enumerate(SO107_MOTOR_NAMES):
        if name == "gripper":
            continue
        assert abs(out[f"{name}.pos"] - motor_vec[i]) < 1e-2, (
            f"{name}: in={motor_vec[i]} out={out[f'{name}.pos']}"
        )


def test_prefixed_step_ignores_other_arms_observation_keys(left_step):
    """left_step should only read left_<motor>.pos; right_<motor>.pos must be skipped."""
    obs = {**_prefixed_obs("left_"), **_prefixed_obs("right_")}
    # Make right arm wildly different so it would corrupt the IK seed if accidentally read.
    for name in SO107_MOTOR_NAMES:
        obs[f"right_{name}.pos"] = 999.0
    q_seed = np.array([float(obs[f"left_{n}.pos"]) for n in SO107_MOTOR_NAMES])
    target = left_step.kinematics.forward_kinematics(q_seed)
    out = _run_step(left_step, obs, _prefixed_ee_action(target, "left_"))

    # IK should still return ~seed (target = FK(seed)); confirms right_* was ignored.
    for i, name in enumerate(SO107_MOTOR_NAMES):
        if name == "gripper":
            continue
        np.testing.assert_allclose(out[f"left_{name}.pos"], q_seed[i], atol=1e-2)
