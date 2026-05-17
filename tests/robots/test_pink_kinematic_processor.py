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
