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

"""Tests for the Cartesian IK pipeline auto-composition logic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lerobot.processor.cartesian_ik_pipeline import (
    CartesianIKRobotConfig,
    get_cartesian_ik_robot_config,
    is_cartesian_teleop,
    make_cartesian_ik_pipeline,
    register_cartesian_ik_robot,
)
from lerobot.utils.import_utils import _pin_pink_available


def test_is_cartesian_teleop_true_for_target_xyz():
    """Teleops whose action_features include target_x/y/z should be detected."""
    teleop = SimpleNamespace(
        action_features={
            "shape": (8,),
            "names": {
                "enabled": 0,
                "target_x": 1,
                "target_y": 2,
                "target_z": 3,
                "target_wx": 4,
                "target_wy": 5,
                "target_wz": 6,
                "gripper_vel": 7,
            },
        }
    )
    assert is_cartesian_teleop(teleop)


def test_is_cartesian_teleop_false_for_joint_teleop():
    """Joint-space teleops (leader arms etc.) should not match."""
    teleop = SimpleNamespace(
        action_features={
            "shape": (7,),
            "names": {f"motor_{i}.pos": i for i in range(7)},
        }
    )
    assert not is_cartesian_teleop(teleop)


def test_is_cartesian_teleop_false_for_missing_features():
    """Teleop without action_features at all shouldn't crash."""
    teleop = SimpleNamespace()
    assert not is_cartesian_teleop(teleop)


def test_so107_follower_is_registered():
    """SO-107 follower's Cartesian config should be auto-registered at import time."""
    # Triggers the package's _register_cartesian_ik() call.
    import lerobot.robots.so107_description  # noqa: F401

    cfg = get_cartesian_ik_robot_config("so107_follower")
    assert cfg is not None
    assert cfg.ee_frame_name == "L7_1"
    assert cfg.motor_names[0] == "shoulder_pan"
    assert cfg.motor_names[-1] == "gripper"
    assert cfg.joint_names == ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def test_unknown_robot_returns_none():
    """Robots without a registered config should yield None (callers fall back)."""
    fake = SimpleNamespace(name="some_robot_that_does_not_exist")
    assert make_cartesian_ik_pipeline(fake) is None


def test_re_register_overwrites_and_warns():
    """Second registration with the same name should overwrite."""
    cfg1 = CartesianIKRobotConfig(urdf_path="/tmp/a.urdf", ee_frame_name="ee", motor_names=["m"])
    cfg2 = CartesianIKRobotConfig(urdf_path="/tmp/b.urdf", ee_frame_name="ee", motor_names=["m"])
    register_cartesian_ik_robot("test_overwrite", cfg1)
    register_cartesian_ik_robot("test_overwrite", cfg2)
    assert get_cartesian_ik_robot_config("test_overwrite").urdf_path == "/tmp/b.urdf"


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_pipeline_builds_for_so107_follower():
    """Smoke test: make_cartesian_ik_pipeline returns a working pipeline for SO-107."""
    import lerobot.robots.so107_description  # noqa: F401

    fake_robot = SimpleNamespace(name="so107_follower")
    pipeline = make_cartesian_ik_pipeline(fake_robot)
    assert pipeline is not None
    # Pipeline should have the expected steps.
    step_names = [type(s).__name__ for s in pipeline.steps]
    assert "EEReferenceAndDelta" in step_names
    assert "EEBoundsAndSafety" in step_names
    assert "GripperVelocityToJoint" in step_names
    assert "PinkInverseKinematicsEEToJoints" in step_names
