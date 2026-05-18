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
    CartesianIKArmConfig,
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


def test_so107_follower_predictive_shares_config():
    """200Hz predictive variant uses the same hardware/URDF; should also be registered."""
    import lerobot.robots.so107_description  # noqa: F401

    cfg_plain = get_cartesian_ik_robot_config("so107_follower")
    cfg_predictive = get_cartesian_ik_robot_config("so107_follower_predictive")
    assert cfg_predictive is not None
    # Same hardware: same URDF, EE frame, motor names, joint names.
    assert cfg_predictive.urdf_path == cfg_plain.urdf_path
    assert cfg_predictive.ee_frame_name == cfg_plain.ee_frame_name
    assert cfg_predictive.motor_names == cfg_plain.motor_names
    assert cfg_predictive.joint_names == cfg_plain.joint_names


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


# ── Bimanual detection / config / composition ─────────────────────────────


def test_is_cartesian_teleop_true_for_bimanual():
    """Teleops emitting both left_target_xyz and right_target_xyz should be detected."""
    names = {}
    for prefix in ("left_", "right_"):
        for i, k in enumerate(
            (
                "enabled",
                "target_x",
                "target_y",
                "target_z",
                "target_wx",
                "target_wy",
                "target_wz",
                "gripper_vel",
            )
        ):
            names[f"{prefix}{k}"] = len(names) * 1 + i  # values don't matter
    teleop = SimpleNamespace(action_features={"shape": (16,), "names": names})
    assert is_cartesian_teleop(teleop)


def test_is_cartesian_teleop_false_for_left_only():
    """Only left_ prefixed keys (missing right_) shouldn't match bimanual."""
    names = {f"left_target_{c}": i for i, c in enumerate("xyz")}
    teleop = SimpleNamespace(action_features={"shape": (3,), "names": names})
    assert not is_cartesian_teleop(teleop)


def test_bimanual_config_marks_is_bimanual():
    """CartesianIKRobotConfig with two arms reports is_bimanual=True."""
    left = CartesianIKArmConfig(
        urdf_path="/tmp/a.urdf", ee_frame_name="ee", motor_names=["m"], key_prefix="left_"
    )
    right = CartesianIKArmConfig(
        urdf_path="/tmp/a.urdf", ee_frame_name="ee", motor_names=["m"], key_prefix="right_"
    )
    cfg = CartesianIKRobotConfig(arms=[left, right])
    assert cfg.is_bimanual
    assert len(cfg.arms) == 2


def test_unimanual_shortcut_builds_one_arm():
    """The flat-field shortcut produces a single arm with empty prefix."""
    cfg = CartesianIKRobotConfig(urdf_path="/tmp/a.urdf", ee_frame_name="ee", motor_names=["m"])
    assert not cfg.is_bimanual
    assert len(cfg.arms) == 1
    assert cfg.arms[0].key_prefix == ""


def test_bi_so107_follower_is_registered():
    """SO-107 bimanual variants should be auto-registered with two arms."""
    import lerobot.robots.so107_description  # noqa: F401

    for name in ("bi_so107_follower", "bi_so107_follower_predictive"):
        cfg = get_cartesian_ik_robot_config(name)
        assert cfg is not None, f"{name} not registered"
        assert cfg.is_bimanual
        prefixes = sorted(a.key_prefix for a in cfg.arms)
        assert prefixes == ["left_", "right_"]


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_pipeline_builds_for_bi_so107_follower():
    """Bimanual pipeline has 8 steps (4 per arm) with prefixed key_prefix fields."""
    import lerobot.robots.so107_description  # noqa: F401

    fake_robot = SimpleNamespace(name="bi_so107_follower")
    pipeline = make_cartesian_ik_pipeline(fake_robot)
    assert pipeline is not None
    assert len(pipeline.steps) == 8

    # Inspect prefixes: first 4 steps are "left_", next 4 are "right_".
    left_steps = pipeline.steps[:4]
    right_steps = pipeline.steps[4:]
    for s in left_steps:
        assert s.key_prefix == "left_", f"{type(s).__name__} has prefix {s.key_prefix!r}"
    for s in right_steps:
        assert s.key_prefix == "right_", f"{type(s).__name__} has prefix {s.key_prefix!r}"
