#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.bi_openarm_follower.bi_openarm_follower import BiOpenArmFollower
from lerobot.robots.openarm_description.mjcf import (
    OPENARM_CARTESIAN_ACTION_KEYS,
    OPENARM_MOTOR_NAMES,
    BimanualOpenArmMJCFIKTransform,
    MJCFArmKinematics,
    build_openarm_bimanual_mjcf_ik_transform,
)


class FakeKinematics:
    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        del joint_pos_deg
        return np.eye(4)

    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray:
        del target
        return np.asarray(seed_deg, dtype=float).copy()


class JumpKinematics(FakeKinematics):
    def inverse_kinematics(self, seed_deg: np.ndarray, target: np.ndarray) -> np.ndarray:
        del target
        result = np.asarray(seed_deg, dtype=float).copy()
        result[0] += 30.0
        return result


class FakeArm:
    def __init__(self, seed: list[float]) -> None:
        self.is_connected = True
        self.config = SimpleNamespace(gravity_ff_xml=None)
        self._seed = seed

    def get_observation(self) -> dict[str, float]:
        return {f"{motor}.pos": value for motor, value in zip(OPENARM_MOTOR_NAMES, self._seed, strict=True)}


class FakeCartesianTeleop:
    def __init__(self) -> None:
        names = [f"{side}_{key}" for side in ("left", "right") for key in OPENARM_CARTESIAN_ACTION_KEYS]
        self.action_features = {"names": {name: index for index, name in enumerate(names)}}
        self.transform = None

    def set_action_transform(self, transform) -> None:
        self.transform = transform


def raw_action(*, enabled: float = 0.0, gripper: float = 0.0) -> dict[str, float]:
    action = {
        f"{side}_{key}": 0.0
        for side in ("left", "right")
        for key in OPENARM_CARTESIAN_ACTION_KEYS
    }
    for side in ("left", "right"):
        action[f"{side}_enabled"] = enabled
        action[f"{side}_gripper_pos"] = gripper
    return action


def build_fake_transform(*, jump_right: bool = False) -> BimanualOpenArmMJCFIKTransform:
    left_arm = FakeArm([1.0, 2.0, 3.0, 20.0, 5.0, 6.0, 7.0, 8.0])
    right_arm = FakeArm([-1.0, -2.0, -3.0, 20.0, -5.0, -6.0, -7.0, -8.0])
    return build_openarm_bimanual_mjcf_ik_transform(
        {"left": FakeKinematics(), "right": JumpKinematics() if jump_right else FakeKinematics()},
        left_arm,
        right_arm,
    )


def test_transform_emits_exactly_sixteen_finite_motor_positions_and_idle_holds():
    transform = build_fake_transform()
    output = transform(raw_action(enabled=0.0, gripper=4.0))

    assert set(output) == {
        f"{side}_{motor}.pos" for side in ("left", "right") for motor in OPENARM_MOTOR_NAMES
    }
    assert len(output) == 16
    assert output["left_joint_1.pos"] == pytest.approx(1.0)
    assert output["right_joint_1.pos"] == pytest.approx(-1.0)
    assert output["left_gripper.pos"] == pytest.approx(4.0)
    assert output["right_gripper.pos"] == pytest.approx(4.0)


def test_transform_rejects_bad_or_non_finite_raw_action_before_ik():
    transform = build_fake_transform()
    missing = raw_action()
    missing.pop("right_target_x")
    with pytest.raises(ValueError, match="keys mismatch"):
        transform(missing)

    non_finite = raw_action()
    non_finite["left_target_x"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        transform(non_finite)


def test_implausible_ik_jump_holds_only_the_affected_arm():
    transform = build_fake_transform(jump_right=True)
    output = transform(raw_action(enabled=1.0))

    assert transform.hold_per_arm == (False, True)
    assert output["right_joint_1.pos"] == pytest.approx(-1.0)


def test_bi_follower_attach_and_detach_without_hardware():
    robot = object.__new__(BiOpenArmFollower)
    robot.left_arm = FakeArm([0.0] * 8)
    robot.right_arm = FakeArm([0.0] * 8)
    robot._ik_kinematics = {"left": FakeKinematics(), "right": FakeKinematics()}
    robot._attached_cartesian_teleop = None
    robot.config = SimpleNamespace(ik_max_iterations=10, ik_damping=0.05)
    teleop = FakeCartesianTeleop()

    robot.attach_teleop(teleop)
    assert isinstance(teleop.transform, BimanualOpenArmMJCFIKTransform)
    robot.attach_teleop(None)
    assert teleop.transform is None


@pytest.mark.parametrize(
    ("side", "goal"),
    [
        ("left", [0.0, -20.0, 5.0, 30.0, 0.0, 10.0, 0.0, 0.0]),
        ("right", [0.0, 20.0, -5.0, 30.0, 0.0, 10.0, 0.0, 0.0]),
    ],
)
def test_real_mjcf_fk_ik_round_trip(side: str, goal: list[float]):
    pytest.importorskip("mujoco")
    pytest.importorskip("openarm_mujoco")
    kinematics = MJCFArmKinematics(side, max_iterations=120, damping=0.03)
    goal_array = np.asarray(goal, dtype=float)
    target = kinematics.forward_kinematics(goal_array)
    seed = goal_array.copy()
    seed[:7] += np.asarray([0.5, -0.5, 0.5, -0.5, 0.25, -0.25, 0.25])

    solved = kinematics.inverse_kinematics(seed, target)
    achieved = kinematics.forward_kinematics(solved)
    np.testing.assert_allclose(achieved[:3, 3], target[:3, 3], atol=5e-4)
    np.testing.assert_allclose(achieved[:3, :3], target[:3, :3], atol=5e-3)
    assert solved[7] == pytest.approx(seed[7])
