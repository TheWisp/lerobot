# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Hardware-free wiring tests for the OpenArm 2.0 record paths.

Covers the two data-collection setups documented in
``examples/openarm2/README.md``:

- ``bi_openarm_follower`` + ``bi_openarm_leader`` (joint space), and
- ``bi_openarm_follower`` + ``quest_vr`` (Cartesian IK).

Proves, without touching CAN or cameras, that (a) draccus parses a
``RecordConfig`` from the documented CLI flags, (b) the factories construct
the robot and teleop unconnected, and (c) ``attach_teleop`` installs an IK
transform whose output keys are exactly what ``BiOpenArmFollower.send_action``
consumes. The Cartesian-IK math itself is covered by
``tests/robots/test_openarm_cartesian_ik.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco", reason="openarm-ff extra not installed")
# RecordConfig imports lerobot.datasets at module level.
pytest.importorskip("datasets", reason="dataset extra not installed")

import draccus  # noqa: E402

from lerobot.robots.bi_openarm_follower.config_bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.openarm_description.cartesian_ik import MOTOR_NAMES
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.teleoperators.quest_vr.configuration_quest_vr import QuestVRTeleopConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.import_utils import _pin_pink_available

# Mirrors Option A of examples/openarm2/README.md (leader-follower).
_LEADER_ARGS = [
    "--robot.type=bi_openarm_follower",
    "--robot.id=bi_openarm_follower",
    "--robot.left_arm_config.port=can0",
    "--robot.left_arm_config.side=left",
    "--robot.left_arm_config.gravity_ff_gain=0.9",
    "--robot.left_arm_config.use_velocity_and_torque=true",
    "--robot.right_arm_config.port=can1",
    "--robot.right_arm_config.side=right",
    "--robot.right_arm_config.gravity_ff_gain=0.9",
    "--robot.right_arm_config.use_velocity_and_torque=true",
    "--teleop.type=bi_openarm_leader",
    "--teleop.id=bi_openarm_leader",
    "--teleop.left_arm_config.port=can2",
    "--teleop.right_arm_config.port=can3",
    "--dataset.repo_id=user/openarm2_leader",
    "--dataset.single_task=Pick the cube.",
    "--dataset.fps=30",
    "--dataset.num_episodes=2",
    "--dataset.push_to_hub=false",
]

# Mirrors Option B of examples/openarm2/README.md (Quest VR, Cartesian IK).
_QUEST_VR_ARGS = [
    "--robot.type=bi_openarm_follower",
    "--robot.id=bi_openarm_follower",
    "--robot.left_arm_config.port=can0",
    "--robot.left_arm_config.side=left",
    "--robot.left_arm_config.gravity_ff_gain=0.9",
    "--robot.left_arm_config.use_velocity_and_torque=true",
    "--robot.right_arm_config.port=can1",
    "--robot.right_arm_config.side=right",
    "--robot.right_arm_config.gravity_ff_gain=0.9",
    "--robot.right_arm_config.use_velocity_and_torque=true",
    "--teleop.type=quest_vr",
    "--teleop.id=quest_vr",
    "--teleop.port=8443",
    "--teleop.robot_forward_in_urdf=[1,0,0]",
    "--teleop.robot_up_in_urdf=[0,0,1]",
    "--teleop.left_gripper_open_motor=45",
    "--teleop.left_gripper_closed_motor=0",
    "--teleop.right_gripper_open_motor=-45",
    "--teleop.right_gripper_closed_motor=0",
    "--dataset.repo_id=user/openarm2_quest_vr",
    "--dataset.single_task=Pick the cube.",
    "--dataset.fps=30",
    "--dataset.num_episodes=2",
    "--dataset.push_to_hub=false",
]

# Bent-elbow IK seed poses (motor degrees), matching test_openarm_cartesian_ik.
_LEFT_READY = dict(zip(MOTOR_NAMES, [0.0, -45.0, 0.0, 90.0, 0.0, 30.0, 0.0, 0.0], strict=True))
_RIGHT_READY = dict(zip(MOTOR_NAMES, [0.0, 45.0, 0.0, 90.0, 0.0, 30.0, 0.0, 0.0], strict=True))


def _parse(args: list[str]) -> RecordConfig:
    return draccus.parse(RecordConfig, args=args)


def _assert_bi_openarm_follower(cfg: RecordConfig) -> None:
    assert isinstance(cfg.robot, BiOpenArmFollowerConfig)
    for side_name, port in (("left", "can0"), ("right", "can1")):
        arm = getattr(cfg.robot, f"{side_name}_arm_config")
        assert arm.port == port
        assert arm.side == side_name
        assert arm.gravity_ff_gain == pytest.approx(0.9)
        assert arm.use_velocity_and_torque is True


def test_record_config_parses_bi_openarm_follower_with_leader():
    cfg = _parse(_LEADER_ARGS)
    _assert_bi_openarm_follower(cfg)
    assert cfg.teleop.type == "bi_openarm_leader"
    assert cfg.teleop.left_arm_config.port == "can2"
    assert cfg.teleop.right_arm_config.port == "can3"
    assert cfg.dataset.repo_id == "user/openarm2_leader"
    assert cfg.dataset.fps == 30


def test_record_config_parses_bi_openarm_follower_with_quest_vr():
    cfg = _parse(_QUEST_VR_ARGS)
    _assert_bi_openarm_follower(cfg)
    assert isinstance(cfg.teleop, QuestVRTeleopConfig)
    assert cfg.teleop.robot_forward_in_urdf == [1.0, 0.0, 0.0]
    assert cfg.teleop.robot_up_in_urdf == [0.0, 0.0, 1.0]
    # Standard-edition mapping from the pinned dora IK implementation:
    # trigger released opens to +/-45 degrees; trigger pulled closes to zero.
    assert cfg.teleop.left_gripper_open_motor == pytest.approx(45.0)
    assert cfg.teleop.left_gripper_closed_motor == pytest.approx(0.0)
    assert cfg.teleop.right_gripper_open_motor == pytest.approx(-45.0)
    assert cfg.teleop.right_gripper_closed_motor == pytest.approx(0.0)


def test_factories_construct_robot_and_teleop_without_hardware():
    """Construction must not touch CAN/cameras/servers — only connect() may."""
    cfg = _parse(_QUEST_VR_ARGS)
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    assert not robot.is_connected
    assert not teleop.is_connected

    # The FF/ramp regime set on the bi config must reach both arm
    # controllers (regression: BiOpenArmFollower used to drop these fields).
    for side_name in ("left", "right"):
        arm = getattr(robot, f"{side_name}_arm")
        assert arm.config.gravity_ff_gain == pytest.approx(0.9)
        assert arm._gravity_ff is not None
        assert arm.config.use_velocity_and_torque is True

    # Cartesian teleop advertises the per-arm EE-delta keys.
    names = set(teleop.action_features["names"])
    assert {f"{s}_target_x" for s in ("left", "right")} <= names
    assert {f"{s}_gripper_pos" for s in ("left", "right")} <= names


@pytest.mark.skipif(not _pin_pink_available, reason="pin-pink (optional) not installed")
def test_attach_teleop_installs_ik_transform_with_send_action_keys():
    """attach_teleop on the real constructed pair: a synthetic action dict
    with exactly the teleop's advertised keys must come out of the installed
    transform as exactly the ``{left,right}_{joint_1..7,gripper}.pos`` keys
    send_action consumes."""
    cfg = _parse(_QUEST_VR_ARGS)
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    # Stand in for connected arms: is_connected + a get_observation seed.
    robot.left_arm = SimpleNamespace(
        is_connected=True, get_observation=lambda: {f"{m}.pos": v for m, v in _LEFT_READY.items()}
    )
    robot.right_arm = SimpleNamespace(
        is_connected=True, get_observation=lambda: {f"{m}.pos": v for m, v in _RIGHT_READY.items()}
    )

    robot.attach_teleop(teleop)
    assert teleop._action_transform is not None

    # Feed a synthetic all-zero action with exactly the keys the teleop
    # advertises (9 per arm: enabled/reset/6-DOF targets/gripper).
    action = dict.fromkeys(teleop.action_features["names"], 0.0)
    joints = teleop._action_transform(action)

    expected = {f"{side}_{m}.pos" for side in ("left", "right") for m in MOTOR_NAMES}
    assert set(joints) == expected
