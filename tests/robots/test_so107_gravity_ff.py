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

"""The gravity-compensated SO-107 (``so107_follower_gravity``): its feed-forward offsets, their units, its gains, and
that the plain follower has none of it."""

from unittest.mock import MagicMock, patch

import draccus
import numpy as np
import pytest

pytest.importorskip("pinocchio")

from lerobot.motors import MotorCalibration
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so107_description.gravity import GravityFeedForward
from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES
from lerobot.robots.so107_follower_gravity import SO107FollowerGravity, SO107FollowerGravityRobotConfig
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO107Follower
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

# A gripper-down pose over the tray, motor degrees: the upper arm forward, forearm folded.
EASY_POSE = dict(zip(MOTOR_NAMES, (0.0, -45.0, 74.0, 2.0, -41.0, -12.0, 93.0), strict=True))


def test_offsets_scale_with_alpha_and_spare_the_gripper():
    a = GravityFeedForward(1.0, LEFT_ARM_ALIGNMENT).offset_deg(EASY_POSE)
    b = GravityFeedForward(3.0, LEFT_ARM_ALIGNMENT).offset_deg(EASY_POSE)
    assert set(a) == set(MOTOR_NAMES)
    assert a["gripper"] == 0.0 and b["gripper"] == 0.0
    for m in MOTOR_NAMES:
        assert b[m] == pytest.approx(3.0 * a[m])


def test_the_loaded_joints_are_pushed_against_the_droop():
    """A sagging joint settles at a larger motor angle than commanded (measured on the rig:
    lift/elbow present - commanded > 0 under load), so the offset must be negative there and
    grow with the moment the pose puts on the joint."""
    ff = GravityFeedForward(1.0, LEFT_ARM_ALIGNMENT)
    folded = ff.offset_deg(EASY_POSE)
    extended = ff.offset_deg({**EASY_POSE, "elbow_flex": 20.0, "shoulder_lift": -20.0})
    assert folded["elbow_flex"] < 0 and folded["shoulder_lift"] < 0
    assert abs(extended["shoulder_lift"]) > abs(folded["shoulder_lift"])
    # The rolls carry no gravity moment: nothing to pre-empt there.
    assert abs(folded["forearm_roll"]) < 0.05 and abs(folded["wrist_roll"]) < 0.05


def _bus():
    bus = MagicMock()
    writes: list[tuple[str, dict]] = []
    bus.sync_write.side_effect = lambda name, goal, **_: writes.append((name, dict(goal)))
    bus.writes = writes
    return bus


def _robot(cls, config):
    bus = _bus()

    def _make(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        return bus

    with (
        patch("lerobot.robots.so_follower.so_follower.FeetechMotorsBus", side_effect=_make),
        patch.object(cls, "configure", lambda self: None),
    ):
        robot = cls(config)
        robot.bus = bus
        # Symmetric 200-degree ranges so normalized units are exactly one per degree.
        robot.calibration = {
            m: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=0, range_min=910, range_max=3185)
            for i, m in enumerate(MOTOR_NAMES)
        }
        return robot


def _follower(**cfg):
    return _robot(
        SO107FollowerGravity, SO107FollowerGravityRobotConfig(port="/dev/null", id="test_arm", **cfg)
    )


def test_send_action_shifts_the_goal_by_the_offset_in_action_units():
    for use_degrees in (True, False):
        robot = _follower(use_degrees=use_degrees)
        action = {f"{m}.pos": v for m, v in EASY_POSE.items()}
        sent = robot.send_action(action)
        expected = robot.gravity.offset_deg(EASY_POSE)
        for m in MOTOR_NAMES:
            units = robot.units_per_motor_degree(m)
            assert sent[f"{m}.pos"] == pytest.approx(EASY_POSE[m] + expected[m] * units, abs=1e-9)
        assert robot.bus.writes[-1][0] == "Goal_Position"


def test_the_plain_follower_sends_its_goals_as_they_are_and_a_partial_action_passes_through():
    plain = _robot(SO107Follower, SOFollowerRobotConfig(port="/dev/null", id="test_arm"))
    action = {f"{m}.pos": v for m, v in EASY_POSE.items()}
    assert plain.send_action(action) == action
    # A gripper-only command has no pose to compensate at; it must go through untouched.
    assert _follower().send_action({"gripper.pos": 50.0}) == {"gripper.pos": 50.0}


def test_units_per_degree_follows_the_calibration_range():
    robot = _follower(use_degrees=False)
    assert robot.units_per_motor_degree("shoulder_lift") == pytest.approx(1.0, rel=1e-3)
    robot.calibration["shoulder_lift"] = MotorCalibration(
        id=2, drive_mode=0, homing_offset=0, range_min=1479, range_max=2616
    )
    assert robot.units_per_motor_degree("shoulder_lift") == pytest.approx(2.0, rel=1e-3)
    assert _follower(use_degrees=True).units_per_motor_degree("shoulder_lift") == 1.0


def test_it_writes_its_own_gains_and_shares_the_plain_followers_calibration():
    """The gains the feed-forward's alpha was measured at, written at connect; the motors' calibration is the arm's,
    so the plain follower's file is read, not a copy that could drift from it."""
    robot = _follower()
    robot.bus = _bus()
    robot.bus.motors = dict.fromkeys(MOTOR_NAMES)
    SO107FollowerGravity.configure(robot)
    written = {(c.args[0], c.args[1]): c.args[2] for c in robot.bus.write.call_args_list}
    for m in MOTOR_NAMES:
        assert (
            written[("P_Coefficient", m)],
            written[("I_Coefficient", m)],
            written[("D_Coefficient", m)],
        ) == (32, 0, 32)
    assert robot.calibration_dir == HF_LEROBOT_CALIBRATION / ROBOTS / "so107_follower"


def test_only_the_gravity_type_takes_gains_and_a_feed_forward():
    """A robot profile names the type; the plain follower refuses the gravity fields, so a profile cannot half-set
    them, and the gravity type is built by LeRobot's own factory."""
    config = draccus.decode(
        RobotConfig, {"type": "so107_follower_gravity", "port": "/dev/null", "id": "test_arm", "arm": "left"}
    )
    assert isinstance(config, SO107FollowerGravityRobotConfig) and config.p_coefficient == 32
    with patch("lerobot.robots.so_follower.so_follower.FeetechMotorsBus"):
        assert isinstance(make_robot_from_config(config), SO107FollowerGravity)
    with pytest.raises(draccus.utils.DecodingError):
        draccus.decode(RobotConfig, {"type": "so107_follower", "port": "/dev/null", "gravity_ff_alpha": 3.2})
    with pytest.raises(ValueError):
        SO107FollowerGravityRobotConfig(port="/dev/null", arm="middle")


def test_alpha_scales_with_the_gain_we_measured():
    """Guard the two rig-measured facts the defaults rely on: one compliance for every loaded
    joint, and the folded easy pose loads the elbow more than the lift."""
    g = GravityFeedForward(1.0, LEFT_ARM_ALIGNMENT).gravity_torque_nm(
        np.array([EASY_POSE[m] for m in MOTOR_NAMES])
    )
    assert abs(g[MOTOR_NAMES.index("elbow_flex")]) > abs(g[MOTOR_NAMES.index("shoulder_lift")])
    assert np.all(np.abs(g) < 2.0), "masses are placeholders if any joint sees > 2 N*m at rest"
