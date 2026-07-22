#!/usr/bin/env python

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

"""Hardware-free tests for OpenArmFollower.send_action control features.

Uses a stub CAN bus (no python-can, no hardware): the follower is constructed
with DamiaoMotorsBus monkeypatched out, and the stub records every batch
command so the MIT slots (kp, kd, pos, vel, torque) can be asserted.
"""

import logging
import time

import numpy as np
import pytest

from lerobot.robots.openarm_follower import openarm_follower as of
from lerobot.robots.openarm_follower.config_openarm_follower import OpenArmFollowerConfig
from lerobot.utils.import_utils import _mujoco_available, _openarm_mujoco_available

MOTORS = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]

FRONT_MID_RIGHT_RAD = [0.9007, -0.1745, 0.1079, 0.0, 0.1078, 0.7854, -0.2766]


class StubBus:
    """Records MIT batch commands; serves configurable measured states."""

    def __init__(self, port, motors, **kwargs):
        self.motors = motors
        self.is_connected = True
        self.is_calibrated = True
        self.states = {
            name: {"position": 0.0, "velocity": 0.0, "torque": 0.0, "temp_mos": 40.0, "temp_rotor": 30.0}
            for name in motors
        }
        self.sent: list[dict] = []
        self.posforce_sent: list[dict] = []
        self.zero_calls = 0

    def connect(self, handshake=True):
        self.is_connected = True

    def configure_motors(self):
        pass

    def set_zero_position(self, motors=None):
        self.zero_calls += 1

    def enable_torque(self, motors=None, num_retry=0):
        pass

    def sync_read_all_states(self, motors=None, *, num_retry=0):
        return {n: dict(s) for n, s in self.states.items()}

    def get_cached_states(self):
        return {n: dict(s) for n, s in self.states.items()}

    def sync_read(self, data_name, motors=None):
        assert data_name == "Present_Position"
        return {n: s["position"] for n, s in self.states.items()}

    def mit_control_batch(self, commands):
        self.sent.append(dict(commands))

    def posforce_control(self, motor, position_rad, speed_rad_s, current_pu):
        self.posforce_sent.append(
            {
                "motor": motor,
                "position_rad": position_rad,
                "speed_rad_s": speed_rad_s,
                "current_pu": current_pu,
            }
        )


def make_follower(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(of, "DamiaoMotorsBus", StubBus)
    side = overrides.pop("side", "right")
    config = OpenArmFollowerConfig(
        port="can0",
        id="test_arm",
        calibration_dir=tmp_path,
        side=side,
        **overrides,
    )
    follower = of.OpenArmFollower(config)
    return follower


def action(**positions_deg):
    return {f"{name}.pos": float(pos) for name, pos in positions_deg.items()}


def full_action(**positions_deg):
    """Action covering every motor (zeros by default), like a real teleop cycle."""
    return action(**{**dict.fromkeys(MOTORS, 0.0), **positions_deg})


def last_cmd(follower, motor):
    return follower.bus.sent[-1][motor]


def last_gripper_cmd(follower):
    return follower.bus.posforce_sent[-1]


def test_default_sends_zero_feedforward(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    follower.send_action(action(joint_1=5.0, gripper=-10.0))
    kp, kd, pos, vel, torque = last_cmd(follower, "joint_1")
    assert (kp, kd) == (70.0, 2.75)  # OpenArm 2.0 standard gains
    assert pos == pytest.approx(5.0)
    assert vel == 0.0 and torque == 0.0
    assert "gripper" not in follower.bus.sent[-1]
    assert last_gripper_cmd(follower) == {
        "motor": "gripper",
        "position_rad": pytest.approx(np.radians(-10.0)),
        "speed_rad_s": 50.0,
        "current_pu": pytest.approx(1.0 / 4.5),
    }


def test_connect_does_not_rewrite_motor_zero(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch)
    follower.bus.is_connected = False

    follower.connect(calibrate=False)

    assert follower.bus.zero_calls == 0


def test_explicit_mit_gripper_compatibility_mode(tmp_path, monkeypatch):
    follower = make_follower(tmp_path, monkeypatch, gripper_control_mode="mit")

    follower.send_action(action(gripper=-10.0))

    assert last_cmd(follower, "gripper")[2] == pytest.approx(-10.0)
    assert follower.bus.posforce_sent == []


def test_side_selects_standard_joint_limits(tmp_path, monkeypatch):
    right = make_follower(tmp_path, monkeypatch, side="right")
    assert right.config.joint_limits["joint_1"] == (-80.0, 200.0)
    assert right.config.joint_limits["gripper"] == (-45.0, 0.0)
    left = make_follower(tmp_path / "left", monkeypatch, side="left")
    assert left.config.joint_limits["joint_1"] == (-200.0, 80.0)
    assert left.config.joint_limits["joint_2"] == (-190.0, 10.0)
    assert left.config.joint_limits["gripper"] == (0.0, 45.0)


class TestAlignRamp:
    def test_first_command_ramps_from_measured_pose(self, tmp_path, monkeypatch):
        step_deg = np.degrees(0.003)
        follower = make_follower(tmp_path, monkeypatch, align_step_limit=0.003)
        follower.bus.states["joint_1"]["position"] = 2.0  # measured pose
        follower.send_action(action(joint_1=10.0, joint_2=10.0, gripper=-30.0))
        # Arm joints move one step from the MEASURED pose, not from zero.
        assert last_cmd(follower, "joint_1")[2] == pytest.approx(2.0 + step_deg)
        assert last_cmd(follower, "joint_2")[2] == pytest.approx(0.0 + step_deg)
        # Gripper is EXCLUDED from the clamp: it rides unclipped to the target.
        assert last_gripper_cmd(follower)["position_rad"] == pytest.approx(np.radians(-30.0))

    def test_ramp_converges_step_by_step(self, tmp_path, monkeypatch):
        step_deg = np.degrees(0.003)
        follower = make_follower(tmp_path, monkeypatch, align_step_limit=0.003)
        target = 1.0
        prev = 0.0
        for _ in range(50):
            follower.send_action(action(joint_1=target))
            cmd = last_cmd(follower, "joint_1")[2]
            assert abs(cmd - prev) <= step_deg + 1e-9
            prev = cmd
        assert prev == pytest.approx(target)

    def test_ramp_disabled_by_default(self, tmp_path, monkeypatch):
        follower = make_follower(tmp_path, monkeypatch)
        follower.send_action(action(joint_1=5.0))
        assert last_cmd(follower, "joint_1")[2] == pytest.approx(5.0)


class TestJumpGuard:
    def test_jump_logged_with_joint_name(self, tmp_path, monkeypatch, caplog):
        follower = make_follower(tmp_path, monkeypatch, align_step_limit=0.003, align_jump_threshold=0.05)
        follower.send_action(action(joint_1=0.0))  # establishes last command
        with caplog.at_level(logging.WARNING, logger=of.__name__):
            follower.send_action(action(joint_1=5.0))  # ~0.09 rad jump > 0.05
        assert "target jumped" in caplog.text
        assert "joint_1" in caplog.text
        # The ramp still rate-limits the actual command.
        assert last_cmd(follower, "joint_1")[2] == pytest.approx(np.degrees(0.003))

    def test_jump_log_rate_limited(self, tmp_path, monkeypatch, caplog):
        follower = make_follower(tmp_path, monkeypatch, align_step_limit=0.003, align_jump_threshold=0.05)
        follower.send_action(action(joint_1=0.0))
        with caplog.at_level(logging.WARNING, logger=of.__name__):
            follower.send_action(action(joint_1=5.0))
            follower.send_action(action(joint_1=10.0))  # another jump < 2 s later
        assert caplog.text.count("target jumped") == 1

    def test_small_move_not_logged(self, tmp_path, monkeypatch, caplog):
        follower = make_follower(tmp_path, monkeypatch, align_step_limit=0.003, align_jump_threshold=0.5)
        follower.send_action(action(joint_1=0.0))
        with caplog.at_level(logging.WARNING, logger=of.__name__):
            follower.send_action(action(joint_1=5.0))  # ~0.09 rad < 0.5 threshold
        assert "target jumped" not in caplog.text


class TestVelocityFF:
    def test_unclamped_finite_difference(self, tmp_path, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: clock[0])
        follower = make_follower(tmp_path, monkeypatch, velocity_ff_gain=1.0)
        follower.send_action(action(joint_1=0.0))
        assert last_cmd(follower, "joint_1")[3] == 0.0  # no previous command
        clock[0] += 0.5
        follower.send_action(action(joint_1=1.0))  # 1 deg in 0.5 s = 2 deg/s
        assert last_cmd(follower, "joint_1")[3] == pytest.approx(2.0)

    def test_gain_scales_velocity(self, tmp_path, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: clock[0])
        follower = make_follower(tmp_path, monkeypatch, velocity_ff_gain=0.5)
        follower.send_action(action(joint_1=0.0))
        clock[0] += 0.5
        follower.send_action(action(joint_1=1.0))
        assert last_cmd(follower, "joint_1")[3] == pytest.approx(1.0)

    def test_clamped_to_joint_delta_limit(self, tmp_path, monkeypatch):
        follower = make_follower(tmp_path, monkeypatch, velocity_ff_gain=1.0)
        follower.send_action(action(joint_1=0.0))
        follower.send_action(action(joint_1=45.0))  # huge step in ~ms -> clamp
        # joint_1 delta limit: 1.8 rad/s.
        assert last_cmd(follower, "joint_1")[3] == pytest.approx(np.degrees(1.8))

    def test_gripper_uses_posforce_speed_not_mit_velocity_feedforward(self, tmp_path, monkeypatch):
        follower = make_follower(tmp_path, monkeypatch, velocity_ff_gain=1.0)
        follower.send_action(action(gripper=0.0))
        follower.send_action(action(gripper=-45.0))
        assert "gripper" not in follower.bus.sent[-1]
        assert last_gripper_cmd(follower)["position_rad"] == pytest.approx(np.radians(-45.0))
        assert last_gripper_cmd(follower)["speed_rad_s"] == 50.0


class TestTelemetryWiring:
    def test_send_action_accumulates_from_cache(self, tmp_path, monkeypatch):
        follower = make_follower(tmp_path, monkeypatch)
        follower.bus.states["joint_1"]["torque"] = 3.0
        follower.bus.states["joint_1"]["temp_mos"] = 47.0
        follower.send_action(full_action(joint_1=5.0))
        t = follower._telemetry
        assert t.count == 1
        assert t.err_max[0] == pytest.approx(5.0)  # cmd 5.0 vs measured 0.0
        assert t.tau_abs_max[0] == pytest.approx(3.0)
        assert t.tmax == 47.0

    def test_report_emitted_via_logging(self, tmp_path, monkeypatch, caplog):
        follower = make_follower(tmp_path, monkeypatch)
        follower.send_action(full_action(joint_1=5.0))
        with caplog.at_level(logging.INFO, logger="lerobot.robots.openarm_follower.telemetry"):
            assert follower._telemetry.maybe_report(now=follower._telemetry.t0 + 31.0)
        assert "[test_arm] telem n=1" in caplog.text


@pytest.mark.skipif(
    not (_mujoco_available and _openarm_mujoco_available),
    reason="mujoco / openarm-mujoco not available (extra: openarm-ff)",
)
class TestGravityFFWiring:
    def test_torque_slot_carries_gravity_ff(self, tmp_path, monkeypatch):
        follower = make_follower(tmp_path, monkeypatch, gravity_ff_gain=0.9)
        follower._gravity_ff.fade_secs = 0.0  # skip fade-in for determinism
        for i, q in enumerate(FRONT_MID_RIGHT_RAD):
            follower.bus.states[f"joint_{i + 1}"]["position"] = float(np.degrees(q))
        follower.send_action(full_action(joint_1=30.0, gripper=-10.0))
        expected = 0.9 * follower._gravity_ff.raw_tau(FRONT_MID_RIGHT_RAD)
        assert last_cmd(follower, "joint_1")[4] == pytest.approx(expected[0], rel=1e-3)
        assert last_cmd(follower, "joint_2")[4] == pytest.approx(expected[1], rel=1e-3)
        # Gripper is sent through its independent POS_FORCE frame.
        assert "gripper" not in follower.bus.sent[-1]
        assert last_gripper_cmd(follower)["position_rad"] == pytest.approx(np.radians(-10.0))
        # tff actually sent is tracked for telemetry.
        np.testing.assert_allclose(follower._last_tff, expected, rtol=1e-3)

    def test_arms_down_pose_gives_near_zero_torque(self, tmp_path, monkeypatch):
        follower = make_follower(tmp_path, monkeypatch, gravity_ff_gain=0.9)
        follower._gravity_ff.fade_secs = 0.0
        follower.send_action(full_action(joint_1=0.0))
        assert abs(last_cmd(follower, "joint_1")[4]) < 0.2

    def test_gain_zero_disables(self, tmp_path, monkeypatch):
        follower = make_follower(tmp_path, monkeypatch, gravity_ff_gain=0.0)
        assert follower._gravity_ff is None
        follower.send_action(action(joint_1=5.0))
        assert last_cmd(follower, "joint_1")[4] == 0.0
