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

"""Hardware-free tests for the Damiao MIT control public API and packet encoding."""

import struct

import numpy as np
import pytest

from lerobot.utils.import_utils import _can_available

if not _can_available:
    pytest.skip("python-can not available", allow_module_level=True)

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MIT_KD_RANGE, MIT_KP_RANGE, MOTOR_LIMIT_PARAMS, MotorType

PMAX, VMAX, TMAX = MOTOR_LIMIT_PARAMS[MotorType.DM8009]


@pytest.fixture
def bus():
    motor = Motor(0x01, "dm8009", MotorNormMode.DEGREES)
    motor.recv_id = 0x11
    motor.motor_type_str = "dm8009"
    # Never connected: encoding and delegation need no CAN hardware.
    return DamiaoMotorsBus(port="can0", motors={"joint_1": motor})


def _unpack_mit_packet(bus, data):
    """Invert the bit packing of _encode_mit_packet (unsigned ints)."""
    q_uint = (data[0] << 8) | data[1]
    dq_uint = (data[2] << 4) | (data[3] >> 4)
    kp_uint = ((data[3] & 0xF) << 8) | data[4]
    kd_uint = (data[5] << 4) | (data[6] >> 4)
    tau_uint = ((data[6] & 0xF) << 8) | data[7]
    return (
        bus._uint_to_float(kp_uint, *MIT_KP_RANGE, 12),
        bus._uint_to_float(kd_uint, *MIT_KD_RANGE, 12),
        bus._uint_to_float(q_uint, -PMAX, PMAX, 16),
        bus._uint_to_float(dq_uint, -VMAX, VMAX, 12),
        bus._uint_to_float(tau_uint, -TMAX, TMAX, 12),
    )


def test_encode_roundtrip_with_velocity_and_torque(bus):
    kp, kd, pos_deg, vel_deg_s, torque = 70.0, 2.75, 30.0, 45.0, 5.0
    data = bus._encode_mit_packet(MotorType.DM8009, kp, kd, pos_deg, vel_deg_s, torque)
    assert len(data) == 8
    kp_rt, kd_rt, pos_rt, vel_rt, tau_rt = _unpack_mit_packet(bus, data)
    assert kp_rt == pytest.approx(kp, abs=500 / 4095)
    assert kd_rt == pytest.approx(kd, abs=5 / 4095)
    assert np.degrees(pos_rt) == pytest.approx(pos_deg, abs=np.degrees(25 / 65535))
    assert np.degrees(vel_rt) == pytest.approx(vel_deg_s, abs=np.degrees(2 * VMAX / 4095))
    assert tau_rt == pytest.approx(torque, abs=2 * TMAX / 4095)


def test_encode_zero_feedforwards_hit_range_midpoints(bus):
    data = bus._encode_mit_packet(MotorType.DM8009, 0.0, 0.0, 0.0, 0.0, 0.0)
    _, _, pos_rt, vel_rt, tau_rt = _unpack_mit_packet(bus, data)
    assert pos_rt == pytest.approx(0.0, abs=25 / 65535)
    assert vel_rt == pytest.approx(0.0, abs=2 * VMAX / 4095)
    assert tau_rt == pytest.approx(0.0, abs=2 * TMAX / 4095)


def test_encode_clamps_to_motor_limits(bus):
    data = bus._encode_mit_packet(MotorType.DM8009, 1e4, 1e4, 1e4, 1e4, 1e4)
    _, _, pos_rt, vel_rt, tau_rt = _unpack_mit_packet(bus, data)
    assert pos_rt == pytest.approx(PMAX, abs=25 / 65535)
    assert vel_rt == pytest.approx(VMAX, abs=2 * VMAX / 4095)
    assert tau_rt == pytest.approx(TMAX, abs=2 * TMAX / 4095)


def test_public_mit_control_delegates(bus, monkeypatch):
    calls = []
    monkeypatch.setattr(bus, "_mit_control", lambda *args: calls.append(args))
    bus.mit_control("joint_1", 70.0, 2.75, 30.0, 10.0, 1.5)
    bus.mit_control("joint_1", 70.0, 2.75, 30.0)  # default vel/torque feedforwards
    assert calls == [
        ("joint_1", 70.0, 2.75, 30.0, 10.0, 1.5),
        ("joint_1", 70.0, 2.75, 30.0, 0.0, 0.0),
    ]


def test_public_mit_control_batch_delegates(bus, monkeypatch):
    calls = []
    monkeypatch.setattr(bus, "_mit_control_batch", lambda commands: calls.append(commands))
    commands = {"joint_1": (70.0, 2.75, 30.0, 10.0, 1.5)}
    bus.mit_control_batch(commands)
    assert calls == [commands]


def test_get_cached_states_returns_copies(bus):
    states = bus.get_cached_states()
    assert set(states) == {"joint_1"}
    assert states["joint_1"]["position"] == 0.0
    states["joint_1"]["position"] = 999.0
    assert bus.get_cached_states()["joint_1"]["position"] == 0.0


def test_encode_posforce_packet_uses_official_little_endian_layout(bus):
    payload = bus._encode_posforce_packet(
        "joint_1",
        position_rad=0.25,
        speed_rad_s=5.5,
        current_pu=0.12,
    )

    assert len(payload) == 8
    position, speed_scaled, current_scaled = struct.unpack("<fHH", payload)
    assert position == pytest.approx(0.25)
    assert speed_scaled == 550
    assert current_scaled == 1200


@pytest.mark.parametrize(
    ("position", "speed", "current", "message"),
    [
        (float("nan"), 1.0, 0.1, "finite"),
        (0.0, -0.1, 0.1, "speed_rad_s"),
        (0.0, 100.1, 0.1, "speed_rad_s"),
        (0.0, 1.0, -0.1, "current_pu"),
        (0.0, 1.0, 1.1, "current_pu"),
    ],
)
def test_encode_posforce_packet_rejects_invalid_values(bus, position, speed, current, message):
    with pytest.raises(ValueError, match=message):
        bus._encode_posforce_packet("joint_1", position, speed, current)


def test_posforce_control_uses_offset_arbitration_id(bus, monkeypatch):
    class FakeCanBus:
        def __init__(self):
            self.sent = []

        def send(self, message):
            self.sent.append(message)

    fake = FakeCanBus()
    bus.canbus = fake
    bus._is_connected = True
    bus.posforce_control("joint_1", position_rad=0.25, speed_rad_s=5.0, current_pu=0.1)

    assert len(fake.sent) == 1
    assert fake.sent[0].arbitration_id == 0x301
    assert bytes(fake.sent[0].data) == bus._encode_posforce_packet("joint_1", 0.25, 5.0, 0.1)
