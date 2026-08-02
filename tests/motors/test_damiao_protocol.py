# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Damiao wire-protocol contract, pinned by frames captured from real hardware.

These tests exist because the driver's correctness depends on facts about the
*motor* that the source cannot state and a reader cannot derive. The specific
one that bit us: a parameter write goes out on the broadcast parameter channel
(``0x7FF``), but the firmware echoes its acknowledgement on the motor's **master
ID** — the same ID that carries state feedback. Read the code alone and
``set_control_mode`` looks like it listens on the wrong channel, because every
other path treats master-ID frames as state feedback. It is not wrong; the
firmware simply multiplexes both onto that ID.

A 2026-08-02 review reported that as a blocking bug on exactly that reasoning.
It cost a round trip to the robot to disprove. Encoding the frame here means the
next reader — human or automated — gets the answer from the test suite instead
of from a robot.

Fixture provenance
------------------
``ACK_CTRL_MODE_TORQUE_POS`` is a byte-for-byte capture from an OpenArm 2.0 rig,
2026-07-25, both CAN ports, taken from the driver's own diagnostics::

    CAN_CTRL_MODE_TX  port=can0 motor=gripper tx_id=0x7FF expected_rx_id=0x18
                      mode=TORQUE_POS data=0800550a04000000
    CAN_CTRL_MODE_ACK port=can0 motor=gripper rx_id=0x18
                      mode=TORQUE_POS data=0800550a04000000

These bytes are a property of the motor firmware, not of this repository, so they
do not go stale as the driver is refactored. If a firmware revision ever moves
the acknowledgement to another CAN ID, these tests are the intended place to
record that — update the fixture and the reason, do not delete the coverage.
"""

from __future__ import annotations

import struct

import pytest

from lerobot.utils.import_utils import _can_available

if not _can_available:
    pytest.skip("python-can not available", allow_module_level=True)

import can  # noqa: E402

from lerobot.motors.damiao.damiao import DamiaoMotorsBus  # noqa: E402
from lerobot.motors.damiao.tables import CAN_PARAM_ID, ControlMode  # noqa: E402

GRIPPER_MOTOR_ID = 0x08
GRIPPER_MASTER_ID = 0x18  # where BOTH state feedback and parameter acks arrive

# id_lo, id_hi, 0x55 (write-param), 0x0a (CTRL_MODE), then the mode as uint32-le.
ACK_CTRL_MODE_TORQUE_POS = bytes.fromhex("0800550a04000000")


class FakeCanBus:
    """A CAN bus that sends into a list and replays queued frames in order."""

    def __init__(self, replies: list[can.Message] | None = None):
        self.sent: list[can.Message] = []
        self._replies = list(replies or [])

    def send(self, msg):
        self.sent.append(msg)

    def recv(self, timeout=None):
        return self._replies.pop(0) if self._replies else None


def _frame(arbitration_id: int, data: bytes) -> can.Message:
    return can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False)


def _bus(replies: list[can.Message] | None = None) -> DamiaoMotorsBus:
    """A bus wired for the gripper only, with no hardware and no connect()."""
    bus = DamiaoMotorsBus.__new__(DamiaoMotorsBus)
    bus.port = "can-test"
    bus.canbus = FakeCanBus(replies)
    bus.use_can_fd = True
    bus._is_connected = True  # posforce_control is guarded by @check_if_not_connected
    bus._recv_id_to_motor = {GRIPPER_MASTER_ID: "gripper"}
    bus._get_motor_id = lambda motor: GRIPPER_MOTOR_ID
    bus._get_motor_recv_id = lambda motor: GRIPPER_MASTER_ID
    bus._get_motor_name = lambda motor: "gripper"
    return bus


class TestControlModeAcknowledgement:
    """The protocol fact a code reader cannot derive."""

    def test_ack_arrives_on_the_master_id_not_the_parameter_channel(self):
        """THE fixture test: the write goes to 0x7FF, the ack comes back on 0x18."""
        bus = _bus([_frame(GRIPPER_MASTER_ID, ACK_CTRL_MODE_TORQUE_POS)])

        bus.set_control_mode("gripper", ControlMode.TORQUE_POS)  # must not raise

        assert len(bus.canbus.sent) == 1
        assert bus.canbus.sent[0].arbitration_id == CAN_PARAM_ID, (
            "the CTRL_MODE write must go out on the broadcast parameter channel"
        )

    def test_ack_for_a_different_mode_is_rejected(self):
        """A motor that acknowledges a mode we did not request must fail loudly."""
        wrong_mode = ACK_CTRL_MODE_TORQUE_POS[:4] + struct.pack("<I", int(ControlMode.MIT))
        bus = _bus([_frame(GRIPPER_MASTER_ID, wrong_mode)])

        with pytest.raises(RuntimeError, match="invalid CTRL_MODE acknowledgement"):
            bus.set_control_mode("gripper", ControlMode.TORQUE_POS)

    def test_missing_ack_warns_but_does_not_raise(self, caplog):
        """Deliberate: a lost ack must not abort a connect that is otherwise fine.

        The cost is that a genuinely unapplied mode is only a warning, so this
        line is the single signal that the gripper may still be in MIT mode.
        """
        bus = _bus([])  # motor says nothing

        with caplog.at_level("WARNING"):
            bus.set_control_mode("gripper", ControlMode.TORQUE_POS)

        assert "No CTRL_MODE acknowledgement" in caplog.text

    def test_a_frame_on_the_parameter_channel_does_not_satisfy_the_ack(self, caplog):
        """Pins the observed firmware behaviour, not an assumption about it.

        This is the exact inverse of what the 2026-08 review assumed. If a future
        firmware really does reply on 0x7FF, this test fails and is the place to
        record the change — together with the rig capture that proves it.
        """
        bus = _bus([_frame(CAN_PARAM_ID, ACK_CTRL_MODE_TORQUE_POS)])

        with caplog.at_level("WARNING"):
            bus.set_control_mode("gripper", ControlMode.TORQUE_POS)

        assert "No CTRL_MODE acknowledgement" in caplog.text


class TestPosForceFrame:
    """POS_FORCE command encoding and its deliberate fire-and-forget design."""

    def test_encoding_matches_the_documented_layout(self):
        bus = _bus()
        bus._motor_types = {"gripper": next(iter(_motor_types_for_limits()))}

        payload = bus._encode_posforce_packet("gripper", 0.5, 20.0, 0.25)

        position, speed, current = struct.unpack("<fHH", payload)
        assert len(payload) == 8, "POS_FORCE payload is exactly one CAN frame"
        assert position == pytest.approx(0.5)
        assert speed == 2000, "speed is rad/s scaled by 100"
        assert current == 2500, "current is per-unit scaled by 10000"

    def test_command_is_addressed_to_the_posforce_offset(self):
        bus = _bus()
        bus._motor_types = {"gripper": next(iter(_motor_types_for_limits()))}

        bus.posforce_control("gripper", 0.5, 20.0, 0.25)

        assert bus.canbus.sent[0].arbitration_id == GRIPPER_MOTOR_ID + 0x300

    def test_command_deliberately_does_not_consume_a_reply(self):
        """Documents the design, so the asymmetry is not read as an oversight.

        Every other command path consumes exactly one feedback frame; this one
        does not. Any reply the motor sends is drained by the next state refresh
        — observed on hardware as a single extra frame (messages_seen=9 against
        received=8) that never accumulates, with no motor ever serving a stale
        reading. If that ever changes, the CAN_REFRESH diagnostic reports it as
        a rising messages_seen or a non-empty unexpected_ids.
        """
        reply = _frame(GRIPPER_MASTER_ID, ACK_CTRL_MODE_TORQUE_POS)
        bus = _bus([reply])
        bus._motor_types = {"gripper": next(iter(_motor_types_for_limits()))}

        bus.posforce_control("gripper", 0.5, 20.0, 0.25)

        assert bus.canbus._replies == [reply], "posforce_control must not read the bus"

    @pytest.mark.parametrize(
        ("position", "speed", "current"),
        [
            (0.5, 20.0, 1.5),  # current above the per-unit ceiling
            (0.5, 200.0, 0.25),  # speed above the documented range
            (float("nan"), 20.0, 0.25),  # non-finite
        ],
    )
    def test_out_of_range_values_are_refused_before_transmission(self, position, speed, current):
        """A malformed frame reaches a torque-producing motor; refuse it here."""
        bus = _bus()
        bus._motor_types = {"gripper": next(iter(_motor_types_for_limits()))}

        with pytest.raises(ValueError):
            bus._encode_posforce_packet("gripper", position, speed, current)
        assert bus.canbus.sent == []


def _motor_types_for_limits():
    """Any motor type present in MOTOR_LIMIT_PARAMS; the tests use its pmax only."""
    from lerobot.motors.damiao.tables import MOTOR_LIMIT_PARAMS

    return MOTOR_LIMIT_PARAMS
