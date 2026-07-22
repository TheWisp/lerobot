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

"""Hardware-free protocol and fail-closed tests for the Damiao CAN bus."""

import struct
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import (
    DamiaoMotorsBus,
    MotorFeedbackError,
    MotorStateUnavailableError,
    damiao as damiao_module,
)
from lerobot.motors.damiao.tables import (
    CAN_CMD_ENABLE,
    CAN_CMD_REFRESH,
    CAN_PARAM_ID,
    MIT_KD_RANGE,
    MIT_KP_RANGE,
    MOTOR_LIMIT_PARAMS,
    MotorType,
)


class FakeMessage:
    def __init__(
        self,
        arbitration_id: int,
        data: list[int] | bytes | bytearray,
        is_extended_id: bool = False,
        is_fd: bool = True,
        dlc: int | None = None,
        is_error_frame: bool = False,
        is_remote_frame: bool = False,
        is_rx: bool = True,
    ) -> None:
        self.arbitration_id = arbitration_id
        self.data = bytearray(data)
        self.is_extended_id = is_extended_id
        self.is_fd = is_fd
        self.dlc = len(self.data) if dlc is None else dlc
        self.is_error_frame = is_error_frame
        self.is_remote_frame = is_remote_frame
        self.is_rx = is_rx


class FakeCanBus:
    def __init__(self, response_factory: Callable[[FakeMessage], FakeMessage | None] | None = None) -> None:
        self.sent: list[FakeMessage] = []
        self._responses: list[FakeMessage] = []
        self._response_factory = response_factory
        self.was_shutdown = False

    def send(self, message: FakeMessage) -> None:
        self.sent.append(message)
        if self._response_factory is not None:
            response = self._response_factory(message)
            if response is not None:
                self._responses.append(response)

    def recv(self, timeout: float) -> FakeMessage | None:
        del timeout
        return self._responses.pop(0) if self._responses else None

    def shutdown(self) -> None:
        self.was_shutdown = True


@pytest.fixture(autouse=True)
def fake_python_can(monkeypatch):
    monkeypatch.setattr(damiao_module, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(damiao_module, "can", SimpleNamespace(Message=FakeMessage))


@pytest.fixture
def bus() -> DamiaoMotorsBus:
    motors = {
        "joint_1": Motor(
            id=0x01,
            model="dm8009",
            norm_mode=MotorNormMode.DEGREES,
            motor_type_str="dm8009",
            recv_id=0x11,
        ),
        "joint_2": Motor(
            id=0x02,
            model="dm8009",
            norm_mode=MotorNormMode.DEGREES,
            motor_type_str="dm8009",
            recv_id=0x12,
        ),
    }
    return DamiaoMotorsBus(port="virtual", motors=motors, state_timeout_s=0.1)


def make_feedback(
    bus: DamiaoMotorsBus,
    recv_id: int,
    *,
    position_degrees: float = 30.0,
    velocity_deg_per_sec: float = 5.0,
    torque: float = 1.0,
    status: int = 1,
    embedded_id: int | None = None,
) -> FakeMessage:
    pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[MotorType.DM8009]
    q_uint = bus._float_to_uint(np.radians(position_degrees), -pmax, pmax, 16)
    dq_uint = bus._float_to_uint(np.radians(velocity_deg_per_sec), -vmax, vmax, 12)
    tau_uint = bus._float_to_uint(torque, -tmax, tmax, 12)
    data = [
        ((status & 0x0F) << 4) | ((recv_id if embedded_id is None else embedded_id) & 0x0F),
        (q_uint >> 8) & 0xFF,
        q_uint & 0xFF,
        (dq_uint >> 4) & 0xFF,
        ((dq_uint & 0x0F) << 4) | ((tau_uint >> 8) & 0x0F),
        tau_uint & 0xFF,
        31,
        32,
    ]
    return FakeMessage(arbitration_id=recv_id, data=data)


def make_control_mode_response(
    *,
    recv_id: int = 0x11,
    motor_id: int = 0x01,
    opcode: int = 0x33,
    rid: int = 10,
    mode: int = 2,
    dlc: int | None = None,
    is_error_frame: bool = False,
    is_remote_frame: bool = False,
    is_rx: bool = True,
) -> FakeMessage:
    data = [
        motor_id & 0xFF,
        (motor_id >> 8) & 0xFF,
        opcode,
        rid,
        *mode.to_bytes(4, byteorder="little", signed=False),
    ]
    return FakeMessage(
        arbitration_id=recv_id,
        data=data,
        dlc=dlc,
        is_error_frame=is_error_frame,
        is_remote_frame=is_remote_frame,
        is_rx=is_rx,
    )


def attach_fake_bus(bus: DamiaoMotorsBus, fake_bus: FakeCanBus) -> None:
    bus.canbus = fake_bus
    bus._is_connected = True


def unpack_mit_command(bus: DamiaoMotorsBus, data: bytearray) -> tuple[float, float, float, float, float]:
    pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[MotorType.DM8009]
    q_uint = (data[0] << 8) | data[1]
    dq_uint = (data[2] << 4) | (data[3] >> 4)
    kp_uint = ((data[3] & 0x0F) << 8) | data[4]
    kd_uint = (data[5] << 4) | (data[6] >> 4)
    tau_uint = ((data[6] & 0x0F) << 8) | data[7]
    return (
        bus._uint_to_float(kp_uint, *MIT_KP_RANGE, 12),
        bus._uint_to_float(kd_uint, *MIT_KD_RANGE, 12),
        np.degrees(bus._uint_to_float(q_uint, -pmax, pmax, 16)),
        np.degrees(bus._uint_to_float(dq_uint, -vmax, vmax, 12)),
        bus._uint_to_float(tau_uint, -tmax, tmax, 12),
    )


def test_handshake_is_read_only_and_decodes_received_feedback(bus: DamiaoMotorsBus) -> None:
    def response_for(message: FakeMessage) -> FakeMessage | None:
        if message.arbitration_id != CAN_PARAM_ID:
            return None
        return make_feedback(bus, message.data[0] + 0x10, position_degrees=42.0)

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)

    bus._handshake()

    assert len(fake_bus.sent) == 4
    assert all(message.arbitration_id == CAN_PARAM_ID for message in fake_bus.sent)
    assert all(message.data[2] == CAN_CMD_REFRESH for message in fake_bus.sent)
    assert all(message.data[-1] != CAN_CMD_ENABLE for message in fake_bus.sent)
    assert bus.get_cached_states()["joint_1"]["position"] == pytest.approx(42.0, abs=0.05)


def test_handshake_failure_only_latches_software_fault(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus()
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(ConnectionError, match="did not respond"):
        bus._handshake()

    assert bus.fault_latched
    assert all(message.arbitration_id == CAN_PARAM_ID for message in fake_bus.sent)
    assert all(message.data[2] == CAN_CMD_REFRESH for message in fake_bus.sent)


def test_public_mit_control_returns_received_feedback(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus(
        lambda message: make_feedback(bus, message.arbitration_id + 0x10, position_degrees=17.0)
    )
    attach_fake_bus(bus, fake_bus)

    state = bus.mit_control("joint_1", 70.0, 2.75, 17.0, 45.0, 5.0)

    assert state["position"] == pytest.approx(17.0, abs=0.05)
    assert fake_bus.sent[0].arbitration_id == 0x01
    assert len(fake_bus.sent[0].data) == 8
    kp, kd, position, velocity, torque = unpack_mit_command(bus, fake_bus.sent[0].data)
    assert kp == pytest.approx(70.0, abs=500 / 4095)
    assert kd == pytest.approx(2.75, abs=5 / 4095)
    assert position == pytest.approx(17.0, abs=np.degrees(25 / 65535))
    assert velocity == pytest.approx(45.0, abs=np.degrees(90 / 4095))
    assert torque == pytest.approx(5.0, abs=80 / 4095)


def test_mit_control_requires_enabled_status_and_disables_on_failure(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus(lambda message: make_feedback(bus, 0x11, status=0))
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorFeedbackError, match="expected enabled status 1"):
        bus.mit_control("joint_1", 10.0, 0.5, 0.0)

    assert bus.fault_latched
    assert fake_bus.sent[0].arbitration_id == 0x01
    assert fake_bus.sent[-1].data[-1] == 0xFD


@pytest.mark.parametrize("reply_opcode", [0x33, 0x55])
def test_query_control_mode_is_read_only_and_requires_two_matching_replies(
    bus: DamiaoMotorsBus, reply_opcode: int
) -> None:
    fake_bus = FakeCanBus(lambda message: make_control_mode_response(opcode=reply_opcode, mode=3))
    attach_fake_bus(bus, fake_bus)

    assert bus.query_control_mode("joint_1") == 3

    assert len(fake_bus.sent) == 2
    assert all(message.arbitration_id == CAN_PARAM_ID for message in fake_bus.sent)
    assert all(message.data == bytearray([0x01, 0x00, 0x33, 0x0A, 0, 0, 0, 0]) for message in fake_bus.sent)
    assert not bus.fault_latched


@pytest.mark.parametrize(
    "invalid_reply",
    [
        make_control_mode_response(opcode=0x44),
        make_control_mode_response(rid=11),
        make_control_mode_response(motor_id=2),
        make_control_mode_response(mode=5),
        make_control_mode_response(dlc=7),
        make_control_mode_response(is_error_frame=True),
        make_control_mode_response(is_remote_frame=True),
        make_control_mode_response(is_rx=False),
    ],
    ids=[
        "opcode",
        "rid",
        "embedded-id",
        "mode-range",
        "dlc",
        "error-frame",
        "remote-frame",
        "tx-echo",
    ],
)
def test_invalid_control_mode_reply_only_latches_software_fault(
    bus: DamiaoMotorsBus, invalid_reply: FakeMessage
) -> None:
    fake_bus = FakeCanBus(lambda message: invalid_reply)
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorStateUnavailableError):
        bus.query_control_mode("joint_1")

    assert bus.fault_latched
    assert len(fake_bus.sent) == 1
    assert fake_bus.sent[0].arbitration_id == CAN_PARAM_ID
    assert fake_bus.sent[0].data == bytearray([0x01, 0x00, 0x33, 0x0A, 0, 0, 0, 0])


def test_inconsistent_control_mode_replies_never_send_control_frames(bus: DamiaoMotorsBus) -> None:
    modes = iter([1, 2])
    fake_bus = FakeCanBus(lambda message: make_control_mode_response(mode=next(modes)))
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorFeedbackError, match="Inconsistent"):
        bus.query_control_mode("joint_1")

    assert bus.fault_latched
    assert len(fake_bus.sent) == 2
    assert all(message.arbitration_id == CAN_PARAM_ID for message in fake_bus.sent)
    assert all(message.data[-1] not in (0xFC, 0xFD, 0xFE) for message in fake_bus.sent)


def test_write_control_mode_requires_disabled_refresh_and_verifies_twice(
    bus: DamiaoMotorsBus,
) -> None:
    def response_for(message: FakeMessage) -> FakeMessage | None:
        if message.arbitration_id != CAN_PARAM_ID:
            return None
        if message.data[2] == CAN_CMD_REFRESH:
            return make_feedback(bus, 0x11, status=0)
        if message.data[2] == 0x33:
            return make_control_mode_response(mode=4)
        return None

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)

    assert bus.write_control_mode("joint_1", 4) == 4

    assert len(fake_bus.sent) == 4
    assert fake_bus.sent[0].data == bytearray([0x01, 0x00, CAN_CMD_REFRESH, 0, 0, 0, 0, 0])
    assert fake_bus.sent[1].data == bytearray([0x01, 0x00, 0x55, 0x0A, 4, 0, 0, 0])
    assert fake_bus.sent[2].data == bytearray([0x01, 0x00, 0x33, 0x0A, 0, 0, 0, 0])
    assert fake_bus.sent[3].data == fake_bus.sent[2].data
    assert all(message.arbitration_id == CAN_PARAM_ID for message in fake_bus.sent)
    assert not bus.fault_latched


def test_write_control_mode_refuses_enabled_motor_without_control_side_effects(
    bus: DamiaoMotorsBus,
) -> None:
    fake_bus = FakeCanBus(lambda message: make_feedback(bus, 0x11, status=1))
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorFeedbackError, match="expected disabled status 0"):
        bus.write_control_mode("joint_1", 2)

    assert bus.fault_latched
    assert len(fake_bus.sent) == 1
    assert fake_bus.sent[0].arbitration_id == CAN_PARAM_ID
    assert fake_bus.sent[0].data[2] == CAN_CMD_REFRESH


def test_write_control_mode_verification_failure_never_sends_control_or_disable(
    bus: DamiaoMotorsBus,
) -> None:
    query_modes = iter([2, 3])

    def response_for(message: FakeMessage) -> FakeMessage | None:
        if message.data[2] == CAN_CMD_REFRESH:
            return make_feedback(bus, 0x11, status=0)
        if message.data[2] == 0x33:
            return make_control_mode_response(mode=next(query_modes))
        return None

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorFeedbackError, match="Inconsistent"):
        bus.write_control_mode("joint_1", 2)

    assert bus.fault_latched
    assert len(fake_bus.sent) == 4
    assert all(message.arbitration_id == CAN_PARAM_ID for message in fake_bus.sent)
    assert all(message.data[-1] not in (0xFC, 0xFD, 0xFE) for message in fake_bus.sent)


def test_write_control_mode_rejects_invalid_mode_without_can_io(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus()
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(ValueError, match=r"within \[0, 4\]"):
        bus.write_control_mode("joint_1", 5)

    assert bus.fault_latched
    assert fake_bus.sent == []


def test_posforce_control_uses_official_id_and_little_endian_payload(
    bus: DamiaoMotorsBus, monkeypatch
) -> None:
    fake_bus = FakeCanBus(lambda message: make_feedback(bus, 0x11))
    attach_fake_bus(bus, fake_bus)
    receive_timeouts = []
    original_receive = bus._recv_motor_response

    def receive(*, expected_recv_id=None, timeout=damiao_module.SHORT_TIMEOUT_SEC):
        receive_timeouts.append(timeout)
        return original_receive(expected_recv_id=expected_recv_id, timeout=timeout)

    monkeypatch.setattr(bus, "_recv_motor_response", receive)

    state = bus.posforce_control("joint_1", position_rad=0.25, speed_rad_s=1.5, current_pu=0.2)

    command = fake_bus.sent[0]
    assert command.arbitration_id == 0x301
    assert struct.unpack("<fHH", command.data) == pytest.approx((0.25, 150, 2000))
    assert state["status"] == 1
    assert receive_timeouts[-1] == damiao_module.MEDIUM_TIMEOUT_SEC


def test_posforce_requires_enabled_status_and_disables_on_failure(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus(lambda message: make_feedback(bus, 0x11, status=0))
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorFeedbackError, match="expected enabled status 1"):
        bus.posforce_control("joint_1", position_rad=0.25, speed_rad_s=1.5, current_pu=0.2)

    assert bus.fault_latched
    assert fake_bus.sent[0].arbitration_id == 0x301
    assert fake_bus.sent[-1].data[-1] == 0xFD


def test_posforce_command_sends_without_waiting_for_feedback(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus()
    attach_fake_bus(bus, fake_bus)

    bus.posforce_command("joint_1", position_rad=0.25, speed_rad_s=1.5, current_pu=0.2)

    assert len(fake_bus.sent) == 1
    assert fake_bus.sent[0].arbitration_id == 0x301
    assert struct.unpack("<fHH", fake_bus.sent[0].data) == pytest.approx((0.25, 150, 2000))


@pytest.mark.parametrize(
    ("position", "speed", "current"),
    [
        (float("nan"), 1.0, 0.1),
        (0.0, -0.1, 0.1),
        (0.0, 100.1, 0.1),
        (0.0, 1.0, -0.1),
        (0.0, 1.0, 1.1),
        (13.0, 1.0, 0.1),
    ],
)
def test_posforce_prevalidation_sends_nothing(
    bus: DamiaoMotorsBus, position: float, speed: float, current: float
) -> None:
    fake_bus = FakeCanBus()
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(ValueError):
        bus.posforce_control("joint_1", position, speed, current)

    assert fake_bus.sent == []


@pytest.mark.parametrize(
    "invalid_feedback",
    [
        FakeMessage(0x11, [0] * 8),
        FakeMessage(0x11, [0x12, 1, 2, 3, 4, 5, 6, 7]),
        FakeMessage(0x11, [0x81, 1, 2, 3, 4, 5, 6, 7]),
        FakeMessage(0x11, [0x11, 1, 2, 3, 4, 5, 6], dlc=7),
        FakeMessage(0x12, [0x11, 1, 2, 3, 4, 5, 6, 7]),
        FakeMessage(0x11, [0x11, 1, 2, 3, 4, 5, 6, 7], is_error_frame=True),
    ],
    ids=["all-zero", "wrong-embedded-id", "fault-status", "wrong-dlc", "wrong-can-id", "error-frame"],
)
def test_invalid_feedback_latches_and_sends_fail_safe_disable(
    bus: DamiaoMotorsBus, invalid_feedback: FakeMessage
) -> None:
    fake_bus = FakeCanBus(lambda message: invalid_feedback)
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorStateUnavailableError):
        bus.mit_control("joint_1", 10.0, 0.5, 0.0)

    assert bus.fault_latched
    assert fake_bus.sent[-1].arbitration_id == 0x01
    assert fake_bus.sent[-1].data[-1] == 0xFD


def test_non_finite_decoded_feedback_is_rejected(bus: DamiaoMotorsBus, monkeypatch) -> None:
    feedback = make_feedback(bus, 0x11)
    monkeypatch.setattr(bus, "_decode_motor_state", lambda *args: (float("nan"), 0.0, 0.0, 30, 30))

    with pytest.raises(MotorFeedbackError, match="non-finite"):
        bus._decode_validated_response("joint_1", feedback)


def test_enable_failure_rolls_back_every_target_and_latches(bus: DamiaoMotorsBus) -> None:
    def response_for(message: FakeMessage) -> FakeMessage | None:
        if message.arbitration_id == 0x01 and message.data[-1] == 0xFC:
            return make_feedback(bus, 0x11, status=1)
        return None

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorStateUnavailableError, match="joint_2"):
        bus.enable_torque()

    disable_ids = [message.arbitration_id for message in fake_bus.sent if message.data[-1] == 0xFD]
    assert disable_ids == [0x01, 0x02]
    assert bus.fault_latched


def test_configure_motors_is_a_no_io_compatibility_noop(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus()
    attach_fake_bus(bus, fake_bus)

    bus.configure_motors()

    assert fake_bus.sent == []


def test_torque_disabled_never_reenables_after_context_exception(bus: DamiaoMotorsBus) -> None:
    def response_for(message: FakeMessage) -> FakeMessage | None:
        if message.data[-1] == 0xFD:
            return make_feedback(bus, 0x11, status=0)
        if message.data[-1] == 0xFC:
            return make_feedback(bus, 0x11, status=1)
        return None

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(RuntimeError, match="body failed"):
        with bus.torque_disabled("joint_1", reenable_on_success=True):
            raise RuntimeError("body failed")

    assert [message.data[-1] for message in fake_bus.sent] == [0xFD]


def test_torque_disabled_requires_opt_in_to_reenable_on_clean_exit(bus: DamiaoMotorsBus) -> None:
    def response_for(message: FakeMessage) -> FakeMessage | None:
        return make_feedback(bus, 0x11, status=0 if message.data[-1] == 0xFD else 1)

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)

    with bus.torque_disabled("joint_1"):
        pass

    assert [message.data[-1] for message in fake_bus.sent] == [0xFD]


def test_torque_disabled_can_explicitly_reenable_after_clean_exit(bus: DamiaoMotorsBus) -> None:
    def response_for(message: FakeMessage) -> FakeMessage | None:
        return make_feedback(bus, 0x11, status=0 if message.data[-1] == 0xFD else 1)

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)

    with bus.torque_disabled("joint_1", reenable_on_success=True):
        pass

    assert [message.data[-1] for message in fake_bus.sent] == [0xFD, 0xFC]


def test_record_ranges_leaves_motors_disabled_even_when_range_is_invalid(
    bus: DamiaoMotorsBus, monkeypatch
) -> None:
    disabled = []
    enabled = []
    monkeypatch.setattr(bus, "disable_torque", lambda motors=None: disabled.append(motors))
    monkeypatch.setattr(bus, "enable_torque", lambda motors=None: enabled.append(motors))
    monkeypatch.setattr(
        bus,
        "sync_read",
        lambda data_name, motors=None: dict.fromkeys(motors or bus.motors, 0.0),
    )
    monkeypatch.setattr(damiao_module, "enter_pressed", lambda: True)

    with pytest.raises(ValueError, match="insufficient range"):
        bus.record_ranges_of_motion(["joint_1"], display_values=False)

    assert disabled == [["joint_1"]]
    assert enabled == []


def test_handshake_requires_consistent_consecutive_samples(bus: DamiaoMotorsBus) -> None:
    count = 0

    def response_for(message: FakeMessage) -> FakeMessage | None:
        nonlocal count
        count += 1
        return make_feedback(bus, message.data[0] + 0x10, position_degrees=0.0 if count == 1 else 20.0)

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorFeedbackError, match="Inconsistent"):
        bus._handshake()

    assert bus.fault_latched
    assert all(message.arbitration_id == CAN_PARAM_ID for message in fake_bus.sent)


def test_stale_frame_drain_is_bounded(bus: DamiaoMotorsBus) -> None:
    class NeverEmptyBus(FakeCanBus):
        def recv(self, timeout: float) -> FakeMessage:
            del timeout
            return make_feedback(bus, 0x11)

    fake_bus = NeverEmptyBus()
    attach_fake_bus(bus, fake_bus)

    assert bus._drain_pending_messages() <= damiao_module.MAX_DRAIN_MESSAGES


def test_batch_control_fails_when_any_feedback_is_missing(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus(
        lambda message: make_feedback(bus, 0x11) if message.arbitration_id == 0x01 else None
    )
    attach_fake_bus(bus, fake_bus)
    commands = {
        "joint_1": (10.0, 0.5, 10.0, 0.0, 0.0),
        "joint_2": (10.0, 0.5, 20.0, 0.0, 0.0),
    }

    with pytest.raises(MotorStateUnavailableError, match="joint_2"):
        bus.mit_control_batch(commands)

    assert bus.fault_latched
    assert bus.get_cached_states("joint_1")["joint_1"]["position"] == pytest.approx(30.0, abs=0.05)
    with pytest.raises(MotorStateUnavailableError, match="joint_2"):
        bus.get_cached_states()

    sent_count = len(fake_bus.sent)
    with pytest.raises(MotorFeedbackError, match="fault-latched"):
        bus.mit_control("joint_1", 10.0, 0.5, 0.0)
    assert len(fake_bus.sent) == sent_count


def test_batch_control_requires_enabled_status_from_every_motor(bus: DamiaoMotorsBus) -> None:
    def response_for(message: FakeMessage) -> FakeMessage | None:
        if message.arbitration_id == 0x01:
            return make_feedback(bus, 0x11, status=1)
        if message.arbitration_id == 0x02:
            return make_feedback(bus, 0x12, status=0)
        return None

    fake_bus = FakeCanBus(response_for)
    attach_fake_bus(bus, fake_bus)
    commands = {
        "joint_1": (10.0, 0.5, 10.0, 0.0, 0.0),
        "joint_2": (10.0, 0.5, 20.0, 0.0, 0.0),
    }

    with pytest.raises(MotorFeedbackError, match="joint_2.*expected enabled status 1"):
        bus.mit_control_batch(commands)

    assert bus.fault_latched
    assert [message.data[-1] for message in fake_bus.sent[-2:]] == [0xFD, 0xFD]


def test_batch_is_fully_validated_before_any_frame_is_sent(bus: DamiaoMotorsBus) -> None:
    fake_bus = FakeCanBus()
    attach_fake_bus(bus, fake_bus)
    commands = {
        "joint_1": (10.0, 0.5, 10.0, 0.0, 0.0),
        "joint_2": (10.0, 0.5, float("nan"), 0.0, 0.0),
    }

    with pytest.raises(ValueError, match="must be finite"):
        bus.mit_control_batch(commands)

    assert fake_bus.sent == []


def test_batch_refresh_does_not_substitute_an_old_state_for_a_drop(bus: DamiaoMotorsBus) -> None:
    bus._process_response("joint_2", make_feedback(bus, 0x12, position_degrees=55.0))
    fake_bus = FakeCanBus(
        lambda message: make_feedback(bus, 0x11) if message.data[0] == 0x01 else None
    )
    attach_fake_bus(bus, fake_bus)

    with pytest.raises(MotorStateUnavailableError, match="joint_2"):
        bus.sync_read_all_states(num_retry=1)

    refreshes_for_joint_2 = [message for message in fake_bus.sent if message.data[0] == 0x02]
    assert len(refreshes_for_joint_2) == 2
    assert bus.get_cached_states("joint_2", allow_stale=True)["joint_2"]["position"] == pytest.approx(
        55.0, abs=0.05
    )


def test_cache_rejects_missing_and_stale_feedback(bus: DamiaoMotorsBus) -> None:
    with pytest.raises(MotorStateUnavailableError, match="No feedback"):
        bus.get_cached_states("joint_1")

    bus._process_response("joint_1", make_feedback(bus, 0x11))
    bus._state_updated_at["joint_1"] -= 1.0

    with pytest.raises(MotorStateUnavailableError, match="stale"):
        bus.get_cached_states("joint_1")
    assert bus.get_cached_states("joint_1", allow_stale=True)["joint_1"]["position"] == pytest.approx(
        30.0, abs=0.05
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_mit_encoder_rejects_non_finite_values(bus: DamiaoMotorsBus, value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        bus._encode_mit_packet(MotorType.DM8009, 10.0, 0.5, value, 0.0, 0.0)
