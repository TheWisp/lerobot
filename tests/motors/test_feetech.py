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

import re
import sys
import time
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.encoding_utils import encode_sign_magnitude
from lerobot.motors.feetech import MODEL_NUMBER, MODEL_NUMBER_TABLE, BusReplyError, FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE

try:
    import scservo_sdk as scs

    from tests.mocks.mock_feetech import MockInstructionPacket, MockMotors, MockPortHandler, MockStatusPacket
except (ImportError, ModuleNotFoundError):
    pytest.skip("scservo_sdk not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def patch_port_handler():
    if sys.platform == "darwin":
        with patch.object(scs, "PortHandler", MockPortHandler):
            yield
    else:
        yield


@pytest.fixture
def mock_motors() -> Generator[MockMotors, None, None]:
    motors = MockMotors()
    motors.open()
    yield motors
    motors.close()


@pytest.fixture
def dummy_motors() -> dict[str, Motor]:
    return {
        "dummy_1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    }


@pytest.fixture
def dummy_calibration(dummy_motors) -> dict[str, MotorCalibration]:
    homings = [-709, -2006, 1624]
    mins = [43, 27, 145]
    maxes = [1335, 3608, 3999]
    calibration = {}
    for motor, m in dummy_motors.items():
        calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=homings[m.id - 1],
            range_min=mins[m.id - 1],
            range_max=maxes[m.id - 1],
        )
    return calibration


@pytest.mark.skipif(sys.platform != "darwin", reason=f"No patching needed on {sys.platform=}")
def test_autouse_patch():
    """Ensures that the autouse fixture correctly patches scs.PortHandler with MockPortHandler."""
    assert scs.PortHandler is MockPortHandler


@pytest.mark.parametrize(
    "protocol, value, length, expected",
    [
        (0, 0x12,       1, [0x12]),
        (1, 0x12,       1, [0x12]),
        (0, 0x1234,     2, [0x34, 0x12]),
        (1, 0x1234,     2, [0x12, 0x34]),
        (0, 0x12345678, 4, [0x78, 0x56, 0x34, 0x12]),
        (1, 0x12345678, 4, [0x56, 0x78, 0x12, 0x34]),
    ],
    ids=[
        "P0: 1 byte",
        "P1: 1 byte",
        "P0: 2 bytes",
        "P1: 2 bytes",
        "P0: 4 bytes",
        "P1: 4 bytes",
    ],
)  # fmt: skip
def test__split_into_byte_chunks(protocol, value, length, expected):
    bus = FeetechMotorsBus("", {}, protocol_version=protocol)
    assert bus._split_into_byte_chunks(value, length) == expected


def test_abc_implementation(dummy_motors):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    FeetechMotorsBus(port="/dev/dummy-port", motors=dummy_motors)


@pytest.mark.parametrize("id_", [1, 2, 3])
def test_ping(id_, mock_motors, dummy_motors):
    expected_model_nb = MODEL_NUMBER_TABLE[dummy_motors[f"dummy_{id_}"].model]
    addr, length = MODEL_NUMBER
    ping_stub = mock_motors.build_ping_stub(id_)
    mobel_nb_stub = mock_motors.build_read_stub(addr, length, id_, expected_model_nb)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    ping_model_nb = bus.ping(id_)

    assert ping_model_nb == expected_model_nb
    assert mock_motors.stubs[ping_stub].called
    assert mock_motors.stubs[mobel_nb_stub].called


def test_broadcast_ping(mock_motors, dummy_motors):
    models = {m.id: m.model for m in dummy_motors.values()}
    addr, length = MODEL_NUMBER
    ping_stub = mock_motors.build_broadcast_ping_stub(list(models))
    mobel_nb_stubs = []
    expected_model_nbs = {}
    for id_, model in models.items():
        model_nb = MODEL_NUMBER_TABLE[model]
        stub = mock_motors.build_read_stub(addr, length, id_, model_nb)
        expected_model_nbs[id_] = model_nb
        mobel_nb_stubs.append(stub)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    ping_model_nbs = bus.broadcast_ping()

    assert ping_model_nbs == expected_model_nbs
    assert mock_motors.stubs[ping_stub].called
    assert all(mock_motors.stubs[stub].called for stub in mobel_nb_stubs)


@pytest.mark.parametrize(
    "addr, length, id_, value",
    [
        (0, 1, 1, 2),
        (10, 2, 2, 999),
        (42, 4, 3, 1337),
    ],
)
def test__read(addr, length, id_, value, mock_motors, dummy_motors):
    stub = mock_motors.build_read_stub(addr, length, id_, value)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    read_value, _, _ = bus._read(addr, length, id_)

    assert mock_motors.stubs[stub].called
    assert read_value == value


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__read_error(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value, error = (10, 4, 1, 1337, scs.ERRBIT_VOLTAGE)
    stub = mock_motors.build_read_stub(addr, length, id_, value, error=error)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(RuntimeError, match=re.escape("[RxPacketError] Input voltage error!")):
            bus._read(addr, length, id_, raise_on_error=raise_on_error)
    else:
        _, _, read_error = bus._read(addr, length, id_, raise_on_error=raise_on_error)
        assert read_error == error

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__read_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value = (10, 4, 1, 1337)
    stub = mock_motors.build_read_stub(addr, length, id_, value, reply=False)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._read(addr, length, id_, raise_on_error=raise_on_error)
    else:
        _, read_comm, _ = bus._read(addr, length, id_, raise_on_error=raise_on_error)
        assert read_comm == scs.COMM_RX_TIMEOUT

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize(
    "addr, length, id_, value",
    [
        (0, 1, 1, 2),
        (10, 2, 2, 999),
        (42, 4, 3, 1337),
    ],
)
def test__write(addr, length, id_, value, mock_motors, dummy_motors):
    stub = mock_motors.build_write_stub(addr, length, id_, value)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    comm, error = bus._write(addr, length, id_, value)

    assert mock_motors.stubs[stub].wait_called()
    assert comm == scs.COMM_SUCCESS
    assert error == 0


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__write_error(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value, error = (10, 4, 1, 1337, scs.ERRBIT_VOLTAGE)
    stub = mock_motors.build_write_stub(addr, length, id_, value, error=error)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(RuntimeError, match=re.escape("[RxPacketError] Input voltage error!")):
            bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
    else:
        _, write_error = bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
        assert write_error == error

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__write_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value = (10, 4, 1, 1337)
    stub = mock_motors.build_write_stub(addr, length, id_, value, reply=False)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
    else:
        write_comm, _ = bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
        assert write_comm == scs.COMM_RX_TIMEOUT

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize(
    "addr, length, ids_values",
    [
        (0, 1, {1: 4}),
        (10, 2, {1: 1337, 2: 42}),
        (42, 4, {1: 1337, 2: 42, 3: 4016}),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)
def test__sync_read(addr, length, ids_values, mock_motors, dummy_motors):
    stub = mock_motors.build_sync_read_stub(addr, length, ids_values)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    read_values, _ = bus._sync_read(addr, length, list(ids_values))

    assert mock_motors.stubs[stub].called
    assert read_values == ids_values


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__sync_read_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, ids_values = (10, 4, {1: 1337})
    stub = mock_motors.build_sync_read_stub(addr, length, ids_values, reply=False)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._sync_read(addr, length, list(ids_values), raise_on_error=raise_on_error)
    else:
        _, read_comm = bus._sync_read(addr, length, list(ids_values), raise_on_error=raise_on_error)
        assert read_comm == scs.COMM_RX_TIMEOUT

    assert mock_motors.stubs[stub].called


STATE = ["Present_Position", "Present_Velocity", "Present_Load", "Present_Voltage", "Present_Temperature"]


def _state_reply(id_: int, pos: int, load: int = 0, temp: int = 30, error: int = 0) -> bytes:
    """A motor's whole reply to a read of registers 56-63: position, speed, load, voltage, temperature."""
    params = [pos & 0xFF, pos >> 8, 0, 0, load & 0xFF, load >> 8, 120, temp]
    return MockStatusPacket.build(scs_id=id_, params=params, length=len(params) + 2, error=error)


def _stub_state_replies(mock_motors: MockMotors, ids: list[int], replies: list[bytes]) -> str:
    """Answer the n-th read of registers 56-63 from ``ids`` with ``replies[n - 1]``."""
    name = "Sync_Read_state_" + "_".join(map(str, ids))
    mock_motors.stub(
        name=name,
        receive_bytes=MockInstructionPacket.sync_read(ids, 56, 8),
        send_fn=lambda n: replies[min(n, len(replies)) - 1],
    )
    return name


def test_sync_read_block_reads_several_registers_in_one_request(mock_motors, dummy_motors):
    stub = _stub_state_replies(
        mock_motors,
        [1, 2, 3],
        [
            _state_reply(1, 1500)
            + _state_reply(2, 2048, load=1024 + 32, error=32)
            + _state_reply(3, 2600, temp=41)
        ],
    )
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    block = bus.sync_read_block(STATE, normalize=False)

    assert mock_motors.stubs[stub].calls == 1, "one request"
    assert block.values["Present_Position"] == {"dummy_1": 1500, "dummy_2": 2048, "dummy_3": 2600}
    assert block.values["Present_Load"]["dummy_2"] == -32, "sign and magnitude, as sync_read decodes it"
    assert block.values["Present_Temperature"] == {"dummy_1": 30, "dummy_2": 30, "dummy_3": 41}
    assert block.errors == {"dummy_1": 0, "dummy_2": 32, "dummy_3": 0}
    assert block.raw["dummy_3"] == [2600 & 0xFF, 2600 >> 8, 0, 0, 0, 0, 120, 41]


def test_sync_read_block_refuses_a_bad_reply_and_clears_what_came_after_it(mock_motors, dummy_motors):
    """2026-10-11: a corrupted reply in the middle of a request, the port "cleared" with the SDK's clearPort (which
    keeps what was received), and the next request read the last motor's leftover reply for the first's. The bad
    reply raises, the rest of its request is dropped from the port, and the next request reads only its own."""
    corrupted = bytearray(_state_reply(2, 2048))
    corrupted[-1] ^= 0xFF  # its checksum no longer matches
    _stub_state_replies(
        mock_motors,
        [1, 2, 3],
        [
            _state_reply(1, 1500) + bytes(corrupted) + _state_reply(3, 2600),
            _state_reply(1, 1501) + _state_reply(2, 2049) + _state_reply(3, 2601),
        ],
    )
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    with pytest.raises(
        BusReplyError, match=re.escape("no reply from dummy_2: [TxRxResult] Incorrect status packet!")
    ) as e:
        bus.sync_read_block(STATE, normalize=False)
    assert e.value.raw["dummy_1"][:2] == [1500 & 0xFF, 1500 >> 8], "what came back, kept for the log"

    block = bus.sync_read_block(STATE, normalize=False)
    assert block.values["Present_Position"] == {"dummy_1": 1501, "dummy_2": 2049, "dummy_3": 2601}


@pytest.mark.parametrize(
    "replies, why",
    [
        (
            lambda: _state_reply(1, 1500) + _state_reply(3, 2600),
            "a reply from motor 3 where dummy_2's was due",
        ),
        (
            lambda: _state_reply(1, 1500)
            + MockStatusPacket.build(scs_id=2, params=[0] * 10, length=12)
            + _state_reply(3, 2600),
            "dummy_2 replied 10 bytes, not 8",
        ),
        (
            lambda: _state_reply(1, 1500) + MockStatusPacket.build(scs_id=2, params=[0, 8], length=4),
            "dummy_2 replied 2 bytes, not 8",
        ),
    ],
    ids=["another motor's", "longer", "a late two-byte one"],
)
def test_sync_read_block_refuses_a_reply_that_is_not_the_one_asked_for(
    replies, why, mock_motors, dummy_motors
):
    """The SDK's own read passes over another motor's reply and cuts a longer one to the length asked."""
    _stub_state_replies(mock_motors, [1, 2, 3], [replies()])
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    with pytest.raises(BusReplyError, match=re.escape(why)):
        bus.sync_read_block(STATE, normalize=False)


def test_clear_input_drops_what_clear_port_keeps(mock_motors, dummy_motors):
    """The SDK's clearPort is pyserial's flush: it waits for the output to go out and keeps every byte received."""
    _stub_state_replies(
        mock_motors, [1, 2, 3], [_state_reply(1, 1500) + _state_reply(2, 2048) + _state_reply(3, 2600)]
    )
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)
    bus.port_handler.writePort(MockInstructionPacket.sync_read([1, 2, 3], 56, 8))
    deadline = time.monotonic() + 2.0
    while bus.port_handler.getBytesAvailable() < 3 * 14 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bus.port_handler.getBytesAvailable() == 3 * 14

    bus.port_handler.clearPort()
    assert bus.port_handler.getBytesAvailable() == 3 * 14, "clearPort kept the replies"
    bus.clear_input()
    assert bus.port_handler.getBytesAvailable() == 0


@pytest.mark.parametrize(
    "addr, length, ids_values",
    [
        (0, 1, {1: 4}),
        (10, 2, {1: 1337, 2: 42}),
        (42, 4, {1: 1337, 2: 42, 3: 4016}),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)
def test__sync_write(addr, length, ids_values, mock_motors, dummy_motors):
    stub = mock_motors.build_sync_write_stub(addr, length, ids_values)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    comm = bus._sync_write(addr, length, ids_values)

    assert mock_motors.stubs[stub].wait_called()
    assert comm == scs.COMM_SUCCESS


def test_is_calibrated(mock_motors, dummy_motors, dummy_calibration):
    mins_stubs, maxes_stubs, homings_stubs = [], [], []
    for cal in dummy_calibration.values():
        mins_stubs.append(
            mock_motors.build_read_stub(
                *STS_SMS_SERIES_CONTROL_TABLE["Min_Position_Limit"], cal.id, cal.range_min
            )
        )
        maxes_stubs.append(
            mock_motors.build_read_stub(
                *STS_SMS_SERIES_CONTROL_TABLE["Max_Position_Limit"], cal.id, cal.range_max
            )
        )
        homings_stubs.append(
            mock_motors.build_read_stub(
                *STS_SMS_SERIES_CONTROL_TABLE["Homing_Offset"],
                cal.id,
                encode_sign_magnitude(cal.homing_offset, 11),
            )
        )

    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
        calibration=dummy_calibration,
    )
    bus.connect(handshake=False)

    is_calibrated = bus.is_calibrated

    assert is_calibrated
    assert all(mock_motors.stubs[stub].called for stub in mins_stubs)
    assert all(mock_motors.stubs[stub].called for stub in maxes_stubs)
    assert all(mock_motors.stubs[stub].called for stub in homings_stubs)


def test_reset_calibration(mock_motors, dummy_motors):
    write_homing_stubs = []
    write_mins_stubs = []
    write_maxes_stubs = []
    for motor in dummy_motors.values():
        write_homing_stubs.append(
            mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Homing_Offset"], motor.id, 0)
        )
        write_mins_stubs.append(
            mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Min_Position_Limit"], motor.id, 0)
        )
        write_maxes_stubs.append(
            mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Max_Position_Limit"], motor.id, 4095)
        )

    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    bus.reset_calibration()

    assert all(mock_motors.stubs[stub].wait_called() for stub in write_homing_stubs)
    assert all(mock_motors.stubs[stub].wait_called() for stub in write_mins_stubs)
    assert all(mock_motors.stubs[stub].wait_called() for stub in write_maxes_stubs)


def test_set_half_turn_homings(mock_motors, dummy_motors):
    """
    For this test, we assume that the homing offsets are already 0 such that
    Present_Position == Actual_Position
    """
    current_positions = {
        1: 1337,
        2: 42,
        3: 3672,
    }
    expected_homings = {
        1: -710,  # 1337 - 2047
        2: -2005,  # 42 - 2047
        3: 1625,  # 3672 - 2047
    }
    read_pos_stub = mock_motors.build_sync_read_stub(
        *STS_SMS_SERIES_CONTROL_TABLE["Present_Position"], current_positions
    )
    write_homing_stubs = []
    for id_, homing in expected_homings.items():
        encoded_homing = encode_sign_magnitude(homing, 11)
        stub = mock_motors.build_write_stub(
            *STS_SMS_SERIES_CONTROL_TABLE["Homing_Offset"], id_, encoded_homing
        )
        write_homing_stubs.append(stub)

    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)
    bus.reset_calibration = MagicMock()

    bus.set_half_turn_homings()

    bus.reset_calibration.assert_called_once()
    assert mock_motors.stubs[read_pos_stub].called
    assert all(mock_motors.stubs[stub].wait_called() for stub in write_homing_stubs)


@pytest.mark.parametrize(
    "initial_phase, expected_phase",
    [
        (0b00010000, 0b00000000),  # bit 4 set - cleared
        (0b11111111, 0b11101111),  # all bits set - bit 4 cleared, others preserved
        (0b00000000, 0b00000000),  # bit 4 already 0 - unchanged
    ],
    ids=["bit4_set", "all_bits_set", "bit4_already_cleared"],
)
def test_configure_motors_clears_sts3215_phase_bit4(initial_phase, expected_phase, mock_motors, dummy_motors):
    """Phase register bit 4 (angle feedback mode) must be cleared for sts3215, other bits preserved."""
    phase_read_stubs = []
    phase_write_stubs = []
    for motor in dummy_motors.values():
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Return_Delay_Time"], motor.id, 0)
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Maximum_Acceleration"], motor.id, 254)
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Acceleration"], motor.id, 254)
        phase_read_stubs.append(
            mock_motors.build_read_stub(*STS_SMS_SERIES_CONTROL_TABLE["Phase"], motor.id, initial_phase)
        )
        if initial_phase != expected_phase:
            phase_write_stubs.append(
                mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Phase"], motor.id, expected_phase)
            )

    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    with patch.object(bus, "write", wraps=bus.write) as mock_write:
        bus.configure_motors()

    assert all(mock_motors.stubs[stub].called for stub in phase_read_stubs)
    if initial_phase != expected_phase:  # ensure that phase is written only if it needs to be changed
        assert all(mock_motors.stubs[stub].wait_called() for stub in phase_write_stubs)
    else:  # If no write should be made, ensure that Phase is not written for any motor
        write_data_names = [call.args[0] for call in mock_write.call_args_list]
        assert "Phase" not in write_data_names


def test_configure_motors_skips_phase_for_non_sts3215(mock_motors):
    """Phase register must not be touched for motors other than sts3215."""
    motors = {
        "dummy_1": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
    }
    for motor in motors.values():
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Return_Delay_Time"], motor.id, 0)
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Maximum_Acceleration"], motor.id, 254)
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Acceleration"], motor.id, 254)

    bus = FeetechMotorsBus(port=mock_motors.port, motors=motors)
    bus.connect(handshake=False)

    with patch.object(bus, "read", wraps=bus.read) as mock_read:
        bus.configure_motors()
        read_data_names = [call.args[0] for call in mock_read.call_args_list]

    assert "Phase" not in read_data_names


def test_record_ranges_of_motion(mock_motors, dummy_motors):
    positions = {
        1: [351, 42, 1337],
        2: [28, 3600, 2444],
        3: [4002, 2999, 146],
    }
    expected_mins = {
        "dummy_1": 42,
        "dummy_2": 28,
        "dummy_3": 146,
    }
    expected_maxes = {
        "dummy_1": 1337,
        "dummy_2": 3600,
        "dummy_3": 4002,
    }
    stub = mock_motors.build_sequential_sync_read_stub(
        *STS_SMS_SERIES_CONTROL_TABLE["Present_Position"], positions
    )
    with patch("lerobot.motors.motors_bus.enter_pressed", side_effect=[False, True]):
        bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
        bus.connect(handshake=False)

        mins, maxes = bus.record_ranges_of_motion(display_values=False)

    assert mock_motors.stubs[stub].calls == 3
    assert mins == expected_mins
    assert maxes == expected_maxes
