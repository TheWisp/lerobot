"""Minimal test script for Damiao motor with ID 3."""

import pytest

from lerobot.utils.import_utils import _can_available

if not _can_available:
    pytest.skip("python-can not available", allow_module_level=True)

from lerobot.motors import Motor
from lerobot.motors.damiao import DamiaoMotorsBus


class SendOnlyCanBus:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def test_batch_refresh_retains_last_state_and_logs_age(monkeypatch, caplog):
    bus = DamiaoMotorsBus.__new__(DamiaoMotorsBus)
    bus.port = "can-test"
    bus.canbus = SendOnlyCanBus()
    bus.use_can_fd = True
    bus._recv_id_to_motor = {0x11: "joint_1", 0x12: "joint_2"}
    bus._last_state_update_monotonic = {"joint_1": None, "joint_2": 10.0}
    monkeypatch.setattr(bus, "_get_motor_id", lambda motor: {"joint_1": 1, "joint_2": 2}[motor])
    monkeypatch.setattr(bus, "_get_motor_recv_id", lambda motor: {"joint_1": 0x11, "joint_2": 0x12}[motor])
    monkeypatch.setattr("lerobot.motors.damiao.damiao.time.monotonic", lambda: 10.05)

    def receive_one(expected_recv_ids, timeout, diagnostics):
        diagnostics.update(
            {
                "receive_ms": 10.1,
                "first_response_ms": 0.3,
                "last_response_ms": 0.3,
                "arrival_ms_by_id": {0x11: 0.3},
                "messages_seen": 1,
                "poll_calls": 64,
                "unexpected_ids": [],
            }
        )
        return {0x11: object()}

    monkeypatch.setattr(bus, "_recv_all_responses", receive_one)
    monkeypatch.setattr(bus, "_process_response", lambda motor, msg: None)

    with caplog.at_level("WARNING"):
        bus._batch_refresh(["joint_1", "joint_2"], context="left.observation")

    assert "port=can-test context=left.observation" in caplog.text
    assert "missing=['joint_2']" in caplog.text
    assert "stale_age_ms={'joint_2': 50.0}" in caplog.text


@pytest.mark.skip(reason="Requires physical Damiao motor and CAN interface")
def test_damiao_motor():
    motors = {
        "joint_3": Motor(
            id=0x03,
            model="damiao",
            norm_mode="degrees",
            motor_type_str="dm4310",
            recv_id=0x13,
        ),
    }

    bus = DamiaoMotorsBus(port="can0", motors=motors)

    try:
        print("Connecting...")
        bus.connect()
        print("✓ Connected")

        print("Enabling torque...")
        bus.enable_torque()
        print("✓ Torque enabled")

        print("Reading all states...")
        states = bus.sync_read_all_states()
        print(f"✓ States: {states}")

        print("Reading position...")
        positions = bus.sync_read("Present_Position")
        print(f"✓ Position: {positions}")

        print("Testing MIT control batch...")
        current_pos = states["joint_3"]["position"]
        commands = {"joint_3": (10.0, 0.5, current_pos, 0.0, 0.0)}
        bus.mit_control_batch(commands)
        print("✓ MIT control batch sent")

        print("Disabling torque...")
        bus.disable_torque()
        print("✓ Torque disabled")

        print("Setting zero position...")
        bus.set_zero_position()
        print("✓ Zero position set")

    finally:
        print("Disconnecting...")
        bus.disconnect(disable_torque=True)
        print("✓ Disconnected")


if __name__ == "__main__":
    test_damiao_motor()
