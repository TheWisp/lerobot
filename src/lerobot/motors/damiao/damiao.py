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

# Portions of this file are derived from DM_Control_Python by cmjang.
# Licensed under the MIT License; see `LICENSE` for the full text:
# https://github.com/cmjang/DM_Control_Python

import logging
import math
import struct
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypedDict

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _can_available, require_package

if TYPE_CHECKING or _can_available:
    import can
else:

    class can:  # noqa: N801
        Message = object
        interface = None


import numpy as np

from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase, NameOrID, Value
from .tables import (
    AVAILABLE_BAUDRATES,
    CAN_CMD_DISABLE,
    CAN_CMD_ENABLE,
    CAN_CMD_REFRESH,
    CAN_CMD_SET_ZERO,
    CAN_PARAM_ID,
    DEFAULT_BAUDRATE,
    DEFAULT_TIMEOUT_MS,
    MIT_KD_RANGE,
    MIT_KP_RANGE,
    MOTOR_LIMIT_PARAMS,
    MotorType,
)

logger = logging.getLogger(__name__)


LONG_TIMEOUT_SEC = 0.1
MEDIUM_TIMEOUT_SEC = 0.01
SHORT_TIMEOUT_SEC = 0.001
PRECISE_TIMEOUT_SEC = 0.0001
MAX_DRAIN_MESSAGES = 256
MAX_DRAIN_DURATION_SEC = 0.02
HANDSHAKE_READ_COUNT = 2
HANDSHAKE_MAX_POSITION_DELTA_DEG = 5.0
HANDSHAKE_MAX_VELOCITY_DELTA_DEG_S = 20.0
HANDSHAKE_MAX_TORQUE_DELTA = 5.0
VALID_MOTOR_STATUSES = {0, 1}


class MotorState(TypedDict):
    status: int
    position: float
    velocity: float
    torque: float
    temp_mos: float
    temp_rotor: float


class MotorStateUnavailableError(ConnectionError):
    """Raised when a command does not produce fresh feedback from every requested motor."""


class MotorFeedbackError(MotorStateUnavailableError):
    """Raised when a received CAN frame is not valid feedback for the requested motor."""


class DamiaoMotorsBus(MotorsBusBase):
    """
    The Damiao implementation for a MotorsBus using CAN bus communication.

    This class uses python-can for CAN bus communication with Damiao motors.
    For more info, see:
    - python-can documentation: https://python-can.readthedocs.io/en/stable/
    - Seedstudio documentation: https://wiki.seeedstudio.com/damiao_series/
    - DM_Control_Python repo: https://github.com/cmjang/DM_Control_Python
    """

    # CAN-specific settings
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        can_interface: str = "auto",
        use_can_fd: bool = True,
        bitrate: int = 1000000,
        data_bitrate: int | None = 5000000,
        state_timeout_s: float = 0.1,
    ):
        """
        Initialize the Damiao motors bus.

        Args:
            port: CAN interface name (e.g., "can0" for Linux, "/dev/cu.usbmodem*" for macOS)
            motors: Dictionary mapping motor names to Motor objects
            calibration: Optional calibration data
            can_interface: CAN interface type - "auto" (default), "socketcan" (Linux), or "slcan" (macOS/serial)
            use_can_fd: Whether to use CAN FD mode (default: True for OpenArms)
            bitrate: Nominal bitrate in bps (default: 1000000 = 1 Mbps)
            data_bitrate: Data bitrate for CAN FD in bps (default: 5000000 = 5 Mbps), ignored if use_can_fd is False
            state_timeout_s: Maximum age of cached feedback accepted by cached-state accessors
        """
        require_package("python-can", extra="damiao", import_name="can")
        super().__init__(port, motors, calibration)
        self.port = port
        self.can_interface = can_interface
        self.use_can_fd = use_can_fd
        self.bitrate = bitrate
        self.data_bitrate = data_bitrate
        if state_timeout_s <= 0:
            raise ValueError("state_timeout_s must be greater than zero")
        self.state_timeout_s = state_timeout_s
        self.canbus: can.interface.Bus | None = None
        self._is_connected = False
        self._fault_latched = False
        self._fault_reason: str | None = None

        # Map motor names to CAN IDs
        self._motor_can_ids: dict[str, int] = {}
        self._recv_id_to_motor: dict[int, str] = {}
        self._motor_types: dict[str, MotorType] = {}

        for name, motor in self.motors.items():
            if motor.motor_type_str is None:
                raise ValueError(f"Motor '{name}' is missing required 'motor_type'")
            self._motor_types[name] = getattr(MotorType, motor.motor_type_str.upper().replace("-", "_"))

            # Map recv_id to motor name for filtering responses
            if motor.recv_id is not None:
                self._recv_id_to_motor[motor.recv_id] = name

        # A motor is absent from this cache until a valid feedback frame has been decoded.
        # Keeping an empty cache prevents an initial all-zero placeholder from being mistaken
        # for a real robot state.
        self._last_known_states: dict[str, MotorState] = {}
        self._state_updated_at: dict[str, float] = {}

        # Dynamic gains storage
        # Defaults: Kp=10.0 (Stiffness), Kd=0.5 (Damping)
        self._gains: dict[str, dict[str, float]] = {name: {"kp": 10.0, "kd": 0.5} for name in self.motors}

    @property
    def is_connected(self) -> bool:
        """Check if the CAN bus is connected."""
        return self._is_connected and self.canbus is not None

    @property
    def fault_latched(self) -> bool:
        """Whether a prior command or feedback failure has inhibited further control."""
        return self._fault_latched

    @property
    def fault_reason(self) -> str | None:
        return self._fault_reason

    def clear_fault_latch(self) -> None:
        """Clear the software latch only while disconnected; reconnect will revalidate hardware."""
        if self.is_connected:
            raise RuntimeError("Disconnect before clearing the Damiao fault latch")
        self._fault_latched = False
        self._fault_reason = None

    def _ensure_control_allowed(self) -> None:
        if self._fault_latched:
            raise MotorFeedbackError(f"Damiao control is fault-latched: {self._fault_reason}")

    def _send_disable_best_effort(self, motors: list[str]) -> None:
        if self.canbus is None:
            return
        for motor in dict.fromkeys(motors):
            try:
                msg = can.Message(
                    arbitration_id=self._get_motor_id(motor),
                    data=[0xFF] * 7 + [CAN_CMD_DISABLE],
                    is_extended_id=False,
                    is_fd=self.use_can_fd,
                )
                self.canbus.send(msg)
            except Exception:
                logger.exception("Failed to send fail-safe disable to %s", motor)

    def _latch_fault(self, reason: str, motors: list[str]) -> None:
        self._fault_latched = True
        self._fault_reason = reason
        self._send_disable_best_effort(motors)

    def _latch_software_fault(self, reason: str) -> None:
        """Inhibit later control without emitting any CAN frame."""
        self._fault_latched = True
        self._fault_reason = reason

    def _drain_pending_messages(self) -> int:
        """Discard a bounded number of stale frames without allowing an endless live bus drain."""
        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")
        count = 0
        deadline = time.monotonic() + MAX_DRAIN_DURATION_SEC
        while count < MAX_DRAIN_MESSAGES and time.monotonic() < deadline:
            if self.canbus.recv(timeout=0) is None:
                break
            count += 1
        if count == MAX_DRAIN_MESSAGES or time.monotonic() >= deadline:
            logger.warning("Stopped draining stale CAN frames at the configured safety bound")
        return count

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        """
        Open the CAN bus and initialize communication.

        Args:
            handshake: If True, ping all motors to verify they're present
        """

        try:
            # Auto-detect interface type based on port name
            if self.can_interface == "auto":
                if self.port.startswith("/dev/"):
                    self.can_interface = "slcan"
                    logger.info(f"Auto-detected slcan interface for port {self.port}")
                else:
                    self.can_interface = "socketcan"
                    logger.info(f"Auto-detected socketcan interface for port {self.port}")

            # Connect to CAN bus
            kwargs = {
                "channel": self.port,
                "bitrate": self.bitrate,
                "interface": self.can_interface,
            }

            if self.can_interface == "socketcan" and self.use_can_fd and self.data_bitrate is not None:
                kwargs.update({"data_bitrate": self.data_bitrate, "fd": True})
                logger.info(
                    f"Connected to {self.port} with CAN FD (bitrate={self.bitrate}, data_bitrate={self.data_bitrate})"
                )
            else:
                logger.info(f"Connected to {self.port} with {self.can_interface} (bitrate={self.bitrate})")

            self.canbus = can.interface.Bus(**kwargs)
            self._is_connected = True
            self._last_known_states.clear()
            self._state_updated_at.clear()

            if handshake:
                self._handshake()

            logger.debug(f"{self.__class__.__name__} connected via {self.can_interface}.")
        except Exception as e:
            self._is_connected = False
            if self.canbus is not None:
                self.canbus.shutdown()
                self.canbus = None
            self._last_known_states.clear()
            self._state_updated_at.clear()
            raise ConnectionError(f"Failed to connect to CAN bus: {e}") from e

    def _handshake(self) -> None:
        """
        Verify all motors are present and populate initial state cache without enabling torque.
        Raises ConnectionError if any motor fails to respond.
        """
        logger.info("Starting handshake with motors...")

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        self._drain_pending_messages()

        missing_motors = []
        try:
            for motor_name in self.motors:
                samples: list[MotorState] = []
                for _ in range(HANDSHAKE_READ_COUNT):
                    response = self._refresh_motor(motor_name, timeout=LONG_TIMEOUT_SEC)
                    if response is None:
                        break
                    samples.append(self._decode_validated_response(motor_name, response))
                    time.sleep(MEDIUM_TIMEOUT_SEC)

                if len(samples) != HANDSHAKE_READ_COUNT:
                    missing_motors.append(motor_name)
                    continue
                if not self._handshake_samples_consistent(samples[0], samples[1]):
                    raise MotorFeedbackError(f"Inconsistent consecutive handshake feedback from '{motor_name}'")

                self._cache_state(motor_name, samples[-1])
        except Exception as e:
            self._latch_software_fault(f"Handshake validation failed: {e}")
            raise

        if missing_motors:
            error = ConnectionError(
                f"Handshake failed. The following motors did not respond: {missing_motors}. "
                "Check power (24V) and CAN wiring."
            )
            self._latch_software_fault(str(error))
            raise error
        logger.info("Handshake successful. All motors ready.")

    @staticmethod
    def _handshake_samples_consistent(first: MotorState, second: MotorState) -> bool:
        return (
            first["status"] == second["status"]
            and abs(first["position"] - second["position"]) <= HANDSHAKE_MAX_POSITION_DELTA_DEG
            and abs(first["velocity"] - second["velocity"]) <= HANDSHAKE_MAX_VELOCITY_DELTA_DEG_S
            and abs(first["torque"] - second["torque"]) <= HANDSHAKE_MAX_TORQUE_DELTA
            and abs(first["temp_mos"] - second["temp_mos"]) <= 5.0
            and abs(first["temp_rotor"] - second["temp_rotor"]) <= 5.0
        )

    @check_if_not_connected
    def disconnect(self, disable_torque: bool = True) -> None:
        """
        Close the CAN bus connection.

        Args:
            disable_torque: If True, disable torque on all motors before disconnecting
        """

        if disable_torque:
            try:
                self.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")

        if self.canbus:
            self.canbus.shutdown()
            self.canbus = None
        self._is_connected = False
        self._last_known_states.clear()
        self._state_updated_at.clear()
        logger.debug(f"{self.__class__.__name__} disconnected.")

    def configure_motors(self) -> None:
        """Deprecated compatibility no-op; configuration must be explicit."""
        logger.warning(
            "Damiao configure_motors() is a safety no-op; use explicit configuration and torque APIs"
        )

    def _send_simple_command(
        self, motor: NameOrID, command_byte: int, *, expected_status: int | None = None
    ) -> MotorState:
        """Helper to send simple 8-byte commands (Enable, Disable, Zero)."""
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [0xFF] * 7 + [command_byte]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        self._drain_pending_messages()
        self.canbus.send(msg)
        if msg := self._recv_motor_response(expected_recv_id=recv_id):
            state = self._process_response(motor_name, msg)
            if expected_status is not None and state["status"] != expected_status:
                raise MotorFeedbackError(
                    f"Motor '{motor_name}' returned status {state['status']} after command "
                    f"0x{command_byte:02X}; expected {expected_status}"
                )
            return state
        raise MotorStateUnavailableError(
            f"No feedback from motor '{motor_name}' after command 0x{command_byte:02X}"
        )

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors."""
        target_motors = self._get_motors_list(motors)
        self._ensure_control_allowed()
        try:
            for motor in target_motors:
                for attempt in range(num_retry + 1):
                    try:
                        self._send_simple_command(motor, CAN_CMD_ENABLE, expected_status=1)
                        break
                    except Exception:
                        if attempt == num_retry:
                            raise
                        time.sleep(MEDIUM_TIMEOUT_SEC)
        except Exception as e:
            self._latch_fault(f"Enable sequence failed: {e}", target_motors)
            raise

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors."""
        target_motors = self._get_motors_list(motors)
        try:
            for motor in target_motors:
                for attempt in range(num_retry + 1):
                    try:
                        self._send_simple_command(motor, CAN_CMD_DISABLE, expected_status=0)
                        break
                    except Exception:
                        if attempt == num_retry:
                            raise
                        time.sleep(MEDIUM_TIMEOUT_SEC)
        except Exception as e:
            self._latch_fault(f"Disable sequence failed: {e}", target_motors)
            raise

    @contextmanager
    def torque_disabled(
        self,
        motors: str | list[str] | None = None,
        *,
        reenable_on_success: bool = False,
    ):
        """
        Disable torque for the context and leave it disabled by default.

        Re-enabling requires an explicit opt-in and occurs only after a clean context exit.
        An exception from the context body never triggers an enable command.
        """
        self.disable_torque(motors)
        try:
            yield
        except BaseException:
            raise
        else:
            if reenable_on_success:
                self.enable_torque(motors)

    def set_zero_position(self, motors: str | list[str] | None = None) -> None:
        """Set current position as zero for selected motors."""
        target_motors = self._get_motors_list(motors)
        self._ensure_control_allowed()
        try:
            for motor in target_motors:
                self._send_simple_command(motor, CAN_CMD_SET_ZERO)
                time.sleep(MEDIUM_TIMEOUT_SEC)
        except Exception as e:
            self._latch_fault(f"Set-zero sequence failed: {e}", target_motors)
            raise

    def _refresh_motor(self, motor: NameOrID, *, timeout: float = SHORT_TIMEOUT_SEC) -> can.Message | None:
        """Refresh motor status and return the response."""
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
        msg = can.Message(arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False, is_fd=self.use_can_fd)

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        self._drain_pending_messages()
        self.canbus.send(msg)
        return self._recv_motor_response(expected_recv_id=recv_id, timeout=timeout)

    def _decode_control_mode_response(self, motor: NameOrID, msg: can.Message) -> int:
        """Validate a Damiao RID 10 parameter reply and return its control mode."""
        motor_name = self._get_motor_name(motor)
        if getattr(msg, "is_error_frame", False) or getattr(msg, "is_remote_frame", False):
            raise MotorFeedbackError(f"Invalid CAN frame type in control-mode reply for '{motor_name}'")
        if getattr(msg, "is_rx", True) is False:
            raise MotorFeedbackError(f"Rejected transmit echo in control-mode reply for '{motor_name}'")

        expected_recv_id = self._get_motor_recv_id(motor)
        if msg.arbitration_id != expected_recv_id:
            raise MotorFeedbackError(
                f"Control-mode reply ID mismatch for '{motor_name}': "
                f"expected 0x{expected_recv_id:X}, got 0x{msg.arbitration_id:X}"
            )
        data = msg.data
        if len(data) != 8 or getattr(msg, "dlc", len(data)) != 8:
            raise MotorFeedbackError(f"Control-mode reply for '{motor_name}' must have DLC 8")
        if data[2] not in (0x33, 0x55):
            raise MotorFeedbackError(
                f"Control-mode reply for '{motor_name}' has invalid parameter opcode 0x{data[2]:02X}"
            )
        if data[3] != 10:
            raise MotorFeedbackError(
                f"Control-mode reply for '{motor_name}' has RID {data[3]}, expected 10"
            )

        embedded_motor_id = data[0] | (data[1] << 8)
        expected_motor_id = self._get_motor_id(motor)
        if embedded_motor_id != expected_motor_id:
            raise MotorFeedbackError(
                f"Control-mode reply identity mismatch for '{motor_name}': "
                f"expected {expected_motor_id}, got {embedded_motor_id}"
            )

        mode = int.from_bytes(bytes(data[4:8]), byteorder="little", signed=False)
        if not 0 <= mode <= 4:
            raise MotorFeedbackError(f"Control-mode reply for '{motor_name}' has invalid value {mode}")
        return mode

    @check_if_not_connected
    def query_control_mode(self, motor: NameOrID) -> int:
        """Read RID 10 twice without changing motor mode or sending any control command."""
        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")
        motor_name = self._get_motor_name(motor)
        motor_id = self._get_motor_id(motor)
        recv_id = self._get_motor_recv_id(motor)
        query_data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0x33, 0x0A, 0, 0, 0, 0]
        samples: list[int] = []

        try:
            for _ in range(2):
                self._drain_pending_messages()
                self.canbus.send(
                    can.Message(
                        arbitration_id=CAN_PARAM_ID,
                        data=query_data,
                        is_extended_id=False,
                        is_fd=self.use_can_fd,
                    )
                )
                response = self._recv_motor_response(expected_recv_id=recv_id, timeout=LONG_TIMEOUT_SEC)
                if response is None:
                    raise MotorStateUnavailableError(
                        f"No RID 10 control-mode reply from motor '{motor_name}'"
                    )
                samples.append(self._decode_control_mode_response(motor, response))
                time.sleep(MEDIUM_TIMEOUT_SEC)

            if samples[0] != samples[1]:
                raise MotorFeedbackError(
                    f"Inconsistent consecutive control-mode replies from '{motor_name}': {samples}"
                )
            return samples[0]
        except Exception as e:
            # This is a read-only diagnostic path. Latch software control inhibition without
            # sending disable or any other control frame as a side effect of a failed query.
            self._latch_software_fault(f"Control-mode query failed: {e}")
            raise

    @check_if_not_connected
    def write_control_mode(self, motor: NameOrID, mode: int) -> int:
        """Write RID 10 only while disabled, then verify the value with a read-only double query."""
        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")
        motor_name = self._get_motor_name(motor)
        motor_id = self._get_motor_id(motor)
        self._ensure_control_allowed()

        try:
            if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 4:
                raise ValueError("Damiao control mode must be an integer within [0, 4]")

            response = self._refresh_motor(motor, timeout=LONG_TIMEOUT_SEC)
            if response is None:
                raise MotorStateUnavailableError(
                    f"No pre-write refresh feedback from motor '{motor_name}'"
                )
            state = self._decode_validated_response(motor_name, response)
            self._cache_state(motor_name, state)
            if state["status"] != 0:
                raise MotorFeedbackError(
                    f"Refusing to write control mode while motor '{motor_name}' has status "
                    f"{state['status']}; expected disabled status 0"
                )

            self._drain_pending_messages()
            self.canbus.send(
                can.Message(
                    arbitration_id=CAN_PARAM_ID,
                    data=[motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0x55, 0x0A, mode, 0, 0, 0],
                    is_extended_id=False,
                    is_fd=self.use_can_fd,
                )
            )
            verified_mode = self.query_control_mode(motor)
            if verified_mode != mode:
                raise MotorFeedbackError(
                    f"Control-mode verification failed for '{motor_name}': "
                    f"wrote {mode}, read {verified_mode}"
                )
            return verified_mode
        except Exception as e:
            self._latch_software_fault(f"Control-mode write failed: {e}")
            raise

    def _recv_motor_response(
        self, expected_recv_id: int | None = None, timeout: float = 0.001
    ) -> can.Message | None:
        """
        Receive a response from a motor.

        Args:
            expected_recv_id: If provided, only return messages from this CAN ID
            timeout: Timeout in seconds (default: 1ms for high-speed operation)
        Returns:
            CAN message if received, None otherwise
        """

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        try:
            start_time = time.time()
            messages_seen = []
            while time.time() - start_time < timeout:
                msg = self.canbus.recv(timeout=PRECISE_TIMEOUT_SEC)
                if msg:
                    messages_seen.append(f"0x{msg.arbitration_id:02X}")
                    if expected_recv_id is None or msg.arbitration_id == expected_recv_id:
                        return msg
                    logger.debug(
                        f"Ignoring message from 0x{msg.arbitration_id:02X}, expected 0x{expected_recv_id:02X}"
                    )

            if logger.isEnabledFor(logging.DEBUG):
                if messages_seen:
                    logger.debug(
                        f"Received {len(messages_seen)} msgs from {set(messages_seen)}, expected 0x{expected_recv_id:02X}"
                    )
                else:
                    logger.debug(f"No CAN messages received (expected 0x{expected_recv_id:02X})")
        except Exception as e:
            logger.debug(f"Failed to receive CAN message: {e}")
        return None

    def _recv_all_responses(
        self, expected_recv_ids: list[int], timeout: float = 0.002
    ) -> dict[int, can.Message]:
        """
        Efficiently receive responses from multiple motors at once.
        Uses the OpenArms pattern: collect all available messages within timeout.

        Args:
            expected_recv_ids: List of CAN IDs we expect responses from
            timeout: Total timeout in seconds (default: 2ms)

        Returns:
            Dictionary mapping recv_id to CAN message
        """
        responses: dict[int, can.Message] = {}
        expected_set = set(expected_recv_ids)
        start_time = time.time()

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        try:
            while len(responses) < len(expected_recv_ids) and (time.time() - start_time) < timeout:
                # 100us poll timeout
                msg = self.canbus.recv(timeout=PRECISE_TIMEOUT_SEC)
                if msg and msg.arbitration_id in expected_set:
                    responses[msg.arbitration_id] = msg
                    if len(responses) == len(expected_recv_ids):
                        break
        except Exception as e:
            logger.debug(f"Error receiving responses: {e}")

        return responses

    def _encode_mit_packet(
        self,
        motor_type: MotorType,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> list[int]:
        """Helper to encode control parameters into 8 bytes for MIT mode."""
        # Convert degrees to radians
        position_rad = np.radians(position_degrees)
        velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

        # Get motor limits
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # Encode parameters
        kp_uint = self._float_to_uint(kp, *MIT_KP_RANGE, 12)
        kd_uint = self._float_to_uint(kd, *MIT_KD_RANGE, 12)
        q_uint = self._float_to_uint(position_rad, -pmax, pmax, 16)
        dq_uint = self._float_to_uint(velocity_rad_per_sec, -vmax, vmax, 12)
        tau_uint = self._float_to_uint(torque, -tmax, tmax, 12)

        # Pack data
        data = [0] * 8
        data[0] = (q_uint >> 8) & 0xFF
        data[1] = q_uint & 0xFF
        data[2] = dq_uint >> 4
        data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
        data[4] = kp_uint & 0xFF
        data[5] = kd_uint >> 4
        data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
        data[7] = tau_uint & 0xFF
        return data

    def _encode_posforce_packet(
        self,
        motor: NameOrID,
        position_rad: float,
        speed_rad_s: float,
        current_pu: float,
    ) -> bytes:
        """Encode the official Damiao POS_FORCE payload with strict physical ranges."""
        values = (position_rad, speed_rad_s, current_pu)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("POS_FORCE values must be finite")
        motor_name = self._get_motor_name(motor)
        pmax, _, _ = MOTOR_LIMIT_PARAMS[self._motor_types[motor_name]]
        if not -pmax <= position_rad <= pmax:
            raise ValueError(f"position_rad must be within [{-pmax}, {pmax}]")
        if not 0.0 <= speed_rad_s <= 100.0:
            raise ValueError("speed_rad_s must be within [0, 100]")
        if not 0.0 <= current_pu <= 1.0:
            raise ValueError("current_pu must be within [0, 1]")
        return struct.pack("<fHH", position_rad, int(speed_rad_s * 100), int(current_pu * 10000))

    @check_if_not_connected
    def posforce_control(
        self,
        motor: NameOrID,
        position_rad: float,
        speed_rad_s: float,
        current_pu: float,
    ) -> MotorState:
        """Send one official POS_FORCE command and return its validated feedback."""
        self._ensure_control_allowed()
        motor_name = self._get_motor_name(motor)
        motor_id = self._get_motor_id(motor)
        data = self._encode_posforce_packet(motor, position_rad, speed_rad_s, current_pu)
        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        try:
            self._drain_pending_messages()
            msg = can.Message(
                arbitration_id=motor_id + 0x300,
                data=data,
                is_extended_id=False,
                is_fd=self.use_can_fd,
            )
            self.canbus.send(msg)
            # A POS_FORCE reply shares the same USB adapter scheduling and
            # Python receive latency as MIT feedback. A 1 ms window is too
            # short when both independent PCAN channels dispatch together.
            response = self._recv_motor_response(
                expected_recv_id=self._get_motor_recv_id(motor),
                timeout=MEDIUM_TIMEOUT_SEC,
            )
            if response is None:
                raise MotorStateUnavailableError(
                    f"No feedback from motor '{motor_name}' after POS_FORCE command"
                )
            state = self._process_response(motor_name, response)
            if state["status"] != 1:
                raise MotorFeedbackError(
                    f"Motor '{motor_name}' returned status {state['status']} after POS_FORCE; "
                    "expected enabled status 1"
                )
            return state
        except Exception as e:
            self._latch_fault(f"POS_FORCE control failed: {e}", [motor_name])
            raise

    @check_if_not_connected
    def posforce_command(
        self,
        motor: NameOrID,
        position_rad: float,
        speed_rad_s: float,
        current_pu: float,
    ) -> None:
        """Send an official POS_FORCE frame without blocking for its reply.

        Normal follower ticks validate a complete J1-J8 state snapshot before
        sending, so the next tick acts as the feedback watchdog. Arming keeps
        using :meth:`posforce_control` for an immediate enabled-state check.
        """
        self._ensure_control_allowed()
        motor_name = self._get_motor_name(motor)
        motor_id = self._get_motor_id(motor)
        data = self._encode_posforce_packet(motor, position_rad, speed_rad_s, current_pu)
        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")
        try:
            self._drain_pending_messages()
            self.canbus.send(
                can.Message(
                    arbitration_id=motor_id + 0x300,
                    data=data,
                    is_extended_id=False,
                    is_fd=self.use_can_fd,
                )
            )
        except Exception as e:
            self._latch_fault(f"POS_FORCE command failed: {e}", [motor_name])
            raise

    @check_if_not_connected
    def mit_control(
        self,
        motor: NameOrID,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float = 0.0,
        torque: float = 0.0,
    ) -> MotorState:
        """Send one MIT command and return the feedback produced by that command."""
        return self._mit_control(motor, kp, kd, position_degrees, velocity_deg_per_sec, torque)

    @check_if_not_connected
    def mit_control_batch(
        self,
        commands: dict[NameOrID, tuple[float, float, float, float, float]],
    ) -> dict[str, MotorState]:
        """Send a batch of MIT commands and return one fresh feedback state per motor."""
        return self._mit_control_batch(commands)

    def _mit_control(
        self,
        motor: NameOrID,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float,
        torque: float,
    ) -> MotorState:
        """Send MIT control command to a motor."""
        motor_id = self._get_motor_id(motor)
        motor_name = self._get_motor_name(motor)
        motor_type = self._motor_types[motor_name]
        self._ensure_control_allowed()

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")

        data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
        try:
            self._drain_pending_messages()
            msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)
            self.canbus.send(msg)

            recv_id = self._get_motor_recv_id(motor)
            if response := self._recv_motor_response(expected_recv_id=recv_id):
                state = self._process_response(motor_name, response)
                if state["status"] != 1:
                    raise MotorFeedbackError(
                        f"Motor '{motor_name}' returned status {state['status']} after MIT control; "
                        "expected enabled status 1"
                    )
                return state
            raise MotorStateUnavailableError(
                f"No feedback from motor '{motor_name}' after MIT control command"
            )
        except Exception as e:
            self._latch_fault(f"MIT control failed: {e}", [motor_name])
            raise

    def _mit_control_batch(
        self,
        commands: dict[NameOrID, tuple[float, float, float, float, float]],
    ) -> dict[str, MotorState]:
        """
        Send MIT control commands to multiple motors in batch.
        Sends all commands first, then collects responses.

        Args:
            commands: Dict mapping motor name/ID to (kp, kd, position_deg, velocity_deg/s, torque)
                     Example: {'joint_1': (10.0, 0.5, 45.0, 0.0, 0.0), ...}
        """
        if not commands:
            return {}

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")
        self._ensure_control_allowed()

        # Validate and encode the entire batch before sending its first frame. This avoids
        # partially applying a batch when a later command is malformed or duplicated.
        recv_id_to_motor: dict[int, str] = {}
        messages: list[can.Message] = []
        for motor, (kp, kd, position_degrees, velocity_deg_per_sec, torque) in commands.items():
            motor_id = self._get_motor_id(motor)
            motor_name = self._get_motor_name(motor)
            motor_type = self._motor_types[motor_name]
            recv_id = self._get_motor_recv_id(motor)
            if recv_id in recv_id_to_motor:
                raise ValueError(f"Duplicate MIT command for receive ID 0x{recv_id:02X}")

            data = self._encode_mit_packet(motor_type, kp, kd, position_degrees, velocity_deg_per_sec, torque)
            recv_id_to_motor[recv_id] = motor_name
            messages.append(
                can.Message(arbitration_id=motor_id, data=data, is_extended_id=False, is_fd=self.use_can_fd)
            )

        target_motors = list(recv_id_to_motor.values())
        try:
            self._drain_pending_messages()
            for msg in messages:
                self.canbus.send(msg)

            # A seven-motor burst fits on the wire quickly, but SocketCAN wakeup
            # and Python scheduling on the target can exceed 1 ms. Use the same
            # bounded 10 ms collection window as the full-state refresh path so
            # later replies are not misclassified as missing feedback.
            responses = self._recv_all_responses(list(recv_id_to_motor.keys()), timeout=MEDIUM_TIMEOUT_SEC)
            states: dict[str, MotorState] = {}
            missing_motors: list[str] = []
            for recv_id, motor_name in recv_id_to_motor.items():
                if msg := responses.get(recv_id):
                    state = self._process_response(motor_name, msg)
                    if state["status"] != 1:
                        raise MotorFeedbackError(
                            f"Motor '{motor_name}' returned status {state['status']} after MIT control; "
                            "expected enabled status 1"
                        )
                    states[motor_name] = state
                else:
                    missing_motors.append(motor_name)

            if missing_motors:
                raise MotorStateUnavailableError(
                    f"Missing MIT feedback from motors: {', '.join(missing_motors)}"
                )
            return states
        except Exception as e:
            self._latch_fault(f"MIT batch control failed: {e}", target_motors)
            raise

    def _float_to_uint(self, x: float, x_min: float, x_max: float, bits: int) -> int:
        """Convert float to unsigned integer for CAN transmission."""
        if not math.isfinite(x):
            raise ValueError(f"MIT control value must be finite, got {x}")
        x = max(x_min, min(x_max, x))  # Clamp to range
        span = x_max - x_min
        data_norm = (x - x_min) / span
        return int(data_norm * ((1 << bits) - 1))

    def _uint_to_float(self, x: int, x_min: float, x_max: float, bits: int) -> float:
        """Convert unsigned integer from CAN to float."""
        span = x_max - x_min
        data_norm = float(x) / ((1 << bits) - 1)
        return data_norm * span + x_min

    def _decode_motor_state(
        self, data: bytearray | bytes, motor_type: MotorType
    ) -> tuple[float, float, float, int, int]:
        """
        Decode motor state from CAN data.
        Returns: (position_deg, velocity_deg_s, torque, temp_mos, temp_rotor)
        """
        if len(data) != 8:
            raise ValueError(f"Invalid motor feedback length: expected 8, got {len(data)}")

        # Extract encoded values
        q_uint = (data[1] << 8) | data[2]
        dq_uint = (data[3] << 4) | (data[4] >> 4)
        tau_uint = ((data[4] & 0x0F) << 8) | data[5]
        t_mos = data[6]
        t_rotor = data[7]

        # Get motor limits
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_type]

        # Decode to physical values
        position_rad = self._uint_to_float(q_uint, -pmax, pmax, 16)
        velocity_rad_per_sec = self._uint_to_float(dq_uint, -vmax, vmax, 12)
        torque = self._uint_to_float(tau_uint, -tmax, tmax, 12)

        return np.degrees(position_rad), np.degrees(velocity_rad_per_sec), torque, t_mos, t_rotor

    def _decode_validated_response(self, motor: str, msg: can.Message) -> MotorState:
        """Strictly validate message metadata, embedded identity, status, and decoded values."""
        if getattr(msg, "is_error_frame", False):
            raise MotorFeedbackError(f"CAN error frame received for motor '{motor}'")
        if getattr(msg, "is_remote_frame", False):
            raise MotorFeedbackError(f"CAN remote frame received for motor '{motor}'")
        if getattr(msg, "is_rx", True) is False:
            raise MotorFeedbackError(f"Rejected transmit echo for motor '{motor}'")

        expected_recv_id = self._get_motor_recv_id(motor)
        if msg.arbitration_id != expected_recv_id:
            raise MotorFeedbackError(
                f"Feedback arbitration ID mismatch for '{motor}': "
                f"expected 0x{expected_recv_id:X}, got 0x{msg.arbitration_id:X}"
            )

        data = msg.data
        if len(data) != 8 or getattr(msg, "dlc", len(data)) != 8:
            raise MotorFeedbackError(
                f"Feedback DLC mismatch for '{motor}': expected 8, got {getattr(msg, 'dlc', len(data))}"
            )
        if not any(data):
            raise MotorFeedbackError(f"Rejected all-zero feedback frame for motor '{motor}'")

        expected_embedded_id = self._get_motor_id(motor) & 0x0F
        embedded_id = data[0] & 0x0F
        if embedded_id != expected_embedded_id:
            raise MotorFeedbackError(
                f"Embedded motor ID mismatch for '{motor}': "
                f"expected {expected_embedded_id}, got {embedded_id}"
            )

        status = (data[0] >> 4) & 0x0F
        if status not in VALID_MOTOR_STATUSES:
            raise MotorFeedbackError(f"Motor '{motor}' reported fault/status code 0x{status:X}")

        motor_type = self._motor_types[motor]
        try:
            pos, vel, torque, t_mos, t_rotor = self._decode_motor_state(data, motor_type)
        except Exception as e:
            raise ValueError(f"Invalid feedback frame from motor '{motor}': {e}") from e

        if not all(math.isfinite(value) for value in (pos, vel, torque, float(t_mos), float(t_rotor))):
            raise MotorFeedbackError(f"Motor '{motor}' returned non-finite feedback")

        state: MotorState = {
            "status": status,
            "position": pos,
            "velocity": vel,
            "torque": torque,
            "temp_mos": float(t_mos),
            "temp_rotor": float(t_rotor),
        }
        return state

    def _cache_state(self, motor: str, state: MotorState) -> MotorState:
        self._last_known_states[motor] = state
        self._state_updated_at[motor] = time.monotonic()
        return state.copy()

    def _process_response(self, motor: str, msg: can.Message) -> MotorState:
        """Decode a strictly validated message and update the motor state cache."""
        try:
            return self._cache_state(motor, self._decode_validated_response(motor, msg))
        except Exception as e:
            # A malformed state makes the mechanical state of the complete arm
            # uncertain. Do not leave the other joints energized while control
            # is inhibited by the bus-wide fault latch.
            self._latch_fault(f"Invalid feedback from '{motor}': {e}", list(self.motors))
            raise

    @check_if_not_connected
    def read(self, data_name: str, motor: str) -> Value:
        """Read a value from a single motor. Positions are always in degrees."""

        # Refresh motor to get latest state
        msg = self._refresh_motor(motor)
        if msg is None:
            motor_id = self._get_motor_id(motor)
            recv_id = self._get_motor_recv_id(motor)
            error = MotorStateUnavailableError(
                f"No response from motor '{motor}' (send ID: 0x{motor_id:02X}, recv ID: 0x{recv_id:02X}). "
                f"Check that: 1) Motor is powered (24V), 2) CAN wiring is correct, "
                f"3) Motor IDs are configured correctly using Damiao Debugging Tools"
            )
            self._latch_fault(str(error), [motor])
            raise error

        self._process_response(motor, msg)
        return self._get_cached_value(motor, data_name)

    def _get_cached_value(self, motor: str, data_name: str) -> Value:
        """Retrieve a specific value from the cache."""
        state = self._get_cached_state(motor)
        mapping: dict[str, Any] = {
            "Present_Position": state["position"],
            "Present_Velocity": state["velocity"],
            "Present_Torque": state["torque"],
            "Temperature_MOS": state["temp_mos"],
            "Temperature_Rotor": state["temp_rotor"],
        }
        if data_name not in mapping:
            raise ValueError(f"Unknown data_name: {data_name}")
        return mapping[data_name]

    def _get_cached_state(
        self,
        motor: str,
        *,
        max_age_s: float | None = None,
        allow_stale: bool = False,
    ) -> MotorState:
        """Return cached feedback only when it exists and satisfies the requested freshness bound."""
        if motor not in self._last_known_states or motor not in self._state_updated_at:
            raise MotorStateUnavailableError(f"No feedback has been received from motor '{motor}'")

        if not allow_stale:
            age_limit = self.state_timeout_s if max_age_s is None else max_age_s
            if age_limit <= 0:
                raise ValueError("max_age_s must be greater than zero")
            age_s = time.monotonic() - self._state_updated_at[motor]
            if age_s > age_limit:
                raise MotorStateUnavailableError(
                    f"Cached feedback for motor '{motor}' is stale "
                    f"(age={age_s:.6f}s, limit={age_limit:.6f}s)"
                )

        return self._last_known_states[motor].copy()

    def get_cached_states(
        self,
        motors: str | list[str] | None = None,
        *,
        max_age_s: float | None = None,
        allow_stale: bool = False,
    ) -> dict[str, MotorState]:
        """
        Return cached feedback without CAN I/O.

        By default, every requested motor must have feedback newer than ``state_timeout_s``.
        ``allow_stale=True`` is an explicit opt-in for diagnostics that need historical data.
        """
        target_motors = self._get_motors_list(motors)
        return {
            motor: self._get_cached_state(motor, max_age_s=max_age_s, allow_stale=allow_stale)
            for motor in target_motors
        }

    @check_if_not_connected
    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
    ) -> None:
        """
        Write a value to a single motor. Positions are always in degrees.
        Can write 'Goal_Position', 'Kp', or 'Kd'.
        """

        if data_name in ("Kp", "Kd"):
            self._gains[motor][data_name.lower()] = float(value)
        elif data_name == "Goal_Position":
            kp = self._gains[motor]["kp"]
            kd = self._gains[motor]["kd"]
            self._mit_control(motor, kp, kd, float(value), 0.0, 0.0)
        else:
            raise ValueError(f"Writing {data_name} not supported in MIT mode")

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
    ) -> dict[str, Value]:
        """
        Read the same value from multiple motors simultaneously.
        """
        target_motors = self._get_motors_list(motors)
        self._batch_refresh(target_motors)

        result = {}
        for motor in target_motors:
            result[motor] = self._get_cached_value(motor, data_name)
        return result

    def sync_read_all_states(
        self,
        motors: str | list[str] | None = None,
        *,
        num_retry: int = 0,
    ) -> dict[str, MotorState]:
        """
        Read ALL motor states (position, velocity, torque) from multiple motors in ONE refresh cycle.

        Returns:
            Dictionary mapping motor names to state dicts with keys: 'position', 'velocity', 'torque'
            Example: {'joint_1': {'position': 45.2, 'velocity': 1.3, 'torque': 0.5}, ...}
        """
        target_motors = self._get_motors_list(motors)
        return self._batch_refresh(target_motors, num_retry=num_retry)

    def _batch_refresh(self, motors: list[str], *, num_retry: int = 0) -> dict[str, MotorState]:
        """Refresh all requested motors, failing if this refresh cycle misses any feedback."""

        if self.canbus is None:
            raise RuntimeError("CAN bus is not initialized.")
        if num_retry < 0:
            raise ValueError("num_retry must be non-negative")

        states: dict[str, MotorState] = {}
        missing_motors = list(dict.fromkeys(motors))

        for _ in range(num_retry + 1):
            if not missing_motors:
                break

            self._drain_pending_messages()
            for motor in missing_motors:
                motor_id = self._get_motor_id(motor)
                data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
                msg = can.Message(
                    arbitration_id=CAN_PARAM_ID, data=data, is_extended_id=False, is_fd=self.use_can_fd
                )
                self.canbus.send(msg)

            expected_recv_ids = [self._get_motor_recv_id(m) for m in missing_motors]
            responses = self._recv_all_responses(expected_recv_ids, timeout=MEDIUM_TIMEOUT_SEC)

            still_missing: list[str] = []
            for motor in missing_motors:
                recv_id = self._get_motor_recv_id(motor)
                msg = responses.get(recv_id)
                if msg:
                    states[motor] = self._process_response(motor, msg)
                else:
                    still_missing.append(motor)
            missing_motors = still_missing

        if missing_motors:
            error = MotorStateUnavailableError(
                f"Missing refresh feedback from motors: {', '.join(missing_motors)}"
            )
            self._latch_fault(str(error), list(dict.fromkeys(motors)))
            raise error
        return states

    @check_if_not_connected
    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:
        """
        Write values to multiple motors simultaneously. Positions are always in degrees.
        """

        if data_name in ("Kp", "Kd"):
            key = data_name.lower()
            for motor, val in values.items():
                self._gains[motor][key] = float(val)

        elif data_name == "Goal_Position":
            commands = {
                motor: (self._gains[motor]["kp"], self._gains[motor]["kd"], float(value), 0.0, 0.0)
                for motor, value in values.items()
            }
            self._mit_control_batch(commands)
        else:
            # Fall back to individual writes
            for motor, value in values.items():
                self.write(data_name, motor, value)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration data from motors."""
        # Damiao motors don't store calibration internally
        # Return existing calibration or empty dict
        return self.calibration if self.calibration else {}

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Write calibration data to motors."""
        # Damiao motors don't store calibration internally
        # Just cache it in memory
        if cache:
            self.calibration = calibration_dict

    def record_ranges_of_motion(
        self,
        motors: str | list[str] | None = None,
        display_values: bool = True,
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        """
        Interactively record the min/max values of each motor in degrees.

        Move the joints by hand (with torque disabled) while the method streams live positions.
        Press Enter to finish.
        """
        target_motors = self._get_motors_list(motors)

        self.disable_torque(target_motors)
        time.sleep(LONG_TIMEOUT_SEC)

        start_positions = self.sync_read("Present_Position", target_motors)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        print("\nMove joints through their full range of motion. Press ENTER when done.")
        user_pressed_enter = False

        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", target_motors)

            for motor in target_motors:
                if motor in positions:
                    mins[motor] = min(positions[motor], mins.get(motor, positions[motor]))
                    maxes[motor] = max(positions[motor], maxes.get(motor, positions[motor]))

            if display_values:
                print("\n" + "=" * 50)
                print(f"{'MOTOR':<20} | {'MIN (deg)':>12} | {'POS (deg)':>12} | {'MAX (deg)':>12}")
                print("-" * 50)
                for motor in target_motors:
                    if motor in positions:
                        print(
                            f"{motor:<20} | {mins[motor]:>12.1f} | {positions[motor]:>12.1f} | {maxes[motor]:>12.1f}"
                        )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                move_cursor_up(len(target_motors) + 4)

            time.sleep(LONG_TIMEOUT_SEC)

        for motor in target_motors:
            if (motor in mins) and (motor in maxes) and (int(abs(maxes[motor] - mins[motor])) < 5):
                raise ValueError(f"Motor {motor} has insufficient range of motion (< 5 degrees)")

        logger.info("Range recording complete; motors remain torque-disabled")
        return mins, maxes

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        """Convert motor specification to list of motor names."""
        if motors is None:
            return list(self.motors.keys())
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, list):
            return motors
        else:
            raise TypeError(f"Invalid motors type: {type(motors)}")

    def _get_motor_id(self, motor: NameOrID) -> int:
        """Get CAN ID for a motor."""
        if isinstance(motor, str):
            if motor in self.motors:
                return self.motors[motor].id
            else:
                raise ValueError(f"Unknown motor: {motor}")
        else:
            return motor

    def _get_motor_name(self, motor: NameOrID) -> str:
        """Get motor name from name or ID."""
        if isinstance(motor, str):
            return motor
        else:
            for name, m in self.motors.items():
                if m.id == motor:
                    return name
            raise ValueError(f"Unknown motor ID: {motor}")

    def _get_motor_recv_id(self, motor: NameOrID) -> int:
        """Get motor recv_id from name or ID."""
        motor_name = self._get_motor_name(motor)
        motor_obj = self.motors.get(motor_name)
        if motor_obj and motor_obj.recv_id is not None:
            return motor_obj.recv_id
        else:
            raise ValueError(f"Motor {motor_obj} doesn't have a valid recv_id (None).")

    @cached_property
    def is_calibrated(self) -> bool:
        """Check if motors are calibrated."""
        return bool(self.calibration)
