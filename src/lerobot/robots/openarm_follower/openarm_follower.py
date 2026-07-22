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

import logging
import math
import time
from functools import cached_property
from numbers import Real
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..openarm_description import MJCFGravityCompensator
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarm_follower import (
    CONSERVATIVE_DEFAULT_JOINT_LIMITS,
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArmFollowerConfig,
)
from .telemetry import FollowerTelemetry

logger = logging.getLogger(__name__)

MOTOR_ORDER = [f"joint_{index}" for index in range(1, 8)] + ["gripper"]
ARM_MOTORS = MOTOR_ORDER[:7]
MOTOR_INDEX = {name: index for index, name in enumerate(MOTOR_ORDER)}
MIT_CONTROL_MODE = 1
POS_FORCE_CONTROL_MODE = 4


class OpenArmFollower(Robot):
    """
    OpenArms Follower Robot which uses CAN bus communication to control 7 DOF arm with a gripper.
    The arm uses Damiao motors in MIT control mode.
    """

    config_class = OpenArmFollowerConfig
    name = "openarm_follower"

    def __init__(self, config: OpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        if config.side not in (None, "left", "right"):
            raise ValueError("config.side must be 'left', 'right', or None")
        if set(config.motor_config) != set(MOTOR_ORDER):
            raise ValueError(f"motor_config must define exactly {MOTOR_ORDER}")
        if len(config.position_kp) != len(MOTOR_ORDER) or len(config.position_kd) != len(MOTOR_ORDER):
            raise ValueError("position_kp and position_kd must each contain eight values")
        if not all(
            math.isfinite(value) and value >= 0.0 for value in config.position_kp + config.position_kd
        ):
            raise ValueError("position gains must be finite and non-negative")
        if not 0.0 <= config.gravity_ff_gain <= 1.0:
            raise ValueError("gravity_ff_gain must be in [0, 1]")
        if config.gripper_control_mode not in ("pos_force", "mit"):
            raise ValueError("gripper_control_mode must be 'pos_force' or 'mit'")
        if not math.isfinite(config.gripper_speed_rad_s) or not 0.0 <= config.gripper_speed_rad_s <= 100.0:
            raise ValueError("gripper_speed_rad_s must be finite and in [0, 100]")
        if not math.isfinite(config.gripper_torque_pu) or not 0.0 <= config.gripper_torque_pu <= 1.0:
            raise ValueError("gripper_torque_pu must be finite and in [0, 1]")
        if (
            isinstance(config.present_position_tolerance_deg, bool)
            or not isinstance(config.present_position_tolerance_deg, Real)
            or not math.isfinite(float(config.present_position_tolerance_deg))
            or not 0.0 <= float(config.present_position_tolerance_deg) <= 5.0
        ):
            raise ValueError("present_position_tolerance_deg must be finite and in [0, 5]")
        config.present_position_tolerance_deg = float(config.present_position_tolerance_deg)
        arming_positive_values = {
            "arming_sample_interval_s": config.arming_sample_interval_s,
            "arming_max_position_delta_deg": config.arming_max_position_delta_deg,
            "arming_max_velocity_deg_s": config.arming_max_velocity_deg_s,
            "arming_max_temperature_c": config.arming_max_temperature_c,
            "arming_hold_timeout_s": config.arming_hold_timeout_s,
        }
        for name, value in arming_positive_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive number")
            setattr(config, name, float(value))
        if config.max_relative_target is not None:
            limits = (
                config.max_relative_target.values()
                if isinstance(config.max_relative_target, dict)
                else [config.max_relative_target]
            )
            if not all(
                isinstance(value, Real) and math.isfinite(float(value)) and value > 0 for value in limits
            ):
                raise ValueError("max_relative_target values must be finite and positive")
            if isinstance(config.max_relative_target, dict):
                config.max_relative_target = {
                    key: float(value) for key, value in config.max_relative_target.items()
                }
            else:
                config.max_relative_target = float(config.max_relative_target)

        # Arm motors
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Always use degrees for Damiao motors
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        if config.joint_limits is None:
            if config.side == "left":
                config.joint_limits = dict(LEFT_DEFAULT_JOINTS_LIMITS)
            elif config.side == "right":
                config.joint_limits = dict(RIGHT_DEFAULT_JOINTS_LIMITS)
            else:
                config.joint_limits = dict(CONSERVATIVE_DEFAULT_JOINT_LIMITS)
        elif set(config.joint_limits) != set(MOTOR_ORDER):
            raise ValueError(f"joint_limits must define exactly {MOTOR_ORDER}")
        else:
            for motor_name, bounds in config.joint_limits.items():
                if (
                    len(bounds) != 2
                    or not all(isinstance(value, Real) and math.isfinite(float(value)) for value in bounds)
                    or float(bounds[0]) >= float(bounds[1])
                ):
                    raise ValueError(f"Invalid joint limits for {motor_name}: {bounds}")
            config.joint_limits = {
                motor_name: (float(bounds[0]), float(bounds[1]))
                for motor_name, bounds in config.joint_limits.items()
            }
        if config.side is None:
            logger.info(
                "Set config.side to either 'left' or 'right' to use pre-configured values for joint limits."
            )
        logger.info(f"Values used for joint limits: {config.joint_limits}.")

        self._gravity_ff: MJCFGravityCompensator | None = None
        if config.gravity_ff_gain > 0.0:
            if config.side is None:
                raise ValueError("gravity_ff_gain requires config.side to be 'left' or 'right'")
            self._gravity_ff = MJCFGravityCompensator(
                config.side,
                xml=config.gravity_ff_xml,
                gain=config.gravity_ff_gain,
            )
        self._telemetry = FollowerTelemetry(f"openarm-{config.side or config.port}")
        self._control_modes_validated = False
        self._torque_enabled = False
        self._gripper_torque_enabled = False

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Position, velocity and torque observation features."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            features[f"{motor}.vel"] = float  # Add this
            features[f"{motor}.torque"] = float  # Add this
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Only motor position targets are accepted as actions."""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        """
        Open the transport and cameras with torque left disabled by default.

        Calibration, motor configuration and torque enable are explicit because
        each can alter physical state. This method never writes motor zero.
        """

        logger.info(f"Connecting arm on {self.config.port}...")
        connected_cameras = []
        self._control_modes_validated = False
        self._torque_enabled = False
        try:
            self.bus.connect(handshake=self.config.handshake_on_connect)

            if not self.is_calibrated and calibrate:
                self.calibrate()

            for cam in self.cameras.values():
                cam.connect()
                connected_cameras.append(cam)

            if self.config.configure_on_connect:
                self.configure()
            if self.config.enable_torque_on_connect:
                self.enable_torque()
        except Exception:
            for cam in reversed(connected_cameras):
                try:
                    cam.disconnect()
                except Exception:
                    logger.exception("Failed to roll back camera connection")
            if self.bus.is_connected:
                try:
                    self.bus.disconnect(disable_torque=True)
                except Exception:
                    logger.exception("Failed to roll back CAN connection")
            raise

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms robot.

        The calibration procedure:
        1. Disable torque
        2. Ask user to position arms in hanging position with grippers closed
        3. Set this as zero position
        4. Record range of motion for each joint
        5. Save calibration
        """
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")
        self.bus.disable_torque()

        # Step 1: Set zero position
        input(
            "\nCalibration: Set Zero Position)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        # Set current position as zero for all motors
        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° for safety by default for all joints")
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Read back and validate motor modes without enabling torque."""
        self.validate_control_modes()

    def validate_control_modes(self) -> dict[str, int]:
        """Require J1-J7 MIT and J8 to match the configured gripper path."""
        expected = dict.fromkeys(MOTOR_ORDER[:7], MIT_CONTROL_MODE)
        expected["gripper"] = (
            POS_FORCE_CONTROL_MODE if self.config.gripper_control_mode == "pos_force" else MIT_CONTROL_MODE
        )
        observed = {motor: self.bus.query_control_mode(motor) for motor in MOTOR_ORDER}
        mismatches = {
            motor: {"expected": expected[motor], "observed": observed[motor]}
            for motor in MOTOR_ORDER
            if observed[motor] != expected[motor]
        }
        if mismatches:
            self._control_modes_validated = False
            raise RuntimeError(f"OpenArm control-mode mismatch: {mismatches}")
        self._control_modes_validated = True
        return observed

    def configure_gripper_control_mode(self) -> int:
        """Explicitly change J8 mode while disabled, then read it back."""
        if self._torque_enabled:
            raise RuntimeError("Disable torque before changing the gripper control mode")
        target = (
            POS_FORCE_CONTROL_MODE if self.config.gripper_control_mode == "pos_force" else MIT_CONTROL_MODE
        )
        self._control_modes_validated = False
        mode = self.bus.write_control_mode("gripper", target)
        self.validate_control_modes()
        return mode

    def _validate_state_snapshot(
        self,
        states: dict[str, dict[str, Any]],
        expected_status: dict[str, int],
        *,
        require_stationary: bool,
    ) -> dict[str, float]:
        """Validate one complete, fresh feedback snapshot and return positions."""
        missing = set(MOTOR_ORDER) - set(states)
        if missing:
            raise ConnectionError(f"Missing fresh motor state for: {sorted(missing)}")

        problems: dict[str, Any] = {}
        positions: dict[str, float] = {}
        for motor in MOTOR_ORDER:
            state = states[motor]
            values = {field: state.get(field) for field in ("position", "velocity", "torque", "temp_mos", "temp_rotor")}
            if not all(
                isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
                for value in values.values()
            ):
                problems[motor] = {"invalid_values": values}
                continue
            status = state.get("status")
            if status != expected_status[motor]:
                problems.setdefault(motor, {})["status"] = {
                    "expected": expected_status[motor],
                    "observed": status,
                }
            velocity = float(values["velocity"])
            if require_stationary and abs(velocity) > self.config.arming_max_velocity_deg_s:
                problems.setdefault(motor, {})["velocity"] = velocity
            temperatures = (float(values["temp_mos"]), float(values["temp_rotor"]))
            if any(temp < 0.0 or temp > self.config.arming_max_temperature_c for temp in temperatures):
                problems.setdefault(motor, {})["temperatures"] = temperatures
            positions[motor] = float(values["position"])

        if problems:
            raise ConnectionError(f"Unsafe OpenArm state snapshot: {problems}")
        self._validate_present_positions(positions)
        return positions

    def _read_stable_state_snapshot(self, expected_status: dict[str, int]) -> dict[str, dict[str, Any]]:
        """Require two consecutive stationary full-state snapshots before arming."""
        first = self.bus.sync_read_all_states()
        first_positions = self._validate_state_snapshot(first, expected_status, require_stationary=True)
        time.sleep(self.config.arming_sample_interval_s)
        second = self.bus.sync_read_all_states()
        second_positions = self._validate_state_snapshot(second, expected_status, require_stationary=True)
        moved = {
            motor: (first_positions[motor], second_positions[motor])
            for motor in MOTOR_ORDER
            if abs(second_positions[motor] - first_positions[motor])
            > self.config.arming_max_position_delta_deg
        }
        if moved:
            raise ConnectionError(f"OpenArm moved during disabled arming preflight: {moved}")
        return second

    def _validate_command_feedback(
        self,
        states: dict[str, dict[str, Any]],
        motors: list[str],
    ) -> None:
        """Require complete finite status=1 feedback for a just-sent hold command."""
        missing = set(motors) - set(states)
        problems: dict[str, Any] = {}
        for motor in motors:
            if motor not in states:
                continue
            state = states[motor]
            values = {field: state.get(field) for field in ("position", "velocity", "torque", "temp_mos", "temp_rotor")}
            if state.get("status") != 1:
                problems.setdefault(motor, {})["status"] = state.get("status")
            if not all(
                isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
                for value in values.values()
            ):
                problems.setdefault(motor, {})["invalid_values"] = values
            elif any(
                float(values[field]) < 0.0
                or float(values[field]) > self.config.arming_max_temperature_c
                for field in ("temp_mos", "temp_rotor")
            ):
                problems.setdefault(motor, {})["temperatures"] = (
                    values["temp_mos"],
                    values["temp_rotor"],
                )
        if missing or problems:
            raise ConnectionError(
                f"Invalid OpenArm hold feedback: missing={sorted(missing)}, problems={problems}"
            )

    def _hold_commands(
        self,
        positions: dict[str, float],
        motors: list[str],
    ) -> dict[str, tuple[float, float, float, float, float]]:
        return {
            motor: (
                self._gain_for_motor(motor, self.config.position_kp, None),
                self._gain_for_motor(motor, self.config.position_kd, None),
                float(positions[motor]),
                0.0,
                0.0,
            )
            for motor in motors
        }

    def _check_arming_deadline(self, started_at: float, stage: str) -> None:
        elapsed = time.perf_counter() - started_at
        if elapsed > self.config.arming_hold_timeout_s:
            raise TimeoutError(
                f"OpenArm {stage} hold feedback exceeded arming deadline "
                f"({elapsed:.3f}s > {self.config.arming_hold_timeout_s:.3f}s)"
            )

    def _fail_safe_disable_all(self, context: str) -> None:
        self._torque_enabled = False
        self._gripper_torque_enabled = False
        try:
            self.bus.disable_torque()
        except Exception:
            logger.exception("Failed to disable all OpenArm motors after %s", context)

    def _enable_gripper_from_position(self, position_degrees: float) -> None:
        started_at = time.perf_counter()
        self.bus.enable_torque(["gripper"])
        if self.config.gripper_control_mode == "pos_force":
            state = self.bus.posforce_control(
                "gripper",
                position_rad=math.radians(position_degrees),
                speed_rad_s=min(self.config.gripper_speed_rad_s, 1.0),
                current_pu=0.0,
            )
            states = {"gripper": state}
        else:
            states = self.bus.mit_control_batch(
                self._hold_commands({"gripper": position_degrees}, ["gripper"])
            )
        self._validate_command_feedback(states, ["gripper"])
        self._check_arming_deadline(started_at, "gripper")
        self._gripper_torque_enabled = True

    def enable_torque(self, *, include_gripper: bool = True) -> None:
        """Safely arm J1-J7, then optionally arm J8 with a zero-current hold frame."""
        if self._torque_enabled or self._gripper_torque_enabled:
            raise RuntimeError("OpenArm torque is already enabled")
        self.validate_control_modes()
        disabled_status = dict.fromkeys(MOTOR_ORDER, 0)
        states = self._read_stable_state_snapshot(disabled_status)
        positions = {motor: float(states[motor]["position"]) for motor in MOTOR_ORDER}
        hold_commands = self._hold_commands(positions, ARM_MOTORS)

        try:
            started_at = time.perf_counter()
            self.bus.enable_torque(ARM_MOTORS)
            hold_states = self.bus.mit_control_batch(hold_commands)
            self._validate_command_feedback(hold_states, ARM_MOTORS)
            self._check_arming_deadline(started_at, "arm")
            self._torque_enabled = True
            if include_gripper:
                self._enable_gripper_from_position(positions["gripper"])
        except Exception:
            self._fail_safe_disable_all("OpenArm enable/hold failure")
            raise

    def enable_gripper_torque(self) -> None:
        """Arm J8 separately after J1-J7 are already holding safely."""
        if not self._torque_enabled:
            raise RuntimeError("Arm joints must be holding before enabling the gripper")
        if self._gripper_torque_enabled:
            raise RuntimeError("OpenArm gripper torque is already enabled")
        try:
            self.validate_control_modes()
            expected_status = dict.fromkeys(ARM_MOTORS, 1)
            expected_status["gripper"] = 0
            states = self._read_stable_state_snapshot(expected_status)
            self._enable_gripper_from_position(float(states["gripper"]["position"]))
        except Exception:
            self._fail_safe_disable_all("OpenArm gripper enable/hold failure")
            raise

    def disable_torque(self) -> None:
        """Explicitly disable motor torque."""
        try:
            self.bus.disable_torque()
        finally:
            self._torque_enabled = False
            self._gripper_torque_enabled = False

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Get current observation from robot including position, velocity, and torque.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle
        instead of 3 separate reads.
        """
        started_at = time.perf_counter()
        obs_dict: dict[str, Any] = {}
        try:
            states = self.bus.sync_read_all_states()
            missing = set(self.bus.motors) - set(states)
            if missing:
                raise ConnectionError(f"Missing fresh motor state for: {sorted(missing)}")
            for motor in self.bus.motors:
                state = states[motor]
                values = {field: state.get(field) for field in ("position", "velocity", "torque")}
                if not all(
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values.values()
                ):
                    raise ConnectionError(f"Invalid motor state for {motor}: {values}")
                obs_dict[f"{motor}.pos"] = float(values["position"])
                obs_dict[f"{motor}.vel"] = float(values["velocity"])
                obs_dict[f"{motor}.torque"] = float(values["torque"])

            for cam_key, cam in self.cameras.items():
                camera_started_at = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - camera_started_at) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        except Exception:
            if self._torque_enabled or self._gripper_torque_enabled or self.bus.fault_latched:
                self._fail_safe_disable_all("OpenArm observation failure")
            raise

        dt_ms = (time.perf_counter() - started_at) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")
        return obs_dict

    def _validate_action(self, action: RobotAction) -> dict[str, float]:
        expected = set(self.action_features)
        received = set(action)
        if received != expected:
            missing = sorted(expected - received)
            unknown = sorted(received - expected)
            raise ValueError(
                "OpenArm actions must contain one position for every motor; "
                f"missing={missing}, unknown={unknown}"
            )

        validated: dict[str, float] = {}
        for key, value in action.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"Action value for {key} must be a finite real number, got {value!r}")
            validated[key.removesuffix(".pos")] = float(value)
        return validated

    def _gain_for_motor(
        self,
        motor_name: str,
        configured: list[float],
        custom: dict[str, float] | None,
    ) -> float:
        value = (
            custom.get(motor_name, configured[MOTOR_INDEX[motor_name]])
            if custom
            else configured[MOTOR_INDEX[motor_name]]
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"Gain for {motor_name} must be finite and non-negative, got {value!r}")
        return float(value)

    def _validate_present_positions(self, present_pos: dict[str, float]) -> None:
        """Reject feedback that cannot safely anchor a relative motion command."""
        missing = set(MOTOR_ORDER) - set(present_pos)
        invalid = {
            motor: value
            for motor, value in present_pos.items()
            if motor in MOTOR_ORDER and (not isinstance(value, Real) or not math.isfinite(float(value)))
        }
        tolerance = self.config.present_position_tolerance_deg
        outside_limits = {
            motor: float(present_pos[motor])
            for motor in MOTOR_ORDER
            if motor in present_pos
            and motor in self.config.joint_limits
            and not (
                self.config.joint_limits[motor][0] - tolerance
                <= float(present_pos[motor])
                <= self.config.joint_limits[motor][1] + tolerance
            )
        }
        if missing or invalid or outside_limits:
            raise ConnectionError(
                "Cannot send action from unsafe current positions; "
                f"missing={sorted(missing)}, invalid={invalid}, outside_limits={outside_limits}"
            )

    def _target_is_within_limits_or_recovery(
        self,
        motor: str,
        target: float,
        present: float | None,
    ) -> bool:
        """Allow a tolerance-corridor target only when it holds or moves inward."""
        low, high = self.config.joint_limits[motor]
        if low <= target <= high:
            return True
        if present is None:
            return False
        tolerance = self.config.present_position_tolerance_deg
        if low - tolerance <= present < low:
            return present <= target <= low
        if high < present <= high + tolerance:
            return high <= target <= present
        return False

    def _ensure_ready_for_action(self) -> None:
        if not self._control_modes_validated:
            raise RuntimeError("Validate OpenArm control modes before sending actions")
        if not self._torque_enabled:
            raise RuntimeError("Enable OpenArm torque explicitly before sending actions")
        if not self._gripper_torque_enabled:
            raise RuntimeError("Enable OpenArm gripper torque explicitly before sending complete actions")
        if self.bus.fault_latched:
            raise RuntimeError(f"Damiao bus is fault-latched: {self.bus.fault_reason}")

    def _prepare_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> tuple[
        dict[str, float],
        dict[str, tuple[float, float, float, float, float]],
        np.ndarray,
    ]:
        """Validate and precompute an action using read-only motor feedback."""
        goal_pos = self._validate_action(action)

        # Apply joint limit clipping to arm
        for motor_name, position in goal_pos.items():
            if motor_name not in self.config.joint_limits:
                raise ValueError(f"No joint limit configured for {motor_name}")
            min_limit, max_limit = self.config.joint_limits[motor_name]
            clipped_position = max(min_limit, min(max_limit, position))
            if clipped_position != position:
                logger.warning(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
            goal_pos[motor_name] = clipped_position

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        expected_status = dict.fromkeys(MOTOR_ORDER, 1)
        states = self.bus.sync_read_all_states()
        present_pos = self._validate_state_snapshot(states, expected_status, require_stationary=False)

        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Defense in depth: the final command, after every transformation,
        # must still satisfy the absolute limits.
        final_outside_limits = {
            motor: position
            for motor, position in goal_pos.items()
            if not self._target_is_within_limits_or_recovery(
                motor,
                position,
                float(present_pos[motor]),
            )
        }
        if final_outside_limits:
            raise RuntimeError(f"Refusing final targets outside joint limits: {final_outside_limits}")

        gravity_torque = np.zeros(7, dtype=np.float64)
        if self._gravity_ff is not None:
            measured_rad = np.radians([present_pos[f"joint_{index}"] for index in range(1, 8)])
            gravity_torque = self._gravity_ff.torque(measured_rad)

        # Use batch MIT control for arm (sends all commands, then collects responses)
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            if motor_name == "gripper" and self.config.gripper_control_mode == "pos_force":
                continue
            kp = self._gain_for_motor(motor_name, self.config.position_kp, custom_kp)
            kd = self._gain_for_motor(motor_name, self.config.position_kd, custom_kd)
            torque = 0.0 if motor_name == "gripper" else float(gravity_torque[MOTOR_INDEX[motor_name]])
            commands[motor_name] = (kp, kd, position_degrees, 0.0, torque)

        return goal_pos, commands, gravity_torque

    def _execute_prepared_action(
        self,
        prepared: tuple[
            dict[str, float],
            dict[str, tuple[float, float, float, float, float]],
            np.ndarray,
        ],
    ) -> RobotAction:
        """Send a fully prevalidated action."""
        self._ensure_ready_for_action()
        goal_pos, commands, gravity_torque = prepared

        try:
            states = self.bus.mit_control_batch(commands)
            if self.config.gripper_control_mode == "pos_force":
                self.bus.posforce_command(
                    "gripper",
                    position_rad=math.radians(goal_pos["gripper"]),
                    speed_rad_s=self.config.gripper_speed_rad_s,
                    current_pu=self.config.gripper_torque_pu,
                )
        except Exception:
            self._fail_safe_disable_all("OpenArm command failure")
            raise

        arm_names = MOTOR_ORDER[:7]
        self._telemetry.update(
            [goal_pos[name] for name in arm_names],
            [states[name]["position"] for name in arm_names],
            [states[name]["torque"] for name in arm_names],
            [states[name]["temp_mos"] for name in arm_names],
            gravity_torque,
        )
        self._telemetry.maybe_report()

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """Validate, limit and send one complete OpenArm position action."""
        self._ensure_ready_for_action()
        try:
            prepared = self._prepare_action(action, custom_kp, custom_kd)
        except ConnectionError:
            self._fail_safe_disable_all("OpenArm action preflight failure")
            raise
        return self._execute_prepared_action(prepared)

    def disconnect(self) -> None:
        """Disconnect every resource, even when one cleanup operation fails."""

        first_error: Exception | None = None
        for cam in reversed(list(self.cameras.values())):
            if not cam.is_connected:
                continue
            try:
                cam.disconnect()
            except Exception as exc:
                first_error = first_error or exc
                logger.exception("Failed to disconnect camera")

        if self.bus.is_connected:
            try:
                self.bus.disconnect(self.config.disable_torque_on_disconnect)
            except Exception as exc:
                first_error = first_error or exc
                logger.exception("Failed to disconnect CAN bus")

        self._torque_enabled = False
        self._gripper_torque_enabled = False
        self._control_modes_validated = False

        logger.info(f"{self} disconnected.")
        if first_error is not None:
            raise first_error
