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
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus, MotorState
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarm_follower import (
    JOINT_DELTA_LIMITS_RAD_S,
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArmFollowerConfig,
)
from .gravity_ff import GravityFF
from .telemetry import FollowerTelemetry

logger = logging.getLogger(__name__)

# Command order used for gains, feedforward arrays and telemetry.
MOTOR_ORDER = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
MOTOR_INDEX = {name: i for i, name in enumerate(MOTOR_ORDER)}
ARM_JOINTS = MOTOR_ORDER[:7]  # gripper excluded: 1 N·m POS_FORCE finger, not MIT


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

        if config.gripper_control_mode not in ("pos_force", "mit"):
            raise ValueError("gripper_control_mode must be 'pos_force' or 'mit'")
        if not math.isfinite(config.gripper_speed_rad_s) or not 0.0 <= config.gripper_speed_rad_s <= 100.0:
            raise ValueError("gripper_speed_rad_s must be finite and in [0, 100]")
        if not math.isfinite(config.gripper_torque_pu) or not 0.0 <= config.gripper_torque_pu <= 1.0:
            raise ValueError("gripper_torque_pu must be finite and in [0, 1]")

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

        if config.side is not None:
            if config.side == "left":
                config.joint_limits = LEFT_DEFAULT_JOINTS_LIMITS
            elif config.side == "right":
                config.joint_limits = RIGHT_DEFAULT_JOINTS_LIMITS
            else:
                raise ValueError(
                    "config.side must be either 'left', 'right' (for default values) or 'None' (for CLI values)"
                )
        else:
            logger.info(
                "Set config.side to either 'left' or 'right' to use pre-configured values for joint limits."
            )
        logger.info(f"Values used for joint limits: {config.joint_limits}.")

        # Gravity feedforward (MIT torque slot). Off by default (gain 0.0);
        # validated value is 0.9. Requires the `openarm-ff` extra.
        self._gravity_ff: GravityFF | None = None
        if config.gravity_ff_gain > 0.0:
            if config.side not in ("left", "right"):
                raise ValueError(
                    "gravity_ff_gain requires config.side to be 'left' or 'right' "
                    "(the gravity model needs to know which arm this is)."
                )
            self._gravity_ff = GravityFF(
                side=config.side,
                xml=config.gravity_ff_xml,
                gain=config.gravity_ff_gain,
            )
            logger.info(f"Gravity feedforward ENABLED (side={config.side}, gain={config.gravity_ff_gain})")

        # Aggregated follower telemetry (always on; a few vector ops per cycle).
        self._telemetry = FollowerTelemetry(self.id)
        self._last_tff = np.zeros(7)

        # Alignment ramp / jump guard / velocity feedforward state.
        self._last_cmd_deg: dict[str, float] = {}  # last command actually sent
        self._last_send_time: float | None = None
        self._last_jump_log = 0.0

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.

        We assume that at connection time, the arms are in a safe rest position,
        and torque can be safely disabled to run calibration if needed.
        """

        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        self.bus.enable_torque()

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
        """Configure motors with appropriate settings."""
        # TODO(Steven, Pepijn): Slightly different from what it is happening in the leader
        with self.bus.torque_disabled():
            self.bus.configure_motors()

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
        start = time.perf_counter()

        obs_dict: dict[str, Any] = {}

        states = self.bus.sync_read_all_states()

        for motor in self.bus.motors:
            state = states.get(motor, {})
            obs_dict[f"{motor}.pos"] = state.get("position", 0.0)
            if self.config.use_velocity_and_torque:
                obs_dict[f"{motor}.vel"] = state.get("velocity", 0.0)
                obs_dict[f"{motor}.torque"] = state.get("torque", 0.0)

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """
        Send action command to robot.

        The action magnitude may be clipped based on safety limits.

        Args:
            action: Dictionary with motor positions (e.g., "joint_1.pos", "joint_2.pos")
            custom_kp: Optional custom kp gains per motor (e.g., {"joint_1": 120.0, "joint_2": 150.0})
            custom_kd: Optional custom kd gains per motor (e.g., {"joint_1": 1.5, "joint_2": 2.0})

        Returns:
            The action actually sent (potentially clipped)
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Apply joint limit clipping to arm
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
                goal_pos[motor_name] = clipped_position

        now = time.monotonic()
        states: dict[str, MotorState] | None = None

        # Jump guard: log (rate-limited) when an arm-joint target jumps vs the
        # last command. The alignment ramp below rate-limits the move itself.
        if self.config.align_step_limit is not None and self._last_cmd_deg:
            for motor_name in ARM_JOINTS:
                if motor_name not in goal_pos or motor_name not in self._last_cmd_deg:
                    continue
                jump_rad = abs(np.radians(goal_pos[motor_name] - self._last_cmd_deg[motor_name]))
                if jump_rad > self.config.align_jump_threshold:
                    if now - self._last_jump_log > 2.0:
                        self._last_jump_log = now
                        logger.warning(
                            f"[{self.id}] target jumped {jump_rad:.3f} rad on {motor_name}"
                            f" (threshold {self.config.align_jump_threshold}), ramping"
                        )
                    break

        # Alignment ramp: rate-limit commanded positions toward the target.
        # The gripper is EXCLUDED from the clamp (POS_FORCE finger — the ramp
        # has no safety value for it and only delays grasps).
        if self.config.align_step_limit is not None:
            step_deg = float(np.degrees(self.config.align_step_limit))
            if not self._last_cmd_deg:
                # First command: ramp from the measured pose, not from zero.
                states = self.bus.sync_read_all_states()
                self._last_cmd_deg = {m: s["position"] for m, s in states.items()}
            for motor_name, position in goal_pos.items():
                if motor_name == "gripper" or motor_name not in self._last_cmd_deg:
                    continue
                prev = self._last_cmd_deg[motor_name]
                goal_pos[motor_name] = prev + float(np.clip(position - prev, -step_deg, step_deg))

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            if states is not None:
                present_pos = {motor: state["position"] for motor, state in states.items()}
            else:
                present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Gravity feedforward torque for the MIT torque slot (arm joints only;
        # the gripper gets 0 — it runs POS_FORCE). Uses the measured pose:
        # one CAN refresh per cycle, reused from the ramp when available.
        tff = np.zeros(7)
        if self._gravity_ff is not None:
            if states is None:
                states = self.bus.sync_read_all_states()
            q_meas_rad = np.radians([states[m]["position"] for m in ARM_JOINTS])
            tff = self._gravity_ff.torque(q_meas_rad)
        self._last_tff = tff

        # Velocity feedforward: finite difference of successive commanded
        # positions, clamped per joint to the OpenArm 2.0 delta limits.
        vff_deg_s = np.zeros(8)
        if self.config.velocity_ff_gain > 0.0 and self._last_send_time is not None and self._last_cmd_deg:
            dt = max(1e-3, now - self._last_send_time)
            for motor_name, position in goal_pos.items():
                idx = MOTOR_INDEX.get(motor_name)
                if idx is None or motor_name not in self._last_cmd_deg:
                    continue
                vel_rad_s = np.radians(position - self._last_cmd_deg[motor_name]) / dt
                vel_rad_s = float(
                    np.clip(vel_rad_s, -JOINT_DELTA_LIMITS_RAD_S[idx], JOINT_DELTA_LIMITS_RAD_S[idx])
                )
                vff_deg_s[idx] = self.config.velocity_ff_gain * float(np.degrees(vel_rad_s))

        # J1-J7 use MIT. OpenArm 2.0 configures J8 separately in POS_FORCE;
        # retain MIT only as an explicit compatibility mode.
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            if motor_name == "gripper" and self.config.gripper_control_mode == "pos_force":
                continue
            idx = MOTOR_INDEX.get(motor_name, 0)
            # Use custom gains if provided, otherwise use config defaults
            if custom_kp is not None and motor_name in custom_kp:
                kp = custom_kp[motor_name]
            else:
                kp = (
                    self.config.position_kp[idx]
                    if isinstance(self.config.position_kp, list)
                    else self.config.position_kp
                )
            if custom_kd is not None and motor_name in custom_kd:
                kd = custom_kd[motor_name]
            else:
                kd = (
                    self.config.position_kd[idx]
                    if isinstance(self.config.position_kd, list)
                    else self.config.position_kd
                )
            torque = float(tff[idx]) if motor_name in ARM_JOINTS else 0.0
            commands[motor_name] = (kp, kd, position_degrees, float(vff_deg_s[idx]), torque)

        self.bus.mit_control_batch(commands)
        if "gripper" in goal_pos and self.config.gripper_control_mode == "pos_force":
            self.bus.posforce_control(
                "gripper",
                position_rad=math.radians(goal_pos["gripper"]),
                speed_rad_s=self.config.gripper_speed_rad_s,
                current_pu=self.config.gripper_torque_pu,
            )

        # Merge so partial actions keep the previous positions of uncommanded motors.
        self._last_cmd_deg = {**self._last_cmd_deg, **goal_pos}
        self._last_send_time = now

        # Telemetry: reuse this cycle's fresh states if we read them, otherwise
        # the cache just updated by the batch responses above (no extra CAN traffic).
        telem_states = states if states is not None else self.bus.get_cached_states()
        if all(m in telem_states for m in MOTOR_ORDER) and all(m in self._last_cmd_deg for m in MOTOR_ORDER):
            self._telemetry.update(
                q_cmd=[self._last_cmd_deg[m] for m in MOTOR_ORDER],
                q_pos=[telem_states[m]["position"] for m in MOTOR_ORDER],
                q_torque=[telem_states[m]["torque"] for m in MOTOR_ORDER],
                t_mos=[telem_states[m]["temp_mos"] for m in MOTOR_ORDER],
                tff=tff,
            )
            self._telemetry.maybe_report(now)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        """Disconnect from robot."""

        # Disconnect CAN bus
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
