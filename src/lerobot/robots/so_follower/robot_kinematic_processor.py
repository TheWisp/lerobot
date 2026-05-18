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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta(RobotActionProcessorStep):
    """
    Computes a target end-effector pose from a relative delta command.

    This step takes a desired change in position and orientation (`target_*`) and applies it to a
    reference end-effector pose to calculate an absolute target pose. The reference pose is derived
    from the current robot joint positions using forward kinematics.

    The processor can operate in two modes:
    1.  `use_latched_reference=True`: The reference pose is "latched" or saved at the moment the action
        is first enabled. Subsequent commands are relative to this fixed reference.
    2.  `use_latched_reference=False`: The reference pose is updated to the robot's current pose at
        every step.

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        end_effector_step_sizes: A dictionary scaling the input delta commands.
        motor_names: A list of motor names required for forward kinematics.
        use_latched_reference: If True, latch the reference pose on enable; otherwise, always use the
            current pose as the reference.
        key_prefix: Optional prefix prepended to every action/observation key read or
            written by this step (e.g., ``"left_"``). Used for bimanual setups so two
            chains can share a single transition without key collisions. Empty string
            (default) preserves the unimanual key set (target_x, ee.x, <motor>.pos).
        world_yaw_deg: Rotation around the URDF +Z axis applied to the incoming
            target_x/y/z (translation) and target_wx/wy/wz (rotvec) BEFORE they
            are composed with the reference pose. Use this to express the
            per-arm mounting orientation when the teleop emits deltas in a
            single user-reference frame but the arms are mounted at different
            yaw angles relative to that frame. Common values:
              * ``0.0``  — arm faces the same direction as the teleop's reference
                (parallel mounting for bimanual setups).
              * ``180.0`` — arm rotated 180 deg around Z (mirrored mounting:
                two arms facing each other across the workspace).
              * Any other angle in degrees for arbitrary mountings.
            Identity for unimanual.
        reference_ee_pose: Internal state storing the latched reference pose.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _command_when_disabled: Internal state to hold the last command while disabled.
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    use_latched_reference: bool = (
        True  # If True, latch reference on enable; if False, always use current pose
    )
    use_ik_solution: bool = False
    key_prefix: str = ""
    world_yaw_deg: float = 0.0

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            pref = self.key_prefix
            q_raw = np.array(
                [
                    float(v)
                    for k, v in observation.items()
                    if isinstance(k, str)
                    and k.startswith(pref)
                    and k.endswith(".pos")
                    and k[len(pref) :].removesuffix(".pos") in self.motor_names
                ],
                dtype=float,
            )

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q_raw)

        p = self.key_prefix
        enabled = bool(action.pop(f"{p}enabled"))
        tx = float(action.pop(f"{p}target_x"))
        ty = float(action.pop(f"{p}target_y"))
        tz = float(action.pop(f"{p}target_z"))
        wx = float(action.pop(f"{p}target_wx"))
        wy = float(action.pop(f"{p}target_wy"))
        wz = float(action.pop(f"{p}target_wz"))
        gripper_vel = float(action.pop(f"{p}gripper_vel"))

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                # Latched reference mode: latch reference at the rising edge
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = t_curr.copy()
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            r_delta = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
            # Per-arm yaw: rotate both the translation delta and the orientation
            # delta around URDF +Z so the teleop's user-reference frame maps to
            # THIS arm's URDF frame. Identity (0 deg) for unimanual / parallel
            # bimanual; 180 deg for mirrored mounting.
            if self.world_yaw_deg != 0.0:
                yaw_rad = np.deg2rad(self.world_yaw_deg)
                c, s = np.cos(yaw_rad), np.sin(yaw_rad)
                r_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
                delta_p = r_yaw @ delta_p
                r_delta = r_yaw @ r_delta @ r_yaw.T
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = ref[:3, :3] @ r_delta
            desired[:3, 3] = ref[:3, 3] + delta_p

            self._command_when_disabled = desired.copy()
        else:
            # While disabled, keep sending the same command to avoid drift.
            if self._command_when_disabled is None:
                # If we've never had an enabled command yet, freeze current FK pose once.
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action[f"{p}ee.x"] = float(pos[0])
        action[f"{p}ee.y"] = float(pos[1])
        action[f"{p}ee.z"] = float(pos[2])
        action[f"{p}ee.wx"] = float(tw[0])
        action[f"{p}ee.wy"] = float(tw[1])
        action[f"{p}ee.wz"] = float(tw[2])
        action[f"{p}ee.gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        p = self.key_prefix
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{p}{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"{p}ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.

    This step ensures that the target end-effector pose remains within a safe operational workspace.
    It also moderates the command to prevent large, sudden movements between consecutive steps.

    Attributes:
        end_effector_bounds: A dictionary with "min" and "max" keys for position clipping.
        max_ee_step_m: The maximum allowed change in position (in meters) between steps.
        key_prefix: Prefix prepended to every ``ee.*`` key (see EEReferenceAndDelta).
        _last_pos: Internal state storing the last commanded position.
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.2
    key_prefix: str = ""
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        p = self.key_prefix
        x = action[f"{p}ee.x"]
        y = action[f"{p}ee.y"]
        z = action[f"{p}ee.z"]
        wx = action[f"{p}ee.wx"]
        wy = action[f"{p}ee.wy"]
        wz = action[f"{p}ee.wz"]
        # TODO(Steven): ee.gripper_vel does not need to be bounded

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # Clip position
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # Check for jumps in position
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)
                raise ValueError(f"EE jump {n:.3f}m > {self.max_ee_step_m}m")

        self._last_pos = pos

        action[f"{p}ee.x"] = float(pos[0])
        action[f"{p}ee.y"] = float(pos[1])
        action[f"{p}ee.z"] = float(pos[2])
        action[f"{p}ee.wx"] = float(twist[0])
        action[f"{p}ee.wy"] = float(twist[1])
        action[f"{p}ee.wz"] = float(twist[2])
        return action

    def reset(self):
        """Resets the last known position and orientation."""
        self._last_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This step translates a Cartesian command (position and orientation of the end-effector) into
    the corresponding joint-space commands for each motor.

    Attributes:
        kinematics: The robot's kinematic model for inverse kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        q_curr: Internal state storing the last joint positions, used as an initial guess for the IK solver.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
            If False, use the solution from the previous step.
        key_prefix: Prefix prepended to ``ee.*`` and ``<motor>.pos`` keys
            (see EEReferenceAndDelta).
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    key_prefix: str = ""

    def action(self, action: RobotAction) -> RobotAction:
        p = self.key_prefix
        x = action.pop(f"{p}ee.x")
        y = action.pop(f"{p}ee.y")
        z = action.pop(f"{p}ee.z")
        wx = action.pop(f"{p}ee.wx")
        wy = action.pop(f"{p}ee.wy")
        wz = action.pop(f"{p}ee.wz")
        gripper_pos = action.pop(f"{p}ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [
                float(v)
                for k, v in observation.items()
                if isinstance(k, str) and k.startswith(p) and k.endswith(".pos")
            ],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{p}{name}.pos"] = float(q_target[i])
            else:
                action[f"{p}gripper.pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        p = self.key_prefix
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"{p}ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{p}{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint(RobotActionProcessorStep):
    """
    Converts a gripper velocity command into a target gripper joint position.

    This step integrates a normalized velocity command over time to produce a position command,
    taking the current gripper position as a starting point. It also supports a discrete mode
    where integer actions map to open, close, or no-op.

    Attributes:
        motor_names: A list of motor names, which must include 'gripper'.
        speed_factor: A scaling factor to convert the normalized velocity command to a position change.
        clip_min: The minimum allowed gripper joint position.
        clip_max: The maximum allowed gripper joint position.
        discrete_gripper: If True, treat the input action as discrete (0: open, 1: close, 2: stay).
        key_prefix: Prefix prepended to ``ee.gripper_vel`` / ``ee.gripper_pos`` and to
            observation ``<motor>.pos`` keys read (see EEReferenceAndDelta).
    """

    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False
    key_prefix: str = ""

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        p = self.key_prefix
        gripper_vel = action.pop(f"{p}ee.gripper_vel")

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [
                float(v)
                for k, v in observation.items()
                if isinstance(k, str) and k.startswith(p) and k.endswith(".pos")
            ],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: open, 1: close, 2: stay
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_vel = (gripper_vel - 1) * self.clip_max

        # Compute desired gripper position
        delta = gripper_vel * float(self.speed_factor)
        # TODO: This assumes gripper is the last specified joint in the robot
        gripper_pos = float(np.clip(q_raw[-1] + delta, self.clip_min, self.clip_max))
        action[f"{p}ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        p = self.key_prefix
        features[PipelineFeatureType.ACTION].pop(f"{p}ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION][f"{p}ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


def compute_forward_kinematics_joints_to_ee(
    joints: dict[str, Any], kinematics: RobotKinematics, motor_names: list[str]
) -> dict[str, Any]:
    motor_joint_values = [joints[f"{n}.pos"] for n in motor_names]

    q = np.array(motor_joint_values, dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    gripper_pos = joints["gripper.pos"]
    for n in motor_names:
        joints.pop(f"{n}.pos")
    joints["ee.x"] = float(pos[0])
    joints["ee.y"] = float(pos[1])
    joints["ee.z"] = float(pos[2])
    joints["ee.wx"] = float(tw[0])
    joints["ee.wy"] = float(tw[1])
    joints["ee.wz"] = float(tw[2])
    joints["ee.gripper_pos"] = float(gripper_pos)
    return joints


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_observation")
@dataclass
class ForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def observation(self, observation: RobotObservation) -> RobotObservation:
        return compute_forward_kinematics_joints_to_ee(observation, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.OBSERVATION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_action")
@dataclass
class ForwardKinematicsJointsToEEAction(RobotActionProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        return compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register(name="forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ProcessorStep):
    kinematics: RobotKinematics
    motor_names: list[str]

    def __post_init__(self):
        self.joints_to_ee_action_processor = ForwardKinematicsJointsToEEAction(
            kinematics=self.kinematics, motor_names=self.motor_names
        )
        self.joints_to_ee_observation_processor = ForwardKinematicsJointsToEEObservation(
            kinematics=self.kinematics, motor_names=self.motor_names
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if transition.get(TransitionKey.ACTION) is not None:
            transition = self.joints_to_ee_action_processor(transition)
        if transition.get(TransitionKey.OBSERVATION) is not None:
            transition = self.joints_to_ee_observation_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if features[PipelineFeatureType.ACTION] is not None:
            features = self.joints_to_ee_action_processor.transform_features(features)
        if features[PipelineFeatureType.OBSERVATION] is not None:
            features = self.joints_to_ee_observation_processor.transform_features(features)
        return features


@ProcessorStepRegistry.register("inverse_kinematics_rl_step")
@dataclass
class InverseKinematicsRLStep(ProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This is modified from the InverseKinematicsEEToJoints step to be used in the RL pipeline.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for InverseKinematicsEEToJoints")
        action = dict(action)

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = new_transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        new_transition[TransitionKey.ACTION] = action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data["IK_solution"] = q_target
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None
