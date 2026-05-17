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

"""
PinkInverseKinematicsEEToJoints — drop-in alternative to InverseKinematicsEEToJoints.

Same input/output action protocol as the placo-based InverseKinematicsEEToJoints
(consumes ee.x/y/z/wx/wy/wz/gripper_pos, emits <motor>.pos), but uses
PinkKinematics under the hood for 6-DOF IK with posture regularization.

Recommended for VR / phone-style teleop where:
- The teleop streams Cartesian deltas at high rate (e.g., 90 Hz from a Quest).
- The IK must produce JOINT trajectories smooth in null-space across ticks —
  otherwise the 7th DOF wanders, leading to chatter or wraparound at limits.

The placo-based InverseKinematicsEEToJoints uses only a FrameTask: it picks
"any" valid joint configuration each tick. With consecutive Cartesian targets
that don't constrain the null space, the chosen config can swing arbitrarily.
PinkInverseKinematicsEEToJoints adds a PostureTask regularizing toward the
seed, so joints stay smooth.
"""

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.pink_kinematics import PinkKinematics
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("pink_inverse_kinematics_ee_to_joints")
@dataclass
class PinkInverseKinematicsEEToJoints(RobotActionProcessorStep):
    """Pink-based IK alternative to InverseKinematicsEEToJoints.

    Drop-in: same action input keys (ee.x/y/z/wx/wy/wz/gripper_pos) and same
    output keys (<motor_name>.pos). Internally uses PinkKinematics which adds
    a PostureTask to suppress 7-DOF null-space drift.

    Attributes:
        kinematics: PinkKinematics instance configured for the target robot.
        motor_names: Ordered motor names. The i-th entry of the IK result is
            assigned to motor_names[i]. The "gripper" entry, if present, is
            passed through from ee.gripper_pos unchanged.
        initial_guess_current_joints: If True, the IK seed is read from the
            current observation each tick (matches the placo step). If False,
            the previous solution is used as the seed (slightly smoother under
            heavy motor lag, but can drift if observations are noisy).
    """

    kinematics: PinkKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, "
                "ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")
        observation = observation.copy()

        # Joint values from observation, ordered by motor_names (gripper inclusive — IK
        # ignores trailing entries past joint_names length, just passes them through).
        q_raw = np.array(
            [
                float(v)
                for k, v in observation.items()
                if isinstance(k, str) and k.endswith(".pos") and k.removesuffix(".pos") in self.motor_names
            ],
            dtype=float,
        )

        if self.initial_guess_current_joints or self.q_curr is None:
            self.q_curr = q_raw

        # Build target SE(3) from position + rotvec.
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # Map IK output to motor names. The IK preserves trailing entries (e.g.,
        # gripper) untouched, but we explicitly overwrite gripper.pos from the
        # incoming ee.gripper_pos for clarity — gripper isn't EE-tracked.
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action

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
