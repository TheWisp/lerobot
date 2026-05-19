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
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.pink_kinematics import PinkKinematics
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    _motor_to_urdf_deg,
    _urdf_to_motor_deg,
)
from lerobot.utils.latency.ik_debug import get_recorder
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
        key_prefix: Prefix prepended to every ``ee.*`` and ``<motor>.pos`` key
            this step reads or writes (and the observation ``<motor>.pos`` keys
            it filters). Empty for unimanual; ``"left_"``/``"right_"`` for the
            two arms of a bimanual setup.
    """

    kinematics: PinkKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    key_prefix: str = ""
    # See EEReferenceAndDelta.joint_map.
    joint_map: Any | None = None

    def action(self, action: RobotAction) -> RobotAction:
        # ``dict.pop`` raises KeyError if the key is missing, so the prior
        # ``if None in (...)`` defensive check never fired in practice.
        p = self.key_prefix
        x = action.pop(f"{p}ee.x")
        y = action.pop(f"{p}ee.y")
        z = action.pop(f"{p}ee.z")
        wx = action.pop(f"{p}ee.wx")
        wy = action.pop(f"{p}ee.wy")
        wz = action.pop(f"{p}ee.wz")
        gripper_pos = action.pop(f"{p}ee.gripper_pos")

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        # Read in motor_names order; do NOT rely on dict iteration order.
        q_raw_motor = np.array(
            [float(observation[f"{p}{name}.pos"]) for name in self.motor_names],
            dtype=float,
        )
        # Pink's IK runs in URDF space, so convert the seed.
        q_raw = _motor_to_urdf_deg(q_raw_motor, self.motor_names, self.joint_map)

        if self.initial_guess_current_joints or self.q_curr is None:
            self.q_curr = q_raw

        # Build target SE(3) from position + rotvec.
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Capture the seed BEFORE the IK call. With initial_guess_current_joints=True
        # this equals q_raw (the observation); with =False it diverges and that's
        # exactly what we want to see in the trace.
        q_seed_used_urdf = self.q_curr.copy()
        q_target_urdf = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target_urdf

        # Convert IK output (URDF) back to motor space for the action stream.
        q_target_motor = _urdf_to_motor_deg(q_target_urdf, self.motor_names, self.joint_map)

        _rec = get_recorder()
        if _rec is not None:
            _rec.record(self.key_prefix, "ik_seed_urdf", q_seed_used_urdf)
            _rec.record(self.key_prefix, "ik_q_obs_urdf", q_raw.copy())
            _rec.record(self.key_prefix, "ik_q_urdf", q_target_urdf.copy())
            _rec.record(self.key_prefix, "ik_q_motor", q_target_motor.copy())
            _rec.record(self.key_prefix, "ik_n_iters", float(getattr(self.kinematics, "last_n_iters", -1)))
            _rec.record(self.key_prefix, "ik_gripper_cmd_motor", float(gripper_pos))
            # Position + orientation error vs target (FK round-trip).
            try:
                fk_after = self.kinematics.forward_kinematics(q_target_urdf)
                _rec.record(
                    self.key_prefix, "ik_pos_err_m", float(np.linalg.norm(fk_after[:3, 3] - t_des[:3, 3]))
                )
                # Orientation error: angle of the rotation R_err = R_des @ R_fk.T
                r_err = t_des[:3, :3] @ fk_after[:3, :3].T
                # trace-of-rotation -> angle (clamped for numerical safety).
                cos_a = max(-1.0, min(1.0, (np.trace(r_err) - 1.0) * 0.5))
                _rec.record(self.key_prefix, "ik_rot_err_rad", float(np.arccos(cos_a)))
            except Exception:
                pass

        # Map IK output to motor names. The "gripper" slot of q_target_motor
        # is overridden by ``ee.gripper_pos``, which is already in motor
        # space — the teleop emits an absolute trigger->motor mapping
        # directly and the gripper isn't EE-tracked through IK.
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{p}{name}.pos"] = float(q_target_motor[i])
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
