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

"""Manual CLI example: Quest VR teleop driving SO-107 via the pink IK pipeline.

Demonstrates the composition needed until ``lerobot-teleoperate`` learns to
auto-build the Cartesian pipeline. The GUI integration depends on whoever
writes that auto-composition layer (probably detect "target_x" in the
teleop's action_features and inject EEReferenceAndDelta + IK).

Usage:
    python examples/quest_vr_to_so107/teleoperate.py \\
        --robot-port /dev/ttyACM2 --robot-id white_right
"""

from __future__ import annotations

import argparse
import logging
import time

from lerobot.model.pink_kinematics import PinkKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so107_description import get_urdf_path
from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig
from lerobot.robots.so_follower.pink_kinematic_processor import (
    PinkInverseKinematicsEEToJoints,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
)
from lerobot.teleoperators.quest_vr import QuestVRTeleop, QuestVRTeleopConfig

logger = logging.getLogger(__name__)


SO107_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-port", default="/dev/ttyACM2")
    parser.add_argument("--robot-id", default="white_right")
    parser.add_argument("--teleop-port", type=int, default=8443)
    parser.add_argument("--max-relative-target", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # --- Robot setup ---
    robot_cfg = SO107FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=True,
        cameras={},
        max_relative_target=args.max_relative_target,
    )
    robot = SO107Follower(robot_cfg)

    # --- Teleop setup ---
    teleop_cfg = QuestVRTeleopConfig(id="quest_vr", port=args.teleop_port)
    teleop = QuestVRTeleop(teleop_cfg)

    # --- IK pipeline ---
    # EEReferenceAndDelta: latches a reference EE pose on engage, applies the
    # incoming target_x/y/z + target_wx/wy/wz to it.
    # EEBoundsAndSafety: clips into a workspace box, vetoes large jumps.
    # GripperVelocityToJoint: integrates gripper_vel into ee.gripper_pos.
    # PinkInverseKinematicsEEToJoints: ee.* -> <motor>.pos via pink + posture.
    pink_kin = PinkKinematics(urdf_path=str(get_urdf_path()), target_frame_name="L7_1")

    # Robot-side FK kinematics for EEReferenceAndDelta (uses placo, but it
    # only needs FK; that part isn't affected by the no-posture-task issue).
    from lerobot.model import RobotKinematics

    fk_kin = RobotKinematics(
        urdf_path=str(get_urdf_path()),
        target_frame_name="L7_1",
        joint_names=[f"S{i}" for i in range(1, 8)],
    )

    pipeline_steps = [
        EEReferenceAndDelta(
            kinematics=fk_kin,
            end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},
            motor_names=SO107_MOTOR_NAMES,
            use_latched_reference=True,
        ),
        EEBoundsAndSafety(
            end_effector_bounds={
                "min": [-0.30, -0.40, +0.02],
                "max": [+0.30, +0.10, +0.40],
            },
            max_ee_step_m=0.10,
        ),
        GripperVelocityToJoint(speed_factor=20.0, clip_min=0.0, clip_max=100.0),
        PinkInverseKinematicsEEToJoints(kinematics=pink_kin, motor_names=SO107_MOTOR_NAMES),
    ]
    pipeline = RobotProcessorPipeline(
        steps=pipeline_steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # --- Run loop ---
    print("\nConnecting to robot ...")
    robot.connect(calibrate=False)
    print("Starting Quest VR teleop server ...")
    teleop.connect()
    print(f"\n  ==> open {teleop.url} on Quest 3 browser; tap Connect + Enter AR.")
    print("     (Self-signed cert: accept once.)")
    print("     Squeeze right grip to engage tracking, trigger for gripper.\n")
    input("Press Enter when ready to start the control loop...")

    period = 1.0 / args.fps
    try:
        next_t = time.perf_counter()
        while True:
            obs = robot.get_observation()
            raw_action = teleop.get_action()
            joint_action = pipeline((raw_action, obs))
            robot.send_action(joint_action)
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        teleop.disconnect()
        robot.disconnect()
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
