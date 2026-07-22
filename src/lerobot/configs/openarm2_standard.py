"""Canonical OpenArm 2 standard-edition LeRobot configuration."""

import math

from lerobot.robots.bi_openarm_follower import BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot.teleoperators.quest_vr import QuestVRTeleopConfig

OPENARM_STANDARD_RATES_RAD_S = (1.8, 1.8, 3.3, 2.3, 3.5, 3.5, 3.5, 3.5)
OPENARM_MOTOR_NAMES = tuple(f"joint_{index}" for index in range(1, 8)) + ("gripper",)


def relative_limits_per_tick(control_fps: int) -> dict[str, float]:
    if isinstance(control_fps, bool) or not isinstance(control_fps, int) or control_fps <= 0:
        raise ValueError("control_fps must be a positive integer")
    return {
        motor: math.degrees(rate) / control_fps
        for motor, rate in zip(OPENARM_MOTOR_NAMES, OPENARM_STANDARD_RATES_RAD_S, strict=True)
    }


def make_openarm2_standard_robot_config(
    *, control_fps: int, enable_torque_on_connect: bool
) -> BiOpenArmFollowerConfig:
    def arm(port: str, side: str) -> OpenArmFollowerConfigBase:
        return OpenArmFollowerConfigBase(
            port=port,
            side=side,
            handshake_on_connect=False,
            configure_on_connect=False,
            enable_torque_on_connect=enable_torque_on_connect,
            disable_torque_on_disconnect=True,
            max_relative_target=relative_limits_per_tick(control_fps),
            gravity_ff_gain=0.9,
            gripper_control_mode="pos_force",
            gripper_speed_rad_s=50.0,
            gripper_torque_pu=0.22,
            cameras={},
        )

    return BiOpenArmFollowerConfig(
        id="openarm2_standard",
        left_arm_config=arm("can1", "left"),
        right_arm_config=arm("can0", "right"),
        cameras={},
        ik_max_iterations=10,
        ik_damping=0.1,
    )


def make_openarm2_quest_config(*, port: int = 8443) -> QuestVRTeleopConfig:
    return QuestVRTeleopConfig(
        id="openarm2_quest",
        port=port,
        position_scale=1.0,
        frame_timeout_s=0.25,
        robot_forward_in_urdf=[1.0, 0.0, 0.0],
        robot_up_in_urdf=[0.0, 0.0, 1.0],
        left_gripper_open_motor=0.0,
        left_gripper_closed_motor=45.0,
        right_gripper_open_motor=0.0,
        right_gripper_closed_motor=-45.0,
    )
