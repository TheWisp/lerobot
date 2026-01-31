#!/usr/bin/env python
"""
Setup a single motor on an SO-107 arm.

Useful for replacing a broken motor without having to disconnect all other motors.

Example:
    python -m lerobot.teleoperators.so_leader.so107_leader.setup_single_motor \
        --port /dev/ttyACM3 \
        --motor elbow_flex
"""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# SO-107 motor configuration (same as in so107_leader.py)
SO107_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "forearm_roll": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
}


def main():
    parser = argparse.ArgumentParser(description="Setup a single SO-107 motor")
    parser.add_argument("--port", required=True, help="Serial port (e.g., /dev/ttyACM3)")
    parser.add_argument(
        "--motor",
        required=True,
        choices=list(SO107_MOTORS.keys()),
        help="Motor name to setup",
    )
    args = parser.parse_args()

    motor_name = args.motor
    motor = SO107_MOTORS[motor_name]

    print(f"Setting up motor '{motor_name}' to ID={motor.id} on port {args.port}")
    print(f"Make sure ONLY the new motor is connected to the controller board!")
    input("Press Enter to continue...")

    # Create bus with just this one motor
    bus = FeetechMotorsBus(
        port=args.port,
        motors={motor_name: motor},
    )

    bus.setup_motor(motor_name)
    print(f"'{motor_name}' motor ID set to {motor.id}")
    print("Done! You can now reconnect all motors and recalibrate.")


if __name__ == "__main__":
    main()
