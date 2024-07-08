
import time
import numpy as np
import pytest
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus


def test_motors_bus():
    # Test instantiating a common motors structure.
    # Here the one from Alexander Koch follower arm.
    motors = {
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    }
    motors_bus = DynamixelMotorsBus(
        port="/dev/tty.usbmodem575E0032081",
        motors=motors,
    )

    # Test reading and writting before connecting raises an error
    with pytest.raises(ValueError):
        motors_bus.read("Torque_Enable")
    with pytest.raises(ValueError):
        motors_bus.write("Torque_Enable")

    motors_bus.connect()

    # Test connecting twice raises an error
    with pytest.raises(ValueError):
        motors_bus.connect()

    # Test reading torque on all motors and its 0 after first connection
    values = motors_bus.read("Torque_Enable")
    assert isinstance(values, np.ndarray)
    assert len(values) == len(motors)
    assert (values == 0).all()

    # Test writing torque on a specific motor
    motors_bus.write("Torque_Enable", 1, "gripper")

    # Test reading torque from this specific motor. It is now 1
    values = motors_bus.read("Torque_Enable", "gripper")
    assert len(values) == 1
    assert values[0] == 1

    # Test reading torque from all motors. It is 1 for the specific motor,
    # and 0 on the others.
    values = motors_bus.read("Torque_Enable")
    gripper_index = motors_bus.motor_names.index("gripper")
    assert values[gripper_index] == 1
    assert values.sum() == 1  # gripper is the only motor to have torque 1

    # Test writing torque on all motors and it is 1 for all.
    motors_bus.write("Torque_Enable", 1)
    values = motors_bus.read("Torque_Enable")
    assert (values == 1).all()

    # Test ordering the motors to move slightly (+1 value among 4096) and this move
    # can be executed and seen by the motor position sensor
    values = motors_bus.read("Present_Position")
    motors_bus.write("Goal_Position", values + 1)
    # Give time for the motors to move to the goal position
    time.sleep(1)
    new_values = motors_bus.read("Present_Position")
    assert new_values == values

    # TODO(rcadene): test calibration
    # TODO(rcadene): test logs?
