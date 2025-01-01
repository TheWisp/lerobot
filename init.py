from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus, TorqueMode

leader_port = "/dev/tty.usbmodem58760429321"
follower_port = "/dev/tty.usbmodem58760431431"

leader_arm = DynamixelMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_arm = DynamixelMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)
#leader_arm.connect()
#follower_arm.connect()

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
robot = ManipulatorRobot(
    robot_type="koch",
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/koch",
)
# Connect motors buses and cameras if any (Required)
robot.connect()
try:
    while True:
        robot.teleop_step()
except KeyboardInterrupt:
    follower_arm.write("Torque_Enable", TorqueMode.DISABLED.value)
    robot.disconnect()
    print("Done")

"""
leader_arm.connect()
follower_arm.connect()
leader_pos = leader_arm.read("Present_Position")
follower_pos = follower_arm.read("Present_Position")
print(leader_pos)
print(follower_pos)
print("done")
"""