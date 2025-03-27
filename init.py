from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus, TorqueMode
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

leader_port = "/dev/tty.usbmodem58760429321"
follower_port = "/dev/tty.usbmodem58760431431"

leader_config = DynamixelMotorsBusConfig(
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

follower_config = DynamixelMotorsBusConfig(
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
leader_arm = DynamixelMotorsBus(leader_config)
follower_arm = DynamixelMotorsBus(follower_config)
#leader_arm.connect()
#follower_arm.connect()

robot_config = KochRobotConfig(
    leader_arms={"main": leader_config},
    follower_arms={"main": follower_config},
    cameras={},  # We don't use any camera for now
)
robot = ManipulatorRobot(robot_config)
# Connect motors buses and cameras if any (Required)
robot.connect()
try:
    while True:
        try:
            leader_pos = robot.leader_arms["main"].read("Present_Position")
            follower_pos = robot.follower_arms["main"].read("Present_Position")
            observation, action = robot.teleop_step(record_data=True)

            print(f'follower_pos = {follower_pos}')
            print(f'observation = {observation}')
            print(f'leader_pos = {leader_pos}')
            print(f'action = {action}')
        except ConnectionError:
            print("Connection error, continue...")
            
except KeyboardInterrupt:
    print("Stopping motors...")
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
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