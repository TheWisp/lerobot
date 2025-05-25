from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import (
    SCS_SERIES_BAUDRATE_TABLE,
    FeetechMotorsBus,
)

def init():
    config = FeetechMotorsBusConfig(port='/dev/ttyACM0', motors={'motor_name': (-1, 'sts3215')})
    motor_bus = FeetechMotorsBus(config=config)

    # Try to connect to the motor bus and handle any connection-specific errors
    try:
        motor_bus.connect()
        print(f"Connected on port {motor_bus.port}")
    except OSError as e:
        print(f"Error occurred when connecting to the motor bus: {e}")
        return
    
    # Motor bus is connected, proceed with the rest of the operations
    try:
        print("Scanning all baudrates and motor indices")
        all_baudrates = set(SCS_SERIES_BAUDRATE_TABLE.values())

        for baudrate in all_baudrates:
            motor_bus.set_bus_baudrate(baudrate)
            present_ids = motor_bus.find_motor_indices(list(range(1, 10)))
            if present_ids:
                print(f"Found motor IDs {present_ids} from baudrate {baudrate}")
            else:
                print(f"No motors found from baudrate {baudrate}")
    except Exception as e:
        print(f"Error occurred during motor configuration: {e}")

    finally:
        motor_bus.disconnect()
        print("Disconnected from motor bus.")
    
init()


