"""Validate FLASH's saved reset profile before the shared rest interpolator moves."""

from lerobot.robots.rest_position import move_to_rest_position as interpolate_to_rest


def move_to_rest_position(robot, rest_position, *, stop_event):
    """Reject unreadable/stale joint keys without changing ordinary robot reset behavior."""

    class ValidatedRestRobot:
        def get_observation(self):
            unknown = set(rest_position) - set(robot.action_features)
            if unknown:
                raise ValueError(f"Rest position keys are not robot action keys: {sorted(unknown)}")
            observation = robot.get_observation()
            missing = set(rest_position) - set(observation)
            if missing:
                raise RuntimeError(f"Cannot read current position for rest joints: {sorted(missing)}")
            return observation

        def send_action(self, action):
            return robot.send_action(action)

    interpolate_to_rest(ValidatedRestRobot(), rest_position, stop_event=stop_event)
