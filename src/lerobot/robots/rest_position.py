"""Rest position recording and playback for any LeRobot robot.

Pure logic module — no disk I/O.  Callers (GUI backend, CLI scripts) are
responsible for persisting / loading the rest-position dict.
"""

import logging
import time

from lerobot.robots.robot import Robot
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


def record_rest_position(robot: Robot) -> dict[str, float]:
    """Snapshot the robot's current joint positions.

    Reads ``get_observation()`` and keeps only the keys present in
    ``action_features`` (i.e. motor positions, not camera frames).

    The robot **must** be connected.

    Returns:
        Dict mapping action-feature keys to their current float values.
    """
    obs = robot.get_observation()
    action_keys = set(robot.action_features.keys())

    rest_pos: dict[str, float] = {}
    for key in action_keys:
        if key in obs:
            rest_pos[key] = float(obs[key])
        else:
            logger.warning("Action key %r not found in observation, skipping", key)

    if not rest_pos:
        raise RuntimeError("No motor positions found in observation matching action features")

    return rest_pos


def move_to_rest_position(
    robot: Robot,
    rest_position: dict[str, float],
    duration_s: float = 3.0,
    steps_per_second: int = 50,
) -> None:
    """Smoothly interpolate the robot from its current position to *rest_position*.

    Sends intermediate waypoints via ``robot.send_action()`` at the
    requested rate. All existing safety limits (``max_relative_target``,
    joint limits) inside ``send_action()`` are honoured. Rest keys must
    match this robot's action and observation keys; a stale profile is an
    error rather than a successful no-op.

    The robot **must** be connected.

    Args:
        robot: A connected :class:`Robot` instance.
        rest_position: Target ``{motor.pos: float}`` dict.
        duration_s: Total movement time in seconds (default 3).
        steps_per_second: Interpolation frequency (default 50 Hz).
    """
    if not rest_position:
        raise ValueError("rest_position is empty")

    # Read current positions
    obs = robot.get_observation()
    action_keys = set(robot.action_features)
    unknown = set(rest_position) - action_keys
    if unknown:
        raise ValueError(f"Rest position keys are not robot action keys: {sorted(unknown)}")
    missing = set(rest_position) - set(obs)
    if missing:
        raise RuntimeError(f"Cannot read current position for rest joints: {sorted(missing)}")
    current_pos = {key: float(obs[key]) for key in rest_position}

    num_steps = max(1, int(duration_s * steps_per_second))
    dt = duration_s / num_steps

    logger.info(
        "Moving to rest position over %.1fs (%d steps at %d Hz)", duration_s, num_steps, steps_per_second
    )

    for step in range(1, num_steps + 1):
        start_t = time.perf_counter()
        alpha = step / num_steps  # ramps 1/N … 1.0

        action = {key: current_pos[key] * (1.0 - alpha) + rest_position[key] * alpha for key in rest_position}

        robot.send_action(action)

        elapsed = time.perf_counter() - start_t
        precise_sleep(max(dt - elapsed, 0.0))

    logger.info("Rest-position command sequence complete")
