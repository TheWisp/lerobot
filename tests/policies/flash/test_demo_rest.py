"""FLASH rejects stale reset profiles before issuing any waypoint."""

from threading import Event
from types import SimpleNamespace

import pytest

from lerobot.policies.flash.demo_rest import move_to_rest_position


@pytest.mark.parametrize("readable", [False, True])
def test_invalid_rest_profile_is_rejected_before_motion(readable):
    sent = []
    robot = SimpleNamespace(
        action_features={"a.pos": float, "missing.pos": float} if readable else {"a.pos": float},
        get_observation=lambda: {"a.pos": 0.0},
        send_action=sent.append,
    )
    with pytest.raises(RuntimeError if readable else ValueError, match="missing.pos"):
        move_to_rest_position(robot, {"a.pos": 10.0, "missing.pos": 5.0}, stop_event=Event())
    assert not sent
