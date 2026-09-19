# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""An empty action must reach this arm's bus as no command at all.

The teleoperation loop calls ``send_action`` every cycle whatever the
teleoperator returned, so a teleoperator that commands nothing — which is how
a run is driven with no operator on it — is only safe if an empty action is
written as nothing. Found by pointing one at a real SO-107: the loop reached
``sync_write`` with an empty goal and the run died there, on hardware, with
the arms connected.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO107Follower


@pytest.fixture
def follower():
    bus = MagicMock()
    writes: list[tuple[str, dict]] = []
    bus.sync_write.side_effect = lambda name, goal, **_: writes.append((name, dict(goal)))
    bus.writes = writes

    def _bus(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        bus.sync_read.return_value = dict.fromkeys(bus.motors, 0.0)
        return bus

    with (
        patch("lerobot.robots.so_follower.so_follower.FeetechMotorsBus", side_effect=_bus),
        patch.object(SO107Follower, "configure", lambda self: None),
    ):
        robot = SO107Follower(SOFollowerRobotConfig(port="/dev/null", id="test_arm"))
        robot.bus = bus
        yield robot


def test_an_empty_action_writes_nothing_to_the_bus(follower):
    """The bus is not asked to write an empty batch.

    Its sync_write reads the model off the first motor in the batch, so an
    empty one does not write nothing — it raises, and takes the run with it.
    """
    follower.send_action({"shoulder_pan.pos": 1.0})
    before = len(follower.bus.writes)

    for _ in range(5):
        assert follower.send_action({}) == {}

    assert follower.bus.writes[before:] == [], follower.bus.writes[before:]


def test_an_action_with_motors_is_still_written(follower):
    """The complement: skipping the write must not skip a real command."""
    follower.bus.writes.clear()
    follower.send_action({"shoulder_pan.pos": 12.0})

    goals = [goal for name, goal in follower.bus.writes if name == "Goal_Position"]
    assert goals, follower.bus.writes
    assert goals[-1]["shoulder_pan"] == 12.0
