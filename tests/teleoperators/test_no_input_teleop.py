"""A teleoperator that commands nothing, and what that has to mean.

It exists so a real robot's whole loop can be run with no input: the
cameras, the control rate, the processors, the observation tap and
everything downstream of them behave as in a session, while the arm is
never told to move. That is only true if the action it produces stays
empty all the way to the motors, so this pins both ends.
"""

from __future__ import annotations

import pytest

from lerobot.processor import make_default_processors
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.no_input import NoInputTeleop, NoInputTeleopConfig


@pytest.fixture
def teleop():
    t = NoInputTeleop(NoInputTeleopConfig())
    yield t
    if t.is_connected:
        t.disconnect()


def test_it_needs_no_hardware_and_no_display(teleop):
    """Everything else in this package opens a device, a bus or an input
    listener; this one must open nothing, or it cannot be the fallback for
    a machine that has neither."""
    assert teleop.is_connected is False
    teleop.connect()
    assert teleop.is_connected is True
    assert teleop.is_calibrated is True
    teleop.calibrate()
    teleop.configure()
    teleop.disconnect()
    assert teleop.is_connected is False


def test_every_action_is_empty(teleop):
    teleop.connect()
    for _ in range(10):
        assert teleop.get_action() == {}


def test_it_declares_that_it_commands_nothing(teleop):
    assert teleop.action_features == {}
    assert teleop.feedback_features == {}
    # Feedback is accepted and dropped: a robot that sends some must not
    # fail against this teleop.
    teleop.connect()
    teleop.send_feedback({"anything": 1.0})


def test_an_action_cannot_be_read_before_connecting(teleop):
    with pytest.raises(AssertionError):
        teleop.get_action()


def test_it_is_reachable_by_name(teleop):
    config = TeleoperatorConfig.get_choice_class("no_input")()
    made = make_teleoperator_from_config(config)
    assert isinstance(made, NoInputTeleop)


def test_the_processors_keep_the_action_empty(teleop):
    """The loop puts the teleop's action through two pipelines before the
    robot sees it. If either filled in defaults, 'commands nothing' would
    become 'commands zero', which is a move."""
    teleop.connect()
    teleop_action_processor, robot_action_processor, _ = make_default_processors()
    observation = {"joint_1.pos": 12.5, "joint_2.pos": -30.0}
    action = teleop_action_processor((teleop.get_action(), observation))
    assert action == {}
    assert robot_action_processor((action, observation)) == {}
