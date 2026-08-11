# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The ladder is where an unattended run either recovers or bends something, so every
(state, event) pair is exercised here rather than discovered on the rig."""

from __future__ import annotations

from itertools import product

import pytest

from lerobot.showservo.card import Budget, Chapter, GoalRelation, Termination
from lerobot.showservo.monitor import _TABLE, ChapterMonitor, Decision, Event, Rung, State

from .conftest import TARGET_UV, make_team


def _servoing(chapter) -> ChapterMonitor:
    mon = ChapterMonitor(chapter)
    mon.feed(Event.START)
    mon.feed(Event.BIND_OK)
    assert mon.state is State.SERVOING
    return mon


def _recover_and_resume(mon: ChapterMonitor) -> None:
    """Finish the rung and re-bind, which is what the executor owes after every retry."""
    mon.feed(Event.RECOVERY_DONE)
    mon.feed(Event.BIND_OK)


def test_the_transition_table_is_total():
    assert set(_TABLE) == set(product(State, Event))


def test_no_event_in_any_state_can_raise(d1_chapter):
    # A monitor that throws mid-attempt leaves the arm wherever it was, under load.
    for event in Event:
        mon = ChapterMonitor(d1_chapter)
        for state in State:
            mon.state = state
            assert isinstance(mon.feed(event), Decision)


def test_the_happy_path(d1_chapter):
    mon = ChapterMonitor(d1_chapter)
    assert mon.feed(Event.START).state is State.BINDING
    assert mon.feed(Event.BIND_OK).state is State.SERVOING
    assert mon.feed(Event.TERMINATED, frame=120).state is State.SUCCEEDED
    assert mon.log.succeeded and mon.log.failure_class is None
    assert "terminated" in [n for _, n in mon.log.timeline]


def test_servoing_never_starts_without_a_bind_certificate(d1_chapter):
    mon = ChapterMonitor(d1_chapter)
    mon.feed(Event.START)
    assert mon.feed(Event.BIND_FAIL).state is State.RECOVERING
    assert mon.state is not State.SERVOING


def test_the_ladder_escalates_in_order_and_skips_regrasp_with_nothing_held(d1_chapter):
    mon = ChapterMonitor(d1_chapter)
    mon.feed(Event.START)
    rungs = []
    for _ in range(3):
        rungs.append(mon.feed(Event.BIND_FAIL).rung)
        mon.feed(Event.RECOVERY_DONE)
    assert rungs == [Rung.BACK_OFF, Rung.REBIND, Rung.CANONICAL_VIEW]


def test_servo_phase_failures_are_ignored_while_still_binding(d1_chapter):
    # Nothing is being servoed yet, so a stale NO_PROGRESS or TRACK_LOST from the
    # previous attempt must not spend a rung of the ladder.
    mon = ChapterMonitor(d1_chapter)
    mon.feed(Event.START)
    for event in (Event.NO_PROGRESS, Event.TRACK_LOST):
        assert mon.feed(event).rung is None
    assert mon.state is State.BINDING and mon.retries_left == d1_chapter.budget.retries


def test_regrasp_is_offered_only_when_something_is_actually_held(d2_chapter):
    mon = _servoing(d2_chapter)
    rungs = []
    for _ in range(4):
        rungs.append(mon.feed(Event.NO_PROGRESS).rung)
        _recover_and_resume(mon)
    assert rungs == [Rung.BACK_OFF, Rung.REBIND, Rung.CANONICAL_VIEW, Rung.REGRASP]


def test_the_retry_budget_is_honoured_exactly(d1_chapter):
    assert d1_chapter.budget.retries == 3
    mon = _servoing(d1_chapter)
    for _ in range(3):
        assert mon.feed(Event.TRACK_LOST).rung is not Rung.ABORT
        _recover_and_resume(mon)
    final = mon.feed(Event.TRACK_LOST)
    assert final.rung is Rung.ABORT and final.state is State.ABORTED


def test_a_recovery_in_flight_does_not_burn_several_rungs_at_once(d1_chapter):
    # Failure reports keep arriving while the arm backs off; they are echoes of the
    # failure already being handled, not new ones.
    mon = _servoing(d1_chapter)
    assert mon.feed(Event.NO_PROGRESS).rung is Rung.BACK_OFF
    for _ in range(5):
        assert mon.feed(Event.TRACK_LOST).rung is None
    assert mon.state is State.RECOVERING
    _recover_and_resume(mon)
    assert mon.feed(Event.NO_PROGRESS).rung is Rung.REBIND


def test_every_abort_carries_a_failure_class(d1_chapter):
    mon = _servoing(d1_chapter)
    for _ in range(4):
        mon.feed(Event.NO_PROGRESS)
        _recover_and_resume(mon)
    assert mon.state is State.ABORTED
    assert mon.log.failure_class == "servo"


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (Event.BIND_FAIL, "bind"),
        (Event.TRACK_LOST, "track"),
        (Event.NO_PROGRESS, "servo"),
    ],
)
def test_failure_classes_survive_to_the_log(d1_chapter, event, expected):
    mon = _servoing(d1_chapter)
    for _ in range(4):
        mon.feed(event)
        _recover_and_resume(mon)
    assert mon.log.failure_class == expected


def test_a_timeout_waiting_for_fission_is_logged_as_a_grasp_failure(d1_chapter):
    # The most actionable failure of a pick: the loop converged and the object still
    # did not come with the gripper. Logging it as "timeout" would hide it.
    mon = ChapterMonitor(d1_chapter)
    mon.feed(Event.START)
    for _ in range(4):
        mon.feed(Event.TIMEOUT)
        mon.feed(Event.RECOVERY_DONE)
    assert mon.log.failure_class == "grasp"


def test_a_timeout_on_a_non_grasp_chapter_stays_a_timeout(d2_chapter):
    mon = ChapterMonitor(d2_chapter)
    mon.feed(Event.START)
    for _ in range(5):
        mon.feed(Event.TIMEOUT)
        mon.feed(Event.RECOVERY_DONE)
    assert mon.log.failure_class == "timeout"


def test_a_late_event_cannot_undo_a_success(d1_chapter):
    mon = ChapterMonitor(d1_chapter)
    mon.feed(Event.START)
    mon.feed(Event.BIND_OK)
    mon.feed(Event.TERMINATED)
    for event in Event:
        assert mon.feed(event).state is State.SUCCEEDED
    assert mon.log.succeeded and mon.log.failure_class is None


def test_an_abort_is_final(d1_chapter):
    mon = ChapterMonitor(d1_chapter)
    mon.feed(Event.START)
    for _ in range(4):
        mon.feed(Event.BIND_FAIL)
        mon.feed(Event.RECOVERY_DONE)
    assert mon.state is State.ABORTED
    for event in Event:
        assert mon.feed(event).state is State.ABORTED


def test_a_chapter_that_terminates_during_recovery_still_counts(d1_chapter):
    # Backing off can itself seat a part or complete a release; refusing to notice
    # would spend the rest of the ladder undoing a success.
    mon = _servoing(d1_chapter)
    mon.feed(Event.NO_PROGRESS)
    assert mon.state is State.RECOVERING
    assert mon.feed(Event.TERMINATED).state is State.SUCCEEDED


def test_the_log_carries_the_rungs_that_were_actually_used(d1_chapter):
    mon = ChapterMonitor(d1_chapter)
    mon.feed(Event.START)
    for _ in range(2):
        mon.feed(Event.BIND_FAIL)
        mon.feed(Event.RECOVERY_DONE)
    assert mon.log.rungs_used == [Rung.BACK_OFF.value, Rung.REBIND.value]


def test_a_zero_retry_card_aborts_on_first_failure():
    chapter = Chapter(
        name="one-shot",
        camera="top",
        teams={"target": make_team(TARGET_UV)},
        goal_relation=GoalRelation(held_uv=[[10.0, 10.0]]),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("contact"),
        budget=Budget(seconds=5.0, retries=0),
    )
    mon = ChapterMonitor(chapter)
    mon.feed(Event.START)
    decision = mon.feed(Event.BIND_FAIL)
    assert decision.rung is Rung.ABORT and mon.log.failure_class == "bind"
