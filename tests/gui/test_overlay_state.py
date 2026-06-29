"""Unit tests for the live-overlay state machine — pure, no IPC/HTTP/GPU, so the transition
table is verified in isolation (the whole point of the abstraction)."""

from __future__ import annotations

from lerobot.overlays.overlay_state import Event, OverlayStateMachine, State


def test_initial_is_inactive():
    assert OverlayStateMachine().state is State.INACTIVE


def test_happy_path_walks_the_table():
    sm = OverlayStateMachine()
    assert sm.fire(Event.START) and sm.state is State.LOADING
    assert sm.fire(Event.LOADED) and sm.state is State.ACTIVE
    assert sm.fire(Event.STOP) and sm.state is State.STOPPING
    assert sm.fire(Event.STOPPED) and sm.state is State.INACTIVE


def test_remove_then_rename_does_not_get_stuck():
    # The user's scenario: removing the concept stops it; the start that follows is serialised
    # by the adapter's lock to fire AFTER teardown, so it's a clean INACTIVE -> LOADING.
    sm = OverlayStateMachine()
    sm.fire(Event.START)
    sm.fire(Event.LOADED)
    assert sm.fire(Event.STOP) and sm.state is State.STOPPING
    assert sm.fire(Event.STOPPED) and sm.state is State.INACTIVE
    assert sm.fire(Event.START) and sm.state is State.LOADING  # re-name reloads, not stuck


def test_invalid_events_are_logged_noops():
    sm = OverlayStateMachine()
    assert not sm.fire(Event.LOADED)  # can't load before starting
    assert sm.state is State.INACTIVE
    sm.fire(Event.START)
    assert not sm.fire(Event.START)  # already loading — no second spawn
    assert not sm.fire(Event.STOPPED)  # not stopping
    assert sm.state is State.LOADING


def test_stop_during_loading():
    sm = OverlayStateMachine()
    sm.fire(Event.START)
    assert sm.fire(Event.STOP) and sm.state is State.STOPPING  # stop while still warming
    assert sm.fire(Event.STOPPED) and sm.state is State.INACTIVE


def test_crash_then_restart_or_reset():
    sm = OverlayStateMachine()
    sm.fire(Event.START)
    assert sm.fire(Event.CRASH) and sm.state is State.ERROR
    assert sm.fire(Event.START) and sm.state is State.LOADING  # restart straight from error
    sm.fire(Event.LOADED)
    sm.fire(Event.CRASH)
    assert sm.fire(Event.RESET) and sm.state is State.INACTIVE


def test_crash_during_stopping_is_clean_inactive():
    sm = OverlayStateMachine()
    sm.fire(Event.START)
    sm.fire(Event.LOADED)
    sm.fire(Event.STOP)
    assert sm.fire(Event.CRASH) and sm.state is State.INACTIVE  # died while already tearing down


def test_can_reflects_the_table():
    sm = OverlayStateMachine()
    assert sm.can(Event.START) and not sm.can(Event.LOADED)
    sm.fire(Event.START)
    assert sm.can(Event.LOADED) and sm.can(Event.STOP) and not sm.can(Event.START)


def test_transition_table_is_exhaustive():
    """Every (state, event) pair is accounted for: the listed pairs transition as specified, and
    EVERY other pair is a logged no-op (fire returns False, state unchanged) — the machine never
    raises on an unexpected event. This walks all |State|*|Event| pairs as the explicit spec."""
    from lerobot.overlays.overlay_state import _TABLE

    for state in State:
        for event in Event:
            sm = OverlayStateMachine()
            sm._state = state  # force the start state
            transitioned = sm.fire(event)
            if (state, event) in _TABLE:
                assert transitioned and sm.state is _TABLE[(state, event)], (
                    f"{state}+{event} should transition"
                )
            else:
                assert not transitioned and sm.state is state, f"{state}+{event} must be a no-op"


def test_loading_crash_goes_to_error():
    """A crash WHILE warming (e.g. a gated-weights load failure) -> error, not a hang."""
    sm = OverlayStateMachine()
    sm.fire(Event.START)
    assert sm.state is State.LOADING
    assert sm.fire(Event.CRASH) and sm.state is State.ERROR


def test_on_transition_fires_only_on_real_transitions():
    seen = []
    sm = OverlayStateMachine(on_transition=lambda p, e, n: seen.append((p.value, e.value, n.value)))
    sm.fire(Event.START)
    sm.fire(Event.LOADED)
    sm.fire(Event.LOADED)  # invalid — must NOT call the hook
    assert seen == [("inactive", "start", "loading"), ("loading", "loaded", "active")]
