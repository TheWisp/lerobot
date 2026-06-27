"""Live overlay lifecycle — an explicit, pure state machine.

This is the single authority for the live overlay's run state. The IPC, HTTP, and badge
layers are *adapters*: they fire `Event`s into the machine and render `machine.state`. None
of them set the state directly — state changes ONLY by firing a valid event through `fire()`.

Keeping the machine pure (no IPC / HTTP / I/O) is what makes the transitions unit-testable in
isolation: requirements fulfilled fires `START` → `LOADING`, the model finishing loading fires
`LOADED` → `ACTIVE`, an invalid event is a logged no-op, etc.

States
------
inactive : no model loaded (requirements unmet, or stopped/cleared)
loading  : requirements met; the model is warming. Guard: nothing is inferred or served, and
           a concurrent start can't spawn a second process (serialised by the caller's lock).
active   : model loaded and running. fps / util are 0 while idle (no input frames) — still active.
stopping : tearing down (subprocess terminating, shm/VRAM being freed). Guard: a queued start
           WAITS for this to finish (the caller's lock), it is never dropped.
error    : the subprocess died abnormally.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger(__name__)


class State(str, Enum):
    INACTIVE = "inactive"
    LOADING = "loading"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"


class Event(str, Enum):
    START = "start"  # requirements fulfilled -> begin loading
    LOADED = "loaded"  # model finished loading
    STOP = "stop"  # stop requested (requirements unmet / turned off)
    STOPPED = "stopped"  # teardown complete
    CRASH = "crash"  # subprocess died abnormally
    RESET = "reset"  # clear a terminal error back to inactive


# (state, event) -> next state. THIS TABLE IS THE WHOLE SPEC: every (state, event) pair NOT listed
# is a deliberate no-op — fire() logs a warning and leaves the state unchanged (it never raises; a
# stray event must not crash the GUI). That default is safe because every unlisted pair is one of:
#   - physically impossible — e.g. (INACTIVE, CRASH) / (INACTIVE, LOADED): no process to crash/load;
#   - idempotent — e.g. (INACTIVE, STOP), (ACTIVE, START), (ACTIVE, LOADED): already in / past it;
#   - prevented by the start/stop lock — (STOPPING, START): a start arriving during teardown waits
#     for STOPPED, so it fires from INACTIVE, never mid-stop.
# (LOADING, CRASH) and (ACTIVE, CRASH) ARE listed -> ERROR: a model can die while warming OR running.
# test_transition_table_is_exhaustive walks all (state, event) pairs to prove every one of them.
_TABLE: dict[tuple[State, Event], State] = {
    (State.INACTIVE, Event.START): State.LOADING,
    (State.LOADING, Event.LOADED): State.ACTIVE,
    (State.LOADING, Event.STOP): State.STOPPING,
    (State.LOADING, Event.CRASH): State.ERROR,
    (State.ACTIVE, Event.STOP): State.STOPPING,
    (State.ACTIVE, Event.CRASH): State.ERROR,
    (State.STOPPING, Event.STOPPED): State.INACTIVE,
    (State.STOPPING, Event.CRASH): State.INACTIVE,  # died while we were already tearing it down — fine
    (State.ERROR, Event.RESET): State.INACTIVE,
    (State.ERROR, Event.START): State.LOADING,  # restart straight from a prior error
}

# Sanity: the table is well-formed (catches an enum typo at import, not at runtime).
assert all(
    isinstance(s, State) and isinstance(e, Event) and isinstance(n, State) for (s, e), n in _TABLE.items()
), "malformed overlay transition table"


class OverlayStateMachine:
    """Authority for the live overlay lifecycle. State changes ONLY via `fire(event)`."""

    def __init__(self, on_transition: Callable[[State, Event, State], None] | None = None):
        self._state = State.INACTIVE
        self._on_transition = on_transition  # side-effect hook, called only on a real transition

    @property
    def state(self) -> State:
        return self._state

    def can(self, event: Event) -> bool:
        """Whether `event` is valid (would transition) in the current state."""
        return (self._state, event) in _TABLE

    def fire(self, event: Event) -> bool:
        """Apply `event`. Returns True if it transitioned, False if invalid (state unchanged).

        Every accepted transition is logged; an invalid one is logged at WARNING and ignored
        (the machine never raises — a wrong event must not crash the GUI), so callers can fire
        defensively and rely on `can()`/the return value.
        """
        nxt = _TABLE.get((self._state, event))
        if nxt is None:
            logger.warning("overlay state: invalid event %s in %s — ignored", event.value, self._state.value)
            return False
        prev = self._state
        self._state = nxt
        assert self._state in State, "transition produced a non-State"  # invariant
        logger.info("overlay state: %s --%s--> %s", prev.value, event.value, nxt.value)
        if self._on_transition is not None:
            self._on_transition(prev, event, nxt)
        return True
