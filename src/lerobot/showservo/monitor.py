# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Stage lifecycle: certificates in, one decision out, nothing implicit.

The monitor is a PURE state machine. It never touches the robot, the camera or the
clock; it is handed events and returns the decision. That is what makes the retry
ladder testable without a rig — the ladder is where an unattended run either recovers
or bends something, and it is the last place a bug should be discoverable only on
hardware.

Two rules the transition table enforces structurally:

* **Every (state, event) pair is spelled out.** ``_TABLE`` is checked for
  completeness at import, so a new event cannot be added without deciding what every
  state does with it. Unreachable-looking pairs are the ones that fire at 2 a.m.
* **No blind action** (invariant 5). Entering SERVOING requires a bind certificate;
  losing it drops back to the ladder rather than continuing on the last known good
  transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product

from lerobot.showservo.card import Stage


class State(str, Enum):
    IDLE = "idle"
    BINDING = "binding"
    SERVOING = "servoing"
    RECOVERING = "recovering"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"


TERMINAL = (State.SUCCEEDED, State.ABORTED)


class Event(str, Enum):
    START = "start"
    BIND_OK = "bind_ok"
    BIND_FAIL = "bind_fail"
    TERMINATED = "terminated"  # the stage's own termination condition fired
    NO_PROGRESS = "no_progress"  # the error-decreasing certificate failed
    TRACK_LOST = "track_lost"  # a team's fit stopped being produceable
    TIMEOUT = "timeout"  # the stage's second budget ran out
    RECOVERY_DONE = "recovery_done"  # the executor finished the rung it was given


class Rung(str, Enum):
    """The ladder of §3.5, in escalation order.

    Every rung is executed FROM A KNOWN BACKED-OFF POSE, so re-running one is
    idempotent — that property is what lets the ladder retry without accumulating
    state, and the executor owes it.
    """

    BACK_OFF = "back_off"  # retreat along travel_dir, re-approach
    REBIND = "rebind"  # fresh designation from the current viewpoint
    CANONICAL_VIEW = "canonical_view"  # retreat to the taught viewpoint, then re-bind
    REGRASP = "regrasp"  # release and take the object again (D2+)
    ABORT = "abort"


_LADDER = (Rung.BACK_OFF, Rung.REBIND, Rung.CANONICAL_VIEW, Rung.REGRASP, Rung.ABORT)

# Which failure class a terminating event is logged under (§3.5). TIMEOUT is resolved
# separately because its meaning depends on what the stage was waiting for.
_FAILURE_CLASS = {
    Event.BIND_FAIL: "bind",
    Event.TRACK_LOST: "track",
    Event.NO_PROGRESS: "servo",
    Event.TIMEOUT: "timeout",
}

_RECOVERABLE = (Event.BIND_FAIL, Event.TRACK_LOST, Event.NO_PROGRESS, Event.TIMEOUT)


@dataclass
class Decision:
    """What the executor should do next. ``rung`` is set only when entering RECOVERING."""

    state: State
    rung: Rung | None = None
    note: str = ""


@dataclass
class AttemptLog:
    """The scientific output of §3.5 — one record per stage attempt.

    Everything a post-hoc analysis needs to separate "the perception was wrong" from
    "the perception was right and the arm could not get there".
    """

    stage: str = ""
    bind_scores: list[float] = field(default_factory=list)
    err_norms: list[float] = field(default_factory=list)
    servo_energy: float = 0.0
    timeline: list[tuple[int, str]] = field(default_factory=list)
    rungs_used: list[str] = field(default_factory=list)
    failure_class: str | None = None
    succeeded: bool = False

    def record_event(self, frame: int, name: str) -> None:
        self.timeline.append((int(frame), str(name)))

    @property
    def final_error(self) -> float:
        return self.err_norms[-1] if self.err_norms else float("inf")


def _build_table() -> dict[tuple[State, Event], State]:
    """Every (state, event) pair, decided once, here.

    Terminal states absorb every event: a late TIMEOUT arriving after the stage
    already succeeded must not un-succeed it.
    """
    t: dict[tuple[State, Event], State] = {}
    for state, event in product(State, Event):
        if state in TERMINAL:
            t[(state, event)] = state
        elif state is State.IDLE:
            t[(state, event)] = State.BINDING if event is Event.START else State.IDLE
        elif state is State.BINDING:
            t[(state, event)] = {
                Event.BIND_OK: State.SERVOING,
                Event.BIND_FAIL: State.RECOVERING,
                Event.TIMEOUT: State.RECOVERING,
            }.get(event, State.BINDING)
        elif state is State.SERVOING:
            t[(state, event)] = {
                Event.TERMINATED: State.SUCCEEDED,
                Event.NO_PROGRESS: State.RECOVERING,
                Event.TRACK_LOST: State.RECOVERING,
                Event.BIND_FAIL: State.RECOVERING,
                Event.TIMEOUT: State.RECOVERING,
            }.get(event, State.SERVOING)
        elif state is State.RECOVERING:
            # A recovery in flight ignores further failure reports: they are echoes of
            # the failure already being recovered from, and letting them re-escalate
            # burns the ladder several rungs at a time.
            t[(state, event)] = {
                Event.RECOVERY_DONE: State.BINDING,
                Event.TERMINATED: State.SUCCEEDED,
            }.get(event, State.RECOVERING)
        else:  # pragma: no cover - exhaustive over State by construction
            raise AssertionError(f"state {state} not handled")
    return t


_TABLE = _build_table()
assert len(_TABLE) == len(State) * len(Event), "the transition table must be total"


class StageMonitor:
    """Drives one stage through bind -> servo -> terminate, or down the ladder.

    Pre: constructed with the stage being attempted; ``feed`` is called with events
    as they are certified. Post: ``state`` is terminal exactly when the attempt is
    over, and ``log.failure_class`` is set on every abort — an abort with no class is
    a bug, not a mystery.
    """

    def __init__(self, stage: Stage):
        self.stage = stage
        self.state = State.IDLE
        self.log = AttemptLog(stage=stage.name or stage.camera)
        self._rung_index = -1
        self._retries_used = 0
        self._frame = 0

    @property
    def retries_left(self) -> int:
        return max(0, self.stage.budget.retries - self._retries_used)

    def feed(self, event: Event, *, frame: int | None = None, note: str = "") -> Decision:
        """Advance the machine. Post: exactly one Decision; never raises on any event."""
        assert isinstance(event, Event), f"expected Event, got {event!r}"
        if frame is not None:
            self._frame = int(frame)

        prev = self.state
        nxt = _TABLE[(prev, event)]

        if prev is not nxt or event in _RECOVERABLE:
            self.log.record_event(self._frame, event.value)

        # Entering recovery is the only place the ladder advances, and the only place
        # the retry budget is spent.
        if nxt is State.RECOVERING and prev is not State.RECOVERING:
            self.state = nxt
            return self._escalate(event, note)

        self.state = nxt
        if nxt is State.SUCCEEDED and prev is not State.SUCCEEDED:
            self.log.succeeded = True
            self.log.failure_class = None
        return Decision(state=nxt, note=note)

    def _escalate(self, event: Event, note: str) -> Decision:
        """Pick the next rung, or abort with the class of the failure that got us here."""
        self._retries_used += 1
        self._rung_index += 1

        if self.stage.held_end == "gripper":
            # Nothing is held, so there is nothing to regrasp; skipping the rung keeps
            # a D1 stage from burning a retry on a no-op.
            while self._rung_index < len(_LADDER) and _LADDER[self._rung_index] is Rung.REGRASP:
                self._rung_index += 1

        # Compared AFTER the increment against the budget itself, not against the
        # remaining count: with retries=3 the third retry must still be granted, and
        # only the fourth escalation aborts.
        exhausted = self._rung_index >= len(_LADDER) or self._retries_used > self.stage.budget.retries
        rung = Rung.ABORT if exhausted else _LADDER[self._rung_index]

        if rung is Rung.ABORT:
            self.state = State.ABORTED
            self.log.failure_class = self._failure_class(event)
            self.log.rungs_used.append(rung.value)
            self.log.record_event(self._frame, "abort")
            return Decision(state=State.ABORTED, rung=Rung.ABORT, note=note or self.log.failure_class or "")

        self.log.rungs_used.append(rung.value)
        return Decision(state=State.RECOVERING, rung=rung, note=note)

    def _failure_class(self, event: Event) -> str:
        """A timeout means different things depending on what was being waited for.

        Waiting on fission and never seeing it is a GRASP failure — the loop converged
        and the object still did not come with the gripper. Logging that as "timeout"
        would hide the single most actionable failure mode of a pick.
        """
        if event is Event.TIMEOUT and self.stage.termination.type in ("fission", "defission"):
            return "grasp"
        return _FAILURE_CLASS.get(event, "servo")
