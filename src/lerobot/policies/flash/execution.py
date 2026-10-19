"""Fixed-rate LeRobot robot dispatch with an atomic, revisable action queue.

The expensive producer publishes whole future chunks. The motor owner only
reads motor state and calls ``Robot.send_action``; neither camera nor IK runs
inside a tick. All times use one monotonic clock.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Protocol


class RobotIO(Protocol):
    def get_state_observation(self) -> dict: ...

    def send_action(self, action: dict) -> dict: ...


@dataclass(frozen=True)
class TimedAction:
    tick: int
    action: dict[str, float]
    interval: str
    interval_complete: bool = False
    event_id: str | None = None
    requires_milestone: str | None = None


@dataclass(frozen=True)
class ActionReceipt:
    tick: int
    scheduled_s: float
    dispatched_s: float
    read_duration_s: float
    write_duration_s: float
    observed: dict
    requested: TimedAction
    accepted: dict
    revision: int


@dataclass(frozen=True)
class MotorTick:
    """Every scheduled tick, including a tick that issues no new command."""

    tick: int
    scheduled_s: float
    observed: dict
    read_duration_s: float
    effective_action: dict[str, float]
    receipt: ActionReceipt | None
    idle_reason: str | None = None
    action_source: str = "commanded"


class PlanRejectedError(RuntimeError):
    pass


class MissedEventError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionQueueSnapshot:
    """Immutable view of the last dispatched API action and unissued suffix."""

    revision: int
    last_tick: int
    last_issued: dict[str, float] | None
    pending: tuple[TimedAction, ...]

    def boundary_before(self, tick: int) -> dict[str, float] | None:
        """Action a new chunk must continue from immediately before ``tick``."""
        if tick <= self.last_tick:
            raise ValueError("planning boundary must be in the future")
        if tick == self.last_tick + 1:
            return None if self.last_issued is None else dict(self.last_issued)
        for action in self.pending:
            if action.tick == tick - 1:
                return dict(action.action)
        return None


class ActionQueue:
    """Short-lock replacement of the unissued suffix of a command sequence."""

    def __init__(self, max_splice_delta: dict[str, float] | None = None) -> None:
        if max_splice_delta is not None and any(value <= 0 for value in max_splice_delta.values()):
            raise ValueError("splice limits must be positive")
        self._lock = Lock()
        self._actions: deque[tuple[int, TimedAction]] = deque()
        self._revision = -1
        self._last_tick = -1
        self._issued_events: set[str] = set()
        self._last_issued: dict[str, float] | None = None
        self._max_splice_delta = max_splice_delta

    def publish(
        self, revision: int, commands: list[TimedAction], *, cancelled_events: frozenset[str] = frozenset()
    ) -> None:
        if not commands or any(b.tick != a.tick + 1 for a, b in zip(commands, commands[1:], strict=False)):
            raise PlanRejectedError("replacement chunk must contain consecutive future ticks")
        with self._lock:
            if revision <= self._revision:
                raise PlanRejectedError("stale plan revision")
            future = [command for command in commands if command.tick > self._last_tick]
            if not future:
                raise PlanRejectedError("replacement expired while being computed")
            first_tick = future[0].tick
            if any(command.event_id in self._issued_events for command in future if command.event_id):
                raise PlanRejectedError("replacement repeats an issued semantic event")
            prefix = [
                (old_revision, command)
                for old_revision, command in self._actions
                if command.tick < first_tick
            ]
            displaced_events = {
                command.event_id
                for _, command in self._actions
                if command.tick >= first_tick and command.event_id
            }
            replacement_events = {command.event_id for command in future if command.event_id}
            if not displaced_events <= replacement_events | cancelled_events:
                raise PlanRejectedError("replacement silently drops a required semantic event")
            if prefix and prefix[-1][1].tick + 1 != first_tick:
                raise PlanRejectedError("replacement leaves a gap at the splice")
            if not prefix and first_tick > self._last_tick + 1:
                raise PlanRejectedError("replacement leaves an empty command interval")
            boundary = prefix[-1][1].action if prefix else self._last_issued
            if boundary is not None and self._max_splice_delta is not None:
                for joint, limit in self._max_splice_delta.items():
                    if (
                        joint in boundary
                        and joint in future[0].action
                        and abs(future[0].action[joint] - boundary[joint]) > limit
                    ):
                        raise PlanRejectedError(f"replacement jumps at {joint}; replan from issued boundary")
            self._actions = deque(prefix + [(revision, command) for command in future])
            self._revision = revision

    def note_issued(self, action: dict[str, float]) -> None:
        """Track continuity in the action space the robot API receives."""
        with self._lock:
            self._last_issued = dict(action)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def snapshot(self) -> ActionQueueSnapshot:
        """Let an asynchronous producer seed IK from the committed prefix."""
        with self._lock:
            return ActionQueueSnapshot(
                self._revision,
                self._last_tick,
                None if self._last_issued is None else dict(self._last_issued),
                tuple(replace(action, action=dict(action.action)) for _, action in self._actions),
            )

    def take(self, tick: int) -> tuple[TimedAction | None, int]:
        with self._lock:
            if tick <= self._last_tick:
                raise ValueError("motor ticks must increase")
            while self._actions and self._actions[0][1].tick < tick:
                _, expired = self._actions.popleft()
                if expired.event_id:
                    raise MissedEventError(f"missed required event {expired.event_id} at tick {expired.tick}")
            self._last_tick = tick
            if not self._actions or self._actions[0][1].tick > tick:
                return None, self._revision
            revision, action = self._actions.popleft()
            if action.event_id:
                if action.event_id in self._issued_events:
                    raise MissedEventError(f"duplicate semantic event {action.event_id}")
                self._issued_events.add(action.event_id)
            return action, revision

    def skip(self, tick: int) -> None:
        """Account for a scheduled idle tick without issuing an action."""
        with self._lock:
            if tick <= self._last_tick:
                raise ValueError("motor ticks must increase")
            if self._actions and self._actions[0][1].tick <= tick:
                raise PlanRejectedError("cannot idle across a queued action or event")
            self._last_tick = tick

    def has_tick(self, tick: int) -> bool:
        with self._lock:
            return bool(self._actions and self._actions[0][1].tick == tick)

    def has_due(self, tick: int) -> bool:
        with self._lock:
            return bool(self._actions and self._actions[0][1].tick <= tick)

    def cancel_unissued(self, reason: str) -> tuple[str, ...]:
        """Explicitly invalidate the suffix; return canceled semantic events."""
        if not reason:
            raise ValueError("plan cancellation needs a reason")
        with self._lock:
            events = tuple(command.event_id for _, command in self._actions if command.event_id)
            self._actions.clear()
            self._revision += 1
            return events


@dataclass
class MotorMetrics:
    ticks: int = 0
    commanded: int = 0
    starved: int = 0
    idle: int = 0
    late_ticks: int = 0
    max_dispatch_gap_s: float = 0.0
    read_s: list[float] = field(default_factory=list)
    write_s: list[float] = field(default_factory=list)


class MotorLoop:
    """Absolute-deadline 30 Hz owner of robot state reads and actions.

    The default sleep and clock are injectable for deterministic scheduling
    tests. A starved tick is reported and continues scheduling; it is never
    interpreted as a successfully completed action.
    """

    def __init__(
        self,
        robot: RobotIO,
        actions: ActionQueue,
        on_receipt: Callable[[ActionReceipt], None],
        allow_action: Callable[[TimedAction], bool] | None = None,
        on_tick: Callable[[MotorTick], None] | None = None,
        *,
        fps: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.robot = robot
        self.actions = actions
        self.on_receipt = on_receipt
        self.on_tick = on_tick
        self.allow_action = allow_action
        self.period_s = 1.0 / fps
        self.clock = clock
        self.sleep = sleep
        self.metrics = MotorMetrics()
        self.epoch_s: float | None = None
        self._last_dispatch_s: float | None = None
        self._last_issued: dict[str, float] | None = None

    def _record_tick(
        self,
        tick: int,
        deadline: float,
        observed: dict,
        read_duration_s: float,
        receipt: ActionReceipt | None,
        idle_reason: str | None = None,
    ) -> None:
        if self.on_tick is None:
            return
        # Save the replayable robot-API action, not a downstream actuator goal
        # that a gravity-compensated robot would compensate a second time.
        effective = self._last_issued or {k: float(v) for k, v in observed.items() if k.endswith(".pos")}
        source = (
            "commanded" if receipt else ("held_goal" if self._last_issued is not None else "observed_hold")
        )
        self.on_tick(
            MotorTick(
                tick, deadline, observed, read_duration_s, dict(effective), receipt, idle_reason, source
            )
        )

    def start(self) -> None:
        self.epoch_s = self.clock()

    def tick(self, *, idle_reason: str | None = None) -> ActionReceipt | None:
        if self.epoch_s is None:
            self.start()
        tick = self.metrics.ticks
        deadline = self.epoch_s + tick * self.period_s
        delay = deadline - self.clock()
        if delay > 0:
            self.sleep(delay)
        else:
            if -delay > self.period_s:
                self.metrics.late_ticks += 1
        read_start = self.clock()
        observed = self.robot.get_state_observation()
        read_end = self.clock()
        if idle_reason is not None:
            self.actions.skip(tick)
            self.metrics.ticks += 1
            self.metrics.idle += 1
            self.metrics.read_s.append(read_end - read_start)
            self._record_tick(tick, deadline, observed, read_end - read_start, None, idle_reason)
            return None
        action, revision = self.actions.take(tick)
        self.metrics.ticks += 1
        self.metrics.read_s.append(read_end - read_start)
        if action is None:
            self.metrics.starved += 1
            self._record_tick(tick, deadline, observed, read_end - read_start, None, "action queue starved")
            return None
        if self.allow_action is not None and not self.allow_action(action):
            self.metrics.idle += 1
            self._record_tick(
                tick, deadline, observed, read_end - read_start, None, "milestone gate rejected"
            )
            raise PlanRejectedError(
                f"milestone gate for {action.event_id or action.interval} is not satisfied"
            )
        accepted = self.robot.send_action(action.action)
        self._last_issued = dict(action.action)
        self.actions.note_issued(action.action)
        dispatched = self.clock()
        if self._last_dispatch_s is not None:
            self.metrics.max_dispatch_gap_s = max(
                self.metrics.max_dispatch_gap_s, dispatched - self._last_dispatch_s
            )
        self._last_dispatch_s = dispatched
        self.metrics.commanded += 1
        self.metrics.write_s.append(dispatched - read_end)
        receipt = ActionReceipt(
            tick,
            deadline,
            dispatched,
            read_end - read_start,
            dispatched - read_end,
            observed,
            action,
            accepted,
            revision,
        )
        self.on_receipt(receipt)
        self._record_tick(tick, deadline, observed, read_end - read_start, receipt)
        return receipt
