"""Explicit interval and milestone progression for a FLASH attempt."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .skill import FlashSkill, Interval, MotionMode


class Verdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class Phase(str, Enum):
    WAITING = "waiting"
    MOVING = "moving"
    VERIFYING = "verifying"
    MOTION_COMPLETE = "motion_complete"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class MilestoneUpdate:
    name: str
    verdict: Verdict
    source_id: str
    source_time_s: float
    reason: str = ""


@dataclass
class Progress:
    skill: FlashSkill
    interval_index: int = 0
    phase: Phase = Phase.WAITING
    interval_start_s: float | None = None
    suspended: bool = False
    entry_alignment: Interval | None = None
    aligned_entries: set[int] = field(default_factory=set)
    last_receipt_s: float | None = None
    attempt: int = 0
    milestones: dict[str, MilestoneUpdate] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    _seen_sources: set[tuple[str, str]] = field(default_factory=set)

    @property
    def interval(self):
        return self.entry_alignment or self.skill.intervals[self.interval_index]

    def align_demonstrated_entry(self, interval: Interval, now_s: float) -> None:
        """Insert a generated approach before starting a demonstrated local clock."""
        if (
            self.phase != Phase.WAITING
            or self.entry_alignment is not None
            or self.interval_index in self.aligned_entries
            or self.interval.mode != MotionMode.DEMONSTRATED
            or interval.mode != MotionMode.CONNECTION
        ):
            raise ValueError("demonstrated entry cannot be aligned in the current state")
        self.entry_alignment = interval
        self.aligned_entries.add(self.interval_index)
        self.events.append(
            {
                "type": "entry_alignment_planned",
                "interval": self.skill.intervals[self.interval_index].name,
                "alignment": interval.name,
                "at": now_s,
            }
        )

    def update_milestone(self, update: MilestoneUpdate) -> bool:
        """Update from a new observation; repeated frames cannot add evidence."""
        identity = (update.name, update.source_id)
        if identity in self._seen_sources:
            return False
        self._seen_sources.add(identity)
        prior = self.milestones.get(update.name)
        if prior is not None and update.source_time_s < prior.source_time_s:
            return False
        self.milestones[update.name] = update
        self.events.append(
            {
                "type": "milestone",
                "name": update.name,
                "verdict": update.verdict.value,
                "source_id": update.source_id,
            }
        )
        return True

    def invalidate_changed_roles(
        self, changed_roles: set[str], source_id: str, source_time_s: float
    ) -> tuple[str, ...]:
        """Withdraw old gate verdicts when the objects they refer to change."""
        invalidated = []
        for spec in self.skill.milestones:
            if not changed_roles.intersection(spec.depends_on_roles) or spec.name not in self.milestones:
                continue
            update = MilestoneUpdate(
                spec.name, Verdict.UNKNOWN, source_id, source_time_s, "dependent pose changed"
            )
            if self.update_milestone(update):
                invalidated.append(spec.name)
        return tuple(invalidated)

    def _satisfied(self, name: str | None) -> bool:
        return (
            name is None
            or self.milestones.get(name, MilestoneUpdate(name, Verdict.UNKNOWN, "", 0)).verdict
            == Verdict.TRUE
        )

    def begin(self, now_s: float) -> bool:
        if not self.can_begin():
            return False
        self.interval_start_s = now_s
        self.suspended = False
        self.phase = Phase.MOVING
        self.events.append(
            {"type": "interval_start", "interval": self.interval.name, "at": now_s, "attempt": self.attempt}
        )
        return True

    def suspend_connection(self, now_s: float, reason: str) -> None:
        """Resume a free-space interval from measured joints after evidence returns."""
        if self.phase != Phase.MOVING or self.interval.mode != MotionMode.CONNECTION:
            raise ValueError("only an active generated connection can wait for new evidence")
        self.phase = Phase.WAITING
        self.interval_start_s = None
        self.suspended = True
        self.events.append(
            {"type": "interval_suspended", "interval": self.interval.name, "at": now_s, "reason": reason}
        )

    def can_begin(self) -> bool:
        return self.phase == Phase.WAITING and self._satisfied(self.interval.entry_milestone)

    def allows_milestone(self, name: str | None) -> bool:
        return self._satisfied(name)

    def receipt(self, at_s: float, interval_complete: bool) -> None:
        """Progress follows accepted motor commands, never planned samples."""
        if self.phase != Phase.MOVING or (self.last_receipt_s is not None and at_s < self.last_receipt_s):
            raise ValueError("receipt outside active motion or out of order")
        self.last_receipt_s = at_s
        if interval_complete:
            self.phase = Phase.VERIFYING
            self.events.append(
                {"type": "interval_motion_complete", "interval": self.interval.name, "at": at_s}
            )

    def advance(self, now_s: float) -> bool:
        if self.phase != Phase.VERIFYING or not self._satisfied(self.interval.completion_milestone):
            return False
        self.events.append({"type": "interval_complete", "interval": self.interval.name, "at": now_s})
        if self.entry_alignment is not None:
            self.entry_alignment = None
            self.interval_start_s = None
            self.suspended = False
            self.phase = Phase.WAITING
            return True
        if self.interval_index + 1 == len(self.skill.intervals):
            self.phase = Phase.MOTION_COMPLETE
            return True
        self.interval_index += 1
        self.interval_start_s = None
        self.suspended = False
        self.phase = Phase.WAITING
        return True

    def retry_from(self, interval_name: str, now_s: float, reason: str, max_attempts: int) -> None:
        """A caller chooses a permitted recovery; no implicit contact reversal."""
        if self.attempt >= max_attempts:
            self.fail(now_s, f"retry exhausted: {reason}")
            return
        names = [i.name for i in self.skill.intervals]
        if interval_name not in names:
            raise ValueError(f"unknown recovery interval: {interval_name}")
        if interval_name not in self.interval.recovery_intervals:
            raise ValueError(f"recovery to {interval_name} is not permitted from {self.interval.name}")
        self.interval_index = names.index(interval_name)
        self.entry_alignment = None
        self.aligned_entries.clear()
        self.attempt += 1
        self.interval_start_s = None
        self.suspended = False
        self.phase = Phase.WAITING
        # A new attempt must re-establish its prerequisites from new evidence.
        # Keeping a prior TRUE would let a retry skip the very check that failed.
        self.milestones.clear()
        self.events.append({"type": "recovery", "interval": interval_name, "at": now_s, "reason": reason})

    def finish(self, now_s: float, verdict: Verdict, reason: str) -> None:
        if self.phase != Phase.MOTION_COMPLETE:
            raise ValueError("task outcome cannot be declared before motion completion")
        if verdict == Verdict.UNKNOWN:
            self.events.append({"type": "outcome_unknown", "at": now_s, "reason": reason})
            return
        self.phase = Phase.SUCCEEDED if verdict == Verdict.TRUE else Phase.FAILED
        self.events.append({"type": "outcome", "at": now_s, "verdict": verdict.value, "reason": reason})

    def fail(self, now_s: float, reason: str) -> None:
        self.phase = Phase.FAILED
        self.events.append({"type": "failure", "at": now_s, "reason": reason})

    def interrupt(self, now_s: float, reason: str) -> None:
        self.phase = Phase.INTERRUPTED
        self.events.append({"type": "interruption", "at": now_s, "reason": reason})
