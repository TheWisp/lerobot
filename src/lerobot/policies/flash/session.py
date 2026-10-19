"""Single-owner FLASH attempt state over an asynchronous plan handoff."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Full, Queue

from .execution import ActionQueue, ActionReceipt, MotorLoop, PlanRejectedError, RobotIO, TimedAction
from .progress import MilestoneUpdate, Phase, Progress, Verdict
from .skill import FlashSkill


@dataclass(frozen=True)
class PoseChange:
    roles: frozenset[str]
    source_id: str
    source_time_s: float


class FlashSession:
    """Motor-loop owner; perception and planning publish without holding it.

    The producer owns IK and calls :meth:`publish_plan` with future commands.
    Pose and milestone workers post versioned evidence; only ``step`` mutates
    progress. A physical host still needs a registered pose backend and a
    robot-specific kinematics adapter before creating this session.
    """

    def __init__(
        self,
        skill: FlashSkill,
        robot: RobotIO,
        *,
        on_receipt: Callable[[ActionReceipt], None] | None = None,
        fps: float = 30.0,
        max_starved_ticks: int = 2,
        max_splice_delta: dict[str, float] | None = None,
        clock=None,
        sleep=None,
    ) -> None:
        self.progress = Progress(skill)
        if max_starved_ticks < 1:
            raise ValueError("max_starved_ticks must be positive")
        self.actions = ActionQueue(max_splice_delta=max_splice_delta)
        self._evidence: Queue[MilestoneUpdate | PoseChange] = Queue(maxsize=128)
        self._receipt_sink = on_receipt
        self._starved_in_a_row = 0
        self.max_starved_ticks = max_starved_ticks
        kwargs = {}
        if clock is not None:
            kwargs["clock"] = clock
        if sleep is not None:
            kwargs["sleep"] = sleep
        self.motor = MotorLoop(robot, self.actions, self._on_receipt, self._allow_action, fps=fps, **kwargs)
        self.motor.start()

    def post_evidence(self, item: MilestoneUpdate | PoseChange) -> None:
        try:
            self._evidence.put_nowait(item)
        except Full as exc:
            raise RuntimeError("FLASH evidence queue overflow") from exc

    def publish_plan(self, revision: int, commands: list[TimedAction]) -> None:
        self.actions.publish(revision, commands)

    def _allow_action(self, action: TimedAction) -> bool:
        return action.interval == self.progress.interval.name and self.progress.allows_milestone(
            action.requires_milestone
        )

    def _on_receipt(self, receipt: ActionReceipt) -> None:
        self.progress.receipt(receipt.dispatched_s, receipt.requested.interval_complete)
        if self._receipt_sink is not None:
            self._receipt_sink(receipt)

    def _take_evidence(self) -> None:
        while True:
            try:
                item = self._evidence.get_nowait()
            except Empty:
                return
            if isinstance(item, MilestoneUpdate):
                self.progress.update_milestone(item)
            else:
                self.progress.invalidate_changed_roles(set(item.roles), item.source_id, item.source_time_s)

    def step(self) -> ActionReceipt | None:
        self._take_evidence()
        now_s = self.motor.clock()
        if self.progress.phase == Phase.VERIFYING:
            self.progress.advance(now_s)
        tick = self.motor.metrics.ticks
        if self.progress.phase == Phase.WAITING:
            if self.progress.can_begin() and self.actions.has_tick(tick):
                self.progress.begin(self.motor.epoch_s + tick * self.motor.period_s)
            else:
                if self.actions.has_due(tick):
                    reason = "entry milestone not satisfied or first command expired"
                    events = self.actions.cancel_unissued(reason)
                    self.progress.events.append(
                        {"type": "plan_cancelled", "reason": reason, "events": events}
                    )
                return self.motor.tick(idle_reason="awaiting interval entry")
        if self.progress.phase != Phase.MOVING:
            if self.actions.has_due(tick):
                reason = f"no motion permitted in {self.progress.phase.value}"
                events = self.actions.cancel_unissued(reason)
                self.progress.events.append({"type": "plan_cancelled", "reason": reason, "events": events})
            return self.motor.tick(idle_reason=self.progress.phase.value)
        try:
            receipt = self.motor.tick()
        except PlanRejectedError as exc:
            self.progress.interrupt(self.motor.clock(), str(exc))
            return None
        if receipt is None:
            self._starved_in_a_row += 1
            if self._starved_in_a_row >= self.max_starved_ticks:
                self.progress.fail(self.motor.clock(), "motor action queue starved")
            return None
        self._starved_in_a_row = 0
        if self.progress.phase == Phase.VERIFYING:
            self.progress.advance(receipt.dispatched_s)
        return receipt

    def finish(self, verdict: Verdict, reason: str) -> None:
        self.progress.finish(self.motor.clock(), verdict, reason)
