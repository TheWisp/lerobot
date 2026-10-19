"""Asynchronous IK chunk production with an explicit motor-queue handoff."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread

import numpy as np

from .execution import PlanRejectedError, TimedAction
from .planner import InfeasibleTrajectoryError, JointPlanner
from .skill import Interval
from .trajectory import AttachmentState, ConnectionTrajectory, MissingEvidenceError, PoseEstimate


@dataclass(frozen=True)
class PlanRequest:
    revision: int
    interval: Interval
    start_tick: int
    count: int
    epoch_s: float
    period_s: float
    interval_start_s: float
    seed_deg: np.ndarray
    reference: PoseEstimate
    attachment: AttachmentState | None
    connection: ConnectionTrajectory | None = None
    issued_events: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        seed = np.asarray(self.seed_deg, dtype=float).copy()
        seed.setflags(write=False)
        object.__setattr__(self, "seed_deg", seed)
        if self.connection is not None:
            # ConnectionTrajectory.retarget mutates its fields. The worker
            # must solve against the snapshot that produced this request.
            connection = self.connection
            object.__setattr__(
                self,
                "connection",
                ConnectionTrajectory(
                    connection.interval,
                    connection.start_pose.copy(),
                    connection.goal_pose.copy(),
                    connection.start_time_s,
                    connection.duration_s,
                ),
            )


@dataclass(frozen=True)
class PlanResult:
    revision: int
    issued_actions: int
    error: str | None


class PlannerWorker:
    """Replan off the motor owner and atomically replace only unissued ticks.

    A newer pending request supersedes an older one; a request already being
    solved may finish, but the queue's tick and revision checks reject an
    obsolete result. Infeasible geometry, missing evidence and a stale splice
    are reported to the host, not turned into a successful action.
    """

    def __init__(
        self,
        planner: JointPlanner,
        publish: Callable[[int, list[TimedAction]], None],
    ) -> None:
        self.planner = planner
        self.publish = publish
        self._requests: Queue[PlanRequest | None] = Queue(maxsize=1)
        self._thread = Thread(target=self._run, name="flash-joint-planner", daemon=False)
        self._lock = Lock()
        self._last_result: PlanResult | None = None
        self._fatal: Exception | None = None
        self.superseded = 0
        self._started = False

    @property
    def last_result(self) -> PlanResult | None:
        with self._lock:
            return self._last_result

    @property
    def fatal_error(self) -> Exception | None:
        with self._lock:
            return self._fatal

    def start(self) -> None:
        if self._started:
            raise RuntimeError("planner worker already started")
        self._thread.start()
        self._started = True

    def submit(self, request: PlanRequest) -> None:
        if not self._started or self.fatal_error is not None:
            raise RuntimeError("planner worker is not running")
        try:
            self._requests.put_nowait(request)
        except Full:
            with suppress(Empty):
                self._requests.get_nowait()
                self.superseded += 1
            try:
                self._requests.put_nowait(request)
            except Full as exc:
                raise RuntimeError("planner request handoff is busy") from exc

    def _run(self) -> None:
        while (request := self._requests.get()) is not None:
            try:
                actions = self.planner.plan(
                    request.interval,
                    start_tick=request.start_tick,
                    count=request.count,
                    epoch_s=request.epoch_s,
                    period_s=request.period_s,
                    interval_start_s=request.interval_start_s,
                    seed_deg=request.seed_deg,
                    reference=request.reference,
                    attachment=request.attachment,
                    connection=request.connection,
                    issued_events=request.issued_events,
                )
                self.publish(request.revision, actions)
            except (MissingEvidenceError, InfeasibleTrajectoryError, PlanRejectedError, ValueError) as exc:
                result = PlanResult(request.revision, 0, str(exc))
            except Exception as exc:
                with self._lock:
                    self._fatal = exc
                return
            else:
                result = PlanResult(request.revision, len(actions), None)
            with self._lock:
                self._last_result = result

    def close(self, timeout_s: float = 5.0) -> None:
        if not self._started:
            return
        try:
            self._requests.put_nowait(None)
        except Full:
            with suppress(Empty):
                self._requests.get_nowait()
            with suppress(Full):
                self._requests.put(None, timeout=min(timeout_s, 0.5))
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("planner worker did not stop")
        self._started = False
