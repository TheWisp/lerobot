"""Asynchronous IK chunk production with an explicit motor-queue handoff."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import Enum
from queue import Empty, Full, Queue
from threading import Lock, Thread

import numpy as np

from .execution import PlanRejectedError, TimedAction
from .planner import DemonstratedEntryDiscontinuityError, InfeasibleTrajectoryError, JointPlanner
from .skill import Interval
from .trajectory import AttachmentState, ConnectionTrajectory, MissingEvidenceError, PoseEstimate
from .turn_choice import select_turn_angle


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
    turn_angle_rad: float | None = None

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
                    connection.joint_retimed,
                    connection.requested_goal_pose.copy(),
                    connection.achieved_fraction,
                    connection.turn_angle_rad,
                ),
            )


@dataclass(frozen=True)
class PlanResult:
    revision: int
    issued_actions: int
    error: str | None
    failure: PlanFailureKind | None = None
    connection: ConnectionTrajectory | None = None
    compute_s: float = 0.0
    turn_angle_rad: float | None = None
    turn_rule_id: str | None = None


class PlanFailureKind(str, Enum):
    SUPERSEDED = "superseded"
    MISSING_EVIDENCE = "missing_evidence"
    ENTRY_DISCONTINUITY = "entry_discontinuity"
    INFEASIBLE = "infeasible"
    REJECTED = "rejected"
    INVALID = "invalid"


class PlannerWorker:
    """Replan off the motor owner and atomically replace only unissued ticks.

    A newer pending request supersedes an older one. If a request already
    being solved becomes stale, its result is discarded before publication;
    the motor keeps the last accepted queue while the newer solve runs.
    Infeasible geometry, missing evidence and a stale splice are reported
    to the host, not turned into a successful action.
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
        self._newest_requested = -1
        self.superseded = 0
        self.completed = 0
        self.compute_total_s = 0.0
        self.compute_max_s = 0.0
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
        with self._lock:
            if request.revision <= self._newest_requested:
                raise ValueError("planner requests must have increasing revisions")
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
            self._newest_requested = request.revision

    def _run(self) -> None:
        while (request := self._requests.get()) is not None:
            started_s = time.perf_counter()
            try:
                attachment_pose = (
                    request.attachment.gripper_from_object if request.attachment is not None else None
                )
                turn_angle = request.turn_angle_rad
                if turn_angle is None:
                    turn_angle = select_turn_angle(
                        self.planner,
                        request.interval,
                        request.seed_deg,
                        request.reference.pose,
                        attachment_pose,
                        request.period_s,
                    )
                connection = request.connection
                if connection is not None and connection.turn_angle_rad != turn_angle:
                    connection = ConnectionTrajectory.begin(
                        request.interval,
                        self.planner.kinematics.forward_kinematics(request.seed_deg),
                        request.reference.pose,
                        attachment_pose,
                        request.epoch_s + request.start_tick * request.period_s,
                        turn_angle,
                    )
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
                    connection=connection,
                    issued_events=request.issued_events,
                    turn_angle_rad=turn_angle,
                )
                with self._lock:
                    if request.revision < self._newest_requested:
                        result = PlanResult(
                            request.revision,
                            0,
                            "superseded by a newer pose or plan",
                            PlanFailureKind.SUPERSEDED,
                        )
                    else:
                        self.publish(request.revision, actions)
                        result = PlanResult(
                            request.revision,
                            len(actions),
                            None,
                            connection=connection,
                            turn_angle_rad=turn_angle,
                            turn_rule_id=(
                                request.interval.turn_at_point.rule_id
                                if request.interval.turn_at_point is not None
                                else None
                            ),
                        )
            except MissingEvidenceError as exc:
                result = PlanResult(
                    request.revision, 0, str(exc), PlanFailureKind.MISSING_EVIDENCE, request.connection
                )
            except DemonstratedEntryDiscontinuityError as exc:
                result = PlanResult(
                    request.revision, 0, str(exc), PlanFailureKind.ENTRY_DISCONTINUITY, request.connection
                )
            except InfeasibleTrajectoryError as exc:
                result = PlanResult(
                    request.revision, 0, str(exc), PlanFailureKind.INFEASIBLE, request.connection
                )
            except PlanRejectedError as exc:
                result = PlanResult(
                    request.revision, 0, str(exc), PlanFailureKind.REJECTED, request.connection
                )
            except ValueError as exc:
                result = PlanResult(
                    request.revision, 0, str(exc), PlanFailureKind.INVALID, request.connection
                )
            except Exception as exc:
                with self._lock:
                    self._fatal = exc
                return
            with self._lock:
                compute_s = time.perf_counter() - started_s
                self.completed += 1
                self.compute_total_s += compute_s
                self.compute_max_s = max(self.compute_max_s, compute_s)
                self._last_result = replace(result, compute_s=compute_s)

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
