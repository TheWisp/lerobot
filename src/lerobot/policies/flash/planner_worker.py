"""Asynchronous IK chunk production with an explicit motor-queue handoff."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import Enum
from multiprocessing import get_context
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any

import numpy as np

from .execution import PlanEvidence, PlanRejectedError, TimedAction
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
    pose_revision: int | None = None

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


def _compute_plan(planner: JointPlanner, request: PlanRequest) -> tuple[PlanResult, list[TimedAction]]:
    """One planner request, shared by the local and isolated workers."""
    try:
        attachment_pose = request.attachment.gripper_from_object if request.attachment is not None else None
        turn_angle = request.turn_angle_rad
        if turn_angle is None:
            turn_angle = select_turn_angle(
                planner,
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
                planner.kinematics.forward_kinematics(request.seed_deg),
                request.reference.pose,
                attachment_pose,
                request.epoch_s + request.start_tick * request.period_s,
                turn_angle,
            )
        actions = planner.plan(
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
        if request.pose_revision is not None:
            evidence = PlanEvidence(
                pose_revision=request.pose_revision,
                reference_role=request.reference.role,
                reference_version=request.reference.reference_version,
                reference_source_frame_id=request.reference.source_frame_id,
                reference_source_time_s=request.reference.source_time_s,
                reference_available_time_s=request.reference.available_time_s,
                attachment_reference_version=(
                    request.attachment.reference_version if request.attachment is not None else None
                ),
                attachment_source_time_s=(
                    request.attachment.source_time_s if request.attachment is not None else None
                ),
                attachment_available_time_s=(
                    request.attachment.available_time_s if request.attachment is not None else None
                ),
            )
            actions = [replace(action, plan_evidence=evidence) for action in actions]
        return (
            PlanResult(
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
            ),
            actions,
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
        result = PlanResult(request.revision, 0, str(exc), PlanFailureKind.INFEASIBLE, request.connection)
    except PlanRejectedError as exc:
        result = PlanResult(request.revision, 0, str(exc), PlanFailureKind.REJECTED, request.connection)
    except ValueError as exc:
        result = PlanResult(request.revision, 0, str(exc), PlanFailureKind.INVALID, request.connection)
    return result, []


def _planner_process_main(factory: Callable[[], JointPlanner], requests: Any, results: Any) -> None:
    try:
        planner = factory()
        results.put(("ready",))
        # This is a multiprocessing queue, not an HTTP request. The parent
        # sends a sentinel on close and terminates the child on timeout.
        while (request := requests.get()) is not None:  # nosec B113
            started_s = time.perf_counter()
            result, actions = _compute_plan(planner, request)
            results.put(("result", request.revision, result, actions, time.perf_counter() - started_s))
        results.put(("stopped",))
    except Exception as exc:
        results.put(("fatal", f"{type(exc).__name__}: {exc}"))


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
        *,
        process_factory: Callable[[], JointPlanner] | None = None,
    ) -> None:
        self.planner = planner
        self.publish = publish
        self.process_factory = process_factory
        self._requests: Queue[PlanRequest | None] = Queue(maxsize=1)
        self._thread = Thread(target=self._run, name="flash-joint-planner", daemon=False)
        self._process: Any = None
        self._process_requests: Any = None
        self._process_results: Any = None
        self._sender_thread: Thread | None = None
        self._closing = False
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
        if self.process_factory is not None:
            context = get_context("spawn")
            self._process_requests = context.Queue(maxsize=1)
            self._process_results = context.Queue()
            self._process = context.Process(
                target=_planner_process_main,
                args=(self.process_factory, self._process_requests, self._process_results),
                name="flash-joint-planner-process",
            )
            self._process.start()
            try:
                ready = self._process_results.get(timeout=30)
            except Empty as exc:
                self._process.terminate()
                self._process.join()
                raise RuntimeError("isolated planner did not initialize") from exc
            if ready != ("ready",):
                self._process.join(timeout=1)
                raise RuntimeError(f"isolated planner initialization failed: {ready}")
            self._thread = Thread(
                target=self._receive_process, name="flash-joint-planner-results", daemon=False
            )
            self._sender_thread = Thread(
                target=self._send_process, name="flash-joint-planner-requests", daemon=False
            )
            self._sender_thread.start()
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

    def _send_process(self) -> None:
        assert self._process_requests is not None
        while (request := self._requests.get()) is not None:
            while not self._closing:
                with self._lock:
                    superseded = request.revision < self._newest_requested
                    if superseded:
                        self.superseded += 1
                if superseded:
                    break
                try:
                    self._process_requests.put(request, timeout=0.05)
                    break
                except Full:
                    continue
        while self._process is not None and self._process.is_alive():
            try:
                self._process_requests.put(None, timeout=0.05)
                return
            except Full:
                continue

    def _commit_result(
        self,
        revision: int,
        result: PlanResult,
        actions: list[TimedAction],
        compute_s: float,
    ) -> None:
        with self._lock:
            if revision < self._newest_requested:
                result = PlanResult(
                    revision,
                    0,
                    "superseded by a newer pose or plan",
                    PlanFailureKind.SUPERSEDED,
                )
            elif result.error is None:
                try:
                    self.publish(revision, actions)
                except PlanRejectedError as exc:
                    result = PlanResult(revision, 0, str(exc), PlanFailureKind.REJECTED, result.connection)
            self.completed += 1
            self.compute_total_s += compute_s
            self.compute_max_s = max(self.compute_max_s, compute_s)
            self._last_result = replace(result, compute_s=compute_s)

    def _run(self) -> None:
        while (request := self._requests.get()) is not None:
            started_s = time.perf_counter()
            try:
                result, actions = _compute_plan(self.planner, request)
                self._commit_result(request.revision, result, actions, time.perf_counter() - started_s)
            except Exception as exc:
                with self._lock:
                    self._fatal = exc
                return

    def _receive_process(self) -> None:
        assert self._process_results is not None
        assert self._process is not None
        while True:
            try:
                message = self._process_results.get(timeout=0.2)
            except Empty:
                if not self._process.is_alive():
                    if not self._closing:
                        with self._lock:
                            self._fatal = RuntimeError("isolated planner exited without a result")
                    return
                continue
            if message[0] == "stopped":
                return
            if message[0] == "fatal":
                with self._lock:
                    self._fatal = RuntimeError(message[1])
                return
            _, revision, result, actions, compute_s = message
            try:
                self._commit_result(revision, result, actions, compute_s)
            except Exception as exc:
                with self._lock:
                    self._fatal = exc
                return

    def close(self, timeout_s: float = 5.0) -> None:
        if not self._started:
            return
        self._closing = True
        try:
            self._requests.put_nowait(None)
        except Full:
            with suppress(Empty):
                self._requests.get_nowait()
            with suppress(Full):
                self._requests.put(None, timeout=min(timeout_s, 0.5))
        if self._process is not None:
            assert self._sender_thread is not None
            self._sender_thread.join(timeout_s)
            if self._sender_thread.is_alive():
                self._process.terminate()
                self._process.join(timeout_s)
                raise TimeoutError("isolated planner request sender did not stop")
            self._process.join(timeout_s)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout_s)
                raise TimeoutError("isolated planner did not stop")
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("planner worker did not stop")
        if self._process_requests is not None:
            self._process_requests.close()
            self._process_results.close()
        self._started = False
