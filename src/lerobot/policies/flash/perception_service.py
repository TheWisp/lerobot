"""Live camera and pose handoffs that never execute inside a motor tick."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import Protocol

from .perception import LatestPoseStore, LatestRGBDFrameStore, RGBDFrame
from .perception_worker import PoseProcess
from .trajectory import Evidence, PoseEstimate


class RGBDSource(Protocol):
    def read(self) -> RGBDFrame: ...


class PerceptionServiceError(RuntimeError):
    pass


class PerceptionService:
    """Capture, estimate and publish without giving the motor loop a camera handle.

    ``start`` only means process initialization completed. ``wait_ready``
    separately waits for actual observed roles after model/track seeding.
    The host must stop motion on ``error``; a failed camera or estimator never
    silently turns the last observation into a fresh one.
    """

    def __init__(
        self,
        source: RGBDSource,
        worker: PoseProcess,
        store: LatestPoseStore,
        *,
        clock=time.perf_counter,
        collector_period_s: float = 0.01,
        on_pose_update: Callable[[int, dict[str, PoseEstimate]], None] | None = None,
    ) -> None:
        if collector_period_s <= 0:
            raise ValueError("collector period must be positive")
        if worker.store is not store:
            raise ValueError("service and pose worker must share one pose store")
        self.source = source
        self.worker = worker
        self.store = store
        self.frames = LatestRGBDFrameStore()
        self.clock = clock
        self.collector_period_s = collector_period_s
        self.on_pose_update = on_pose_update
        self.captured = 0
        self._stop = Event()
        self._lock = Lock()
        self._error: Exception | None = None
        self._capture_thread = Thread(target=self._capture, name="flash-rgbd-capture", daemon=True)
        self._collect_thread = Thread(target=self._collect, name="flash-pose-collector", daemon=True)
        self._started = False

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def _fail(self, error: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
        self._stop.set()

    def start(self, timeout_s: float = 60.0) -> None:
        if self._started:
            raise PerceptionServiceError("perception service already started")
        self.worker.start(timeout_s=timeout_s)
        self._started = True
        self._capture_thread.start()
        self._collect_thread.start()

    def _capture(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.source.read()
                self.frames.publish(frame)
                self.captured += 1
                if not self.worker.submit(frame) and not self.worker.alive:
                    raise PerceptionServiceError("pose process stopped during camera capture")
        except Exception as exc:
            self._fail(exc)

    def _collect(self) -> None:
        try:
            while not self._stop.is_set():
                self.worker.poll()
                if self.on_pose_update is not None:
                    self.on_pose_update(*self.store.snapshot())
                self._stop.wait(self.collector_period_s)
        except Exception as exc:
            self._fail(exc)

    def wait_ready(
        self,
        roles: tuple[str, ...],
        *,
        max_pose_age_s: float,
        timeout_s: float,
        allow_predicted: bool = False,
    ) -> None:
        """Block the host before motion until each required role has usable evidence."""
        if not self._started or not roles or max_pose_age_s <= 0 or timeout_s <= 0:
            raise ValueError("wait_ready needs a started service, roles and positive bounds")
        deadline = time.monotonic() + timeout_s
        missing = list(roles)
        while time.monotonic() < deadline:
            if self.error is not None:
                raise PerceptionServiceError("live perception failed") from self.error
            _, poses = self.store.snapshot()
            now_s = self.clock()
            missing = [
                role
                for role in roles
                if role not in poses
                or (not allow_predicted and poses[role].evidence != Evidence.OBSERVED)
                or poses[role].evidence == Evidence.LOST
                or not 0 <= now_s - poses[role].source_time_s <= max_pose_age_s
            ]
            if not missing:
                return
            self._stop.wait(min(0.01, max(0, deadline - time.monotonic())))
        kind = "usable" if allow_predicted else "observed"
        failures = self.worker.metrics.last_failures
        detail = "; ".join(f"{role}: {failures[role]}" for role in missing if role in failures)
        reason = f"no fresh {kind} pose for: {', '.join(missing)}"
        raise PerceptionServiceError(f"{reason} (last estimator failure: {detail})" if detail else reason)

    def close(self, timeout_s: float = 5.0) -> None:
        if not self._started:
            return
        self._stop.set()
        self._capture_thread.join(timeout_s)
        self._collect_thread.join(timeout_s)
        self.worker.close(timeout_s=timeout_s)
        if self._capture_thread.is_alive() or self._collect_thread.is_alive():
            raise PerceptionServiceError("perception threads did not stop")
        self._started = False
