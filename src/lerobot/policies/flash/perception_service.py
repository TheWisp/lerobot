"""Live camera and pose handoffs that never execute inside a motor tick."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import Protocol

from .perception import LatestPoseStore, LatestRGBDFrameStore, RGBDFrame
from .perception_worker import PoseProcess
from .trajectory import Evidence, PoseEstimate

logger = logging.getLogger(__name__)


class RGBDSource(Protocol):
    def read(self) -> RGBDFrame: ...


class PerceptionServiceError(RuntimeError):
    pass


class PerceptionService:
    """Capture, estimate and publish without giving the motor loop a camera handle.

    ``start`` only means process initialization completed. ``wait_ready``
    separately waits for current direct position evidence after model/track seeding;
    an unobserved orientation may still be predicted.
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
        on_frame: Callable[[RGBDFrame], None] | None = None,
        causal_frames: Callable[[RGBDFrame], None] | None = None,
        causal_error: Callable[[], Exception | None] | None = None,
        max_camera_silence_s: float | None = None,
    ) -> None:
        if collector_period_s <= 0:
            raise ValueError("collector period must be positive")
        if causal_frames is None and worker.store is not store:
            raise ValueError("legacy service and pose worker must share one pose store")
        if causal_frames is not None and worker.store is not None:
            raise ValueError("causal service pose worker must not publish directly to a store")
        if max_camera_silence_s is not None and max_camera_silence_s <= 0:
            raise ValueError("camera silence bound must be positive")
        self.source = source
        self.worker = worker
        self.store = store
        self.frames = LatestRGBDFrameStore()
        self.clock = clock
        self.collector_period_s = collector_period_s
        self.on_pose_update = on_pose_update
        self.on_frame = on_frame
        self.causal_frames = causal_frames
        self.causal_error = causal_error
        self.max_camera_silence_s = max_camera_silence_s
        self.captured = 0
        self.camera_timeouts = 0
        self.max_capture_handoff_s = 0.0
        self.max_capture_gap_s = 0.0
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
            last_frame_s = self.clock()
            has_frame = False
            while not self._stop.is_set():
                try:
                    frame = self.source.read()
                except TimeoutError:
                    # One missed camera read does not mean the camera is
                    # lost. The runtime's camera-age bound defines when the
                    # interruption becomes unsafe for planning and recording.
                    self.camera_timeouts += 1
                    silence_s = self.clock() - last_frame_s
                    logger.warning("FLASH RGB-D read timed out after %.3fs without a frame", silence_s)
                    if self.max_camera_silence_s is None or silence_s > self.max_camera_silence_s:
                        raise
                    self._stop.wait(self.collector_period_s)
                    continue
                if has_frame:
                    self.max_capture_gap_s = max(self.max_capture_gap_s, frame.source_time_s - last_frame_s)
                last_frame_s = frame.source_time_s
                has_frame = True
                handoff_start_s = self.clock()
                self.frames.publish(frame)
                self.captured += 1
                frozen = self.frames.snapshot()
                assert frozen is not None
                if self.on_frame is not None:
                    self.on_frame(frozen)
                if self.causal_frames is not None:
                    self.causal_frames(frozen)
                if not self.worker.submit(frame) and not self.worker.alive:
                    raise PerceptionServiceError("pose process stopped during camera capture")
                self.max_capture_handoff_s = max(self.max_capture_handoff_s, self.clock() - handoff_start_s)
        except Exception as exc:
            self._fail(exc)

    def _collect(self) -> None:
        try:
            while not self._stop.is_set():
                self.worker.poll()
                if self.causal_error is not None and (error := self.causal_error()) is not None:
                    raise PerceptionServiceError(
                        f"causal point tracking failed: {type(error).__name__}: {error}"
                    ) from error
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

        def fresh_direct(pose: PoseEstimate, now_s: float) -> bool:
            direct_s = pose.source_time_s if pose.has_direct_translation else pose.last_direct_observation_s
            return direct_s is not None and 0 <= now_s - direct_s <= max_pose_age_s

        while time.monotonic() < deadline:
            if self.error is not None:
                raise PerceptionServiceError("live perception failed") from self.error
            _, poses = self.store.snapshot()
            now_s = self.clock()
            missing = [
                role
                for role in roles
                if role not in poses
                or (not allow_predicted and not fresh_direct(poses[role], now_s))
                or poses[role].evidence == Evidence.LOST
                or not 0 <= now_s - poses[role].source_time_s <= max_pose_age_s
                or now_s < poses[role].available_time_s
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
