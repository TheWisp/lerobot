"""Process-isolated pose proposals with a latest-frame handoff.

The motor owner never calls this worker. A camera/host producer submits aligned
frames, and a separate host collector routes each result to exactly one sink:
the scene estimator or a legacy ``LatestPoseStore``.
Slow perception may drop superseded camera frames; it cannot delay a motor
read or command. The backend is constructed *inside* a spawned process so
CUDA state is never inherited through ``fork``.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from queue import Empty, Full
from typing import cast

import numpy as np

from .perception import InstanceMaskPoseBackend, LatestPoseStore, ObjectReference, PoseBackend, RGBDFrame
from .trajectory import PoseEstimate


@dataclass(frozen=True)
class _Ready:
    pass


@dataclass(frozen=True)
class _Result:
    estimates: tuple[PoseEstimate, ...]
    failures: dict[str, str]
    frame_id: int
    source_time_s: float
    calibration_version: str
    masks: dict[str, np.ndarray]


@dataclass(frozen=True)
class _Failure:
    detail: str


def _publish_latest(queue, message: _Result | _Failure) -> None:
    try:
        queue.put_nowait(message)
    except Full:
        with suppress(Empty):
            queue.get_nowait()
        # A Queue feeder may still own the full slot. A dropped pose result
        # is preferable to blocking the heavy worker behind a slow collector.
        with suppress(Full):
            queue.put_nowait(message)


def _run_backend(
    factory: Callable[[], PoseBackend],
    references: tuple[ObjectReference, ...],
    frames,
    results,
    require_masks: bool,
):
    try:
        backend = factory()
        if require_masks and not hasattr(backend, "last_masks"):
            raise TypeError("causal tracking requires a pose backend that exposes aligned instance masks")
        backend.initialize(references)
        known_roles = {reference.role for reference in references}
        results.put(_Ready())
        while (frame := frames.get()) is not None:
            estimates = backend.update(frame)
            masks = cast(InstanceMaskPoseBackend, backend).last_masks if require_masks else {}
            for role, mask in masks.items():
                if (
                    role not in known_roles
                    or np.asarray(mask).shape != frame.depth.shape
                    or np.asarray(mask).dtype != bool
                ):
                    raise ValueError(f"{role}: causal mask does not align with RGB-D")
            _publish_latest(
                results,
                _Result(
                    estimates,
                    dict(getattr(backend, "last_failure", {})),
                    frame.frame_id,
                    frame.source_time_s,
                    frame.calibration_version,
                    {role: np.asarray(mask, dtype=bool).copy() for role, mask in masks.items()},
                ),
            )
    except Exception:
        _publish_latest(results, _Failure(traceback.format_exc()))


@dataclass
class PoseProcessMetrics:
    submitted: int = 0
    superseded: int = 0
    rejected: int = 0
    delivered: int = 0
    last_source_time_s: float | None = None
    last_failures: dict[str, str] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)


class PoseProcessError(RuntimeError):
    pass


class PoseProcess:
    """Own one isolated backend and deliver newest results to one sink.

    The supplied factory must be spawn-picklable and load GPU models only when
    called by the child. ``submit`` is for a camera/host thread, and ``poll``
    is for a collector outside the motor loop. A tracker consuming every
    camera frame is incompatible with this latest-frame queue and must use a
    different input handoff.
    """

    def __init__(
        self,
        factory: Callable[[], PoseBackend],
        references: tuple[ObjectReference, ...],
        store: LatestPoseStore | None,
        *,
        on_result: Callable[[_Result], None] | None = None,
        require_masks: bool = False,
    ) -> None:
        if not references:
            raise ValueError("pose worker needs object references")
        if (store is None) == (on_result is None):
            raise ValueError("pose worker needs exactly one result sink")
        self.factory = factory
        self.references = references
        self.store = store
        self.on_result = on_result
        self.require_masks = require_masks
        self.metrics = PoseProcessMetrics()
        self._context = mp.get_context("spawn")
        self._frames = self._context.Queue(maxsize=1)
        self._results = self._context.Queue(maxsize=2)
        self._process = self._context.Process(
            target=_run_backend,
            args=(factory, references, self._frames, self._results, require_masks),
            name="flash-pose",
            daemon=True,
        )
        self._started = False
        self._failure: str | None = None

    @property
    def alive(self) -> bool:
        return self._started and self._process.is_alive() and self._failure is None

    def start(self, timeout_s: float = 60.0) -> None:
        if self._started or timeout_s <= 0:
            raise ValueError("pose worker already started or timeout is invalid")
        self._process.start()
        self._started = True
        try:
            first = self._results.get(timeout=timeout_s)
        except Empty as exc:
            self.close()
            raise PoseProcessError("pose backend did not initialize before timeout") from exc
        if isinstance(first, _Failure):
            self._failure = first.detail
            self.close()
            raise PoseProcessError(f"pose backend initialization failed:\n{first.detail}")
        if not isinstance(first, _Ready):
            self.close()
            raise PoseProcessError("unexpected pose backend initialization reply")

    def submit(self, frame: RGBDFrame) -> bool:
        """Copy one camera snapshot; replace a pending older frame if necessary."""
        if not self.alive:
            return False
        snapshot = replace(
            frame,
            rgb=frame.rgb.copy(),
            depth=frame.depth.copy(),
            intrinsics=frame.intrinsics.copy(),
            base_from_camera=frame.base_from_camera.copy(),
        )
        try:
            self._frames.put_nowait(snapshot)
        except Full:
            try:
                self._frames.get_nowait()
                self.metrics.superseded += 1
            except Empty:
                # The feeder may hold the queued item before it is readable.
                self.metrics.rejected += 1
                return False
            try:
                self._frames.put_nowait(snapshot)
            except Full:
                self.metrics.rejected += 1
                return False
        self.metrics.submitted += 1
        return True

    def poll(self) -> int:
        """Publish completed snapshots; call from the host collector, never a motor tick."""
        if not self._started:
            raise PoseProcessError("pose worker is not started")
        count = 0
        while True:
            try:
                message = self._results.get_nowait()
            except Empty:
                break
            if isinstance(message, _Failure):
                self._failure = message.detail
                break
            if isinstance(message, _Result):
                # A scene-estimator route never has access to the pose store.
                if self.store is not None:
                    self.store.publish(message.estimates)
                if self.on_result is not None:
                    self.on_result(message)
                self.metrics.delivered += 1
                self.metrics.last_failures = message.failures
                for role, reason in message.failures.items():
                    key = f"{role}: {reason}"
                    self.metrics.failure_counts[key] = self.metrics.failure_counts.get(key, 0) + 1
                if message.estimates:
                    self.metrics.last_source_time_s = max(e.source_time_s for e in message.estimates)
                count += 1
        if self._failure:
            raise PoseProcessError(f"pose backend failed:\n{self._failure}")
        if not self._process.is_alive():
            raise PoseProcessError(f"pose backend exited with code {self._process.exitcode}")
        return count

    def close(self, timeout_s: float = 5.0) -> None:
        if not self._started:
            return
        try:
            self._frames.put_nowait(None)
        except Full:
            with suppress(Empty):
                self._frames.get_nowait()
            with suppress(Full):
                self._frames.put(None, timeout=min(timeout_s, 0.5))
        self._process.join(timeout_s)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout_s)
        self._started = False
