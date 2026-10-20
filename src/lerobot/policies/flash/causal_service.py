"""Every-frame point motion alongside slower, replaceable pose estimation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from math import isfinite
from queue import Full, Queue
from threading import Lock, Thread

import numpy as np

from .causal_points import CausalRigidPointTracker
from .perception import LatestPoseStore, RGBDFrame
from .trajectory import Evidence, PoseEstimate


@dataclass(frozen=True)
class _Anchor:
    frame_id: int
    estimates: tuple[PoseEstimate, ...]
    masks: dict[str, np.ndarray]


@dataclass(frozen=True)
class CausalTrackingConfig:
    history_frames: int
    queue_frames: int
    min_inliers: int
    max_frame_gap_s: float
    fit_threshold_m: float
    image_scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.history_frames < 2
            or self.queue_frames < 2
            or self.min_inliers < 3
            or not isfinite(self.max_frame_gap_s)
            or self.max_frame_gap_s <= 0
            or not isfinite(self.fit_threshold_m)
            or self.fit_threshold_m <= 0
            or not isfinite(self.image_scale)
            or not 0 < self.image_scale <= 1
        ):
            raise ValueError("invalid causal tracking configuration")


class CausalPoseService:
    """Consume each camera frame, replaying delayed anchors on retained frames.

    A slow estimator may skip frames, but this queue may not. The caller stops
    the attempt on overflow. An anchor older than retained frame history is
    refused instead of being transplanted onto a newer image.
    """

    def __init__(
        self,
        store: LatestPoseStore,
        *,
        history_frames: int = 60,
        queue_frames: int = 64,
        min_inliers: int = 12,
        max_frame_gap_s: float = 0.1,
        fit_threshold_m: float = 0.005,
        image_scale: float = 1.0,
    ) -> None:
        if history_frames < 2 or queue_frames < 2:
            raise ValueError("causal pose history and queue need at least two frames")
        self.store = store
        self.history: deque[RGBDFrame] = deque(maxlen=history_frames)
        self.queue: Queue[RGBDFrame | _Anchor | None] = Queue(maxsize=queue_frames)
        self.tracker_kwargs = {
            "min_inliers": min_inliers,
            "max_frame_gap_s": max_frame_gap_s,
            "fit_threshold_m": fit_threshold_m,
            "image_scale": image_scale,
        }
        self.active: dict[str, tuple[CausalRigidPointTracker, PoseEstimate]] = {}
        self._thread = Thread(target=self._run, name="flash-causal-points", daemon=True)
        self._lock = Lock()
        self._error: Exception | None = None
        self._started = False
        self.frames = 0
        self.anchors = 0
        self.missed_anchors = 0
        self.published = 0

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self._started:
            raise RuntimeError("causal pose service already started")
        self._started = True
        self._thread.start()

    def submit_frame(self, frame: RGBDFrame) -> None:
        self._submit(frame)

    def submit_anchor(
        self,
        frame_id: int,
        estimates: tuple[PoseEstimate, ...],
        masks: dict[str, np.ndarray],
    ) -> None:
        self._submit(_Anchor(frame_id, estimates, masks))

    def _submit(self, item: RGBDFrame | _Anchor) -> None:
        if self.error is not None:
            raise RuntimeError("causal pose service failed") from self.error
        try:
            self.queue.put_nowait(item)
        except Full as exc:
            raise RuntimeError("causal point-frame queue overflow") from exc

    def _run(self) -> None:
        try:
            while (item := self.queue.get()) is not None:
                if isinstance(item, RGBDFrame):
                    self._frame(item)
                else:
                    self._anchor(item)
        except Exception as exc:
            with self._lock:
                self._error = exc

    def _frame(self, frame: RGBDFrame) -> None:
        if self.history and frame.frame_id <= self.history[-1].frame_id:
            raise ValueError("causal camera frames must increase")
        self.history.append(frame)
        self.frames += 1
        for role, (tracker, anchor) in list(self.active.items()):
            try:
                motion = tracker.update(frame)
            except ValueError:
                del self.active[role]
                continue
            self._publish(role, anchor, motion)

    def _anchor(self, result: _Anchor) -> None:
        source = next((frame for frame in self.history if frame.frame_id == result.frame_id), None)
        if source is None:
            self.missed_anchors += 1
            return
        for estimate in result.estimates:
            mask = result.masks.get(estimate.role)
            if (
                mask is None
                or estimate.pose is None
                or estimate.evidence == Evidence.LOST
                or estimate.source_time_s != source.source_time_s
            ):
                continue
            tracker = CausalRigidPointTracker(**self.tracker_kwargs)
            if tracker.seed(source, mask) < tracker.min_inliers:
                continue
            self.active[estimate.role] = (tracker, estimate)
            self.anchors += 1
            for frame in self.history:
                if frame.frame_id <= source.frame_id:
                    continue
                try:
                    motion = tracker.update(frame)
                except ValueError:
                    del self.active[estimate.role]
                    break
                self._publish(estimate.role, estimate, motion)

    def _publish(self, role: str, anchor: PoseEstimate, motion) -> None:
        if motion.base_motion_from_seed is None:
            return
        assert anchor.pose is not None
        current = PoseEstimate(
            role,
            motion.base_motion_from_seed @ anchor.pose,
            motion.source_time_s,
            max(motion.available_time_s, time.perf_counter()),
            # The current RGB-D frame measures point motion. It does not
            # independently recognize the object's absolute pose or identity;
            # those still come from the older anchor and can drift or be
            # corrupted by a bad seed mask. Keep that distinction visible to
            # the planner and the recorded evidence stream.
            Evidence.PREDICTED,
            anchor.reference_version,
            anchor.translation_supported,
            anchor.rotation_supported,
        )
        before, _ = self.store.snapshot()
        after = self.store.publish((current,))
        if after != before:
            self.published += 1

    def close(self, timeout_s: float = 5.0) -> None:
        if not self._started:
            return
        try:
            self.queue.put(None, timeout=timeout_s)
        except Full as exc:
            raise RuntimeError("causal pose service did not drain") from exc
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("causal pose service did not stop")
        self._started = False
