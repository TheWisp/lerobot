"""Aligned camera frames and timestamped object observations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import numpy as np

from .geometry import transform
from .pose import Evidence, PoseEstimate


@dataclass(frozen=True)
class RGBDFrame:
    """One aligned camera observation; source time is on the motor clock."""

    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    depth_m_per_unit: float
    base_from_camera: np.ndarray
    source_time_s: float
    available_time_s: float
    calibration_version: str
    depth_processing: dict[str, bool] | None = None
    depth_filter_revision: int = 0
    depth_before_filters: np.ndarray | None = None
    infrared_left: np.ndarray | None = None
    infrared_right: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("RGB must be HxWx3 uint8")
        if self.depth.shape != self.rgb.shape[:2] or self.depth.dtype not in (np.uint16, np.float32):
            raise ValueError("depth must be aligned HxW uint16 or float32")
        if self.depth_before_filters is not None and (
            self.depth_before_filters.shape != self.depth.shape
            or self.depth_before_filters.dtype != self.depth.dtype
        ):
            raise ValueError("pre-filter depth must share the aligned depth layout and dtype")
        if (self.infrared_left is None) != (self.infrared_right is None):
            raise ValueError("infrared images must be a synchronized pair")
        if self.infrared_left is not None and (
            self.infrared_left.shape != self.infrared_right.shape
            or self.infrared_left.ndim != 2
            or self.infrared_left.dtype != np.uint8
            or self.infrared_right.dtype != np.uint8
        ):
            raise ValueError("infrared images must share a grayscale uint8 grid")
        if self.intrinsics.shape != (3, 3) or not np.isfinite(self.intrinsics).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        if not np.isfinite(self.depth_m_per_unit) or self.depth_m_per_unit <= 0:
            raise ValueError("invalid depth scale")
        if self.available_time_s < self.source_time_s:
            raise ValueError("availability cannot precede capture")
        object.__setattr__(self, "base_from_camera", transform(self.base_from_camera))

    def metric_depth(self) -> np.ndarray:
        return self.depth.astype(np.float32) * self.depth_m_per_unit


class LatestRGBDFrameStore:
    """Retain a capture-thread snapshot for asynchronous episode recording.

    The capture thread copies frame buffers before publication so a camera
    driver may reuse its next read buffer. Readers only take a short lock and
    receive immutable-by-convention arrays; they never wait for the sensor.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: RGBDFrame | None = None

    def publish(self, frame: RGBDFrame) -> None:
        rgb = frame.rgb.copy()
        depth = frame.depth.copy()
        before = frame.depth_before_filters.copy() if frame.depth_before_filters is not None else None
        infrared_left = frame.infrared_left.copy() if frame.infrared_left is not None else None
        infrared_right = frame.infrared_right.copy() if frame.infrared_right is not None else None
        rgb.setflags(write=False)
        depth.setflags(write=False)
        if before is not None:
            before.setflags(write=False)
        if infrared_left is not None:
            infrared_left.setflags(write=False)
            infrared_right.setflags(write=False)
        snapshot = RGBDFrame(
            frame.frame_id,
            rgb,
            depth,
            frame.intrinsics.copy(),
            frame.depth_m_per_unit,
            frame.base_from_camera.copy(),
            frame.source_time_s,
            frame.available_time_s,
            frame.calibration_version,
            dict(frame.depth_processing) if frame.depth_processing is not None else None,
            frame.depth_filter_revision,
            before,
            infrared_left,
            infrared_right,
        )
        snapshot.intrinsics.setflags(write=False)
        snapshot.base_from_camera.setflags(write=False)
        with self._lock:
            if self._latest is not None and (
                snapshot.frame_id <= self._latest.frame_id
                or snapshot.source_time_s <= self._latest.source_time_s
            ):
                raise ValueError("camera frame IDs and capture times must increase")
            self._latest = snapshot

    def snapshot(self, *, no_later_than_s: float | None = None) -> RGBDFrame | None:
        with self._lock:
            frame = self._latest
        if frame is not None and no_later_than_s is not None and frame.source_time_s > no_later_than_s:
            return None
        return frame


@dataclass(frozen=True)
class ReferenceSnippet:
    """One annotated view registered into a stable object frame."""

    repo_id: str
    episode: int
    frame_index: int
    object_role: str
    object_from_camera: np.ndarray
    mask_artifact: str
    calibration_version: str

    def __post_init__(self) -> None:
        if not self.repo_id or not self.object_role or self.episode < 0 or self.frame_index < 0:
            raise ValueError("invalid reference snippet identity")
        object.__setattr__(self, "object_from_camera", transform(self.object_from_camera))


@dataclass(frozen=True)
class ObjectReference:
    role: str
    version: str
    snippets: tuple[ReferenceSnippet, ...]

    def __post_init__(self) -> None:
        if not self.role or not self.version or not self.snippets:
            raise ValueError("object reference needs role, version and snippets")
        if any(snippet.object_role != self.role for snippet in self.snippets):
            raise ValueError("reference snippets must share one object role")


class PoseBackend(Protocol):
    """Heavy implementations live outside the motor process.

    A backend may use Point2Pose, a different model, or a trained checkpoint;
    all return pose estimates with provenance and supported dimensions.
    """

    def initialize(self, references: tuple[ObjectReference, ...]) -> None: ...

    def update(self, frame: RGBDFrame) -> tuple[PoseEstimate, ...]: ...


class InstanceMaskPoseBackend(PoseBackend, Protocol):
    """A pose backend that also exposes masks from its most recent frame."""

    @property
    def last_masks(self) -> dict[str, np.ndarray]: ...


class LatestPoseStore:
    """Short-lock handoff from an asynchronous estimator to the planner."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._version = 0
        self._poses: dict[str, PoseEstimate] = {}

    def publish(self, estimates: tuple[PoseEstimate, ...]) -> int:
        with self._lock:
            changed = False
            for estimate in estimates:
                previous = self._poses.get(estimate.role)
                if previous is not None:
                    if estimate.source_time_s < previous.source_time_s:
                        continue
                    if estimate.source_time_s == previous.source_time_s:
                        if estimate.reference_version != previous.reference_version:
                            continue
                        priority = (
                            estimate.evidence != Evidence.LOST,
                            estimate.has_direct_translation,
                            estimate.evidence == Evidence.OBSERVED,
                        )
                        previous_priority = (
                            previous.evidence != Evidence.LOST,
                            previous.has_direct_translation,
                            previous.evidence == Evidence.OBSERVED,
                        )
                        if priority < previous_priority or (
                            priority == previous_priority
                            and estimate.available_time_s <= previous.available_time_s
                        ):
                            continue
                self._poses[estimate.role] = estimate
                changed = True
            if changed:
                self._version += 1
            return self._version

    def snapshot(self) -> tuple[int, dict[str, PoseEstimate]]:
        with self._lock:
            return self._version, dict(self._poses)

    def revise(self, estimate: PoseEstimate) -> int:
        """Replace a same-time belief after replaying earlier source-frame evidence.

        Ordinary publication favors a direct observation over a prediction at
        the same camera time. Event replay may invalidate that observation, so
        its revised belief needs an explicit, versioned replacement instead.
        """
        with self._lock:
            previous = self._poses.get(estimate.role)
            if previous is not None and estimate.source_time_s < previous.source_time_s:
                raise ValueError("a replay revision cannot move scene time backwards")
            self._poses[estimate.role] = estimate
            self._version += 1
            return self._version
