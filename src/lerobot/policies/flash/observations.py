"""Aligned camera frames and timestamped object observations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from .geometry import transform


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
