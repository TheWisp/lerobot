"""Timestamped aligned RGB-D input from LeRobot's existing camera API."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal, Protocol

import numpy as np

from .perception import RGBDFrame


class AlignedRGBDCamera(Protocol):
    def read_color_and_aligned_depth_with_timestamp(
        self, color_mode=None, timeout_ms: int = 200
    ) -> tuple[np.ndarray, np.ndarray, float]: ...


class LeRobotRGBDSource:
    """Wrap one aligned RealSense frameset without a second RGB/depth read.

    The camera timestamp, ``clock`` and motor scheduler must share a monotonic
    clock. ``base_from_camera_at`` is evaluated at capture time, which also
    permits a calibrated moving camera when its robot-pose history exists.
    Calibration must correspond to the returned image's crop and rotation.
    """

    def __init__(
        self,
        camera: AlignedRGBDCamera,
        *,
        intrinsics: np.ndarray,
        depth_m_per_unit: float,
        base_from_camera_at: Callable[[float], np.ndarray],
        calibration_version: str,
        color_order: Literal["rgb", "bgr"],
        max_capture_age_s: float,
        clock: Callable[[], float] = time.perf_counter,
        on_frame: Callable[[RGBDFrame], None] | None = None,
    ) -> None:
        if not calibration_version or color_order not in ("rgb", "bgr"):
            raise ValueError("RGB-D source needs calibration identity and color order")
        if not np.isfinite(max_capture_age_s) or max_capture_age_s <= 0:
            raise ValueError("RGB-D source needs a positive capture-age bound")
        self.camera = camera
        self.intrinsics = np.asarray(intrinsics, dtype=float).copy()
        self.depth_m_per_unit = depth_m_per_unit
        self.base_from_camera_at = base_from_camera_at
        self.calibration_version = calibration_version
        self.color_order = color_order
        self.max_capture_age_s = max_capture_age_s
        self.clock = clock
        self.on_frame = on_frame
        self._next_frame_id = 0

    def read(self, timeout_ms: int = 200) -> RGBDFrame:
        color, depth, captured_s = self.camera.read_color_and_aligned_depth_with_timestamp(
            timeout_ms=timeout_ms
        )
        available_s = self.clock()
        if not np.isfinite(captured_s) or not 0 <= available_s - captured_s <= self.max_capture_age_s:
            raise ValueError("aligned RGB-D frame is stale or its timestamp is on another clock")
        rgb = np.asarray(color)
        if self.color_order == "bgr":
            rgb = rgb[:, :, ::-1]
        frame = RGBDFrame(
            self._next_frame_id,
            np.ascontiguousarray(rgb),
            np.ascontiguousarray(depth),
            self.intrinsics,
            self.depth_m_per_unit,
            self.base_from_camera_at(captured_s),
            captured_s,
            available_s,
            self.calibration_version,
        )
        self._next_frame_id += 1
        if self.on_frame is not None:
            self.on_frame(frame)
        return frame
