"""Causal RGB-D point tracks for rigid objects and environmental groups.

This fast path consumes consecutive camera frames. It does not use semantic
identity or claim that co-moving bodies are attached. A caller supplies a seed
mask and decides what the resulting rigid motion means.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .depth_quality import depth_support_mask
from .perception import RGBDFrame
from .registration import DegenerateRegistrationError, register_object, transform_points, unproject_pixels


@dataclass(frozen=True)
class RigidPointMotion:
    base_motion_from_seed: np.ndarray | None
    source_time_s: float
    available_time_s: float
    tracked_points: int
    depth_points: int
    inliers: int
    median_error_m: float | None


class CausalRigidPointTracker:
    """Track stable 2D point identities and fit their current metric motion.

    ``seed`` receives a mask from any segmenter. ``update`` must receive every
    subsequent frame; an unobserved temporal gap invalidates the tracks rather
    than silently fitting point identities across an unknown interval.
    """

    def __init__(
        self,
        *,
        max_points: int = 120,
        min_inliers: int = 12,
        max_frame_gap_s: float = 0.1,
        fb_threshold_px: float = 1.0,
        fit_threshold_m: float = 0.005,
        image_scale: float = 1.0,
        depth_support_radius_px: int = 1,
    ) -> None:
        if (
            max_points < min_inliers
            or min_inliers < 3
            or not np.isfinite(max_frame_gap_s)
            or max_frame_gap_s <= 0
            or not np.isfinite(fb_threshold_px)
            or fb_threshold_px <= 0
            or not np.isfinite(fit_threshold_m)
            or fit_threshold_m <= 0
            or not np.isfinite(image_scale)
            or not 0 < image_scale <= 1
            or isinstance(depth_support_radius_px, bool)
            or not isinstance(depth_support_radius_px, int)
            or depth_support_radius_px < 0
        ):
            raise ValueError("invalid causal point-tracker limits")
        self.max_points = max_points
        self.min_inliers = min_inliers
        self.max_frame_gap_s = max_frame_gap_s
        self.fb_threshold_px = fb_threshold_px
        self.fit_threshold_m = fit_threshold_m
        self.image_scale = image_scale
        self.depth_support_radius_px = depth_support_radius_px
        self._gray: np.ndarray | None = None
        self._uv = np.empty((0, 2), dtype=np.float32)
        self._base_seed = np.empty((0, 3), dtype=float)
        self._valid = np.zeros(0, dtype=bool)
        self._last_source_time_s: float | None = None

    def seed(self, frame: RGBDFrame, mask: np.ndarray) -> int:
        if mask.shape != frame.depth.shape or mask.dtype != bool:
            raise ValueError("seed mask must align with RGB-D pixels")
        # Detect before resizing: a small visible object can lose most of its
        # corners when the image is downsampled for faster optical flow.
        full_gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            full_gray,
            maxCorners=self.max_points,
            qualityLevel=0.005,
            minDistance=5,
            mask=mask.astype(np.uint8),
        )
        full_uv = corners.reshape(-1, 2) if corners is not None else np.empty((0, 2), dtype=np.float32)
        if len(full_uv):
            px = np.rint(full_uv).astype(int)
            depth_supported = depth_support_mask(frame.depth, radius_px=self.depth_support_radius_px)
            valid = depth_supported[px[:, 1], px[:, 0]]
            full_uv = full_uv[valid]
        self._uv = (full_uv * self.image_scale).astype(np.float32)
        self._base_seed = (
            transform_points(
                frame.base_from_camera,
                unproject_pixels(frame.depth, frame.intrinsics, full_uv, frame.depth_m_per_unit),
            )
            if len(full_uv)
            else np.empty((0, 3), dtype=float)
        )
        self._valid = np.ones(len(full_uv), dtype=bool)
        self._gray = self._gray_image(frame.rgb)
        self._last_source_time_s = frame.source_time_s
        return len(full_uv)

    def update(self, frame: RGBDFrame, mask: np.ndarray | None = None) -> RigidPointMotion:
        if mask is not None and (mask.shape != frame.depth.shape or mask.dtype != bool):
            raise ValueError("current mask must align with RGB-D pixels")
        if self._gray is None or self._last_source_time_s is None:
            raise RuntimeError("seed point tracks before updating")
        dt = frame.source_time_s - self._last_source_time_s
        if not 0 < dt <= self.max_frame_gap_s:
            self._valid[:] = False
            raise ValueError("causal point tracks lost a camera-frame interval")
        gray = self._gray_image(frame.rgb)
        live = np.flatnonzero(self._valid)
        if len(live):
            before = self._uv[live].reshape(-1, 1, 2)
            after, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
                self._gray, gray, before, None, winSize=(21, 21), maxLevel=3
            )
            backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._gray, after, None, winSize=(21, 21), maxLevel=3
            )
            xy = after.reshape(-1, 2)
            height, width = gray.shape
            good = (
                (forward_ok.ravel() == 1)
                & (backward_ok.ravel() == 1)
                & (np.linalg.norm((before - backward).reshape(-1, 2), axis=1) < self.fb_threshold_px)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < width)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < height)
            )
            self._valid[live[~good]] = False
            self._uv[live[good]] = xy[good]
        self._gray = gray
        self._last_source_time_s = frame.source_time_s
        live = np.flatnonzero(self._valid)
        if len(live):
            px = self._full_pixels(self._uv[live], frame.depth.shape)
            depth_supported = depth_support_mask(frame.depth, radius_px=self.depth_support_radius_px)
            depth_ok = depth_supported[px[:, 1], px[:, 0]]
            if mask is not None:
                # Validate ownership before fitting, not by counting unrelated
                # in-mask tracks after an out-of-mask majority wins the fit.
                depth_ok &= mask[px[:, 1], px[:, 0]]
            live = live[depth_ok]
        if len(live) < self.min_inliers:
            return RigidPointMotion(
                None, frame.source_time_s, frame.available_time_s, int(self._valid.sum()), len(live), 0, None
            )
        current = transform_points(
            frame.base_from_camera,
            unproject_pixels(
                frame.depth, frame.intrinsics, self._uv[live] / self.image_scale, frame.depth_m_per_unit
            ),
        )
        try:
            fit = register_object(
                self._base_seed[live],
                current,
                threshold_m=self.fit_threshold_m,
                min_inliers=self.min_inliers,
                trials=100,
            )
        except DegenerateRegistrationError:
            return RigidPointMotion(
                None, frame.source_time_s, frame.available_time_s, int(self._valid.sum()), len(live), 0, None
            )
        return RigidPointMotion(
            fit.base_from_demo_object,
            frame.source_time_s,
            frame.available_time_s,
            int(self._valid.sum()),
            len(live),
            len(fit.inlier_indices),
            fit.median_error_m,
        )

    def current_pixels(self, depth_shape: tuple[int, int]) -> np.ndarray:
        """Depth-image coordinates for surviving tracked point identities."""
        return self._full_pixels(self._uv[self._valid], depth_shape)

    def _gray_image(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.image_scale != 1:
            gray = cv2.resize(
                gray, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA
            )
        return gray

    def _full_pixels(self, uv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        px = np.rint(uv / self.image_scale).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, shape[1] - 1)
        px[:, 1] = np.clip(px[:, 1], 0, shape[0] - 1)
        return px
