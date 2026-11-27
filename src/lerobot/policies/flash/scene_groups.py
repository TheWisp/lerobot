"""Causal scene points and conservative rigid environment relationships.

An environment group is a set of *non-object* scene tracks whose measured
motion agrees with an object's independently observed motion. Mere proximity
or stillness never establishes the relationship. Once established, surviving
group points may carry the object's last observed pose through occlusion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .depth_quality import depth_support_mask
from .image_matching import LKMatchConfig, prepare_image_pair
from .observations import RGBDFrame
from .registration import (
    transform_points,
    unproject_pixels,
)


@dataclass(frozen=True)
class SceneTracks:
    frame_id: int
    source_time_s: float
    ids: np.ndarray
    xy: np.ndarray
    base_xyz: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.ids.ndim != 1
            or self.xy.shape != (len(self.ids), 2)
            or self.base_xyz.shape != (len(self.ids), 3)
        ):
            raise ValueError("scene tracks need aligned point IDs, pixels and metric positions")
        if len(np.unique(self.ids)) != len(self.ids) or not np.isfinite(self.base_xyz).all():
            raise ValueError("scene tracks need unique IDs and finite metric positions")


class ScenePointTracker:
    """Track and replenish RGB-D point identities across every camera frame.

    This frontend does not label objects or infer rigid membership. It only
    supplies timestamped measured point pairs for a later group hypothesis.
    """

    def __init__(
        self,
        *,
        max_points: int = 800,
        max_frame_gap_s: float = 0.1,
        image_scale: float = 0.5,
        fb_threshold_px: float = 1.0,
        depth_support_radius_px: int = 1,
    ) -> None:
        if (
            max_points < 16
            or not np.isfinite(max_frame_gap_s)
            or max_frame_gap_s <= 0
            or not 0 < image_scale <= 1
            or not np.isfinite(fb_threshold_px)
            or fb_threshold_px <= 0
            or isinstance(depth_support_radius_px, bool)
            or not isinstance(depth_support_radius_px, int)
            or depth_support_radius_px < 0
        ):
            raise ValueError("invalid scene point tracking limits")
        self.max_points = max_points
        self.max_frame_gap_s = max_frame_gap_s
        self.image_scale = image_scale
        self.fb_threshold_px = fb_threshold_px
        self.depth_support_radius_px = depth_support_radius_px
        self._gray: np.ndarray | None = None
        self._uv = np.empty((0, 2), dtype=np.float32)
        self._ids = np.empty(0, dtype=np.int64)
        self._next_id = 0
        self._last_time_s: float | None = None
        self.gaps = 0
        self.recent_gap_durations_s: deque[float] = deque(maxlen=16)

    def update(self, frame: RGBDFrame) -> SceneTracks:
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        if self.image_scale != 1:
            gray = cv2.resize(
                gray, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA
            )
        if self._gray is not None:
            assert self._last_time_s is not None
            dt = frame.source_time_s - self._last_time_s
            if dt <= 0:
                # Non-monotone camera time has no causal optical-flow meaning.
                self._uv = np.empty((0, 2), dtype=np.float32)
                self._ids = np.empty(0, dtype=np.int64)
                self._gray = None
                self.gaps += 1
                self.recent_gap_durations_s.append(float(dt))
            elif dt > self.max_frame_gap_s:
                # Time alone cannot prove that a point moved: a stationary
                # scene remains trackable after a dropped camera frame. Keep
                # only identities that pass the same forward/backward image
                # check and later 3D group fit as on every other frame.
                self.gaps += 1
                self.recent_gap_durations_s.append(float(dt))
            if self._gray is not None and len(self._uv):
                pair = prepare_image_pair(
                    self._gray, gray, LKMatchConfig(forward_backward_limit_px=self.fb_threshold_px)
                )
                matches = pair.match(self._uv)
                xy, good = matches.target_xy, matches.accepted
                self._uv, self._ids = xy[good], self._ids[good]
        self._gray = gray
        self._last_time_s = frame.source_time_s

        # Feature replenishment is causal and bounded. Existing identities are
        # never renamed, and new points cannot appear in earlier snapshots.
        depth_supported = depth_support_mask(frame.depth, radius_px=self.depth_support_radius_px)
        if len(self._uv) < self.max_points:
            valid_depth = depth_supported.astype(np.uint8)
            valid_depth = cv2.resize(
                valid_depth, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            for x, y in self._uv:
                cv2.circle(valid_depth, (round(float(x)), round(float(y))), 5, 0, -1)
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_points - len(self._uv),
                qualityLevel=0.005,
                minDistance=5,
                mask=valid_depth,
            )
            if corners is not None:
                fresh = corners.reshape(-1, 2)
                new_ids = np.arange(self._next_id, self._next_id + len(fresh), dtype=np.int64)
                self._next_id += len(fresh)
                self._uv = np.concatenate((self._uv, fresh))
                self._ids = np.concatenate((self._ids, new_ids))

        pixels = np.rint(self._uv / self.image_scale).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, frame.depth.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, frame.depth.shape[0] - 1)
        valid = depth_supported[pixels[:, 1], pixels[:, 0]]
        measured_pixels = pixels[valid]
        camera_xyz = unproject_pixels(frame.depth, frame.intrinsics, measured_pixels, frame.depth_m_per_unit)
        return SceneTracks(
            frame.frame_id,
            frame.source_time_s,
            self._ids[valid].copy(),
            measured_pixels.copy(),
            transform_points(frame.base_from_camera, camera_xyz),
        )
