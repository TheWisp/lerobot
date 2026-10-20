"""Causal scene points and conservative rigid environment relationships.

An environment group is a set of *non-object* scene tracks whose measured
motion agrees with an object's independently observed motion. Mere proximity
or stillness never establishes the relationship. Once established, surviving
group points may carry the object's last observed pose through occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import inverse, transform
from .perception import RGBDFrame
from .registration import DegenerateRegistrationError, register_object, transform_points, unproject_pixels


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
    ) -> None:
        if (
            max_points < 16
            or not np.isfinite(max_frame_gap_s)
            or max_frame_gap_s <= 0
            or not 0 < image_scale <= 1
            or not np.isfinite(fb_threshold_px)
            or fb_threshold_px <= 0
        ):
            raise ValueError("invalid scene point tracking limits")
        self.max_points = max_points
        self.max_frame_gap_s = max_frame_gap_s
        self.image_scale = image_scale
        self.fb_threshold_px = fb_threshold_px
        self._gray: np.ndarray | None = None
        self._uv = np.empty((0, 2), dtype=np.float32)
        self._ids = np.empty(0, dtype=np.int64)
        self._next_id = 0
        self._last_time_s: float | None = None
        self.gaps = 0

    def update(self, frame: RGBDFrame) -> SceneTracks:
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        if self.image_scale != 1:
            gray = cv2.resize(
                gray, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA
            )
        if self._gray is not None:
            assert self._last_time_s is not None
            dt = frame.source_time_s - self._last_time_s
            if not 0 < dt <= self.max_frame_gap_s:
                # A capture gap ends this track generation. Keep monotonically
                # increasing IDs so an old relationship cannot attach to new
                # pixels when tracking resumes.
                self._uv = np.empty((0, 2), dtype=np.float32)
                self._ids = np.empty(0, dtype=np.int64)
                self._gray = None
                self.gaps += 1
            if self._gray is not None and len(self._uv):
                before = self._uv.reshape(-1, 1, 2)
                after, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
                    self._gray, gray, before, None, winSize=(21, 21), maxLevel=3
                )
                backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self._gray, after, None, winSize=(21, 21), maxLevel=3
                )
                xy = after.reshape(-1, 2)
                good = (
                    (forward_ok.ravel() == 1)
                    & (backward_ok.ravel() == 1)
                    & (np.linalg.norm((before - backward).reshape(-1, 2), axis=1) < self.fb_threshold_px)
                    & (xy[:, 0] >= 0)
                    & (xy[:, 0] < gray.shape[1])
                    & (xy[:, 1] >= 0)
                    & (xy[:, 1] < gray.shape[0])
                )
                self._uv, self._ids = xy[good], self._ids[good]
        self._gray = gray
        self._last_time_s = frame.source_time_s

        # Feature replenishment is causal and bounded. Existing identities are
        # never renamed, and new points cannot appear in earlier snapshots.
        if len(self._uv) < self.max_points:
            valid_depth = (np.isfinite(frame.depth) & (frame.depth > 0)).astype(np.uint8)
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
        valid = np.isfinite(frame.depth[pixels[:, 1], pixels[:, 0]]) & (
            frame.depth[pixels[:, 1], pixels[:, 0]] > 0
        )
        self._uv, self._ids, pixels = self._uv[valid], self._ids[valid], pixels[valid]
        camera_xyz = unproject_pixels(frame.depth, frame.intrinsics, pixels, frame.depth_m_per_unit)
        return SceneTracks(
            frame.frame_id,
            frame.source_time_s,
            self._ids.copy(),
            pixels.copy(),
            transform_points(frame.base_from_camera, camera_xyz),
        )


def _paired(before: SceneTracks, after: SceneTracks) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    before_indices = {int(point_id): index for index, point_id in enumerate(before.ids)}
    left, right = [], []
    for index, point_id in enumerate(after.ids):
        if (old := before_indices.get(int(point_id))) is not None:
            left.append(old)
            right.append(index)
    left_indices, right_indices = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    return (
        before.ids[left_indices],
        before.xy[left_indices],
        before.base_xyz[left_indices],
        after.base_xyz[right_indices],
    )


@dataclass(frozen=True)
class EnvironmentRelationship:
    """Measured environmental point group carrying one last observed pose."""

    ids: np.ndarray
    anchor_xyz: np.ndarray
    anchor_pose: np.ndarray
    anchor_time_s: float
    reference_version: str

    def predict(
        self,
        tracks: SceneTracks,
        *,
        fit_threshold_m: float,
        min_inliers: int,
    ) -> np.ndarray | None:
        lookup = {int(point_id): index for index, point_id in enumerate(tracks.ids)}
        left, right = [], []
        for index, point_id in enumerate(self.ids):
            if (current := lookup.get(int(point_id))) is not None:
                left.append(index)
                right.append(current)
        if len(left) < min_inliers:
            return None
        try:
            fit = register_object(
                self.anchor_xyz[left],
                tracks.base_xyz[right],
                threshold_m=fit_threshold_m,
                min_inliers=min_inliers,
                trials=100,
            )
        except DegenerateRegistrationError:
            return None
        return fit.base_from_demo_object @ self.anchor_pose


def establish_environment_relationship(
    before: SceneTracks,
    after: SceneTracks,
    before_pose: np.ndarray,
    after_pose: np.ndarray,
    before_object_masks: tuple[np.ndarray, ...],
    *,
    context_radius_m: float,
    fit_threshold_m: float,
    min_motion_m: float,
    min_inliers: int,
    reference_version: str,
) -> EnvironmentRelationship | None:
    """Link only if non-object scene points share a *measured* object motion."""
    if after.source_time_s <= before.source_time_s:
        return None
    ids, xy, old_xyz, new_xyz = _paired(before, after)
    if len(ids) < min_inliers:
        return None
    for mask in before_object_masks:
        if mask.dtype != bool or mask.ndim != 2:
            raise ValueError("object exclusion mask must be boolean HxW")
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
        excluded = np.zeros(len(ids), dtype=bool)
        excluded[inside] = mask[xy[inside, 1], xy[inside, 0]]
        keep = ~excluded
        ids, xy, old_xyz, new_xyz = ids[keep], xy[keep], old_xyz[keep], new_xyz[keep]
    old_pose, new_pose = transform(before_pose), transform(after_pose)
    old_center = old_pose[:3, 3]
    nearby = np.linalg.norm(old_xyz - old_center, axis=1) <= context_radius_m
    ids, old_xyz, new_xyz = ids[nearby], old_xyz[nearby], new_xyz[nearby]
    if len(ids) < min_inliers:
        return None
    expected_motion = new_pose @ inverse(old_pose)
    displaced_center = expected_motion[:3, :3] @ old_center + expected_motion[:3, 3]
    if np.linalg.norm(displaced_center - old_center) < min_motion_m:
        return None
    residual = np.linalg.norm(transform_points(expected_motion, old_xyz) - new_xyz, axis=1)
    agree = residual <= fit_threshold_m
    if int(agree.sum()) < min_inliers:
        return None
    ids, old_xyz, new_xyz = ids[agree], old_xyz[agree], new_xyz[agree]
    try:
        fit = register_object(
            old_xyz,
            new_xyz,
            threshold_m=fit_threshold_m,
            min_inliers=min_inliers,
            trials=100,
        )
    except DegenerateRegistrationError:
        return None
    predicted_center = transform_points(fit.base_from_demo_object, old_center[None])[0]
    if np.linalg.norm(predicted_center - new_pose[:3, 3]) > fit_threshold_m:
        return None
    indices = np.asarray(fit.inlier_indices, dtype=int)
    return EnvironmentRelationship(
        ids[indices], new_xyz[indices], new_pose, after.source_time_s, reference_version
    )
