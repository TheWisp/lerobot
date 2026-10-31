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
from .geometry import inverse, transform
from .perception import RGBDFrame
from .registration import (
    DegenerateRegistrationError,
    register_object,
    rigid_fit,
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


def _outside_masks(xy: np.ndarray, masks: tuple[np.ndarray, ...]) -> np.ndarray:
    keep = np.ones(len(xy), dtype=bool)
    for mask in masks:
        if mask.dtype != bool or mask.ndim != 2:
            raise ValueError("object exclusion mask must be boolean HxW")
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
        keep[inside] &= ~mask[xy[inside, 1], xy[inside, 0]]
    return keep


def _support_center(pose: np.ndarray, object_center_object_m: np.ndarray | None) -> np.ndarray:
    local_center = (
        np.zeros(3) if object_center_object_m is None else np.asarray(object_center_object_m, dtype=float)
    )
    if local_center.shape != (3,) or not np.isfinite(local_center).all():
        raise ValueError("object support center must be a finite 3D point")
    return transform_points(transform(pose), local_center[None])[0]


@dataclass(frozen=True)
class EnvironmentRelationship:
    """Measured environmental point group carrying one last observed pose."""

    ids: np.ndarray
    anchor_xyz: np.ndarray
    anchor_pose: np.ndarray
    anchor_time_s: float
    reference_version: str
    last_group_measurement_s: float | None = None

    def predict(
        self,
        tracks: SceneTracks,
        *,
        fit_threshold_m: float,
        min_inliers: int,
    ) -> np.ndarray | None:
        tracked = self.track(tracks, fit_threshold_m=fit_threshold_m, min_inliers=min_inliers)
        return tracked[0] if tracked is not None else None

    def track(
        self,
        tracks: SceneTracks,
        *,
        fit_threshold_m: float,
        min_inliers: int,
    ) -> tuple[np.ndarray, EnvironmentRelationship] | None:
        lookup = {int(point_id): index for index, point_id in enumerate(tracks.ids)}
        left, right = [], []
        for index, point_id in enumerate(self.ids):
            if (current := lookup.get(int(point_id))) is not None:
                left.append(index)
                right.append(current)
        if len(left) < min_inliers:
            return None
        source = self.anchor_xyz[left]
        target = tracks.base_xyz[right]
        # When the measured support points have not moved beyond the 3-D fit
        # resolution, their orientation is unobservable. A least-squares rigid
        # fit can still invent a large rotation on a nearly planar support,
        # which rotates a distant object-frame origin and the planner goal.
        # Preserve the anchor orientation and carry only measured translation.
        translation = np.median(target - source, axis=0)
        translation_inliers = np.flatnonzero(
            np.linalg.norm(source + translation - target, axis=1) <= fit_threshold_m
        )
        if np.linalg.norm(translation) <= fit_threshold_m and len(translation_inliers) >= max(
            min_inliers, len(source) // 2 + 1
        ):
            motion = np.eye(4)
            motion[:3, 3] = translation
            inliers = translation_inliers
        else:
            fitted = self._fit_motion(source, target, fit_threshold_m, min_inliers)
            if fitted is None:
                return None
            motion, inliers = fitted
        source_indices = np.asarray(left, dtype=int)[inliers]
        relationship = EnvironmentRelationship(
            self.ids[source_indices],
            self.anchor_xyz[source_indices],
            self.anchor_pose,
            self.anchor_time_s,
            self.reference_version,
            tracks.source_time_s,
        )
        return motion @ self.anchor_pose, relationship

    @staticmethod
    def _fit_motion(
        source: np.ndarray, target: np.ndarray, fit_threshold_m: float, min_inliers: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            # Once point identities are associated, a consensus fit should
            # be cheap. Trim depth/flow outliers directly when a majority
            # agrees; use robust search only when the scene split is real.
            direct = rigid_fit(source, target)
            residual = np.linalg.norm(transform_points(direct, source) - target, axis=1)
            inliers = np.flatnonzero(residual <= fit_threshold_m)
            if len(inliers) >= max(min_inliers, len(source) // 2 + 1):
                refined = rigid_fit(source[inliers], target[inliers])
                residual = np.linalg.norm(transform_points(refined, source) - target, axis=1)
                inliers = np.flatnonzero(residual <= fit_threshold_m)
                if len(inliers) >= max(min_inliers, len(source) // 2 + 1):
                    motion = refined
                else:
                    raise DegenerateRegistrationError("direct group refit lost its majority")
            else:
                raise DegenerateRegistrationError("direct group fit lacked a majority")
        except DegenerateRegistrationError:
            try:
                fit = register_object(
                    source,
                    target,
                    threshold_m=fit_threshold_m,
                    min_inliers=min_inliers,
                    trials=100,
                )
            except DegenerateRegistrationError:
                return None
            motion = fit.base_from_demo_object
            inliers = np.asarray(fit.inlier_indices, dtype=int)
        return motion, inliers


def seed_environment_relationship(
    scene: SceneTracks,
    pose: np.ndarray,
    object_masks: tuple[np.ndarray, ...],
    *,
    context_radius_m: float,
    min_inliers: int,
    reference_version: str,
    object_center_object_m: np.ndarray | None = None,
) -> EnvironmentRelationship | None:
    """Propose nearby support points from one direct object observation.

    This is a hypothesis until a second direct view agrees. It may still
    carry an explicitly predicted pose while the object is briefly hidden.
    """
    center = _support_center(pose, object_center_object_m)
    nearby = np.linalg.norm(scene.base_xyz - center, axis=1) <= context_radius_m
    keep = nearby & _outside_masks(scene.xy, object_masks)
    if int(keep.sum()) < min_inliers:
        return None
    return EnvironmentRelationship(
        scene.ids[keep].copy(),
        scene.base_xyz[keep].copy(),
        transform(pose),
        scene.source_time_s,
        reference_version,
        scene.source_time_s,
    )


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
    allow_stationary: bool = False,
    object_center_object_m: np.ndarray | None = None,
) -> EnvironmentRelationship | None:
    """Link an object to nearby non-object points after independent pose agreement.

    A stationary support can be verified from distinct direct object views:
    both the object and its surrounding points must stay within the measured
    registration error. Moving associations still require measurable co-motion.
    """
    if after.source_time_s <= before.source_time_s:
        return None
    ids, xy, old_xyz, new_xyz = _paired(before, after)
    if len(ids) < min_inliers:
        return None
    keep = _outside_masks(xy, before_object_masks)
    ids, old_xyz, new_xyz = ids[keep], old_xyz[keep], new_xyz[keep]
    old_pose, new_pose = transform(before_pose), transform(after_pose)
    old_center = _support_center(old_pose, object_center_object_m)
    nearby = np.linalg.norm(old_xyz - old_center, axis=1) <= context_radius_m
    ids, old_xyz, new_xyz = ids[nearby], old_xyz[nearby], new_xyz[nearby]
    if len(ids) < min_inliers:
        return None
    expected_motion = new_pose @ inverse(old_pose)
    displaced_center = transform_points(expected_motion, old_center[None])[0]
    displacement = np.linalg.norm(displaced_center - old_center)
    if displacement < min_motion_m and (not allow_stationary or displacement > fit_threshold_m):
        return None
    residual = np.linalg.norm(transform_points(expected_motion, old_xyz) - new_xyz, axis=1)
    agree = residual <= fit_threshold_m
    if int(agree.sum()) < min_inliers:
        return None
    ids, old_xyz, new_xyz = ids[agree], old_xyz[agree], new_xyz[agree]
    new_center = _support_center(new_pose, object_center_object_m)
    # These correspondences already agree with the independently observed
    # object motion. If one rigid fit explains every remaining measured point
    # and carries the object centre correctly, random three-point trials
    # cannot find a larger consensus. Keep the robust search for other cases.
    try:
        direct = rigid_fit(old_xyz, new_xyz)
        residual = np.linalg.norm(transform_points(direct, old_xyz) - new_xyz, axis=1)
        if (
            np.all(residual <= fit_threshold_m)
            and np.linalg.norm(transform_points(direct, old_center[None])[0] - new_center) <= fit_threshold_m
        ):
            return EnvironmentRelationship(
                ids, new_xyz, new_pose, after.source_time_s, reference_version, after.source_time_s
            )
    except DegenerateRegistrationError:
        pass
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
    if np.linalg.norm(predicted_center - new_center) > fit_threshold_m:
        return None
    indices = np.asarray(fit.inlier_indices, dtype=int)
    return EnvironmentRelationship(
        ids[indices],
        new_xyz[indices],
        new_pose,
        after.source_time_s,
        reference_version,
        after.source_time_s,
    )
