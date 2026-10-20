"""Metric pose updates from matched RGB-D pixels, independent of the matcher."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from .perception import ObjectReference, ReferenceSnippet, RGBDFrame
from .registration import (
    DegenerateRegistrationError,
    register_object,
    transform_points,
    unproject_pixels,
    valid_depth_pixels,
)
from .shape_registration import register_visible_shape
from .trajectory import Evidence, PoseEstimate


@dataclass(frozen=True)
class ObjectPointModel:
    """Surface points from any number of demo views in one stable object frame."""

    role: str
    reference_version: str
    points_object_m: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_object_m, dtype=float)
        if not self.role or not self.reference_version or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("invalid object point model")
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("object point model needs at least three finite 3D points")
        object.__setattr__(self, "points_object_m", points)


@dataclass(frozen=True)
class PixelMatches:
    """Matcher output: model-point indices and aligned live RGB/depth pixels."""

    role: str
    reference_version: str
    model_indices: np.ndarray
    live_xy: np.ndarray


class PixelMatchProvider(Protocol):
    def matches(self, frame: RGBDFrame) -> tuple[PixelMatches, ...]: ...


class MaskProvidingPixelMatchProvider(PixelMatchProvider, Protocol):
    @property
    def last_masks(self) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True)
class ReferencePixels:
    snippet: ReferenceSnippet
    frame: RGBDFrame
    pixels_xy: np.ndarray


def build_object_point_model(
    reference: ObjectReference, observations: tuple[ReferencePixels, ...]
) -> ObjectPointModel:
    """Merge sampled depth from demo snippets into their shared object frame."""
    allowed = {(s.repo_id, s.episode, s.frame_index) for s in reference.snippets}
    groups = []
    for observation in observations:
        snippet, frame = observation.snippet, observation.frame
        if (snippet.repo_id, snippet.episode, snippet.frame_index) not in allowed:
            raise ValueError("reference pixels do not belong to this object reference")
        if snippet.calibration_version != frame.calibration_version or snippet.frame_index != frame.frame_id:
            raise ValueError("reference pixels and RGB-D frame have different calibration or frame id")
        points_camera = unproject_pixels(
            frame.depth, frame.intrinsics, observation.pixels_xy, frame.depth_m_per_unit
        )
        groups.append(transform_points(snippet.object_from_camera, points_camera))
    if not groups:
        raise ValueError("object reference has no sampled surface points")
    return ObjectPointModel(reference.role, reference.version, np.concatenate(groups))


class MatchedRGBDPoseBackend:
    """PoseBackend implementation for any pixel matcher or human correspondence source.

    The provider may use P2P, a feature matcher, SAM masks, or annotations.
    This class performs only the common metric depth and robust rigid fit.
    A failed fit publishes LOST, never a confident pose. It cannot solve
    unobserved rotation about an object's symmetry axis by itself.
    """

    def __init__(
        self,
        models: tuple[ObjectPointModel, ...],
        provider: PixelMatchProvider,
        *,
        threshold_m: float,
        min_inliers: int,
        trials: int,
        visible_shape_fallback: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.models = {model.role: model for model in models}
        if not models or len(self.models) != len(models):
            raise ValueError("each object role needs one unique point model")
        if threshold_m <= 0 or min_inliers < 3 or trials < 1:
            raise ValueError("invalid registration settings")
        self.provider = provider
        self.threshold_m = threshold_m
        self.min_inliers = min_inliers
        self.trials = trials
        self.visible_shape_fallback = visible_shape_fallback
        self.clock = clock
        self.last_failure: dict[str, str] = {}
        self._last_accepted_pose: dict[str, np.ndarray] = {}
        self._initialized = False

    def initialize(self, references: tuple[ObjectReference, ...]) -> None:
        versions = {reference.role: reference.version for reference in references}
        if len(versions) != len(references) or set(versions) != set(self.models):
            raise ValueError("point models and references must cover the same unique roles")
        for role, model in self.models.items():
            if versions[role] != model.reference_version:
                raise ValueError(f"{role}: point model and reference versions differ")
        self._last_accepted_pose.clear()
        self._initialized = True

    @property
    def last_masks(self) -> dict[str, np.ndarray]:
        return cast(MaskProvidingPixelMatchProvider, self.provider).last_masks

    def update(self, frame: RGBDFrame) -> tuple[PoseEstimate, ...]:
        if not self._initialized:
            raise RuntimeError("initialize references before updating poses")
        matches = self.provider.matches(frame)
        by_role = {item.role: item for item in matches}
        if len(by_role) != len(matches) or not set(by_role) <= set(self.models):
            raise ValueError("matcher returned a duplicate or unknown object role")
        available_s = max(frame.available_time_s, self.clock())
        estimates = []
        self.last_failure.clear()
        for role, model in self.models.items():
            match = by_role.get(role)
            estimator = "appearance_rgbd"
            try:
                if match is None:
                    raise DegenerateRegistrationError("no matched pixels")
                pose = self._fit(frame, model, match)
            except DegenerateRegistrationError as match_error:
                if not self.visible_shape_fallback:
                    pose = None
                    failure = str(match_error)
                else:
                    mask = getattr(self.provider, "last_masks", {}).get(role)
                    try:
                        if mask is None:
                            raise DegenerateRegistrationError("no aligned instance mask")
                        shape = register_visible_shape(
                            model.points_object_m,
                            frame.depth,
                            mask,
                            frame.intrinsics,
                            frame.depth_m_per_unit,
                            tolerance_m=self.threshold_m,
                            min_points=self.min_inliers,
                        )
                        pose = frame.base_from_camera @ shape.camera_from_object
                        previous = self._last_accepted_pose.get(role)
                        if previous is not None:
                            # Nearest-surface ICP can slide along a uniform or
                            # partly hidden object while retaining a low depth
                            # residual. Without matched points, such a fit is
                            # only a local continuation of an established
                            # object frame, not independent evidence of a
                            # large object-frame motion.
                            origin_shift = np.linalg.norm(pose[:3, 3] - previous[:3, 3])
                            old_surface = model.points_object_m @ previous[:3, :3].T + previous[:3, 3]
                            new_surface = model.points_object_m @ pose[:3, :3].T + pose[:3, 3]
                            surface_shift = np.max(np.linalg.norm(new_surface - old_surface, axis=1))
                            if max(origin_shift, surface_shift) > 2 * self.threshold_m:
                                raise DegenerateRegistrationError(
                                    "shape fit changes the object frame without matched-point support"
                                )
                        estimator = "visible_shape_rgbd"
                    except (DegenerateRegistrationError, ValueError) as shape_error:
                        pose = None
                        failure = f"appearance: {match_error}; visible shape: {shape_error}"
            except (ValueError, IndexError) as exc:
                pose = None
                failure = str(exc)
            if pose is None:
                self.last_failure[role] = failure
                estimates.append(
                    PoseEstimate(
                        role,
                        None,
                        frame.source_time_s,
                        available_s,
                        Evidence.LOST,
                        model.reference_version,
                        (False, False, False),
                        (False, False, False),
                    )
                )
            else:
                self._last_accepted_pose[role] = pose.copy()
                estimates.append(
                    PoseEstimate(
                        role,
                        pose,
                        frame.source_time_s,
                        available_s,
                        Evidence.OBSERVED,
                        model.reference_version,
                        (True, True, True),
                        (True, True, True),
                        direct_translation_observed=True,
                        estimator=estimator,
                        last_direct_observation_s=frame.source_time_s,
                    )
                )
        return tuple(estimates)

    def _fit(self, frame: RGBDFrame, model: ObjectPointModel, match: PixelMatches) -> np.ndarray:
        if match.reference_version != model.reference_version:
            raise ValueError("matcher reference version differs from object model")
        raw_indices = np.asarray(match.model_indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("matcher model indices must be integers")
        indices = raw_indices.astype(int)
        pixels = np.asarray(match.live_xy, dtype=float)
        if indices.ndim != 1 or pixels.shape != (len(indices), 2):
            raise ValueError("matcher indices and pixels must have matching lengths")
        if (
            not np.isfinite(pixels).all()
            or np.any(indices < 0)
            or np.any(indices >= len(model.points_object_m))
        ):
            raise ValueError("matcher returned non-finite pixels or invalid model indices")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("matcher repeated a model point")
        if len(indices) < self.min_inliers:
            raise DegenerateRegistrationError(
                f"too few pixel correspondences ({len(indices)}/{self.min_inliers})"
            )
        valid = valid_depth_pixels(frame.depth, pixels)
        if int(valid.sum()) < self.min_inliers:
            raise DegenerateRegistrationError(
                f"too few correspondences with aligned depth ({int(valid.sum())}/{len(indices)})"
            )
        source = model.points_object_m[indices[valid]]
        camera_points = unproject_pixels(frame.depth, frame.intrinsics, pixels[valid], frame.depth_m_per_unit)
        base_points = transform_points(frame.base_from_camera, camera_points)
        result = register_object(
            source,
            base_points,
            threshold_m=self.threshold_m,
            min_inliers=self.min_inliers,
            trials=self.trials,
        )
        return result.base_from_demo_object
