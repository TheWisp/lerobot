"""Registered measured RGB-D surfaces for a prepared object reference.

Each view stays separate, with the object-frame transform and provenance in
the matching ``ObjectReference.snippets`` entry. Registration is supplied by
the annotation/preparation path; this builder does not claim to estimate it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
from scipy.spatial import cKDTree

from .appearance import ReferenceAppearanceFrame
from .perception import ObjectReference
from .pose_evidence import SurfaceAgreement
from .registration import transform_points
from .surface_observation import SurfaceObservationConfig, move, observation, spatial_sample
from .surface_tracking import reference_depth_directional_support


@dataclass(frozen=True)
class RegisteredSurfaceModel:
    role: str
    reference_version: str
    view_points_object_m: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not self.role or not self.reference_version or not self.view_points_object_m:
            raise ValueError("registered surface needs an object identity and views")
        views = tuple(np.asarray(points, dtype=float).copy() for points in self.view_points_object_m)
        if any(
            points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all()
            for points in views
        ):
            raise ValueError("registered surface views need finite metric Nx3 points")
        object.__setattr__(self, "view_points_object_m", views)


@dataclass(frozen=True)
class PreparedSurfaceAgreement:
    """Object-frame search trees reused across camera frames."""

    model: RegisteredSurfaceModel
    first_tree: cKDTree
    all_tree: cKDTree
    all_points: np.ndarray

    @classmethod
    def build(cls, model: RegisteredSurfaceModel) -> PreparedSurfaceAgreement:
        first = model.view_points_object_m[0]
        all_points = (
            first if len(model.view_points_object_m) == 1 else np.concatenate(model.view_points_object_m)
        )
        first_tree = cKDTree(first)
        return cls(
            model,
            first_tree,
            first_tree if all_points is first else cKDTree(all_points),
            all_points,
        )


def registered_surface_agreement(
    model: RegisteredSurfaceModel,
    camera_from_object: np.ndarray,
    measured_camera_points: np.ndarray,
    clean_depth_pixels: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    *,
    prepared: PreparedSurfaceAgreement | None = None,
) -> tuple[SurfaceAgreement, SurfaceAgreement]:
    """Score the first view and all registered views against one live RGB-D frame.

    A low measured-to-reference distance can come from a tiny matching patch.
    The reverse depth comparison exposes a projected surface that disagrees
    with measured depth. A reference surface behind the measurement may be
    hidden or misplaced; neither direction alone establishes the correct 6D
    pose.
    """
    pose = np.asarray(camera_from_object, dtype=float)
    measured = np.asarray(measured_camera_points, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or measured.ndim != 2 or measured.shape[1] != 3:
        raise ValueError("surface agreement needs a finite camera pose and Nx3 measured points")
    if len(measured) < 3 or not np.isfinite(measured).all():
        raise ValueError("surface agreement needs at least three finite measured points")
    prepared = prepared or PreparedSurfaceAgreement.build(model)
    if prepared.model is not model:
        raise ValueError("prepared surface agreement belongs to another model")
    measured_object = move(measured, np.linalg.inv(pose))
    first = move(model.view_points_object_m[0], pose)
    all_views = (
        first if prepared.all_points is model.view_points_object_m[0] else move(prepared.all_points, pose)
    )

    def score(reference: np.ndarray, tree: cKDTree) -> SurfaceAgreement:
        distance = tree.query(measured_object)[0]
        support, depth_p90_mm, front_p90_mm, behind_p90_mm = reference_depth_directional_support(
            reference, clean_depth_pixels, depth_m, intrinsics
        )
        return SurfaceAgreement(
            float(np.percentile(distance, 90)),
            support,
            depth_p90_mm / 1000 if depth_p90_mm is not None else None,
            front_p90_mm / 1000 if front_p90_mm is not None else None,
            behind_p90_mm / 1000 if behind_p90_mm is not None else None,
        )

    first_score = score(first, prepared.first_tree)
    return first_score, first_score if all_views is first else score(all_views, prepared.all_tree)


def build_registered_surface_model(
    reference: ObjectReference,
    views: tuple[ReferenceAppearanceFrame, ...],
    *,
    observation_config: SurfaceObservationConfig,
    max_points_per_view: int,
) -> RegisteredSurfaceModel:
    """Lift every declared mask/depth view into the reference's object frame.

    No view is silently dropped: a missing mask or insufficient reliable depth
    must be fixed in preparation. The per-view cap bounds stored geometry while
    retaining spatial coverage. The model is still partial, not a CAD mesh.
    """
    if (
        len(views) != len(reference.snippets)
        or isinstance(max_points_per_view, bool)
        or not isinstance(max_points_per_view, Integral)
        or max_points_per_view < 3
    ):
        raise ValueError("surface views must match the reference and have a positive point budget")
    stored = []
    for snippet, view in zip(reference.snippets, views, strict=True):
        observed = view.snippet
        if (
            (observed.repo_id, observed.episode, observed.frame_index, observed.object_role)
            != (snippet.repo_id, snippet.episode, snippet.frame_index, snippet.object_role)
            or not np.array_equal(observed.object_from_camera, snippet.object_from_camera)
            or view.frame.frame_id != snippet.frame_index
            or view.frame.calibration_version != snippet.calibration_version
            or view.mask.shape != view.frame.depth.shape
            or view.mask.dtype != bool
            or not view.mask.any()
        ):
            raise ValueError("surface view does not match its registered RGB-D snippet")
        measured = observation(
            view.frame.metric_depth(), view.mask, view.frame.intrinsics, asdict(observation_config)
        )["points"]
        if len(measured) < 3:
            raise ValueError(f"{reference.role}: reference frame {snippet.frame_index} lacks reliable depth")
        points = transform_points(snippet.object_from_camera, measured)
        stored.append(points[spatial_sample(points, max_points_per_view)])
    return RegisteredSurfaceModel(reference.role, reference.version, tuple(stored))
