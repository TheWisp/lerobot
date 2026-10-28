"""Registered measured RGB-D surfaces for a prepared object reference.

Each view stays separate, with the object-frame transform and provenance in
the matching ``ObjectReference.snippets`` entry. Registration is supplied by
the annotation/preparation path; this builder does not claim to estimate it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np

from .appearance import ReferenceAppearanceFrame
from .perception import ObjectReference
from .registration import transform_points
from .surface_observation import SurfaceObservationConfig, observation, spatial_sample


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
