"""Candidate registration of a new RGB-D view to an existing object frame.

The input is one saved partial surface and one later measured view. A prepared
pose trace is not required. Geometry alone may leave several object-frame
assignments plausible, so this module reports candidates rather than promoting
one into the active reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .geometry import transform
from .reference_view_candidates import MeasuredReferenceView
from .surface_grounding import PreparedReferenceSurface, ShapeConfig, WholeSurfaceFit, pca_candidates
from .surface_observation import move
from .surface_reference import PreparedSurfaceAgreement, RegisteredSurfaceModel, registered_surface_agreement


@dataclass(frozen=True)
class ReferenceViewRegistrationCandidate:
    seed: str
    camera_from_object: np.ndarray
    converged: bool
    fit_cost: float
    evaluations: int
    measured_to_reference_p90_m: float
    reference_depth_pixels: int
    reference_to_depth_p90_m: float | None
    reference_in_front_p90_m: float | None
    reference_behind_p90_m: float | None
    surface_center_camera_m: tuple[float, float, float]


def fit_reference_view_candidates(
    reference_points_object_m: np.ndarray,
    measured: MeasuredReferenceView,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    config: ShapeConfig,
    *,
    prior_poses_camera_from_object: Mapping[str, np.ndarray] | None = None,
) -> tuple[ReferenceViewRegistrationCandidate, ...]:
    """Refine independent geometry seeds and an optional explicitly named prior.

    Scores compare candidates only within this measured frame. A low fit cost
    or surface distance is not a certificate of landmark correspondence.
    """
    reference = np.asarray(reference_points_object_m, dtype=float)
    mask = np.asarray(mask)
    intrinsics = np.asarray(intrinsics, dtype=float)
    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or len(reference) < 3
        or not np.isfinite(reference).all()
        or mask.shape != measured.depth_m.shape
        or mask.dtype != bool
        or intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or len(measured.points_camera_m) < 3
    ):
        raise ValueError("view registration needs a finite saved surface and aligned measured RGB-D view")
    if not mask.any():
        raise ValueError("view registration needs an object mask")
    prepared = PreparedReferenceSurface.build(reference, config)
    fit = WholeSurfaceFit(
        reference,
        {"mask": mask, "depth": measured.depth_m, "obs": {"points": measured.points_camera_m}},
        intrinsics,
        config,
        prepared_reference=prepared,
    )
    seeds = [
        (f"geometry_{index}", seed)
        for index, seed in enumerate(pca_candidates(reference, measured.points_camera_m))
    ]
    for name, pose in (prior_poses_camera_from_object or {}).items():
        if not name or name.startswith("geometry_"):
            raise ValueError("prior seeds need distinct names")
        seeds.append((name, transform(pose)))
    model = RegisteredSurfaceModel("candidate", "candidate", (reference,))
    prepared_agreement = PreparedSurfaceAgreement.build(model)
    results = []
    for name, seed in seeds:
        pose, detail = fit.refine(seed)
        score, _ = registered_surface_agreement(
            model,
            pose,
            measured.points_camera_m,
            measured.clean_depth_pixels,
            measured.depth_m,
            intrinsics,
            prepared=prepared_agreement,
        )
        center = move(reference, pose).mean(axis=0)
        results.append(
            ReferenceViewRegistrationCandidate(
                name,
                pose,
                detail["converged"],
                float(detail["cost"]),
                int(detail["evaluations"]),
                score.measured_to_reference_p90_m,
                score.reference_depth_pixels,
                score.reference_to_depth_p90_m,
                score.reference_in_front_p90_m,
                score.reference_behind_p90_m,
                tuple(float(value) for value in center),
            )
        )
    return tuple(sorted(results, key=lambda item: item.fit_cost))
