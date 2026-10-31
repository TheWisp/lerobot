"""Review measured views before adding them to an object's reference.

The prepared pose trace is only a registration hypothesis. This module never
promotes a candidate or treats cross-frame agreement as independent truth.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np

from .geometry import inverse
from .pose_evidence import SurfaceAgreement
from .surface_observation import move, spatial_sample
from .surface_reference import PreparedSurfaceAgreement, RegisteredSurfaceModel, registered_surface_agreement
from .trajectory import Evidence, PoseEstimate


@dataclass(frozen=True)
class MeasuredReferenceView:
    """One aligned RGB-D observation, with cleaned points inside its object mask."""

    frame_id: int
    depth_m: np.ndarray
    clean_depth_pixels: np.ndarray
    points_camera_m: np.ndarray
    mask_pixels: int

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m)
        clean = np.asarray(self.clean_depth_pixels)
        points = np.asarray(self.points_camera_m)
        if (
            self.frame_id < 0
            or depth.ndim != 2
            or clean.shape != depth.shape
            or clean.dtype != bool
            or points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
            or self.mask_pixels < 0
        ):
            raise ValueError("reference view needs aligned depth, clean pixels and finite metric points")


def _agreement(agreement: SurfaceAgreement) -> dict:
    return {
        "measured_to_reference_p90_m": agreement.measured_to_reference_p90_m,
        "reference_depth_pixels": agreement.reference_depth_pixels,
        "reference_to_depth_p90_m": agreement.reference_to_depth_p90_m,
    }


def _summary(rows: list[dict]) -> dict:
    def middle(view: str, metric: str) -> float | None:
        values = [row[view][metric] for row in rows if row[view][metric] is not None]
        return float(median(values)) if values else None

    return {
        "frames": len(rows),
        "registration_source": "same_prepared_pose_trace",
        "first_view_measured_p90_median_m": middle("first_view", "measured_to_reference_p90_m"),
        "combined_views_measured_p90_median_m": middle("combined_views", "measured_to_reference_p90_m"),
        "first_view_reverse_p90_median_m": middle("first_view", "reference_to_depth_p90_m"),
        "combined_views_reverse_p90_median_m": middle("combined_views", "reference_to_depth_p90_m"),
        "measured_gap_improved_frames": sum(
            row["combined_views"]["measured_to_reference_p90_m"]
            < row["first_view"]["measured_to_reference_p90_m"]
            for row in rows
        ),
    }


def review_reference_view_candidates(
    frame_estimates: Mapping[int, Mapping[str, PoseEstimate]],
    surface_models: Sequence[RegisteredSurfaceModel],
    candidate_frames_by_role: Mapping[str, Sequence[int]],
    measured_view: Callable[[str, int], MeasuredReferenceView | None],
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    max_points_per_view: int,
) -> tuple[list[dict], list[dict]]:
    """Compare first-view geometry with first-plus-candidate geometry.

    Both registration and cross-frame scoring use the supplied pose trace.
    Scores detect internal contradictions but cannot prove pose accuracy. A
    candidate remains unregistered for execution until separately reviewed.
    """
    models = {model.role: model for model in surface_models}
    if (
        len(models) != len(surface_models)
        or set(candidate_frames_by_role) - set(models)
        or isinstance(max_points_per_view, bool)
        or not isinstance(max_points_per_view, int)
        or max_points_per_view < 3
    ):
        raise ValueError("reference-view candidates need known roles and a positive point budget")
    intrinsics = np.asarray(intrinsics, dtype=float)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("reference-view review needs finite camera intrinsics")
    camera_from_base = inverse(base_from_camera)
    summaries, details = [], []
    cache: dict[tuple[str, int], MeasuredReferenceView | None] = {}

    def measured(role: str, index: int) -> MeasuredReferenceView | None:
        key = role, index
        if key not in cache:
            frame = measured_view(role, index)
            if frame is not None and frame.frame_id != index:
                raise ValueError("reference-view measurement has a different source frame")
            cache[key] = frame
        return cache[key]

    for role, indices in candidate_frames_by_role.items():
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index not in frame_estimates
            for index in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError(f"{role}: candidate frames must be distinct source frames in this demo")
        first = models[role]
        if len(first.view_points_object_m) != 1:
            raise ValueError("view-candidate review currently needs a single active reference view")
        for index in indices:
            estimate = frame_estimates[index].get(role)
            source = measured(role, index)
            item = {
                "role": role,
                "frame_index": index,
                "status": "unregistered",
                "registration_source": "same_prepared_pose_trace",
                "pose_estimator": estimate.estimator if estimate else None,
                "pose_evidence": estimate.evidence.value if estimate else None,
                "mask_pixels": source.mask_pixels if source else 0,
                "clean_depth_points": len(source.points_camera_m) if source else 0,
                "object_from_camera": None,
                "cross_frame": _summary([]),
            }
            rows = []
            if source is None:
                item["reason"] = "object mask missing at selected frame"
            elif len(source.points_camera_m) < 3:
                item["reason"] = "selected frame has fewer than three clean depth points"
            elif (
                estimate is None
                or estimate.evidence != Evidence.OBSERVED
                or not estimate.supports()
                or estimate.pose is None
            ):
                item["reason"] = "selected frame lacks an observed, fully supported object pose"
            elif estimate.source_frame_id is not None and estimate.source_frame_id != index:
                raise ValueError("reference-view pose has a different source frame")
            else:
                object_from_camera = inverse(estimate.pose) @ base_from_camera
                points = move(source.points_camera_m, object_from_camera)
                points = points[spatial_sample(points, max_points_per_view)]
                combined = RegisteredSurfaceModel(
                    role, first.reference_version, (first.view_points_object_m[0], points)
                )
                prepared = PreparedSurfaceAgreement.build(combined)
                for other, estimates in sorted(frame_estimates.items()):
                    if other == index:
                        continue
                    other_estimate = estimates.get(role)
                    if (
                        other_estimate is None
                        or other_estimate.evidence != Evidence.OBSERVED
                        or not other_estimate.supports()
                        or other_estimate.pose is None
                    ):
                        continue
                    if other_estimate.source_frame_id is not None and other_estimate.source_frame_id != other:
                        raise ValueError("reference-view pose has a different source frame")
                    frame = measured(role, other)
                    if frame is None or len(frame.points_camera_m) < 3:
                        continue
                    first_score, combined_score = registered_surface_agreement(
                        combined,
                        camera_from_base @ other_estimate.pose,
                        frame.points_camera_m,
                        frame.clean_depth_pixels,
                        frame.depth_m,
                        intrinsics,
                        prepared=prepared,
                    )
                    rows.append(
                        {
                            "frame_index": other,
                            "first_view": _agreement(first_score),
                            "combined_views": _agreement(combined_score),
                        }
                    )
                item["status"] = "registration_hypothesis"
                item["object_from_camera"] = object_from_camera.tolist()
                item["cross_frame"] = _summary(rows)
            summaries.append(item)
            details.append({"role": role, "frame_index": index, "comparisons": rows})
    return summaries, details
