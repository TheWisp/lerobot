"""Evaluation-only distance from a reconstructed partial surface to a labelled model.

Nearest-surface agreement deliberately ignores point identity (and hence geometric
symmetries). It supplements pose error; a small matching patch cannot certify the
pose. The comparison is one-way because unobserved parts of a full model need not
exist in the reconstruction. Model sampling and label accuracy limit this metric.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class LabelledSurface:
    def __init__(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("Labelled surface must contain finite 3D points in metres")
        self.tree = cKDTree(points)

    def distances_mm(self, reference, camera_from_reference, camera_from_model):
        reference = np.asarray(reference, dtype=float)
        model_from_reference = np.linalg.inv(camera_from_model) @ camera_from_reference
        points = reference @ model_from_reference[:3, :3].T + model_from_reference[:3, 3]
        return self.tree.query(points)[0] * 1000


def distance_statistics(values):
    return {
        "points_scored": len(values),
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.quantile(values, 0.9)),
        "within_5mm_percent": float(np.mean(values <= 5) * 100),
        "within_10mm_percent": float(np.mean(values <= 10) * 100),
    }


def save_surface_version(run, method, role, track, details):
    """Retain only changed point clouds, outside the inference timer, for scoring."""
    if details["map_version_after"] != details["map_version_used"]:
        models = Path(run) / "surface_models" / method / role
        models.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(models / f"{track.store.version:06d}.npz", points=track.store.points)


def score_surface_trace(spec, truth, details, method, reference, *, run=None, role=None):
    """Score returned surfaces after initialization; no pose or model is altered."""
    result = {"frames_total": len(details) - 1, "frames_scored": 0}
    if truth is None:
        return {**result, "reason": "No reference poses"}
    if not spec:
        return {**result, "reason": "No full labelled surface model"}
    path = Path(spec["points_m"]).expanduser().resolve()
    if not path.is_file():
        return {**result, "reason": "Labelled surface model file unavailable"}
    surface = LabelledSurface(np.load(path, allow_pickle=False))
    distances = []
    model_version = None
    points = reference
    missing_surface_frames = 0
    for index, row in enumerate(details[1:], start=1):
        pose = row[method]
        if pose is None:
            continue
        version = row.get(method + "_details", {}).get("map_version_after", 0)
        if version:
            if run is None:
                missing_surface_frames += 1
                continue
            model_path = Path(run) / "surface_models" / method / role / f"{int(version):06d}.npz"
            if not model_path.is_file():
                missing_surface_frames += 1
                continue
            if version != model_version:
                with np.load(model_path, allow_pickle=False) as saved:
                    points = saved["points"].copy()
                model_version = version
        else:
            points = reference
            model_version = None
        if not len(points):
            missing_surface_frames += 1
            continue
        distances.append(surface.distances_mm(points, np.asarray(pose), truth[index]))
    result.update(
        frames_scored=len(distances),
        missing_surface_frames=missing_surface_frames,
        model_points=len(surface.tree.data),
        metric="partial_to_labelled_surface_v1",
    )
    if not distances:
        return {**result, "reason": "No saved surfaces with a returned pose after initialization"}
    values = np.concatenate(distances)
    return {**result, **distance_statistics(values)}
