"""Apply one measured 3D rigid estimator to every raw point-tracker output.

The fit receives first-frame depth seeds and predicted 3D point positions,
never an evaluation pose or mask. Annotated object motion is read only after
fitting to measure the transported source-cloud error.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from benchmarks.flash_surface_benchmark import _load, validate_prediction


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Proper-rotation Kabsch transform; three non-collinear points suffice."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("rigid fit needs at least three corresponding 3D points")
    centered_source = source - source.mean(axis=0)
    centered_target = target - target.mean(axis=0)
    # A collinear sample leaves rotation about its line unobservable.
    if np.linalg.svd(centered_source, compute_uv=False)[1] <= np.finfo(float).eps:
        return None
    left, _, right = np.linalg.svd(centered_source.T @ centered_target)
    correction = np.diag([1.0, 1.0, np.linalg.det(right.T @ left.T)])
    rotation = right.T @ correction @ left.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target.mean(axis=0) - rotation @ source.mean(axis=0)
    return transform


def robust_fit(
    source: np.ndarray,
    target: np.ndarray,
    residual_m: float,
    success_probability: float,
    prior_inlier_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """RANSAC sample count follows its 3-point success probability formula."""
    if not (residual_m > 0 and 0 < success_probability < 1 and 0 < prior_inlier_fraction < 1):
        raise ValueError("invalid rigid-fit confidence or residual parameters")
    if source.shape != target.shape or len(source) < 3:
        return None, np.zeros(len(source), bool), 0
    trials = math.ceil(math.log1p(-success_probability) / math.log1p(-(prior_inlier_fraction**3)))
    best = np.zeros(len(source), bool)
    best_median = float("inf")
    for _ in range(trials):
        candidate = rigid_transform(source[ids := rng.choice(len(source), 3, replace=False)], target[ids])
        if candidate is None:
            continue
        error = np.linalg.norm((candidate[:3, :3] @ source.T).T + candidate[:3, 3] - target, axis=1)
        inliers = error <= residual_m
        if inliers.sum() < 3:
            continue
        median = float(np.median(error[inliers]))
        if inliers.sum() > best.sum() or (inliers.sum() == best.sum() and median < best_median):
            best, best_median = inliers, median
    if best.sum() < 3:
        return None, best, trials
    transform = rigid_transform(source[best], target[best])
    if transform is None:
        return None, best, trials
    error = np.linalg.norm((transform[:3, :3] @ source.T).T + transform[:3, 3] - target, axis=1)
    refined = error <= residual_m
    if refined.sum() >= 3:
        updated = rigid_transform(source[refined], target[refined])
        if updated is not None:
            return updated, refined, trials
    return transform, best, trials


def run(
    experiment: Path,
    prediction_path: Path,
    output: Path,
    residual_mm: float = 10.0,
    success_probability: float = 0.99,
    prior_inlier_fraction: float = 0.5,
) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    if not str(manifest.get("ground_truth", "")).startswith("YCBInEOAT annotated object poses"):
        raise ValueError("rigid motion evaluation currently requires annotated YCBInEOAT poses")
    reference, prediction = _load(experiment / "queries.npz"), _load(prediction_path)
    validate_prediction(reference, prediction)
    if "xyz" not in prediction:
        raise ValueError("predicted 3D positions required")
    if np.any(reference["queries"][:, 0] != 0):
        raise ValueError("common first-frame fit requires queries with the same material anchor frame")
    k, queries, z = reference["intrinsics"], reference["queries"], reference["initial_depth_m"]
    source = np.column_stack(
        ((queries[:, 1] - k[0, 2]) * z / k[0, 0], (queries[:, 2] - k[1, 2]) * z / k[1, 1], z)
    )
    pose_paths = sorted((Path(manifest["source"]) / "annotated_poses").glob("*.txt"))
    object_poses = np.stack(
        [
            np.loadtxt(pose_paths[manifest["source_first_index"] + int(index)]).reshape(4, 4)
            for index in reference["frames"]
        ]
    )
    relative_truth = object_poses @ np.linalg.inv(object_poses[0])
    t = len(reference["frames"])
    transforms = np.full((t, 4, 4), np.nan)
    available = np.zeros(t, bool)
    inlier_count, usable_count, fit_s = (np.zeros(t) for _ in range(3))
    cloud_error_mm = np.full(t, np.nan)
    angular_error_deg = np.full(t, np.nan)
    rng = np.random.default_rng(20261023)
    for index in range(t):
        usable = prediction["visible"][index] & np.isfinite(prediction["xyz"][index]).all(axis=1)
        usable_count[index] = int(usable.sum())
        started = time.perf_counter()
        transform, inliers, trials = robust_fit(
            source[usable],
            prediction["xyz"][index, usable],
            residual_mm * 0.001,
            success_probability,
            prior_inlier_fraction,
            rng,
        )
        fit_s[index] = time.perf_counter() - started
        inlier_count[index] = int(inliers.sum())
        if transform is None:
            continue
        available[index] = True
        transforms[index] = transform
        estimated = (transform[:3, :3] @ source.T).T + transform[:3, 3]
        expected = (relative_truth[index, :3, :3] @ source.T).T + relative_truth[index, :3, 3]
        cloud_error_mm[index] = np.median(np.linalg.norm(estimated - expected, axis=1)) * 1000
        relative_rotation = transform[:3, :3] @ relative_truth[index, :3, :3].T
        angular_error_deg[index] = np.rad2deg(
            np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=reference["frames"],
        transforms=transforms,
        available=available,
        inlier_count=inlier_count,
        usable_count=usable_count,
        fit_s=fit_s,
        cloud_error_mm=cloud_error_mm,
        angular_error_deg=angular_error_deg,
    )
    report = {
        "model": prediction_path.stem,
        "clip": experiment.name,
        "residual_mm": residual_mm,
        "success_probability": success_probability,
        "prior_inlier_fraction": prior_inlier_fraction,
        "ransac_trials_per_frame": trials,
        "available_frames": int(available.sum()),
        "total_frames": t,
        "cloud_error_mm_median_when_available": float(np.median(cloud_error_mm[available]))
        if available.any()
        else None,
        "cloud_error_mm_p90_when_available": float(np.quantile(cloud_error_mm[available], 0.9))
        if available.any()
        else None,
        "angular_error_deg_median_when_available": float(np.median(angular_error_deg[available]))
        if available.any()
        else None,
        "fit_ms_median": float(np.median(fit_s) * 1000),
        "fit_ms_p90": float(np.quantile(fit_s, 0.9) * 1000),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-mm", type=float, default=10.0)
    parser.add_argument("--success-probability", type=float, default=0.99)
    parser.add_argument("--prior-inlier-fraction", type=float, default=0.5)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.experiment,
                args.prediction,
                args.output,
                args.residual_mm,
                args.success_probability,
                args.prior_inlier_fraction,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
