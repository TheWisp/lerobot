"""Isolate the value of a causal multiview surface map using annotated poses.

This is an *oracle map registration* ablation, not a deployable pose tracker:
the annotation places each preceding measured view in the initial frame. The
current frame never enters the map until after it is scored. Both fixed and
growing fits receive the same annotated current-pose proposal.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_surface_map import CandidateSurfaceMap, SurfaceMapConfig
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import (
    SurfaceObservationConfig,
    move,
    observation,
    voxel_sample,
)


def score_pose(reference: np.ndarray, candidate: np.ndarray, truth: np.ndarray) -> dict:
    distance = np.linalg.norm(move(reference, candidate) - move(reference, truth), axis=1)
    angle = Rotation.from_matrix(candidate[:3, :3] @ truth[:3, :3].T).magnitude()
    return {
        "surface_point_error_mm": float(np.median(distance) * 1000),
        "rotation_error_deg": float(np.degrees(angle)),
    }


def run(
    case: Path, *, fit_every: int, spacing_m: float, max_map_points: int, admission: str, confirmations: int
) -> list[dict]:
    manifest = json.loads((case / "manifest.json").read_text())
    source = Path(manifest["source"])
    names = sorted(path.stem for path in (source / "annotated_poses").glob("*.txt"))
    names = names[manifest["source_first_index"] : manifest["source_last_index"] + 1]
    poses = [np.loadtxt(source / "annotated_poses" / f"{name}.txt") for name in names]
    k = np.loadtxt(case / "recording" / "cam_K.txt").reshape(3, 3)
    observations = []
    for index in range(len(names)):
        depth = (
            cv2.imread(str(case / "recording" / "depth" / f"{index:06d}.png"), -1).astype(np.float32) * 0.001
        )
        mask = cv2.imread(str(case / "masks" / f"{index:06d}.png"), 0) > 0
        obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
        observations.append({"depth": depth, "mask": mask, "obs": obs, "all_surface_points": obs["points"]})
    reference = observations[0]["obs"]["points"]
    model = reference.copy()
    candidate = (
        CandidateSurfaceMap(
            reference,
            SurfaceMapConfig(
                spacing_m=spacing_m,
                novelty_distance_m=max(spacing_m, ShapeConfig().depth_tolerance_m),
                confirmations=confirmations,
                max_points=max_map_points,
            ),
            source_frame=0,
            reference_normals=observations[0]["obs"]["normals"],
        )
        if admission == "confirmed"
        else None
    )
    rows = []
    for index, (current, camera_from_object) in enumerate(zip(observations, poses, strict=True)):
        truth = camera_from_object @ np.linalg.inv(poses[0])
        measured = current["obs"]["points"]
        fixed = move(reference, truth)
        grown = move(model, truth)
        fixed_distances = cKDTree(fixed).query(measured)[0]
        grown_distances = cKDTree(grown).query(measured)[0]
        row = {
            "frame": index,
            "map_points_before": len(model),
            "measured_points": len(measured),
            "fixed_surface_p90_mm": float(np.percentile(fixed_distances, 90) * 1000),
            "earlier_views_p90_mm": float(np.percentile(grown_distances, 90) * 1000),
        }
        if index and (index % fit_every == 0 or index == len(poses) - 1):
            shape = ShapeConfig()
            for name, points in (("fixed_fit", reference), ("earlier_views_fit", model)):
                fitter = WholeSurfaceFit(points, current, k, shape)
                local, local_detail = fitter.refine(truth)
                fitted, detail = fitter.correct(truth)
                row[name] = {
                    **score_pose(reference, fitted, truth),
                    "cost": detail["cost"],
                    "cost_at_annotation": fitter.score(truth),
                    "used_coarse": detail["used_coarse"],
                    "local_only": score_pose(reference, local, truth),
                    "local_cost": local_detail["cost"],
                }
        rows.append(row)
        # Only earlier observations are available to score the next frame.
        if candidate is not None:
            if index:
                row["admission"] = candidate.observe(
                    measured,
                    truth,
                    index,
                    measured_normals_camera=current["obs"]["normals"],
                )
                model = candidate.points
        else:
            local = move(measured, np.linalg.inv(truth))
            model = np.concatenate((model, local))
            model = model[voxel_sample(model, spacing_m)]
            if len(model) > max_map_points:
                from lerobot.policies.flash.surface_observation import spatial_sample

                model = model[spatial_sample(model, max_map_points)]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-every", type=int, default=5)
    parser.add_argument("--spacing-m", type=float, default=0.003)
    parser.add_argument("--max-map-points", type=int, default=15000)
    parser.add_argument("--admission", choices=("all", "confirmed"), default="all")
    parser.add_argument("--confirmations", type=int, default=3)
    args = parser.parse_args()
    if args.fit_every < 1 or args.spacing_m <= 0 or args.max_map_points < 3:
        parser.error("fit-every, spacing-m and max-map-points must be positive")
    cv2.setNumThreads(1)
    rows = run(
        args.case,
        fit_every=args.fit_every,
        spacing_m=args.spacing_m,
        max_map_points=args.max_map_points,
        admission=args.admission,
        confirmations=args.confirmations,
    )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
