"""Evaluate motion-constrained surface completion on a labeled RGB-D case.

The annotations score object motion; they never enter the tracker or map.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from flash_surface_motion_trace import MotionConstrainedFit, angle_deg
from scipy.spatial import cKDTree

from lerobot.policies.flash.experimental_surface_map import CandidateSurfaceMap, SurfaceMapConfig
from lerobot.policies.flash.surface_correspondence import (
    SurfaceMotionConfig,
    crop_pair,
    robust_fit,
    ufm_pairs,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, observation
from lerobot.policies.flash.surface_tracking import load_ufm


def run(case: Path) -> list[dict]:
    manifest = json.loads((case / "manifest.json").read_text())
    source = Path(manifest["source"])
    names = sorted(path.stem for path in (source / "annotated_poses").glob("*.txt"))
    names = names[manifest["source_first_index"] : manifest["source_last_index"] + 1]
    ground_truth = [np.loadtxt(source / "annotated_poses" / f"{name}.txt") for name in names]
    intrinsics = np.loadtxt(case / "recording" / "cam_K.txt").reshape(3, 3)
    observation_config, motion_config = SurfaceObservationConfig(), SurfaceMotionConfig()

    def frame(index):
        stem = f"{index:06d}"
        rgb = cv2.cvtColor(cv2.imread(str(case / "recording" / "rgb" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(case / "recording" / "depth" / f"{stem}.png"), -1).astype(np.float32) * 0.001
        mask = cv2.imread(str(case / "masks" / f"{stem}.png"), 0) > 0
        obs = observation(depth, mask, intrinsics, asdict(observation_config))
        return {"rgb": rgb, "depth": depth, "mask": mask, "obs": obs, "usable": obs["valid_image"]}

    previous = frame(0)
    reference = previous["obs"]["points"]
    surface = CandidateSurfaceMap(
        reference,
        SurfaceMapConfig(
            spacing_m=previous["obs"]["spacing_m"],
            novelty_distance_m=max(previous["obs"]["spacing_m"], ShapeConfig().depth_tolerance_m),
            max_points=15000,
        ),
        source_frame=0,
        reference_normals=previous["obs"]["normals"],
    )
    model = load_ufm()
    raw_pose = np.eye(4)
    constrained_pose = np.eye(4)
    result = []
    for index in range(1, len(names)):
        current = frame(index)
        first, second, crop = crop_pair(previous, current, motion_config)
        source_points, target_points = ufm_pairs(
            model, first, second, crop, previous, current, intrinsics, motion_config
        )
        step, inliers, residual = robust_fit(source_points, target_points, motion_config)
        if step is not None:
            raw_pose = step @ raw_pose
            errors = np.linalg.norm(move(source_points, step) - target_points, axis=1)
            keep = errors < motion_config.ransac_distance_m
            fitter = MotionConstrainedFit(
                surface.points,
                {**current, "all_surface_points": current["obs"]["points"]},
                intrinsics,
                ShapeConfig(),
                previous_pose=constrained_pose,
                source=source_points[keep],
                target=target_points[keep],
                tolerance_m=motion_config.ransac_distance_m,
            )
            constrained_pose, fit = fitter.correct(step @ constrained_pose)
        else:
            fit = None
        truth = ground_truth[index] @ np.linalg.inv(ground_truth[0])
        measured = current["obs"]["points"]
        row = {"frame": index, "inliers": inliers, "step_residual_mm": float(residual * 1000)}
        for name, pose in (("chain", raw_pose), ("constrained", constrained_pose)):
            row[f"{name}_angle_error_deg"] = angle_deg(pose, truth)
            row[f"{name}_point_error_mm"] = float(
                np.median(np.linalg.norm(move(reference, pose) - move(reference, truth), axis=1)) * 1000
            )
        row["map_p90_before_mm"] = float(
            np.percentile(cKDTree(move(surface.points, constrained_pose)).query(measured)[0], 90) * 1000
        )
        row["map_points_before"] = len(surface.points)
        if step is not None:
            row["map"] = surface.observe(
                measured,
                constrained_pose,
                index,
                measured_normals_camera=current["obs"]["normals"],
            )
        row["fit"] = fit
        result.append(row)
        previous = current
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = run(args.case)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
