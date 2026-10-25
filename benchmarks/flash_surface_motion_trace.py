"""Compare consecutive RGB-D motion with the fixed-surface pose on a saved trace.

This is a diagnostic, not a pose backend. Chaining pairwise matches can drift;
it is useful here to see whether fixed-surface correction erases a real turn.
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
from lerobot.policies.flash.surface_correspondence import (
    SurfaceMotionConfig,
    crop_pair,
    robust_fit,
    ufm_pairs,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, observation
from lerobot.policies.flash.surface_tracking import load_ufm


def load(trace: Path, row: dict, role: str, config: SurfaceObservationConfig) -> dict:
    stem = f"{row['frame_id']:06d}.png"
    rgb = cv2.cvtColor(cv2.imread(str(trace / "rgb" / stem)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(trace / "depth" / stem), -1).astype(np.float32) * row["depth_m_per_unit"]
    mask = cv2.imread(str(trace / "masks" / role / stem), 0) > 0
    k = np.asarray(row["intrinsics"])
    obs = observation(depth, mask, k, asdict(config))
    return {"rgb": rgb, "depth": depth, "mask": mask, "obs": obs, "usable": obs["valid_image"]}


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))


def score(reference: np.ndarray, measured: np.ndarray, pose: np.ndarray) -> float:
    return float(np.percentile(cKDTree(move(reference, pose)).query(measured)[0], 90) * 1000)


class MotionConstrainedFit(WholeSurfaceFit):
    """Ablate equal-weight pairwise correspondence evidence during shape fit."""

    def __init__(self, reference, current, k, config, *, previous_pose, source, target, tolerance_m):
        super().__init__(reference, current, k, config)
        self.previous_inverse = np.linalg.inv(previous_pose)
        self.source = source
        self.correspondence_target = target
        self.tolerance_m = tolerance_m

    def residual(self, pose):
        surface = super().residual(pose)
        step = pose @ self.previous_inverse
        motion = (move(self.source, step) - self.correspondence_target).ravel()
        return np.concatenate((surface, motion / (self.tolerance_m * np.sqrt(len(motion)))))


def run(
    trace: Path,
    *,
    role: str,
    first: int,
    last: int,
    max_map_points: int,
    map_snapshot_frame: int | None = None,
    map_snapshot_output: Path | None = None,
    motion_constrained_fit: bool = False,
) -> list[dict]:
    rows = [json.loads(line) for line in (trace / "frames.jsonl").read_text().splitlines()]
    rows = [row for row in rows if first <= row["frame_id"] <= last]
    if len(rows) < 2:
        raise ValueError("selected interval needs at least two saved frames")
    observation_config, motion_config = SurfaceObservationConfig(), SurfaceMotionConfig()
    previous = load(trace, rows[0], role, observation_config)
    reference = previous["obs"]["points"]
    spacing = previous["obs"]["spacing_m"]
    surface = CandidateSurfaceMap(
        reference,
        SurfaceMapConfig(
            spacing_m=spacing,
            novelty_distance_m=max(spacing, ShapeConfig().depth_tolerance_m),
            max_points=max_map_points,
        ),
        source_frame=rows[0]["frame_id"],
        reference_normals=previous["obs"]["normals"],
    )
    first_live = np.asarray(rows[0]["objects"][role]["camera_from_reference"])
    pose = np.eye(4)
    constrained_pose = np.eye(4)
    previous_live = np.eye(4)
    model = load_ufm()
    result = []
    for row in rows[1:]:
        current = load(trace, row, role, observation_config)
        rgb_a, rgb_b, crop = crop_pair(previous, current, motion_config)
        source, target = ufm_pairs(
            model, rgb_a, rgb_b, crop, previous, current, np.asarray(row["intrinsics"]), motion_config
        )
        step, inliers, residual = robust_fit(source, target, motion_config)
        if step is not None:
            pose = step @ pose
        live = np.asarray(row["objects"][role]["camera_from_reference"]) @ np.linalg.inv(first_live)
        live_step = live @ np.linalg.inv(previous_live)
        measured = current["obs"]["points"]
        item = {
            "frame_id": row["frame_id"],
            "source_time_s": row["source_time_s"],
            "step_available": step is not None,
            "step_inliers": inliers,
            "step_residual_mm": float(residual * 1000) if np.isfinite(residual) else None,
            "step_rotation_deg": angle_deg(step, np.eye(4)) if step is not None else None,
            "chain_rotation_deg": angle_deg(pose, np.eye(4)),
            "live_rotation_deg": angle_deg(live, np.eye(4)),
            "chain_vs_live_rotation_deg": angle_deg(pose, live),
            "chain_pose": pose.tolist(),
            "live_pose": live.tolist(),
            "chain_fixed_p90_mm": score(reference, measured, pose),
            "chain_map_p90_mm": score(surface.points, measured, pose),
            "live_fixed_p90_mm": score(reference, measured, live),
            "live_accepted": row["objects"][role]["accepted"],
        }
        if step is not None:
            pair_errors = np.linalg.norm(move(source, step) - target, axis=1)
            inlier_pairs = pair_errors < motion_config.ransac_distance_m
            final_errors = np.linalg.norm(
                move(source[inlier_pairs], live_step) - target[inlier_pairs], axis=1
            )
            item["live_match_residual_median_mm"] = float(np.median(final_errors) * 1000)
            item["live_match_residual_p90_mm"] = float(np.percentile(final_errors, 90) * 1000)
            if motion_constrained_fit:
                current_for_fit = {**current, "all_surface_points": measured}
                fitter = MotionConstrainedFit(
                    reference,
                    current_for_fit,
                    np.asarray(row["intrinsics"]),
                    ShapeConfig(),
                    previous_pose=constrained_pose,
                    source=source[inlier_pairs],
                    target=target[inlier_pairs],
                    tolerance_m=motion_config.ransac_distance_m,
                )
                constrained_pose, fit_detail = fitter.correct(step @ constrained_pose)
                item["constrained_pose"] = constrained_pose.tolist()
                item["constrained_rotation_deg"] = angle_deg(constrained_pose, np.eye(4))
                item["constrained_vs_live_deg"] = angle_deg(constrained_pose, live)
                item["constrained_fixed_p90_mm"] = score(reference, measured, constrained_pose)
                item["constrained_fit"] = fit_detail
        # This intentionally uses the pairwise chain, never the fixed-surface
        # answer. It is still self-consistency evidence, not ground truth.
        if step is not None:
            item["map"] = surface.observe(
                measured,
                pose,
                row["frame_id"],
                measured_normals_camera=current["obs"]["normals"],
            )
        if row["frame_id"] == map_snapshot_frame and map_snapshot_output is not None:
            np.savez_compressed(map_snapshot_output, points=surface.points, pose=pose)
        result.append(item)
        previous = current
        previous_live = live
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--role", default="object")
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--max-map-points", type=int, default=8000)
    parser.add_argument("--map-snapshot-frame", type=int)
    parser.add_argument("--map-snapshot-output", type=Path)
    parser.add_argument("--motion-constrained-fit", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = run(
        args.trace,
        role=args.role,
        first=args.first,
        last=args.last,
        max_map_points=args.max_map_points,
        map_snapshot_frame=args.map_snapshot_frame,
        map_snapshot_output=args.map_snapshot_output,
        motion_constrained_fit=args.motion_constrained_fit,
    )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
