"""Test whether past held-object RGB-D surfaces can repair attachment error.

Recorded end-effector motion supplies a known rigid-body prediction.  We inject
a configurable error into that prediction, then ask masked RGB-D registration
against earlier visible views to recover it.  This is an offline experiment;
the encoder motion is evaluation ground truth, not a claimed object ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from flash_geometry_goal import fit_metrics
from flash_geometry_sequence import camera_motion_from_ee, sampled_surface, transform_matrix
from flash_rgbd_registration import apply, trimmed_icp
from scipy.spatial.transform import Rotation


def parse_frames(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def run(args: argparse.Namespace) -> dict:
    k = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
    calibration = json.loads(args.calibration.read_text())
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=np.float64)
    table = pq.read_table(args.demo_data_parquet, columns=["frame_index", "tip.pose"])
    ee_by_frame = {
        int(frame): {"demo_ee": np.asarray(pose).reshape(4, 4)}
        for frame, pose in zip(table["frame_index"].to_pylist(), table["tip.pose"].to_pylist(), strict=True)
    }
    clouds = {
        frame: sampled_surface(
            args.demo_recording / "depth" / f"{frame:06}.png",
            args.demo_held_masks / f"{frame:06}.png",
            k,
            args.demo_depth_scale,
            args.mask_interior_px,
            args.max_points,
        )
        for frame in set(args.anchor_frames) | set(args.query_frames)
    }
    axis = np.asarray(args.perturb_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    offset = np.asarray(args.perturb_direction, dtype=float)
    offset /= np.linalg.norm(offset)
    rows = []
    for query in args.query_frames:
        observed = clouds[query]
        if observed is None:
            rows.append({"query_frame": query, "status": "no_observed_surface"})
            continue
        fits = []
        for anchor in args.anchor_frames:
            if anchor >= query or clouds[anchor] is None:
                continue
            source = clouds[anchor]
            truth = camera_motion_from_ee(ee_by_frame[anchor], ee_by_frame[query], base_from_camera)
            true_surface = apply(source, truth[:3, :3], truth[:3, 3])
            center = true_surface.mean(axis=0)
            error_rotation = Rotation.from_rotvec(axis * np.radians(args.perturb_deg)).as_matrix()
            error_translation = offset * args.perturb_mm / 1000
            perturb = transform_matrix(error_rotation, center + error_translation - error_rotation @ center)
            seed = perturb @ truth
            fitted_rotation, fitted_translation = trimmed_icp(
                source,
                observed,
                seed[:3, :3],
                seed[:3, 3],
                iterations=args.icp_iterations,
                trim_fraction=args.trim_fraction,
            )
            fitted_surface = apply(source, fitted_rotation, fitted_translation)
            seed_surface = apply(source, seed[:3, :3], seed[:3, 3])
            fits.append(
                {
                    "anchor_frame": anchor,
                    "truth_observation_fit": fit_metrics(
                        source, observed, true_surface, args.diagnostic_radius_m
                    ),
                    "seed_observation_fit": fit_metrics(
                        source, observed, seed_surface, args.diagnostic_radius_m
                    ),
                    "refined_observation_fit": fit_metrics(
                        source, observed, fitted_surface, args.diagnostic_radius_m
                    ),
                    "seed_error_at_surface_mm": float(
                        np.sqrt(np.mean(np.sum((seed_surface - true_surface) ** 2, axis=1))) * 1000
                    ),
                    "refined_error_at_surface_mm": float(
                        np.sqrt(np.mean(np.sum((fitted_surface - true_surface) ** 2, axis=1))) * 1000
                    ),
                    "refined_rotation_error_deg": float(
                        np.degrees(Rotation.from_matrix(fitted_rotation @ truth[:3, :3].T).magnitude())
                    ),
                }
            )
        fits.sort(key=lambda fit: -fit["refined_observation_fit"]["balanced_coverage"])
        rows.append({"query_frame": query, "status": "measured" if fits else "no_past_surface", "fits": fits})
    result = {
        "purpose": "offline attachment-error recovery from a bank of earlier held surfaces; no robot command",
        "perturbation": {
            "translation_mm": args.perturb_mm,
            "rotation_deg": args.perturb_deg,
            "translation_direction_camera": offset.tolist(),
            "rotation_axis_camera": axis.tolist(),
        },
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-data-parquet", type=Path, required=True)
    parser.add_argument("--demo-held-masks", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--anchor-frames", type=parse_frames, required=True)
    parser.add_argument("--query-frames", type=parse_frames, required=True)
    parser.add_argument("--perturb-mm", type=float, required=True)
    parser.add_argument("--perturb-deg", type=float, required=True)
    parser.add_argument("--perturb-axis", type=float, nargs=3, default=[0, 1, 0])
    parser.add_argument("--perturb-direction", type=float, nargs=3, default=[1, 0, 0])
    parser.add_argument("--demo-depth-scale", type=float, default=0.001)
    parser.add_argument("--mask-interior-px", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--diagnostic-radius-m", type=float, default=0.006)
    parser.add_argument("--icp-iterations", type=int, default=15)
    parser.add_argument("--trim-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(
        json.dumps(
            [
                {
                    "query_frame": item["query_frame"],
                    "best_anchor": item["fits"][0]["anchor_frame"] if item.get("fits") else None,
                    "seed_error_mm": round(item["fits"][0]["seed_error_at_surface_mm"], 2)
                    if item.get("fits")
                    else None,
                    "refined_error_mm": round(item["fits"][0]["refined_error_at_surface_mm"], 2)
                    if item.get("fits")
                    else None,
                    "coverage": round(item["fits"][0]["refined_observation_fit"]["balanced_coverage"], 3)
                    if item.get("fits")
                    else None,
                }
                for item in result["queries"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
