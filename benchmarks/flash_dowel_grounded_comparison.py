"""Compare frozen point trackers with the same measured-mask/depth rigid fitter.

This is a same-camera consistency check, not six-degree-of-freedom ground truth.
All candidates use the same first-frame queries, depth lift, mask gate, rigid
fit, and optional local surface correction. Model-native depths are deliberately
ignored so this measures the point correspondence frontend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _load, lift_measured_depth, validate_prediction
from lerobot.policies.flash.registration import transform_points
from lerobot.policies.flash.visible_points import surface_support
from lerobot.policies.flash.visible_rigid_tracking import (
    VisibleRGBDFrame,
    _surface_evidence,
    fit_visible_trace,
    visible_surface_cloud,
)


def _fraction(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def evaluate(
    experiment: Path,
    predictions: list[Path],
    gripper_prior: Path | None,
    *,
    track_tolerance_m: float,
    surface_tolerance_m: float,
    min_inliers: int,
    fit_trials: int,
    edge_pixels: int,
    max_surface_points: int,
    icp_iterations: int,
    trim_fraction: float,
) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    reference = _load(experiment / "queries.npz")
    frames, queries, k = reference["frames"], reference["queries"], reference["intrinsics"]
    masks, depths, surfaces, rgbd = [], [], [], []
    for frame in frames:
        depth = cv2.imread(
            str(Path(manifest["recording"]) / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED
        )
        mask = cv2.imread(str(Path(manifest["mask_dir"]) / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if depth is None or mask is None or depth.shape != mask.shape:
            raise ValueError(f"missing aligned RGB-D/mask at frame {frame}")
        depth_m = depth.astype(float) * 0.001
        surface = visible_surface_cloud(
            depth_m, mask > 0, k, edge_pixels=edge_pixels, max_points=max_surface_points
        )
        masks.append(mask > 0)
        depths.append(depth_m)
        surfaces.append(surface)
        rgbd.append(VisibleRGBDFrame(surface, mask > 0, depth_m, k))
    source = surfaces[0]
    gripper = None
    if gripper_prior is not None:
        data = _load(gripper_prior)
        if not np.all(np.isin(frames, data["frames"])):
            raise ValueError("gripper prior does not span the selected frames")
        lookup = {int(frame): pose for frame, pose in zip(data["frames"], data["poses"], strict=True)}
        gripper = np.stack([lookup[int(frame)] @ np.linalg.inv(lookup[int(frames[0])]) for frame in frames])
    report = {
        "frames": [int(frames[0]), int(frames[-1])],
        "queries": len(queries),
        "settings": {
            "track_tolerance_m": track_tolerance_m,
            "surface_tolerance_m": surface_tolerance_m,
            "min_inliers": min_inliers,
            "fit_trials": fit_trials,
            "edge_pixels": edge_pixels,
            "max_surface_points": max_surface_points,
            "icp_iterations": icp_iterations,
            "trim_fraction": trim_fraction,
        },
        "results": {},
    }
    for prediction_path in predictions:
        prediction = _load(prediction_path)
        validate_prediction(reference, prediction)
        xyz = np.stack([lift_measured_depth(prediction["xy"][i], depths[i], k) for i in range(len(frames))])
        supported = np.stack(
            [
                surface_support(
                    xyz[i],
                    prediction["visible"][i].astype(float),
                    masks[i],
                    depths[i],
                    k,
                    surface_tolerance_m,
                )[0]
                for i in range(len(frames))
            ]
        )
        paths = {}
        for name, grounded in (("points_only", False), ("mask_depth_surface", True)):
            result = fit_visible_trace(
                xyz,
                supported,
                tolerance_m=track_tolerance_m,
                min_inliers=min_inliers,
                trials=fit_trials,
                **(
                    {
                        "reference_surface": source,
                        "rgbd_frames": rgbd,
                        "depth_m_per_unit": 1.0,
                        "surface_tolerance_m": surface_tolerance_m,
                        "icp_iterations": icp_iterations,
                        "trim_fraction": trim_fraction,
                    }
                    if grounded
                    else {}
                ),
            )
            mask_fraction, support_fraction, front_fraction, gripper_disagreement = [], [], [], []
            per_frame = []
            for i, pose in enumerate(result["poses"]):
                if result["lost"][i]:
                    per_frame.append(None)
                    continue
                evidence = _surface_evidence(
                    source,
                    pose,
                    masks[i],
                    depths[i],
                    k,
                    depth_m_per_unit=1.0,
                    tolerance_m=surface_tolerance_m,
                )
                moved = transform_points(pose, source)
                z = moved[:, 2]
                in_front = z > 0
                x = np.full(len(z), -1, int)
                y = np.full(len(z), -1, int)
                x[in_front] = np.rint(k[0, 0] * moved[in_front, 0] / z[in_front] + k[0, 2]).astype(int)
                y[in_front] = np.rint(k[1, 1] * moved[in_front, 1] / z[in_front] + k[1, 2]).astype(int)
                inside = in_front & (x >= 0) & (x < masks[i].shape[1]) & (y >= 0) & (y < masks[i].shape[0])
                in_mask = np.zeros(len(z), bool)
                ids = np.flatnonzero(inside)
                in_mask[ids] = masks[i][y[ids], x[ids]]
                mask_fraction.append(float(in_mask.mean()))
                support_fraction.append(evidence.surface_supported / evidence.projected_points)
                front_fraction.append(evidence.in_front_of_depth / evidence.projected_points)
                row = {
                    "frame": int(frames[i]),
                    "mask_fraction": mask_fraction[-1],
                    "mask_depth_support": support_fraction[-1],
                    "foreground_conflict": front_fraction[-1],
                    "inliers": int(result["inliers"][i]),
                    "pose_camera_from_source": pose.tolist(),
                }
                if gripper is not None:
                    expected = transform_points(gripper[i], source)
                    distance_mm = float(np.median(np.linalg.norm(moved - expected, axis=1)) * 1000)
                    gripper_disagreement.append(distance_mm)
                    row["gripper_delta_disagreement_mm"] = distance_mm
                per_frame.append(row)
            paths[name] = {
                "pose_frames": int((~result["lost"]).sum()),
                "median_mask_fraction": _fraction(mask_fraction),
                "median_mask_depth_support": _fraction(support_fraction),
                "median_foreground_conflict": _fraction(front_fraction),
                "median_gripper_delta_disagreement_mm": _fraction(gripper_disagreement),
                "surface_corrections": int(result["surface_refined"].sum()) if grounded else 0,
                "last_frame": per_frame[-1],
                "rows": per_frame,
            }
        report["results"][prediction_path.stem] = {
            "depth_backed_mask_points_median": float(np.median(supported.sum(axis=1))),
            "depth_backed_mask_points_last": int(supported[-1].sum()),
            **paths,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--gripper-prior", type=Path)
    parser.add_argument("--track-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--surface-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--fit-trials", type=int, default=100)
    parser.add_argument("--edge-pixels", type=int, default=2)
    parser.add_argument("--max-surface-points", type=int, default=1000)
    parser.add_argument("--icp-iterations", type=int, default=20)
    parser.add_argument("--trim-fraction", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.experiment,
        args.prediction,
        args.gripper_prior,
        track_tolerance_m=args.track_tolerance_mm * 0.001,
        surface_tolerance_m=args.surface_tolerance_mm * 0.001,
        min_inliers=args.min_inliers,
        fit_trials=args.fit_trials,
        edge_pixels=args.edge_pixels,
        max_surface_points=args.max_surface_points,
        icp_iterations=args.icp_iterations,
        trim_fraction=args.trim_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    key: value["mask_depth_surface"][key]
                    for key in (
                        "pose_frames",
                        "median_mask_depth_support",
                        "median_foreground_conflict",
                        "median_gripper_delta_disagreement_mm",
                    )
                }
                for name, value in result["results"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
