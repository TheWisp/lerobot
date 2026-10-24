"""Evaluate the unchanged shape correction over a full cached matcher sequence.

The reference and landmarks are fixed from one observed frame. Missing observations
hold the previous estimate and are labelled explicitly, not counted as tracking.
No gripper motion or later reference frame enters the estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from compare import RECORDING, frame_data, move
from run_surface_tracker import initial_landmarks, unproject, voxel_sample
from scipy.spatial import cKDTree
from shape_grounding import ShapeConfig, WholeSurfaceFit, pca_candidates, project


def available_geometry(observed: dict) -> str:
    if not observed["mask"].any():
        return "no_mask"
    points = observed["obs"]["points"]
    if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(0)) < 2:
        return "insufficient_depth"
    return "observed"


def diagnostics(source, pose, observed, k):
    points = move(source, pose)
    target = observed["all_surface_points"]
    uv = project(points, k)
    height, width = observed["mask"].shape
    pixels = np.rint(uv).astype(int)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    mask_support = np.zeros(len(points), bool)
    mask_support[inside] = observed["mask"][pixels[inside, 1], pixels[inside, 0]]
    _, _, basis = np.linalg.svd(source - source.mean(0), full_matrices=False)
    _, _, current_basis = np.linalg.svd(target - target.mean(0), full_matrices=False)
    axis_angle = np.degrees(np.arccos(np.clip(abs((pose[:3, :3] @ basis[0]) @ current_basis[0]), 0, 1)))
    distance = cKDTree(points).query(target)[0] * 1000
    return {
        "axis_vs_current_patch_deg": float(axis_angle),
        "observed_to_reference_median_mm": float(np.median(distance)),
        "observed_to_reference_p90_mm": float(np.percentile(distance, 90)),
        # Includes legitimate foreground occlusion: this is not an error rate.
        "reference_projected_inside_mask_fraction": float(mask_support.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--reference-frame", type=int)
    args = parser.parse_args()
    config = ShapeConfig(**json.loads(args.config.read_text())) if args.config else ShapeConfig()
    baseline = np.load(args.baseline / "trace.npz")
    baseline_summary = json.loads((args.baseline / "summary.json").read_text())
    selection = np.ones(len(baseline["frames"]), bool)
    if args.start is not None:
        selection &= baseline["frames"] >= args.start
    if args.stop is not None:
        selection &= baseline["frames"] <= args.stop
    indices, original = baseline["frames"][selection], baseline["poses"][selection]
    if not len(indices):
        parser.error("Selected interval is empty")
    original = original @ np.linalg.inv(original[0])
    reference_frame = int(indices[0]) if args.reference_frame is None else args.reference_frame
    if reference_frame > indices[0]:
        parser.error("The reference must not come from a future frame")
    baseline_rows = {int(r["frame"]): r for r in csv.DictReader((args.baseline / "trace.csv").open())}
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    times = np.loadtxt(RECORDING / "times.txt")
    first = frame_data(int(indices[0]), k)
    if available_geometry(first) != "observed":
        raise ValueError("The first frame must have usable reference geometry")
    reference = first if reference_frame == indices[0] else frame_data(reference_frame, k)
    source = reference["obs"]["points"]
    anchors, _ = initial_landmarks(source)
    last = np.eye(4)
    if reference_frame != indices[0]:
        all_points = unproject(first["depth"], k)[first["obs"]["valid_image"]]
        first["all_surface_points"] = all_points[voxel_sample(all_points, config.surface_spacing_m)]
        fitter = WholeSurfaceFit(source, first, k, config)
        candidates = [fitter.refine(pose) for pose in pca_candidates(source, first["obs"]["points"])]
        last, _ = min(candidates, key=lambda item: item[1]["cost"])
    poses, rows = [], []
    for index, frame in enumerate(indices):
        begin = time.perf_counter()
        observed = first if index == 0 else frame_data(int(frame), k)
        all_points = unproject(observed["depth"], k)[observed["obs"]["valid_image"]]
        observed["all_surface_points"] = all_points[voxel_sample(all_points, config.surface_spacing_m)]
        status = available_geometry(observed)
        detail = {
            "initial_cost": np.nan,
            "cost": np.nan,
            "evaluations": 0,
            "converged": False,
            "used_coarse": False,
        }
        elapsed = 0.0
        if index and status == "observed":
            step = original[index] @ np.linalg.inv(original[index - 1])
            proposal = step @ last
            fitter = WholeSurfaceFit(source, observed, k, config)
            start = time.perf_counter()
            last, detail = fitter.correct(proposal)
            elapsed = (time.perf_counter() - start) * 1000
        poses.append(last.copy())
        row = {
            "frame": int(frame),
            "seconds": float(times[frame] - times[indices[0]]),
            "status": "reference" if index == 0 else status,
            "matcher_valid": int(baseline_rows.get(int(frame), {}).get("valid", 0)),
            "mask_pixels": int(observed["mask"].sum()),
            "clean_depth_pixels": len(all_points),
            "shape_cost_before": detail["initial_cost"],
            "shape_cost_after": detail["cost"],
            "correction_ms": elapsed,
            "evaluations": detail["evaluations"],
            "converged": int(detail["converged"]),
            "used_coarse": int(detail["used_coarse"]),
        }
        measurements = diagnostics(source, last, observed, k) if status == "observed" else {}
        for name in (
            "axis_vs_current_patch_deg",
            "observed_to_reference_median_mm",
            "observed_to_reference_p90_mm",
            "reference_projected_inside_mask_fraction",
        ):
            row[name] = measurements.get(name, np.nan)
        row["input_and_correction_ms"] = (time.perf_counter() - begin) * 1000
        rows.append(row)
        if frame % 30 == 0:
            print(frame, status, "cost", round(detail["cost"], 2), "ms", round(elapsed), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "trace.npz",
        frames=indices,
        poses=poses,
        reference_points=source,
        landmarks_reference=anchors,
        statuses=np.array([r["status"] for r in rows]),
        baseline_poses=original,
    )
    measured = [r for r in rows if r["status"] == "observed"]
    summary = {
        "method": baseline_summary["method"],
        "config": asdict(config),
        "reference_frame": reference_frame,
        "start": int(indices[0]),
        "stop": int(indices[-1]),
        "frames": len(indices),
        "duration_seconds": float(times[indices[-1]] - times[indices[0]]),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "missing_observation_frames": [
            r["frame"] for r in rows if r["status"] not in ("observed", "reference")
        ],
        "optimizer_converged": sum(r["converged"] for r in measured),
        "coarse_reinitializations": sum(r["used_coarse"] for r in measured),
        "median_correction_ms": float(np.median([r["correction_ms"] for r in measured])),
        "p95_correction_ms": float(np.percentile([r["correction_ms"] for r in measured], 95)),
        "warning": "Shape consistency and solver convergence do not establish physical landmark identity or pose accuracy.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
