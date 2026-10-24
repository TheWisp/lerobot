"""Apply whole measured-surface correction after each cached RGB match delta."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from compare import HERE, RECORDING, frame_data, move, reference
from run_surface_tracker import unproject, voxel_sample
from shape_grounding import ShapeConfig, WholeSurfaceFit, pca_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-frame", type=int, default=100)
    parser.add_argument("--method", choices=("ufm", "roma-fast"), default="ufm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = ShapeConfig(**json.loads(args.config.read_text())) if args.config else ShapeConfig()
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    baseline = np.load(HERE / "results" / args.method / "trace.npz")
    indices, original = baseline["frames"], baseline["poses"]
    if args.reference_frame > indices[0]:
        parser.error("reference frame must be observed before the evaluated interval")
    source = frame_data(args.reference_frame, k)["obs"]["points"]

    def shape_frame(frame):
        observed = frame_data(int(frame), k)
        all_points = unproject(observed["depth"], k)[observed["obs"]["valid_image"]]
        observed["all_surface_points"] = all_points[voxel_sample(all_points, config.surface_spacing_m)]
        return observed

    first = shape_frame(int(indices[0]))
    initial_fitter = WholeSurfaceFit(source, first, k, config)
    candidates = [initial_fitter.refine(pose) for pose in pca_candidates(source, first["obs"]["points"])]
    initial, _ = min(candidates, key=lambda item: item[1]["cost"])
    last = initial
    ee, landmarks = reference()
    local_landmarks = move(landmarks, np.linalg.inv(initial))
    _, _, source_basis = np.linalg.svd(source - source.mean(0), full_matrices=False)
    initial_axis = initial[:3, :3] @ source_basis[0]
    poses, rows = [initial], []
    for index, frame in enumerate(indices[1:], start=1):
        observed = shape_frame(int(frame))
        step = original[index] @ np.linalg.inv(original[index - 1])
        proposal = step @ last
        fitter = WholeSurfaceFit(source, observed, k, config)
        start = time.perf_counter()
        last, detail = fitter.correct(proposal)
        elapsed = (time.perf_counter() - start) * 1000
        poses.append(last)
        expected = ee[frame] @ np.linalg.inv(ee[indices[0]])
        relative = last @ np.linalg.inv(initial)
        shaft = landmarks[1] - landmarks[0]
        shaft /= np.linalg.norm(shaft)
        cosine = np.clip((relative[:3, :3] @ shaft) @ (expected[:3, :3] @ shaft), -1, 1)
        _, _, surface_basis = np.linalg.svd(
            observed["obs"]["points"] - observed["obs"]["points"].mean(0), full_matrices=False
        )
        corrected_axis = last[:3, :3] @ source_basis[0]
        visible_angle = np.degrees(np.arccos(np.clip(abs(corrected_axis @ surface_basis[0]), 0, 1)))
        baseline_axis = original[index][:3, :3] @ initial_axis
        baseline_visible_angle = np.degrees(np.arccos(np.clip(abs(baseline_axis @ surface_basis[0]), 0, 1)))
        reference_axis_angle = np.degrees(
            np.arccos(np.clip(abs(corrected_axis @ (expected[:3, :3] @ initial_axis)), 0, 1))
        )
        row = {
            "frame": int(frame),
            "shape_cost_before": detail["initial_cost"],
            "shape_cost_after": detail["cost"],
            "axis_error_vs_gripper_deg": float(np.degrees(np.arccos(cosine))),
            "landmark_error_vs_gripper_mm": float(
                np.mean(np.linalg.norm(move(landmarks, relative) - move(landmarks, expected), axis=1)) * 1000
            ),
            "axis_vs_visible_surface_deg": float(visible_angle),
            "correction_ms": elapsed,
            "evaluations": detail["evaluations"],
            "converged": int(detail["converged"]),
            "used_coarse": int(detail["used_coarse"]),
            "baseline_same_axis_vs_visible_surface_deg": float(baseline_visible_angle),
            "reference_axis_vs_gripper_deg": float(reference_axis_angle),
        }
        rows.append(row)
        if frame % 20 == 0:
            print(
                frame,
                "visible_axis",
                round(visible_angle, 1),
                "cost",
                round(detail["cost"], 2),
                "ms",
                round(elapsed),
                flush=True,
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "trace.npz",
        frames=indices,
        poses=np.array(poses),
        initial_pose=initial,
        reference_points=source,
        landmarks_reference=local_landmarks,
    )
    summary = {"config": asdict(config), "reference_frame": args.reference_frame, "method": args.method}
    for name in (
        "axis_error_vs_gripper_deg",
        "landmark_error_vs_gripper_mm",
        "axis_vs_visible_surface_deg",
        "baseline_same_axis_vs_visible_surface_deg",
        "reference_axis_vs_gripper_deg",
        "correction_ms",
    ):
        summary["median_" + name] = float(np.median([row[name] for row in rows]))
        summary["last_" + name] = rows[-1][name]
    summary["median_clear_axis_vs_visible_surface_deg"] = float(
        np.median([row["axis_vs_visible_surface_deg"] for row in rows if row["frame"] >= 360])
    )
    summary["median_clear_baseline_same_axis_vs_visible_surface_deg"] = float(
        np.median([row["baseline_same_axis_vs_visible_surface_deg"] for row in rows if row["frame"] >= 360])
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
