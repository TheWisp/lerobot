"""Audit the recorded fusion consumer, and report accuracy proxies separately."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from compare import RECORDING, frame_data, move
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def audit(trace, rows):
    poses, priors = trace["poses"], trace["prior_poses"]
    held = np.array([bool(int(r["held"])) for r in rows])
    changed = np.array([bool(int(r["visual_changed"])) for r in rows])
    available = np.array([bool(int(r["output_available"])) for r in rows])
    np.testing.assert_array_equal(available, np.isfinite(poses).all((1, 2)))
    bridged = np.isfinite(priors).all((1, 2))
    np.testing.assert_allclose(
        poses[held & bridged & ~changed], priors[held & bridged & ~changed], atol=1e-10
    )
    ee = trace["end_effector_poses"]
    steps = held[1:] & held[:-1] & available[:-1] & bridged[1:]
    steps &= np.isfinite(ee[1:]).all((1, 2)) & np.isfinite(ee[:-1]).all((1, 2))
    indices = np.flatnonzero(steps) + 1
    predicted = ee[indices] @ np.linalg.inv(ee[indices - 1]) @ poses[indices - 1]
    np.testing.assert_allclose(priors[indices], predicted, atol=1e-9)
    return {
        "held_motion_steps_verified": len(indices),
        "unchanged_held_predictions": int((held & bridged & ~changed).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    # Same interval used by the earlier benchmark, not selected for this result.
    clear_frames = np.arange(360, 400)
    clear_points = [frame_data(int(f), k)["obs"]["points"] for f in clear_frames]
    depth_axes = np.array([np.linalg.svd(p - p.mean(0), full_matrices=False)[2][0] for p in clear_points])
    output = {
        "clear_interval": [360, 399],
        "held_interval": [230, 514],
        "later_held_interval": [350, 514],
        "warning": "Depth agreement is same-sensor consistency, not ground-truth accuracy. Motion prediction assumes no slip. Large visual repairs may be jumps rather than physical motion.",
        "methods": {},
    }
    for path in args.results:
        trace = np.load(path / "trace.npz")
        rows = list(csv.DictReader((path / "trace.csv").open()))
        assert np.array_equal(trace["frames"], np.arange(len(rows)))
        poses, prior, source = trace["poses"], trace["prior_poses"], trace["reference_points"]
        axis = np.linalg.svd(source - source.mean(0), full_matrices=False)[2][0]
        current_axes = poses[clear_frames, :3, :3] @ axis
        angles = np.degrees(np.arccos(np.clip(abs(np.sum(current_axes * depth_axes, axis=1)), 0, 1)))
        distances = [
            float(np.median(cKDTree(move(source, poses[f])).query(p)[0]) * 1000)
            for f, p in zip(clear_frames, clear_points, strict=True)
        ]
        corrections = np.degrees(
            Rotation.from_matrix(
                poses[230:515, :3, :3] @ prior[230:515, :3, :3].transpose(0, 2, 1)
            ).magnitude()
        )
        result = {
            "clear_shaft_vs_connected_depth_median_deg": float(np.median(angles)),
            "clear_observed_to_model_median_mm": float(np.median(distances)),
            "held_visual_correction_p95_deg": float(np.percentile(corrections, 95)),
            "held_visual_correction_max_deg": float(corrections.max()),
            "largest_correction_frame": int(230 + corrections.argmax()),
            "later_held_visual_correction_p95_deg": float(np.percentile(corrections[120:], 95)),
            "later_held_visual_correction_max_deg": float(corrections[120:].max()),
            "frame_300_visual_correction_deg": float(corrections[70]),
        }
        if "end_effector_poses" in trace and "output_available" in rows[0]:
            result["audit"] = audit(trace, rows)
            result["held_corrections"] = [
                int(r["frame"]) for r in rows if int(r["held"]) and int(r["visual_changed"])
            ]
            result["lost_frames"] = [int(r["frame"]) for r in rows if not int(r["output_available"])]
        output["methods"][path.name] = result
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
