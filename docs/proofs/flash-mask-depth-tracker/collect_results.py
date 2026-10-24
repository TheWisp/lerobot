"""Archive this experiment and compare held motion with recorded gripper deltas.

The gripper comparison is an approximate diagnostic, not ground truth, and is
never an input to the surface tracker. It is anchored separately for each trace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from run_surface_tracker import move
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gripper-reference", required=True, type=Path)
    parser.add_argument("--anchor", type=int, default=280)
    parser.add_argument("--end", type=int, default=399)
    args = parser.parse_args()
    reference = np.load(args.gripper_reference)
    camera_from_base = np.linalg.inv(reference["base_from_camera"])
    ee = reference["demo_ee_pose_base"]
    clean_anchor = np.load(args.results / "connected280" / "trace.npz")
    clean_index = int(np.flatnonzero(clean_anchor["frames"] == args.anchor)[0])
    shared_landmarks_camera = move(clean_anchor["landmarks_object"], clean_anchor["poses"][clean_index])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    drift_rows = []
    for name in ("growing", "previous", "visible280", "connected280", "connected_full"):
        source = args.results / name
        target = args.output / name
        target.mkdir(exist_ok=True)
        for filename in ("trace.npz", "metrics.csv", "summary.json", "config.json", "provenance.json"):
            shutil.copy2(source / filename, target / filename)
        rows = list(csv.DictReader((source / "metrics.csv").open()))
        report[name] = json.loads((source / "summary.json").read_text())
        report[name]["timing_ms_median_p95"] = {
            key: np.percentile([float(row[key]) for row in rows], [50, 95]).tolist()
            for key in ("preprocess_ms", "fit_ms", "check_ms", "map_ms", "load_ms")
        }
        trace = np.load(source / "trace.npz")
        index_by_frame = {int(frame): i for i, frame in enumerate(trace["frames"])}
        anchor_index = index_by_frame[args.anchor]
        if trace["state"][anchor_index] not in ("initialized", "surface_supported"):
            continue
        attachment = np.linalg.inv(camera_from_base @ ee[args.anchor]) @ trace["poses"][anchor_index]
        anchor_pose = trace["poses"][anchor_index]
        shared_landmarks_local = (shared_landmarks_camera - anchor_pose[:3, 3]) @ anchor_pose[:3, :3]
        local_rows = []
        for frame in range(args.anchor, args.end + 1):
            index = index_by_frame[frame]
            if trace["state"][index] not in ("initialized", "surface_supported"):
                continue
            expected = camera_from_base @ ee[frame] @ attachment
            estimated = trace["poses"][index]
            errors = (
                np.linalg.norm(
                    move(shared_landmarks_local, estimated) - move(shared_landmarks_local, expected), axis=1
                )
                * 1000
            )
            angle = np.degrees(Rotation.from_matrix(estimated[:3, :3] @ expected[:3, :3].T).magnitude())
            row = {
                "variant": name,
                "frame": frame,
                "a_mm": errors[0],
                "b_mm": errors[1],
                "c_mm": errors[2],
                "mean_landmark_mm": float(np.mean(errors)),
                "rotation_deg": float(angle),
            }
            local_rows.append(row)
            drift_rows.append(row)
        report[name]["gripper_relative_drift"] = {
            "anchor": args.anchor,
            "end": args.end,
            "evaluated_frames": len(local_rows),
            "landmarks": "Same three measured points from connected280 at the anchor, transformed into each candidate's local frame",
            "mean_landmark_error_mm_median_p95": np.percentile(
                [row["mean_landmark_mm"] for row in local_rows], [50, 95]
            ).tolist(),
            "rotation_error_deg_median_p95": np.percentile(
                [row["rotation_deg"] for row in local_rows], [50, 95]
            ).tolist(),
            "interpretation": "Rigid-grasp approximation; includes slip, calibration and timestamp errors. "
            "Full rotation includes weakly observable axial roll. Not pose ground truth.",
        }
    mask_dir = args.results / "sam3-full"
    shutil.copy2(mask_dir / "timing.csv", args.output / "sam3_timing.csv")
    sam_rows = list(csv.DictReader((mask_dir / "timing.csv").open()))
    report["sam3_warm_ms_median_p95"] = np.percentile(
        [float(row["sam_ms"]) for row in sam_rows[1:]], [50, 95]
    ).tolist()
    report["sam3_initialization_ms"] = float(sam_rows[0]["sam_ms"])
    report["mask_source"] = {
        "prompt": "wooden dowel",
        "resolution": 504,
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(mask_dir.glob("*.png"))
        },
    }
    with (args.output / "gripper_drift.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(drift_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(drift_rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: value for name, value in report.items() if name != "mask_source"}, indent=2))


if __name__ == "__main__":
    main()
