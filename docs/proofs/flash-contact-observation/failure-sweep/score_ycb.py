"""Evaluate completed traces only. Pose labels cannot influence tracking."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path.home() / ".cache/flash-failure-sweep-20261029"
DEST = Path(__file__).with_name("results.json.gz")


def moved(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def summarize(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return (
        None
        if not len(values)
        else {
            "n": len(values),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "maximum": float(np.max(values)),
        }
    )


def score(directory):
    provenance = json.loads((directory / "provenance.json").read_text())
    manifest = json.loads((directory / "evidence/manifest.json").read_text())
    assert manifest["complete"] and manifest["frames_completed"] == provenance["count"]
    entries = [
        json.loads(gzip.decompress(p.read_bytes())) for p in sorted((directory / "evidence").glob("*.gz"))
    ]
    source = Path(provenance["source"])
    pose_files = sorted((source / "annotated_poses").glob("*.txt"))
    rgb_files = sorted((source / "rgb").glob("*.png"))
    assert len(pose_files) == len(rgb_files)
    # Most source sequences use ordinal pose names and timestamp image names;
    # the published YCBInEOAT readers pair their sorted lists by ordinal index.
    # One sequence instead uses identical timestamp names: check that case too.
    if int(pose_files[0].stem) > 10**12:
        assert [p.stem for p in pose_files] == [p.stem for p in rgb_files]
    labels = np.array([np.loadtxt(p).reshape(4, 4) for p in pose_files])[
        provenance["start"] : provenance["start"] + len(entries)
    ]
    assert np.isfinite(labels).all() and np.allclose(labels[:, 3], [0, 0, 0, 1])
    first = entries[0]["proposals"]["object"]["raw_pose"]
    if first is None:
        raise ValueError("cannot score without first-frame pose anchor")
    inverse_first = np.linalg.inv(first)
    k = np.array(provenance["intrinsics"])
    name = provenance["names"][0]
    depth = cv2.imread(str(source / "depth" / name), -1).astype(float) * 0.001
    mask = cv2.imread(str(source / "gt_mask" / name), 0) > 0
    interior = cv2.erode(mask.astype("uint8"), np.ones((5, 5), "uint8")) > 0
    y, x = np.where(interior & (depth > 0))
    idx = np.random.default_rng(42).choice(len(x), min(256, len(x)), replace=False)
    x, y = x[idx], y[idx]
    z = depth[y, x]
    points = np.column_stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z))
    inv_label = np.linalg.inv(labels[0])
    state_counts, reasons, estimators = Counter(), Counter(), Counter()
    frames = []
    for i, entry in enumerate(entries):
        assert entry["frame_index"] == i
        proposal, estimate = entry["proposals"]["object"], entry["estimates"]["object"]
        support = proposal["visible_support"]
        target = labels[i] @ inv_label
        truth_points = moved(points, target)
        row = {
            "frame": i,
            "source_index": provenance["start"] + i,
            "state": estimate["evidence"],
            "estimator": estimate["estimator"],
            "failure": proposal["failure"],
            "support": support,
        }
        state_counts[row["state"]] += 1
        estimators[row["estimator"]] += 1
        if row["failure"]:
            reasons[row["failure"]] += 1
        for mode, pose in [("raw", proposal["raw_pose"]), ("published", estimate["pose"])]:
            if pose is None:
                row[mode] = None
                continue
            relative = np.array(pose) @ inverse_first
            row[mode] = {
                "point_mm": float(
                    np.median(np.linalg.norm(moved(points, relative) - truth_points, axis=1)) * 1000
                ),
                "angle_deg": float(
                    Rotation.from_matrix(relative[:3, :3] @ target[:3, :3].T).magnitude() * 180 / np.pi
                ),
            }
        if i:
            step = labels[i] @ np.linalg.inv(labels[i - 1])
            row["label_step_angle_deg"] = float(Rotation.from_matrix(step[:3, :3]).magnitude() * 180 / np.pi)
            row["label_step_translation_mm"] = float(
                np.linalg.norm(labels[i, :3, 3] - labels[i - 1, :3, 3]) * 1000
            )
        frames.append(row)
    errors = {}
    for state in ["all", "observed", "predicted", "lost"]:
        selected = [f for f in frames[1:] if state == "all" or f["state"] == state]
        errors[state] = {
            mode: {
                metric: summarize([f[mode][metric] for f in selected if f[mode] is not None])
                for metric in ["point_mm", "angle_deg"]
            }
            for mode in ["raw", "published"]
        }
    return {
        "sequence": directory.name,
        "start": provenance["start"],
        "frames_count": len(entries),
        "input_sha256": provenance["input_sha256"],
        "config_sha256": provenance["config_sha256"],
        "states": dict(state_counts),
        "failures": dict(reasons),
        "estimators": dict(estimators),
        "errors": errors,
        "label_step_angle_deg": summarize([f["label_step_angle_deg"] for f in frames[1:]]),
        "label_step_translation_mm": summarize([f["label_step_translation_mm"] for f in frames[1:]]),
        "source_dt_s": summarize(np.diff(provenance["source_times_s"])),
        "frames": frames,
        "timing": json.loads((directory / "completed.json").read_text()),
    }


if __name__ == "__main__":
    result = [score(d) for d in sorted(ROOT.iterdir()) if (d / "completed.json").exists()]
    with gzip.open(DEST, "wt", encoding="utf-8") as stream:
        json.dump({"schema": "flash-failure-sweep-v1", "cases": result}, stream, indent=2, allow_nan=False)
    for case in result:
        errors = case["errors"]["all"]["published"]
        print(case["sequence"], case["states"], "point", errors["point_mm"], "angle", errors["angle_deg"])
