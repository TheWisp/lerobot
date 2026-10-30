"""Score completed alternative fits against held-out pose labels."""

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
r = json.loads((out / "clock_objectives.json").read_text())
base = Path.home() / ".cache/flash-labeled-laval/sample"
labels = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.load(base / "clock_interaction_hard/poses.npy").reshape(
    -1, 4, 4
)
vertices = np.asarray(trimesh.load(base / "clock/geometry.ply", process=False).vertices)
points = vertices[np.random.default_rng(42).choice(len(vertices), 256, replace=False)]


def move(p, t):
    return p @ t[:3, :3].T + t[:3, 3]


def error(p, t, pts):
    return {
        "point_mm": float(np.median(np.linalg.norm(move(pts, p) - move(pts, t), axis=1)) * 1000),
        "angle_deg": float(np.rad2deg(Rotation.from_matrix(p[:3, :3] @ t[:3, :3].T).magnitude())),
    }


def summary(rows):
    return {
        key: dict(
            zip(
                ("median", "p90", "max"),
                map(float, np.percentile([row[key] for row in rows], [50, 90, 100])),
                strict=False,
            )
        )
        for key in ("point_mm", "angle_deg")
    }


result = {"full_clip": {}, "selected_pair_tests": {}}
for weight, poses in r["poses"].items():
    rows = [
        {"frame": i, **error(np.array(p), labels[i] @ np.linalg.inv(labels[0]), move(points, labels[0]))}
        for i, p in enumerate(poses)
    ]
    result["full_clip"][weight] = {"summary": summary(rows), "frames": rows}
    print("FULL", weight, summary(rows), flush=True)
# Pairwise scores have different source frames, so these errors measure only
# the local edge. They do not include error in an already estimated source pose.
for mode in ("shape", "shape_plus_matches", "matches", "nearest_frame_raw"):
    rows = []
    for i in sorted({p["target"] for p in r["pairs"]}):
        pairs = [p for p in r["pairs"] if p["target"] == i]
        candidates = [
            {**s, "source": p["source"], "target": i, "matches": p["matches"]}
            for p in pairs
            for s in p["solutions"]
        ]
        if mode == "shape":
            chosen = min(candidates, key=lambda x: x["cost"])
        elif mode == "shape_plus_matches":
            chosen = min(
                candidates,
                key=lambda x: x["cost"]
                + (x["correspondence_cost"] if x["correspondence_cost"] is not None else 25),
            )
        elif mode == "matches":
            chosen = min(
                candidates,
                key=lambda x: x["correspondence_cost"] if x["correspondence_cost"] is not None else 25,
            )
        else:
            chosen = next(x for x in candidates if x["source"] == i - 1 and x["start"] == "ufm-unrefined")
        j = chosen["source"]
        row = {
            **chosen,
            **error(np.array(chosen["pose"]), labels[i] @ np.linalg.inv(labels[j]), move(points, labels[j])),
        }
        rows.append(row)
    result["selected_pair_tests"][mode] = {"summary": summary(rows), "frames": rows}
    print("PAIR", mode, summary(rows), flush=True)
result["frame150"] = []
for pair in r["pairs"]:
    if pair["target"] != 150 or pair["source"] not in (0, 149):
        continue
    for s in pair["solutions"]:
        if s["start"] not in ("ufm", "ufm-unrefined", "continuity", "pca-0", "pca-1", "pca-2", "pca-3"):
            continue
        row = {
            "source": pair["source"],
            **s,
            **error(
                np.array(s["pose"]),
                labels[150] @ np.linalg.inv(labels[pair["source"]]),
                move(points, labels[pair["source"]]),
            ),
        }
        result["frame150"].append(row)
        print(
            "150",
            row["source"],
            s["start"],
            "terms",
            s["shape_terms"],
            "matchcost",
            s["correspondence_cost"],
            "error",
            row["point_mm"],
            row["angle_deg"],
            flush=True,
        )
(out / "clock_objective_scores.json").write_text(json.dumps(result, indent=2))
