"""Independent scoring after audit_clock.py; never supplies labels to tracking."""

import gzip
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
r = json.loads((out / "clock_fits.json").read_text())
base = Path.home() / ".cache/flash-labeled-laval/sample"
labels = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.load(base / "clock_interaction_hard/poses.npy").reshape(
    -1, 4, 4
)
mesh = trimesh.load(base / "clock/geometry.ply", process=False)
vertices = np.asarray(mesh.vertices)
points = vertices[np.random.default_rng(42).choice(len(vertices), 256, replace=False)]
truth = labels @ np.linalg.inv(labels[0])
source = points @ labels[0, :3, :3].T + labels[0, :3, 3]


def error(p, t, pts):
    q = pts @ p[:3, :3].T + p[:3, 3]
    gt = pts @ t[:3, :3].T + t[:3, 3]
    return {
        "point_error_mm": float(np.median(np.linalg.norm(q - gt, axis=1)) * 1000),
        "angle_error_deg": float(np.rad2deg(Rotation.from_matrix(p[:3, :3] @ t[:3, :3].T).magnitude())),
    }


def summary(rows):
    return {
        k: dict(
            zip(
                ["median", "p90", "max"],
                map(float, np.percentile([v[k] for v in rows], [50, 90, 100])),
                strict=False,
            )
        )
        for k in ["point_error_mm", "angle_error_deg"]
    }


root = (
    Path.home()
    / ".cache/huggingface/lerobot/demos/flash_laval_clock_labeled/flash/v1/observation-evidence/episode_000000"
)
old = []
for i in range(len(labels)):
    e = json.loads(gzip.decompress((root / f"{i:06d}.json.gz").read_bytes()))
    old.append(np.array(e["proposals"]["clock"]["raw_pose"]))
old = np.array(old) @ np.linalg.inv(old[0])
results = {}
raw_chain = [np.eye(4)]
missing = []
for row in r["frames"][1:]:
    step = row["rgb_motion"]
    if step is None:
        missing.append(row["frame"])
    raw_chain.append((np.eye(4) if step is None else np.array(step)) @ raw_chain[-1])
print("RAW_MOTION_MISSING", missing)
for name, poses in [
    ("original", old),
    ("rgb_motion_chain", raw_chain),
    ("fixed_first_all_patches", [x["fixed_pose"] for x in r["frames"]]),
    ("previous_frame_all_patches", [x["chain_pose"] for x in r["frames"]]),
]:
    rows = [{"frame": i, **error(np.asarray(pose), truth[i], source)} for i, pose in enumerate(poses)]
    results[name] = {"summary": summary(rows), "frames": rows}
pairs = []
for pair in r["pairs"]:
    j, i = pair["source"], pair["target"]
    pts = points @ labels[j, :3, :3].T + labels[j, :3, 3]
    true = labels[i] @ np.linalg.inv(labels[j])
    sols = [{**s, **error(np.array(s["pose"]), true, pts)} for s in pair["solutions"]]
    chosen = min(sols, key=lambda s: s["cost"])
    oracle = min(sols, key=lambda s: s["point_error_mm"])
    pairs.append({"source": j, "target": i, "chosen": chosen, "oracle": oracle, "solutions": sols})
results["pair_tests"] = pairs
results["target_tests"] = []
for i in sorted({p["target"] for p in pairs}):
    options = [p for p in pairs if p["target"] == i]
    chosen = min(options, key=lambda p: p["chosen"]["cost"])
    oracle = min(options, key=lambda p: p["oracle"]["point_error_mm"])
    item = {
        "target": i,
        "chosen_source": chosen["source"],
        "chosen": chosen["chosen"],
        "oracle_source": oracle["source"],
        "oracle": oracle["oracle"],
    }
    results["target_tests"].append(item)
    print(
        "TARGET",
        i,
        "selected",
        chosen["source"],
        chosen["chosen"]["start"],
        round(chosen["chosen"]["point_error_mm"], 1),
        round(chosen["chosen"]["angle_error_deg"], 1),
        "best available",
        oracle["source"],
        oracle["oracle"]["start"],
        round(oracle["oracle"]["point_error_mm"], 1),
        round(oracle["oracle"]["angle_error_deg"], 1),
    )
for k in ["original", "rgb_motion_chain", "fixed_first_all_patches", "previous_frame_all_patches"]:
    print(k, results[k]["summary"])
(out / "clock_scores.json").write_text(json.dumps(results, indent=2))
