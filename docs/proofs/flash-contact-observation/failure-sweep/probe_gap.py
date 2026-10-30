"""Isolate the point tracker's time-gap rule; no production limit changes."""

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.causal_points import CausalRigidPointTracker
from lerobot.policies.flash.perception import RGBDFrame

directory = Path.home() / ".cache/flash-failure-sweep-20261029/bleach_hard_00_03_chaitanya"
p = json.loads((directory / "provenance.json").read_text())
root = Path(p["source"])
k = np.array(p["intrinsics"])
last = 24
largest_gap = max(np.diff(p["source_times_s"][: last + 1]))
results = []
for label, limit in [
    ("unchanged", 0.1),
    ("allow_this_recorded_gap", float(np.nextafter(largest_gap, np.inf))),
]:
    tracker = CausalRigidPointTracker(max_frame_gap_s=limit)
    rows = []
    for i in range(last + 1):
        name = p["names"][i]
        rgb = cv2.cvtColor(cv2.imread(str(root / "rgb" / name)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(root / "depth" / name), -1)
        mask = cv2.imread(str(root / "gt_mask" / name), 0) > 0
        frame = RGBDFrame(
            i, rgb, depth, k, 0.001, np.eye(4), p["source_times_s"][i], p["source_times_s"][i], "fixed"
        )
        if i == 0:
            count = tracker.seed(frame, mask)
            continue
        try:
            result = asdict(tracker.update(frame, mask))
            if result["base_motion_from_seed"] is not None:
                result["base_motion_from_seed"] = result["base_motion_from_seed"].tolist()
            rows.append(dict(frame=i, **result))
        except ValueError as exc:
            rows.append({"frame": i, "error": str(exc)})
            break
    results.append({"condition": label, "max_frame_gap_s": limit, "seed_points": count, "frames": rows})
target = Path(__file__).with_name("gap_probe.json")
target.write_text(json.dumps(results, indent=2))
# Fitting is finished and saved before annotation loading.
label_paths = sorted((root / "annotated_poses").glob("*.txt"))
initial = np.loadtxt(label_paths[p["start"]]).reshape(4, 4)
for condition in results:
    for row in condition["frames"]:
        pose = row.get("base_motion_from_seed")
        if pose is None:
            continue
        truth = np.loadtxt(label_paths[p["start"] + row["frame"]]).reshape(4, 4) @ np.linalg.inv(initial)
        pose = np.array(pose)
        row["label_angle_error_deg"] = float(
            Rotation.from_matrix(pose[:3, :3] @ truth[:3, :3].T).magnitude() * 180 / np.pi
        )
        row["label_origin_error_mm"] = float(
            np.linalg.norm((pose @ initial)[:3, 3] - (truth @ initial)[:3, 3]) * 1000
        )
target.write_text(json.dumps(results, indent=2, allow_nan=False))
for condition in results:
    print(condition["condition"], condition["frames"][-1])
