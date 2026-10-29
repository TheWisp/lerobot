import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lerobot.policies.flash.registration import rigid_fit
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import reference_depth_support

p = Path(tempfile.gettempdir()) / "flash_dish_surface_audit"
out = []
for i in [8, 15, 35, 70, 120]:
    d = np.load(p / f"{i:06d}.npz")
    ref = d["reference"]
    target = d["observed"]
    t = cKDTree(target)
    pose = d["pose"].copy()
    for iteration in range(80):
        iterations = iteration + 1
        pred = move(ref, pose)
        _, ri = t.query(pred)
        _, li = cKDTree(pred).query(target)
        delta = rigid_fit(np.concatenate([pred, pred[li]]), np.concatenate([target[ri], target]))
        pose = delta @ pose
        if np.linalg.norm(delta - np.eye(4)) < 1e-8:
            break
    pts = move(ref, pose)
    e = cKDTree(pts).query(target)[0] * 1000
    n, dep = reference_depth_support(pts, d["valid"], d["depth"], d["intrinsics"])
    r = {
        "frame": i,
        "iterations": iterations,
        "surface_median_mm": float(np.median(e)),
        "surface_p90_mm": float(np.percentile(e, 90)),
        "depth_p90_mm": dep,
    }
    out.append(r)
    print(r)
(p / "icp_metrics.json").write_text(json.dumps(out, indent=2))
