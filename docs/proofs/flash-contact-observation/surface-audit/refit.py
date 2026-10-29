import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit, pca_candidates
from lerobot.policies.flash.surface_observation import move, observation
from lerobot.policies.flash.surface_tracking import reference_depth_support

p = Path(tempfile.gettempdir()) / "flash_dish_surface_audit"
cfg = json.loads(
    Path(
        "/home/feit/.cache/huggingface/lerobot/demos/flash_external_caryx_device_bowl_aligned/flash/v1/observation-config.json"
    ).read_text()
)["vision_pipeline"]
results = []
for i in [8, 15, 35, 70, 120]:
    d = np.load(p / f"{i:06d}.npz")
    obs = observation(d["depth"], d["mask"], d["intrinsics"], cfg["observation"])
    current = {"obs": obs, "depth": d["depth"], "mask": d["mask"], "all_surface_points": d["observed"]}
    fit = WholeSurfaceFit(d["reference"], current, d["intrinsics"], ShapeConfig(**cfg["shape"]))
    cases = [("recorded", d["pose"]), ("static_camera", d["static_pose"])] + [
        (f"pca_{j}", v) for j, v in enumerate(pca_candidates(d["reference"], obs["points"]))
    ]
    for name, pose in cases:
        refined, detail = fit.refine(pose)
        pts = move(d["reference"], refined)
        dist = cKDTree(pts).query(d["observed"])[0] * 1000
        n, depth = reference_depth_support(pts, obs["valid_image"], d["depth"], d["intrinsics"])
        result = {
            "frame": i,
            "start": name,
            "surface_p90_mm": float(np.percentile(dist, 90)),
            "depth_p90_mm": depth,
            **detail,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        np.savez(p / f"refit_{i:06d}_{name}.npz", pose=refined, predicted=pts)
(p / "refit_metrics.json").write_text(json.dumps(results, indent=2))
