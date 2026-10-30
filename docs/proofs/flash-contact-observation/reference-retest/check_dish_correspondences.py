"""Diagnose whether image-matched dish points support a single rigid movement."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from lerobot.policies.flash.surface_correspondence import crop_pair, robust_fit, ufm_pairs
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession, load_ufm
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
cfg = VisionPipelineConfig.from_dict(
    json.loads((out / "dish_no_span_config.json").read_text())["vision_pipeline"]
)
model = load_ufm()
s = SurfaceTrackingSession(None, observation_config=cfg.observation, shape_config=cfg.shape)
frames = {}
rows = []
for i in (0, 1, 10, 50, 244):
    d = np.load(out / f"dish_{i:03d}.npz")
    k = d["k"]
    s.previous = None
    frames[i] = s.prepare(d["rgb"], d["depth"], d["mask"], k, i)["current"]
for i in (1, 10, 50, 244):
    a, b, crop = crop_pair(frames[0], frames[i], cfg.motion)
    p, q = ufm_pairs(model, a, b, crop, frames[0], frames[i], k, cfg.motion)
    pose, n, med = robust_fit(p, q, cfg.motion)
    if pose is None:
        rows.append({"frame": i, "pairs": len(p), "fit": None})
        continue
    residual = np.linalg.norm(move(p, pose) - q, axis=1) * 1000
    changed_lengths = np.abs(pdist(p) - pdist(q)) * 1000
    dist = cKDTree(move(frames[0]["obs"]["points"], pose)).query(frames[i]["all_surface_points"])[0] * 1000
    row = {
        "frame": i,
        "pairs": len(p),
        "inliers": n,
        "all_correspondence_error_mm": np.percentile(residual, [50, 90, 100]).tolist(),
        "pair_distance_change_mm": np.percentile(changed_lengths, [50, 90, 100]).tolist(),
        "surface_p90_mm": float(np.percentile(dist, 90)),
        "pose": pose.tolist(),
    }
    rows.append(row)
    np.savez(out / f"dish_ufm_0_{i}.npz", p=p, q=q, pose=pose, residual_mm=residual, k=k)
    print({key: val for key, val in row.items() if key != "pose"}, flush=True)
(out / "dish_correspondences.json").write_text(json.dumps(rows, indent=2))
