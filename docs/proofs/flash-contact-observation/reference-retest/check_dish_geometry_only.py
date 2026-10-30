"""Independent geometry-only ICP probe: no UFM residual, mask score or labels.

This intentionally tests a different objective, with multiple starts. A low
surface distance would not certify point identity, and a finite search is not
proof that no better rigid alignment exists.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lerobot.policies.flash.surface_correspondence import rigid_fit
from lerobot.policies.flash.surface_grounding import pca_candidates
from lerobot.policies.flash.surface_observation import move, observation

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
cfg = json.loads((out / "dish_no_span_config.json").read_text())["vision_pipeline"]["observation"]
z = np.load(out / "dish_000.npz")
k = z["k"]
source = observation(z["depth"], z["mask"], k, cfg)["points"]
tree = cKDTree(source)
rows = []
for i in (1, 10, 50, 244):
    d = np.load(out / f"dish_{i:03}.npz")
    target = observation(d["depth"], d["mask"], k, cfg)["points"]
    target_tree = cKDTree(target)
    starts = [np.load(out / f"dish_ufm_0_{i}.npz")["pose"], *pca_candidates(source, target)]
    candidates = []
    for trim in (1.0, 0.9, 0.75):
        for number, initial in enumerate(starts):
            pose = initial.copy()
            for _iteration in range(200):
                moved = move(source, pose)
                a, ib = target_tree.query(moved)
                b, ia = tree.query(move(target, np.linalg.inv(pose)))
                pp = np.r_[moved, moved[ia]]
                qq = np.r_[target[ib], target]
                errors = np.r_[a, b]
                keep = errors <= np.quantile(errors, trim)
                delta = rigid_fit(pp[keep], qq[keep])
                pose = delta @ pose
                if np.max(np.abs(delta - np.eye(4))) < 1e-7:
                    break
            ab = target_tree.query(move(source, pose))[0] * 1000
            ba = tree.query(move(target, np.linalg.inv(pose)))[0] * 1000
            candidates.append(
                {
                    "start": number,
                    "trim_fraction": trim,
                    "iterations": _iteration + 1,
                    "pose": pose.tolist(),
                    "source_to_target_p90_mm": float(np.percentile(ab, 90)),
                    "target_to_source_p90_mm": float(np.percentile(ba, 90)),
                }
            )
    best = min(candidates, key=lambda x: max(x["source_to_target_p90_mm"], x["target_to_source_p90_mm"]))
    rows.append({"frame": i, "best": best, "candidates": candidates})
    print(i, {k: v for k, v in best.items() if k != "pose"}, flush=True)
(out / "dish_geometry_only.json").write_text(json.dumps(rows, indent=2))
