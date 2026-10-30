"""Diagnostic: is the dish mismatch a search failure or a scoring tradeoff?"""

import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit, pca_candidates
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession, reference_depth_support
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
cfg = VisionPipelineConfig.from_dict(
    json.loads((out / "dish_no_span_config.json").read_text())["vision_pipeline"]
)
cfg_shape = replace(cfg.shape, max_evaluations=200)


def ev(i):
    return json.loads(gzip.decompress((out / f"dish_no_span_evidence/{i:06d}.json.gz").read_bytes()))


z = np.load(out / "dish_000.npz")
k = z["k"]
e0 = ev(0)
c0 = np.array(e0["base_from_camera"])
p0 = np.array(e0["proposals"]["object_b"]["raw_pose"])
s = SurfaceTrackingSession(None, observation_config=cfg.observation, shape_config=cfg_shape)
ref = s.prepare(z["rgb"], z["depth"], z["mask"], k, 0)["current"]["obs"]["points"]


class WithoutOutline(WholeSurfaceFit):
    def _residual_with_distances(self, *args):
        r = super()._residual_with_distances(*args)
        r[-len(self.mask_outline) :] = 0
        return r


rows = []
with SearchPool(8) as pool:
    for i in (1, 10, 50, 56, 100, 244):
        d = np.load(out / f"dish_{i:03d}.npz")
        s.previous = None
        cur = s.prepare(d["rgb"], d["depth"], d["mask"], k, i)["current"]
        e = ev(i)
        cam = np.array(e["base_from_camera"])
        prop = e["proposals"]["object_b"]
        initial = np.linalg.inv(cam) @ np.array(prop["raw_pose"]) @ np.linalg.inv(p0) @ c0
        motion = prop.get("motion_prior_pose")
        starts = [("saved_fit", initial)]
        if motion is not None:
            starts.append(("image_motion", np.linalg.inv(cam) @ np.array(motion) @ np.linalg.inv(p0) @ c0))
        starts += [(f"pca_{j}", v) for j, v in enumerate(pca_candidates(ref, cur["obs"]["points"]))]
        for name, klass in [("existing", WholeSurfaceFit), ("without_outline_coverage", WithoutOutline)]:
            fit = klass(ref, cur, k, cfg_shape, fit_kernels=compiled_kernels(), search_pool=pool)
            candidates = []
            for start, pose in starts:
                corrected, diag = fit.refine(pose)
                pts = move(ref, corrected)
                p90 = float(np.percentile(cKDTree(pts).query(cur["all_surface_points"])[0], 90) * 1000)
                count, reverse = reference_depth_support(pts, cur["obs"]["valid_image"], cur["depth"], k)
                candidates.append(
                    {
                        "start": start,
                        "pose": corrected.tolist(),
                        "cost": diag["cost"],
                        "surface_p90_mm": p90,
                        "reverse_z_p90_mm": reverse,
                        "depth_pixels": count,
                    }
                )
            best = min(candidates, key=lambda c: c["cost"])
            rows.append({"frame": i, "method": name, "chosen": best, "candidates": candidates})
            print(i, name, {k: v for k, v in best.items() if k != "pose"}, flush=True)
(out / "dish_objectives.json").write_text(json.dumps(rows, indent=2))
