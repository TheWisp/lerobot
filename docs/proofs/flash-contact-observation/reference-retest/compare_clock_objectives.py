"""Offline alternatives: retain measured RGB-D matches during shape refinement.

All fits and rankings finish before the separate scorer loads labeled poses.
Weights are an explicit experiment sweep, not new production defaults.
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_correspondence import crop_pair, robust_fit, ufm_pairs
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession, load_ufm
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
root = Path.home() / ".cache/huggingface/lerobot/demos/flash_laval_clock_labeled"
cfg = VisionPipelineConfig.from_dict(
    json.loads((root / "flash/v1/observation-config.json").read_text())["vision_pipeline"]
)
cfg_obs = replace(cfg.observation, component_gap_m=None)
k = np.array(json.loads((root / "flash/v1/source.json").read_text())["camera_intrinsics"])
data = np.load(out / "clock_inputs.npz")
model = load_ufm()
kernels = compiled_kernels()
original = json.loads((out / "clock_fits.json").read_text())
variants = {str(w): (w, 1.0) for w in (0.0, 0.25, 1.0, 4.0)}
variants.update({f"no_outline_{w}": (w, 0.0) for w in (0.0, 1.0, 4.0)})


class CorrespondenceFit(WholeSurfaceFit):
    def __init__(self, *args, match_source, match_target, match_weight, outline_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_source = match_source
        self.match_target = match_target
        self.match_weight = match_weight
        self.outline_weight = outline_weight

    def _residual_with_distances(self, pose, predicted, model_distance, target_distance):
        shape = super()._residual_with_distances(pose, predicted, model_distance, target_distance)
        shape[-len(self.mask_outline) :] *= np.sqrt(self.outline_weight)
        if not len(self.match_source) or self.match_weight == 0:
            return shape
        err = (move(self.match_source, pose) - self.match_target) / cfg.motion.ransac_distance_m
        # Smooth robust loss, retaining XYZ correspondence rather than permitting
        # a source point to match whichever surface point becomes nearest.
        robust = np.sign(err) * np.sqrt(2 * (np.sqrt(1 + err**2) - 1))
        return np.r_[shape, robust.ravel() * np.sqrt(self.match_weight / len(err))]


frames = []
poses = {name: [np.eye(4)] for name in variants}
matches = (
    dict(np.load(out / "clock_correspondences.npz")) if (out / "clock_correspondences.npz").exists() else {}
)
pair_scores = []
t0 = time.perf_counter()
with SearchPool(8) as pool:
    session = SurfaceTrackingSession(
        None, observation_config=cfg_obs, shape_config=cfg.shape, motion_config=cfg.motion
    )
    for i, (rgb, depth, mask) in enumerate(zip(data["rgb"], data["depth"], data["mask"], strict=True)):
        session.previous = None
        current = session.prepare(rgb, depth, mask, k, i)["current"]
        frames.append(current)
        if i:
            key = f"{i - 1}_{i}"
            if key + "_p" in matches:
                p, q = matches[key + "_p"], matches[key + "_q"]
            else:
                a, b, crop = crop_pair(frames[i - 1], current, cfg.motion)
                p, q = ufm_pairs(model, a, b, crop, frames[i - 1], current, k, cfg.motion)
            step, n, _ = robust_fit(p, q, cfg.motion)
            if step is None:
                raise RuntimeError(f"missing motion at {i}")
            active = np.linalg.norm(move(p, step) - q, axis=1) < cfg.motion.ransac_distance_m
            matches[f"{i - 1}_{i}_p"] = p
            matches[f"{i - 1}_{i}_q"] = q
            matches[f"{i - 1}_{i}_step"] = step
            for name, (weight, outline_weight) in variants.items():
                prior = poses[name][-1]
                fit = CorrespondenceFit(
                    frames[0]["obs"]["points"],
                    current,
                    k,
                    cfg.shape,
                    match_source=move(p[active], np.linalg.inv(prior)),
                    match_target=q[active],
                    match_weight=weight,
                    outline_weight=outline_weight,
                    fit_kernels=kernels,
                    search_pool=pool,
                )
                pose, detail = fit.correct(step @ prior)
                poses[name].append(pose)
        if i % 20 == 0:
            print("clock full", i, round(time.perf_counter() - t0, 1), flush=True)
    for pair in original["pairs"]:
        j, i = pair["source"], pair["target"]
        key = f"{j}_{i}"
        if key + "_p" in matches:
            p, q = matches[key + "_p"], matches[key + "_q"]
            step = None
        else:
            a, b, crop = crop_pair(frames[j], frames[i], cfg.motion)
            p, q = ufm_pairs(model, a, b, crop, frames[j], frames[i], k, cfg.motion)
            step, n, _ = robust_fit(p, q, cfg.motion)
            matches[key + "_p"] = p
            matches[key + "_q"] = q
        # Score all observed correspondences with a robust cap, without selecting
        # a transform-dependent inlier subset to conceal disagreement.
        fit = WholeSurfaceFit(
            frames[j]["obs"]["points"], frames[i], k, cfg.shape, fit_kernels=kernels, search_pool=pool
        )
        sols = []
        for sol in pair["solutions"]:
            pose = np.array(sol["pose"])
            r = fit.residual(pose)
            nref = len(fit.reference_queries)
            ntarget = len(fit.target_queries)
            blocks = np.split(r, [nref, nref + ntarget, 2 * nref + ntarget])
            distances = np.linalg.norm(move(p, pose) - q, axis=1)
            motion_cost = (
                float(
                    np.mean(
                        np.minimum(
                            distances / cfg.motion.ransac_distance_m,
                            cfg.shape.max_surface_distance_m / cfg.motion.ransac_distance_m,
                        )
                        ** 2
                    )
                )
                if len(p)
                else None
            )
            sols.append(
                {
                    **sol,
                    "shape_terms": dict(
                        zip(
                            (
                                "reference_to_current",
                                "current_to_reference",
                                "outside_mask",
                                "uncovered_mask",
                            ),
                            (float(a @ a) for a in blocks),
                            strict=False,
                        )
                    ),
                    "correspondence_cost": motion_cost,
                    "correspondence_median_mm": float(np.median(distances) * 1000) if len(p) else None,
                    "correspondence_inlier_fraction": float(np.mean(distances < cfg.motion.ransac_distance_m))
                    if len(p)
                    else 0,
                }
            )
        pair_scores.append({"source": j, "target": i, "matches": len(p), "solutions": sols})
np.savez_compressed(out / "clock_correspondences.npz", **matches)
(out / "clock_objectives.json").write_text(
    json.dumps(
        {
            "variants": variants,
            "poses": {w: np.array(ps).tolist() for w, ps in poses.items()},
            "pairs": pair_scores,
            "labels_used_for_fitting": False,
            "elapsed_s": time.perf_counter() - t0,
        }
    )
)
print("DONE", round(time.perf_counter() - t0, 1), flush=True)
