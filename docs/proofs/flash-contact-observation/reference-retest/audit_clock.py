"""Offline clock ablations. Labels are loaded only after every fit is complete.

Runs full first-view and previous-view tracking with the largest-patch rule off.
On selected frames, spends extra work matching earlier views with UFM and all
four PCA starts. This is a bounded search, not a claim of a global optimum.
"""

import gzip
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_correspondence import crop_pair, robust_fit, ufm_pairs
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit, pca_candidates
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession, load_ufm
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

root = Path.home() / ".cache/huggingface/lerobot/demos/flash_laval_clock_labeled"
out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
out.mkdir(exist_ok=True)
source = json.loads((root / "flash/v1/source.json").read_text())
k = np.array(source["camera_intrinsics"])
cfg = VisionPipelineConfig.from_dict(
    json.loads((root / "flash/v1/observation-config.json").read_text())["vision_pipeline"]
)
obs_cfg = replace(cfg.observation, component_gap_m=None)
# Same equations and point budgets; more solver iterations for the slow search.
slow_shape = replace(cfg.shape, max_evaluations=200)
cache = out / "clock_inputs.npz"
if cache.exists():
    data = np.load(cache)
    rgbs = data["rgb"]
    depths = data["depth"]
    masks = data["mask"]
else:
    ds = LeRobotDataset(source["repo_id"], root=root, video_backend="pyav")
    rgbs = []
    depths = []
    masks = []
    for i in range(ds.num_frames):
        row = ds[i]
        rgbs.append(
            np.rint(row["observation.images.rgb"].numpy().transpose(1, 2, 0) * 255)
            .clip(0, 255)
            .astype("uint8")
        )
        depths.append(row["observation.images.depth"].numpy().squeeze().astype(np.float32) * 0.001)
        e = json.loads(
            gzip.decompress(
                (root / f"flash/v1/observation-evidence/episode_000000/{i:06d}.json.gz").read_bytes()
            )
        )
        mask = np.zeros(depths[-1].shape, bool)
        for start, n in e["masks_rle"]["clock"]:
            mask.flat[start : start + n] = True
        masks.append(mask)
    rgbs = np.array(rgbs)
    depths = np.array(depths)
    masks = np.array(masks)
    np.savez(cache, rgb=rgbs, depth=depths, mask=masks)
print("loaded", len(rgbs), "frames", flush=True)
model = load_ufm()
kernels = compiled_kernels()
frames = []
fixed = []
chain = [np.eye(4)]
rows = []
pair_rows = []
targets = {1, 10, 20, 30, 50, 70, 90, 95, 96, 100, 110, 150, 202}
started = time.perf_counter()
with SearchPool(8) as pool:
    session = SurfaceTrackingSession(
        model,
        observation_config=obs_cfg,
        shape_config=cfg.shape,
        motion_config=cfg.motion,
        fit_kernels=kernels,
        search_pool=pool,
    )
    for i, (rgb, depth, mask) in enumerate(zip(rgbs, depths, masks, strict=True)):
        prepared = session.prepare(rgb, depth, mask, k, i)
        current = prepared["current"]
        frames.append(current)
        result = session.finish(prepared)
        session.record_publication(False)
        fixed.append(np.array(result["camera_from_reference"]))
        if i:
            fit = WholeSurfaceFit(
                frames[i - 1]["obs"]["points"], current, k, cfg.shape, fit_kernels=kernels, search_pool=pool
            )
            local, detail = fit.correct(prepared["step"] if prepared["step"] is not None else np.eye(4))
            chain.append(local @ chain[-1])
        rows.append(
            {
                "frame": i,
                "fixed_pose": fixed[-1].tolist(),
                "chain_pose": chain[-1].tolist(),
                "rgb_motion": None if prepared["step"] is None else prepared["step"].tolist(),
                "diagnostics": result["diagnostics"],
            }
        )
        if i in targets:
            refs = {0, max(0, i - 1), max(0, i - 2), max(0, i - 4)}
            if i > 5:
                refs.add(max(range(i - 4), key=lambda j: len(frames[j]["all_surface_points"])))
            for j in sorted(refs):
                a, b, crop = crop_pair(frames[j], current, cfg.motion)
                if j == i - 1:
                    step = prepared["step"]
                    inliers = prepared["diagnostics"].get("inliers", 0)
                else:
                    p, q = ufm_pairs(model, a, b, crop, frames[j], current, k, cfg.motion)
                    step, inliers, _ = robust_fit(p, q, cfg.motion)
                ref = frames[j]["obs"]["points"]
                fit = WholeSurfaceFit(ref, current, k, slow_shape, fit_kernels=kernels, search_pool=pool)
                starts = [("continuity", fixed[i - 1] @ np.linalg.inv(fixed[j]))]
                if step is not None:
                    starts.append(("ufm", step))
                starts.extend(
                    (f"pca-{n}", v) for n, v in enumerate(pca_candidates(ref, current["obs"]["points"]))
                )
                solutions = []
                for name, start in starts:
                    try:
                        pose, detail = fit.refine(start)
                        distances = cKDTree(move(ref, pose)).query(current["all_surface_points"])[0] * 1000
                        solutions.append(
                            {
                                "start": name,
                                "pose": pose.tolist(),
                                "cost": detail["cost"],
                                "converged": detail["converged"],
                                "p90_mm": float(np.percentile(distances, 90)),
                            }
                        )
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        print("fit error", i, j, name, str(exc), flush=True)
                if step is not None:
                    solutions.append(
                        {
                            "start": "ufm-unrefined",
                            "pose": step.tolist(),
                            "cost": fit.score(step),
                            "converged": None,
                        }
                    )
                pair_rows.append(
                    {"source": j, "target": i, "motion_inliers": inliers, "solutions": solutions}
                )
            print(
                "search frame",
                i,
                "sources",
                sorted(refs),
                "elapsed_s",
                round(time.perf_counter() - started, 1),
                flush=True,
            )
        if i % 20 == 0:
            (out / "clock_fit_progress.json").write_text(
                json.dumps({"completed_frame": i, "rows": rows, "pairs": pair_rows})
            )
            print("frame", i, "elapsed_s", round(time.perf_counter() - started, 1), flush=True)
result = {
    "config": {
        "observation": asdict(obs_cfg),
        "shape": asdict(cfg.shape),
        "search_shape": asdict(slow_shape),
        "motion": asdict(cfg.motion),
    },
    "frames": rows,
    "pairs": pair_rows,
    "elapsed_s": time.perf_counter() - started,
    "labels_used_for_fitting": False,
}
(out / "clock_fits.json").write_text(json.dumps(result))
print("DONE", out / "clock_fits.json", round(result["elapsed_s"], 1), flush=True)
