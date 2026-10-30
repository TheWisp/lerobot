"""Locate the actual pixels responsible for the clock's ranking reversal."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from lerobot.policies.flash.surface_grounding import WholeSurfaceFit, project
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
root = Path.home() / ".cache/huggingface/lerobot/demos/flash_laval_clock_labeled"
cfg = VisionPipelineConfig.from_dict(
    json.loads((root / "flash/v1/observation-config.json").read_text())["vision_pipeline"]
)
k = np.array(json.loads((root / "flash/v1/source.json").read_text())["camera_intrinsics"])
data = np.load(out / "clock_inputs.npz")
scores = json.loads((out / "clock_objectives.json").read_text())
frames = {}
s = SurfaceTrackingSession(
    None, observation_config=replace(cfg.observation, component_gap_m=None), shape_config=cfg.shape
)
for i in (0, 149, 150):
    s.previous = None
    frames[i] = s.prepare(data["rgb"][i], data["depth"][i], data["mask"][i], k, i)["current"]
    print(
        i,
        "raw mask",
        int(data["mask"][i].sum()),
        "clean",
        int(frames[i]["obs"]["valid_image"].sum()),
        "cloud",
        len(frames[i]["obs"]["points"]),
    )
for j, mode in [(0, "ufm"), (149, "ufm-unrefined")]:
    pair = next(p for p in scores["pairs"] if p["source"] == j and p["target"] == 150)
    sol = next(p for p in pair["solutions"] if p["start"] == mode)
    pose = np.array(sol["pose"])
    fit = WholeSurfaceFit(frames[j]["obs"]["points"], frames[150], k, cfg.shape)
    uv = project(move(fit.envelope, pose), k)
    poly = uv[ConvexHull(uv).vertices]
    misses = fit.uncovered_mask(uv)
    print("from", j, "mask vertices", fit.mask_outline.tolist(), "misses", misses.tolist())
    np.savez(
        out / f"clock_outline_{j}.npz",
        rgb=data["rgb"][150],
        source_rgb=data["rgb"][j],
        source_mask=data["mask"][j],
        source_valid=frames[j]["obs"]["valid_image"],
        mask=data["mask"][150],
        valid=frames[150]["obs"]["valid_image"],
        polygon=poly,
        outline=fit.mask_outline,
        misses=misses,
    )
