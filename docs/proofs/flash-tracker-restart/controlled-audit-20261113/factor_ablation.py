"""Read-only factor ablation of the `refinement_max_weight` switch on the labeled clock.

Imports the research tree as-is and writes only under --out. The switch gates three
behaviours; each is disabled here by a runtime patch, never by editing the tree:
  R  SurfaceStore._refine            (running-average update of seen points)
  T  SurfaceStore._retire_conflicted (post-acceptance retirement)
  K  object LK motion fed to Kalman.predict (falls back to the group hint's motion)
"""

import argparse
import itertools
import json
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import lerobot.policies.flash.experimental_surface_session as ess
from benchmarks.flash_tracker_experiment import run_case
from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    SurfaceStore,
)
from lerobot.policies.flash.motion_groups import GroupConfig
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_session_config import defaults
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

parser = argparse.ArgumentParser()
parser.add_argument("--cases", required=True, type=Path)
parser.add_argument("--case", default="laval-clock-interaction-hard")
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--arms", nargs="*")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
spec = json.loads(args.cases.read_text())[args.case]
settings = defaults()
settings["retain_motion_reference"] = False
assert settings["config"]["shape"]["registration"]["backend"] == "small_gicp"
assert settings["shared_matching"]["backend"] == "lk"
cv2.setNumThreads(1)
vision = VisionPipelineConfig.from_dict(settings["config"])
kernels = compiled_kernels() if vision.compiled_surface_fit else None

real_update, real_motion = GrowingSurfaceSession.update, ess.MeasuredGroupMotion
local = threading.local()


def no_refine(self, measured, pose, frame, view):
    return np.zeros(len(measured), dtype=bool), {"refined_points": 0, "refinement_mean_shift_mm": 0.0}


def no_retire(self, pose, view, tolerance_m):
    return 0


def stash_hint(self, prepared, time_s, **kwargs):
    local.hint = kwargs.get("hint")
    return real_update(self, prepared, time_s, **kwargs)


def group_motion_only(*a, **k):
    if a and a[-1] == "object_image_matches":
        hint = getattr(local, "hint", None)
        return hint.motion if hint else None
    return real_motion(*a, **k)


arms = [("w0", 0, 1, 1, 1)] + [
    (f"w8_R{r}T{t}K{k}", 8, r, t, k) for r, t, k in itertools.product((0, 1), repeat=3)
]
summary = {}
for name, weight, r, t, k in arms:
    if args.arms and name not in args.arms:
        continue
    recipe = {**settings["surface_experiment"], "refinement_max_weight": weight}
    patches = [patch.object(GrowingSurfaceSession, "update", stash_hint)]
    if not r:
        patches.append(patch.object(SurfaceStore, "_refine", no_refine))
    if not t:
        patches.append(patch.object(SurfaceStore, "_retire_conflicted", no_retire))
    if not k:
        patches.append(patch.object(ess, "MeasuredGroupMotion", group_motion_only))
    pool = SearchPool(vision.shape.batch_query_workers) if vision.reuse_surface_search else None
    try:
        for p in patches:
            p.start()
        result = run_case(
            spec["case"],
            args.out / name,
            None,
            vision,
            kernels,
            pool,
            retain_motion_reference=False,
            record_correspondences=False,
            surface_experiment=GrowingSurfaceConfig(**recipe),
            group_experiment=GroupConfig.from_dict(settings["group_experiment"]),
            shared_matching_config=SharedMatchingConfig.from_dict(settings["shared_matching"]),
            single_method="group_growing",
        )
    finally:
        for p in patches:
            p.stop()
        if pool is not None:
            pool.close()
    (args.out / name / "summary.json").write_text(json.dumps(result, indent=2))
    for role, data in result["roles"].items():
        m = data["methods"]["group_growing"]
        e = m.get("pose_error") or {}
        summary[f"{name}:{role}"] = {
            "available": m["available_after_initialization"],
            "total": m["total_after_initialization"],
            "within_20mm_20deg": e.get("within_20mm_20deg"),
            "within_10mm_10deg": e.get("within_10mm_10deg"),
            "cloud_mm_p90": e.get("cloud_mm_p90"),
            "angle_deg_p90": e.get("angle_deg_p90"),
            "sources": m.get("pose_sources"),
            "input_sha256": result.get("input_sha256"),
        }
        print(name, role, json.dumps(summary[f"{name}:{role}"]), flush=True)
(args.out / "ablation-summary.json").write_text(json.dumps(summary, indent=2))
