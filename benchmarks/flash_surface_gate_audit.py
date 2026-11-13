"""Hold replay state fixed while auditing labeled poses against stored geometry.

Labels are used only by detached diagnostic fits. They never supply the replay's
seed, pose, map, group motion, or acceptance decision. The labeled-placement map
contains exactly the same sampled patch measurements as the tracked-placement
map; only their placement changes. This diagnoses placement error, not a proposed
tracking algorithm. Use append-only surface settings for this comparison.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from benchmarks.flash_tracker_experiment import run_case
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.motion_groups import GroupConfig
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_grounding import SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_session_config import defaults
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def place_measurements(patches, poses):
    """Keep sample identities/counts fixed; replace only camera-to-map placement."""
    return np.concatenate([move(p.measured, np.linalg.inv(poses[p.frame])) for p in patches])


def audit_pose(maps, views, current, intrinsics, shape, limits, pose):
    """A crossed map/view test with one immutable pose, observation and policy."""
    checks = {}
    for map_name, points in maps.items():
        for view_name, references in views.items():
            fit = WholeSurfaceFit(points, current, intrinsics, shape, reference_views=references)
            checks[f"{map_name}/{view_name}"] = fit.check_pose(pose, **limits)
    return checks


def relative_labels(absolute, reference_frame):
    return absolute @ np.linalg.inv(absolute[reference_frame])


def run_audit(case, output, *, focus_frames=(), enabled=True):
    if len(case["roles"]) != 1:
        raise ValueError("This diagnostic requires a single labeled role")
    evaluation = case["evaluation"][case["roles"][0]]
    if evaluation["format"] != "poses_npy":
        raise ValueError("This diagnostic requires poses_npy labels")
    label_path = Path(evaluation["path"]).expanduser()
    absolute = np.asarray(evaluation.get("camera_from_label_camera", np.eye(4))) @ np.load(
        label_path
    ).reshape(-1, 4, 4)
    settings = defaults()
    # Explicitly freeze the coupled refinement trial off. No local behavior
    # is toggled inside a run, and no label participates in map maintenance.
    settings["surface_experiment"].pop("refinement_max_weight", None)
    settings["retain_motion_reference"] = False
    recipe = GrowingSurfaceConfig(**settings["surface_experiment"])
    vision = VisionPipelineConfig.from_dict(settings["config"])
    # Refuse reused destinations before writing a manifest: mixing a new
    # recipe/source hash with an older replay would invalidate the comparison.
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    sources = [*sorted((source_root / "src/lerobot/policies/flash").glob("*.py")), Path(__file__)]
    manifest = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip(),
        "files": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "settings": settings,
        "case": case,
        "diagnostics_enabled": enabled,
        "scope": "Labels change detached diagnostic geometry only; replay remains unlabeled.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    original_update = GrowingSurfaceSession.update
    rows = []

    def checked_update(self, prepared, time_s, **kwargs):
        frame = prepared["frame_id"]
        row = {"frame": frame, "time_s": time_s}
        ready = self.pose is not None and prepared["status"] is None and len(self.store.points) >= 3
        if enabled and ready:
            labels = relative_labels(absolute, self.reference_frame)
            # _fit_views returns latest accepted first under fit_view_limit=1.
            # Reconstruct its frame IDs independently, without guessing from
            # an observation's optional frame_id field.
            view_by_frame = (
                {
                    p.frame: SurfaceFitView(
                        {
                            "depth": p.view.depth,
                            "mask": p.view.mask,
                            "obs": {"points": p.view.measured_points},
                            **({"scene_depth": p.view.scene_depth} if p.view.scene_depth is not None else {}),
                        },
                        p.view.intrinsics,
                        p.camera_from_reference,
                    )
                    for p in self.store.patches.values()
                    if p.active and p.view is not None
                }
                if self.config.partial_surface_matching
                else {}
            )
            if self.config.partial_surface_matching and self.accepted_observation is not None:
                view_by_frame[self.accepted_frame] = SurfaceFitView(
                    self.accepted_observation, self.accepted_intrinsics, self.accepted_pose
                )
            view_frames = list(view_by_frame)
            if self.config.fit_view_limit is not None:
                view_frames = sorted(view_frames, reverse=True)[: self.config.fit_view_limit]
            tracked_views = tuple(view_by_frame[f] for f in view_frames)
            label_views = tuple(
                replace(view_by_frame[f], camera_from_reference=labels[f]) for f in view_frames
            )
            patches = [p for p in self.store.patches.values() if p.active]
            maps = {
                "tracked": self.store.points.copy(),
                "patch_samples_tracked": place_measurements(
                    patches, {p.frame: p.camera_from_reference for p in patches}
                ),
                "patch_samples_labeled": place_measurements(patches, labels),
                "initial": self.reference.copy(),
            }
            views = {"tracked_views": tracked_views, "labeled_views": label_views, "no_retained_views": ()}
            limits = {
                "max_outlier_fraction": self.config.uncertainty.max_visible_outlier_fraction,
                "agreement_m": self.config.agreement_m,
                "min_overlap": self.config.min_overlap,
            }
            row.update(
                reference_frame=self.reference_frame,
                accepted_frame=self.accepted_frame,
                patch_frames=[p.frame for p in patches],
                retained_frames=view_frames,
                map_points={k: len(v) for k, v in maps.items()},
                label_checks=audit_pose(
                    maps,
                    views,
                    prepared["current"],
                    prepared["intrinsics"],
                    self.shape_config,
                    limits,
                    labels[frame],
                ),
            )
            if frame in focus_frames:
                row["tolerance_sweep"] = {
                    str(mm): audit_pose(
                        {"tracked": maps["tracked"]},
                        {"tracked_views": tracked_views},
                        prepared["current"],
                        prepared["intrinsics"],
                        replace(self.shape_config, depth_tolerance_m=mm / 1000),
                        limits,
                        labels[frame],
                    )
                    for mm in (3, 5, 8, 10, 15)
                }
                # Original arrays for repeatable geometry-only analysis. Never
                # resample or re-fit when a new comparison question arises.
                arrays = {
                    **{f"map_{k}": v for k, v in maps.items()},
                    "label_pose": labels[frame],
                    "intrinsics": prepared["intrinsics"],
                    "depth": prepared["current"]["depth"],
                    "mask": prepared["current"]["mask"],
                    "points": prepared["current"]["obs"]["points"],
                }
                arrays.update(
                    {
                        f"obs_{k}": v
                        for k, v in prepared["current"]["obs"].items()
                        if isinstance(v, np.ndarray)
                    }
                )
                if "scene_depth" in prepared["current"]:
                    arrays["scene_depth"] = prepared["current"]["scene_depth"]
                for i, view in enumerate(tracked_views):
                    arrays.update(
                        {
                            f"view_{i}_depth": view.observation["depth"],
                            f"view_{i}_mask": view.observation["mask"],
                            f"view_{i}_points": view.observation["obs"]["points"],
                            f"view_{i}_intrinsics": view.intrinsics,
                            f"view_{i}_pose": view.camera_from_reference,
                            f"view_{i}_label_pose": labels[view_frames[i]],
                        }
                    )
                    arrays.update(
                        {
                            f"view_{i}_obs_{k}": v
                            for k, v in view.observation["obs"].items()
                            if isinstance(v, np.ndarray)
                        }
                    )
                    if "scene_depth" in view.observation:
                        arrays[f"view_{i}_scene_depth"] = view.observation["scene_depth"]
                np.savez_compressed(output / f"frame-{frame}.npz", **arrays)
        result = original_update(self, prepared, time_s, **kwargs)
        row.update(
            status=result["status"],
            pose=result.get("camera_from_reference"),
            reason=result["diagnostics"].get("reason"),
            fit_seed_source=result["diagnostics"].get("fit_seed_source"),
        )
        rows.append(row)
        return result

    cv2.setNumThreads(1)
    kernels = compiled_kernels() if vision.compiled_surface_fit else None
    pool = SearchPool(vision.shape.batch_query_workers) if vision.reuse_surface_search else None
    try:
        with patch.object(GrowingSurfaceSession, "update", checked_update):
            summary = run_case(
                case,
                output / "run",
                None,
                vision,
                kernels,
                pool,
                retain_motion_reference=False,
                record_correspondences=False,
                surface_experiment=recipe,
                group_experiment=GroupConfig.from_dict(settings["group_experiment"]),
                shared_matching_config=SharedMatchingConfig.from_dict(settings["shared_matching"]),
                single_method="group_growing",
            )
    finally:
        if pool is not None:
            pool.close()
    (output / "audit.json").write_text(json.dumps(rows, default=lambda x: x.tolist(), indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["roles"], indent=2), flush=True)
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case", default="laval-clock-interaction-hard")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--focus", type=int, nargs="*", default=[77, 78, 108, 121])
    parser.add_argument(
        "--no-audit", action="store_true", help="Control replay to test that labels have no effect"
    )
    args = parser.parse_args()
    spec = json.loads(args.cases.read_text())[args.case]
    run_audit(spec["case"], args.out, focus_frames=args.focus, enabled=not args.no_audit)
