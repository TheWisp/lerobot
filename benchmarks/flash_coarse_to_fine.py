"""Isolated surface-resolution experiment; production configuration is unchanged.

The coarse solve samples both measured point sets. Full-resolution masks, depth,
outline vertices, image/group constraints and the pose origin remain unchanged.
The final solve and downstream publication/growth checks use the original full
geometry. --fine-evaluations optionally reduces only the final iteration budget.
Neither experiment changes the running server or pipeline defaults.

Examples:
  python -m benchmarks.flash_coarse_to_fine control --output /tmp/coarse-control
  python -m benchmarks.flash_coarse_to_fine replay --points 256 --evaluations 12 -- <flash_runtime_benchmark arguments>
"""

import argparse
import json
import sys
import time
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial import cKDTree

from benchmarks.flash_partial_surface_control import box_observation, perturbed_pose, pose_error
from lerobot.policies.flash.surface_acceleration import PooledTree, SearchPool, compiled_kernels
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation, spatial_sample
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


def coarse_fitter(fitter, points, evaluations, *, simplify_trees=True):
    """An independent coarse objective; no mutation of measured inputs or fine state."""
    coarse = copy(fitter)
    coarse.config = replace(fitter.config, max_fit_points=points, max_evaluations=evaluations)
    coarse.reference_queries = fitter.reference[spatial_sample(fitter.reference, points)]
    coarse.target_queries = fitter.target[spatial_sample(fitter.target, points)]
    if simplify_trees:
        coarse.reference_tree = cKDTree(coarse.reference_queries)
        coarse.target_tree = cKDTree(coarse.target_queries)
        if fitter.search_pool is not None:
            coarse.reference_tree = PooledTree(coarse.reference_tree, fitter.search_pool)
            coarse.target_tree = PooledTree(coarse.target_tree, fitter.search_pool)
    coarse.reference_query_workers = max(
        1, len(coarse.reference_queries) // fitter.config.nn_queries_per_worker
    )
    coarse.target_query_workers = max(1, len(coarse.target_queries) // fitter.config.nn_queries_per_worker)
    coarse.reverse_fits = [
        (camera, coarse_fitter(reverse, points, evaluations, simplify_trees=simplify_trees))
        for camera, reverse in fitter.reverse_fits
    ]
    return coarse


class RefinementTrial:
    """Patch only this benchmark process, keeping every ordinary search starting pose."""

    def __init__(self, points, evaluations, *, simplify_trees=True, fine_evaluations=None):
        if points < 4 or evaluations < 1:
            raise ValueError("positive evaluation budget and at least four coarse points are required")
        self.points, self.evaluations, self.simplify_trees = points, evaluations, simplify_trees
        self.fine_evaluations = fine_evaluations
        self.calls = []

    def __enter__(self):
        original = WholeSurfaceFit.refine

        def refine(fitter, initial, *, admissible=None):
            started = time.perf_counter()
            cached = getattr(fitter, "_coarse_trial_fitter", None)
            key = self.points, self.evaluations, self.simplify_trees
            if cached is None or cached[0] != key:
                coarse = coarse_fitter(
                    fitter, self.points, self.evaluations, simplify_trees=self.simplify_trees
                )
                fitter._coarse_trial_fitter = key, coarse
            else:
                coarse = cached[1]
            prepared = time.perf_counter()
            seed, coarse_detail = original(coarse, initial)
            coarse_done = time.perf_counter()
            fine = copy(fitter)
            if self.fine_evaluations is not None:
                fine.config = replace(fitter.config, max_evaluations=self.fine_evaluations)
            pose, detail = original(fine, seed, admissible=admissible)
            self.calls.append(
                {
                    "prepare_ms": 1000 * (prepared - started),
                    "coarse_ms": 1000 * (coarse_done - prepared),
                    "fine_ms": 1000 * (time.perf_counter() - coarse_done),
                    "coarse_evaluations": coarse_detail["evaluations"],
                    "fine_evaluations": detail["evaluations"],
                    "coarse_reference_queries": len(coarse.reference_queries),
                    "coarse_target_queries": len(coarse.target_queries),
                    "fine_reference_queries": len(fitter.reference_queries),
                    "fine_target_queries": len(fitter.target_queries),
                }
            )
            return pose, detail

        self._patch = patch.object(WholeSurfaceFit, "refine", refine)
        self._patch.start()
        return self

    def __exit__(self, *args):
        self._patch.stop()

    def summary(self):
        return {
            "points": self.points,
            "coarse_max_evaluations": self.evaluations,
            "simplify_trees": self.simplify_trees,
            "fine_max_evaluations": self.fine_evaluations,
            "refinements": len(self.calls),
            "mean_per_refinement": {
                key: float(np.mean([row[key] for row in self.calls])) for key in self.calls[0]
            }
            if self.calls
            else {},
        }


class PointToPlaneTrial:
    """Open3D voxel-pyramid initializer, followed by full-resolution fitting/checks.

    The coarse objective uses local surface planes, not distance to sparse dots.
    Its mask/depth contradictions still have to be checked by the original fitter.
    Normal/search defaults follow the existing RGB-D registration experiment.
    """

    def __init__(
        self, voxel_sizes=(0.006, 0.003), iterations=20, fine_evaluations=None, keep_better_seed=False
    ):
        import open3d as o3d

        self.o3d = o3d
        self.voxel_sizes = voxel_sizes
        self.iterations = iterations
        self.fine_evaluations = fine_evaluations
        self.keep_better_seed = keep_better_seed
        self.calls = []

    def __enter__(self):
        original = WholeSurfaceFit.refine
        o3d = self.o3d
        registration = o3d.pipelines.registration

        def refine(fitter, initial, *, admissible=None):
            started = time.perf_counter()
            cached = getattr(fitter, "_icp_trial_levels", None)
            if cached is None or cached[0] != tuple(self.voxel_sizes):
                full = [
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                    for points in (fitter.reference, fitter.target)
                ]
                levels = []
                for voxel in self.voxel_sizes:
                    source, target = (cloud.voxel_down_sample(voxel) for cloud in full)
                    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
                    levels.append((voxel, source, target))
                fitter._icp_trial_levels = tuple(self.voxel_sizes), levels
            else:
                levels = cached[1]
            prepared = time.perf_counter()
            seed = initial
            for voxel, source, target in levels:
                result = registration.registration_icp(
                    source,
                    target,
                    voxel * 3,
                    seed,
                    registration.TransformationEstimationPointToPlane(),
                    registration.ICPConvergenceCriteria(max_iteration=self.iterations),
                )
                seed = result.transformation
            coarse_seed_used = bool(levels)
            if self.keep_better_seed and fitter.score(seed) >= fitter.score(initial):
                seed = initial
                coarse_seed_used = False
            coarse_done = time.perf_counter()
            fine = copy(fitter)
            if self.fine_evaluations is not None:
                fine.config = replace(fitter.config, max_evaluations=self.fine_evaluations)
            pose, detail = original(fine, seed, admissible=admissible)
            self.calls.append(
                {
                    "prepare_ms": 1000 * (prepared - started),
                    "coarse_ms": 1000 * (coarse_done - prepared),
                    "fine_ms": 1000 * (time.perf_counter() - coarse_done),
                    "fine_evaluations": detail["evaluations"],
                    "coarse_reference_queries": len(levels[0][1].points) if levels else 0,
                    "coarse_target_queries": len(levels[0][2].points) if levels else 0,
                    "coarse_seed_used": coarse_seed_used,
                    "fine_reference_queries": len(fitter.reference_queries),
                    "fine_target_queries": len(fitter.target_queries),
                }
            )
            return pose, detail

        self._patch = patch.object(WholeSurfaceFit, "refine", refine)
        self._patch.start()
        return self

    def __exit__(self, *args):
        self._patch.stop()

    def summary(self):
        return {
            "initializer": "open3d_point_to_plane",
            "voxel_sizes_m": self.voxel_sizes,
            "max_iterations_per_level": self.iterations,
            "fine_max_evaluations": self.fine_evaluations,
            "keep_better_seed": self.keep_better_seed,
            "refinements": len(self.calls),
            "mean_per_refinement": {
                key: float(np.mean([row[key] for row in self.calls])) for key in self.calls[0]
            }
            if self.calls
            else {},
        }


def problems():
    """Known synthetic motion plus recorded difficult surface observations."""
    source, k, _ = box_observation(0)
    for angle in (20, 40, 60):
        current, _, truth = box_observation(angle)
        yield f"box-{angle}", source["obs"]["points"], current, k, perturbed_pose(truth), truth, source
    fixtures = Path(__file__).resolve().parents[1] / "tests/policies/flash/fixtures"
    for name in ("ring_visible_motion_20261016", "dowel_held_frame_315"):
        with np.load(fixtures / f"{name}.npz") as data:
            ring = "source_surface" in data
            reference = data["source_surface" if ring else "model_points"]
            depth = data["target_depth_mm" if ring else "depth"].astype(np.float32) * 0.001
            mask = data["target_mask" if ring else "mask"].astype(bool)
            k = data["intrinsics"]
            initial = np.eye(4) if ring else data["candidate"]
        obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
        current = {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": obs["points"]}
        yield name, reference, current, k, initial, None, None


def controls(
    output, repeats, evaluations, point_to_plane=False, fine_evaluations=None, keep_better_seed=False
):
    from contextlib import nullcontext

    from lerobot.policies.flash.surface_grounding import ShapeConfig

    output.mkdir(parents=True, exist_ok=False)
    kernels = compiled_kernels()
    cv2.setNumThreads(1)
    variants = {
        "baseline": None,
        "coarse128": (128, True),
        "coarse256": (256, True),
        "queries256": (256, False),
    }
    if point_to_plane:
        variants = {"baseline": None, "planes6-3mm": (0.006, 0.003), "planes12-6mm": (0.012, 0.006)}
        if fine_evaluations is not None:
            variants["detail-only-short"] = ()
    rows = []
    with SearchPool(8) as pool:
        for name, reference, current, k, initial, truth, source in problems():
            for repeat in range(repeats):
                # Rotate method order to avoid always measuring baseline first.
                order = list(variants)
                order = order[repeat % len(order) :] + order[: repeat % len(order)]
                for variant in order:
                    spec = variants[variant]
                    fit = WholeSurfaceFit(
                        reference,
                        current,
                        k,
                        ShapeConfig(),
                        fit_kernels=kernels,
                        search_pool=pool,
                        reference_observation=source,
                    )
                    context = nullcontext()
                    if spec is not None:
                        context = (
                            PointToPlaneTrial(spec, evaluations, fine_evaluations, keep_better_seed)
                            if point_to_plane
                            else RefinementTrial(
                                spec[0],
                                evaluations,
                                simplify_trees=spec[1],
                                fine_evaluations=fine_evaluations,
                            )
                        )
                    with context as trial:
                        start = time.perf_counter()
                        pose, detail = fit.correct(initial)
                        elapsed = 1000 * (time.perf_counter() - start)
                    row = {
                        "case": name,
                        "variant": variant,
                        "repeat": repeat,
                        "fit_ms": elapsed,
                        "full_cost": fit.score(pose),
                        "pose": pose.tolist(),
                        "full_check": check_surface_contradiction(fit, pose, PoseUncertaintyConfig()),
                        "fit": detail,
                        "stages": trial.summary() if trial else None,
                        "truth_error": pose_error(reference, pose, truth) if truth is not None else None,
                    }
                    rows.append(row)
                    print(name, variant, round(elapsed, 1), row["truth_error"], flush=True)
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    control = sub.add_parser("control")
    control.add_argument("--output", type=Path, required=True)
    control.add_argument("--repeats", type=int, default=3)
    control.add_argument("--evaluations", type=int, default=12)
    control.add_argument("--point-to-plane", action="store_true")
    control.add_argument("--fine-evaluations", type=int)
    control.add_argument("--keep-better-seed", action="store_true")
    replay = sub.add_parser("replay")
    replay.add_argument("--points", type=int, default=256)
    replay.add_argument("--evaluations", type=int, required=True)
    replay.add_argument("--dense-trees", action="store_true")
    replay.add_argument("--point-to-plane", action="store_true")
    replay.add_argument("--fine-evaluations", type=int)
    replay.add_argument("--keep-better-seed", action="store_true")
    replay.add_argument("--voxel-sizes", type=float, nargs="+", default=[0.006, 0.003])
    replay.add_argument("runtime_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "control":
        return controls(
            args.output,
            args.repeats,
            args.evaluations,
            args.point_to_plane,
            args.fine_evaluations,
            args.keep_better_seed,
        )
    from benchmarks.flash_runtime_benchmark import main as run_runtime

    runtime_args = args.runtime_arguments
    if runtime_args and runtime_args[0] == "--":
        runtime_args = runtime_args[1:]
    output = Path(runtime_args[runtime_args.index("--output") + 1])
    adapter_source = Path(__file__).read_bytes()
    context = (
        PointToPlaneTrial(args.voxel_sizes, args.evaluations, args.fine_evaluations, args.keep_better_seed)
        if args.point_to_plane
        else RefinementTrial(
            args.points,
            args.evaluations,
            simplify_trees=not args.dense_trees,
            fine_evaluations=args.fine_evaluations,
        )
    )
    with context as trial, patch.object(sys, "argv", ["flash_runtime_benchmark", *runtime_args]):
        run_runtime()
    (output / "coarse_to_fine.json").write_text(json.dumps(trial.summary(), indent=2) + "\n")
    # The runtime snapshot does not otherwise include this isolated adapter.
    (output / "code/benchmarks/flash_coarse_to_fine.py").write_bytes(adapter_source)


if __name__ == "__main__":
    main()
