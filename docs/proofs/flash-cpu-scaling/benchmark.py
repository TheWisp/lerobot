"""Compare exact CPU residual batching and threading on recorded FLASH fits.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src from the repo root.
The candidate changes only how SciPy's independent finite-difference residuals
are evaluated; the optimizer, objective, initialization and branch choice stay
the same. This script never changes the live tracker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.surface_grounding import WholeSurfaceFit, project
from lerobot.policies.flash.surface_observation import move

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flash-gpu-surface-fit"))
from compare import cases, fit_input  # noqa: E402
from compare_same_poses import candidate_poses  # noqa: E402


class BatchedNearestFit(WholeSurfaceFit):
    """Research-only SciPy fitter whose Jacobian queries share two tree calls."""

    @classmethod
    def from_cpu(cls, cpu: WholeSurfaceFit, query_workers: int):
        result = object.__new__(cls)
        result.__dict__.update(cpu.__dict__)
        result.batch_query_workers = query_workers
        return result

    def batch_residuals(self, poses: list[np.ndarray]) -> list[np.ndarray]:
        if not poses:
            return []
        predicted = [move(self.reference_queries, pose) for pose in poses]
        target_local = [(self.target_queries - pose[:3, 3]) @ pose[:3, :3] for pose in poses]
        model_distances = self.target_tree.query(np.concatenate(predicted), workers=self.batch_query_workers)[
            0
        ].reshape(len(poses), -1)
        target_distances = self.reference_tree.query(
            np.concatenate(target_local), workers=self.batch_query_workers
        )[0].reshape(len(poses), -1)
        return [
            self._residual_with_distances(pose, points, model, target)
            for pose, points, model, target in zip(
                poses, predicted, model_distances, target_distances, strict=True
            )
        ]

    def _residual_with_distances(self, pose, predicted, model_distance, target_distance):
        # Mirrors WholeSurfaceFit.residual exactly, except its two cKDTree
        # distances were computed with the other Jacobian candidate poses.
        uv = project(predicted, self.k)
        height, width = self.observation["depth"].shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        coordinates = (np.clip(uv[:, 1], 0, height - 1), np.clip(uv[:, 0], 0, width - 1))
        mask_distance = map_coordinates(self.outside_mask, coordinates, order=1, mode="nearest")
        tolerance = self.config.depth_tolerance_m
        cy, cx = coordinates
        x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = cx - x0, cy - y0
        weights = np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
        depth = self.observation["depth"]
        rays = np.column_stack((depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]))
        known = (rays > 0) & inside[:, None]
        ray_visibility = np.clip((rays - predicted[:, 2, None] + tolerance) / tolerance, 0, 1)
        depth_visibility = np.sum(weights * known * ray_visibility, axis=1)
        visibility = depth_visibility + np.sum(weights * ~known, axis=1)
        model_distance = np.minimum(model_distance, self.config.max_surface_distance_m)
        target_distance = np.minimum(target_distance, self.config.max_surface_distance_m)
        silhouette_m = (
            np.maximum(mask_distance - self.config.mask_boundary_tolerance_px, 0)
            * predicted[:, 2]
            / self.k[0, 0]
        )
        return np.concatenate(
            (
                np.sqrt(depth_visibility / len(predicted)) * model_distance / tolerance,
                target_distance / (tolerance * np.sqrt(len(self.target_queries))),
                np.sqrt(visibility / len(predicted)) * silhouette_m / tolerance,
                self.uncovered_mask(project(move(self.envelope, pose), self.k))
                * np.median(predicted[:, 2])
                / (self.k[0, 0] * tolerance * np.sqrt(len(self.mask_outline))),
            )
        )

    def refine(self, initial: np.ndarray) -> tuple[np.ndarray, dict]:
        center = move(self.reference, initial).mean(0)

        def pose_from_delta(delta):
            rotation = Rotation.from_rotvec(delta[:3]).as_matrix()
            pose = initial.copy()
            pose[:3, :3] = rotation @ initial[:3, :3]
            pose[:3, 3] = rotation @ (initial[:3, 3] - center) + center + delta[3:]
            return pose

        def batch_map(_fun, deltas):
            return self.batch_residuals([pose_from_delta(delta) for delta in deltas])

        solution = least_squares(
            lambda delta: self.residual(pose_from_delta(delta)),
            np.zeros(6),
            x_scale="jac",
            max_nfev=self.config.max_evaluations,
            ftol=self.config.solver_tolerance,
            xtol=self.config.solver_tolerance,
            gtol=self.config.solver_tolerance,
            workers=batch_map,
        )
        pose = pose_from_delta(solution.x)
        return pose, {
            "initial_cost": self.score(initial),
            "cost": self.score(pose),
            "evaluations": solution.nfev,
            "converged": bool(solution.success),
            "fit_reference_points": len(self.reference_queries),
            "fit_target_points": len(self.target_queries),
        }


class ThreadedResidualFit(BatchedNearestFit):
    """Use SciPy's finite-difference map with a persistent CPU thread pool."""

    @classmethod
    def from_cpu(cls, cpu: WholeSurfaceFit, pool: ThreadPoolExecutor):
        result = super().from_cpu(cpu, 0)
        result.pool = pool
        return result

    def batch_residuals(self, poses: list[np.ndarray]) -> list[np.ndarray]:
        return list(self.pool.map(self.residual, poses))


def timed(work, repeats: int) -> tuple[object, list[float]]:
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = work()
        durations.append((time.perf_counter() - start) * 1000)
    return result, durations


def measure_case(case, repeats: int) -> dict:
    name, reference, depth, mask, intrinsics, proposal = case
    cpu = fit_input(reference, depth, mask, intrinsics)
    baseline, baseline_ms = timed(lambda: cpu.correct(proposal), repeats)
    sample_poses = candidate_poses(cpu, proposal, 7)
    single_query = WholeSurfaceFit(
        cpu.reference,
        cpu.observation,
        cpu.k,
        replace(cpu.config, nn_queries_per_worker=10**9),
    )
    (single_pose, _), single_query_ms = timed(lambda: single_query.correct(proposal), repeats)
    np.testing.assert_array_equal(single_pose, baseline[0])
    report = {
        "baseline_ms": baseline_ms,
        "single_query_worker_ms": single_query_ms,
        "batched": {},
        "threaded_jacobian": {},
    }
    for workers in (2, 4, 8):
        batched = BatchedNearestFit.from_cpu(cpu, workers)
        expected = [cpu.residual(pose) for pose in sample_poses]
        actual = batched.batch_residuals(sample_poses)
        maximum_difference = max(
            float(np.max(np.abs(left - right))) for left, right in zip(expected, actual, strict=True)
        )
        (pose, detail), milliseconds = timed(lambda fitter=batched: fitter.correct(proposal), repeats)
        position_delta_mm = float(np.linalg.norm(pose[:3, 3] - baseline[0][:3, 3]) * 1000)
        angle_delta_deg = float(
            np.degrees(Rotation.from_matrix(pose[:3, :3] @ baseline[0][:3, :3].T).magnitude())
        )
        report["batched"][str(workers)] = {
            "ms": milliseconds,
            "residual_max_abs_difference": maximum_difference,
            "pose_delta_mm": position_delta_mm,
            "pose_delta_deg": angle_delta_deg,
            "evaluations": detail["evaluations"],
            "cost": detail["cost"],
        }
        if maximum_difference > 1e-9 or position_delta_mm > 1e-6 or angle_delta_deg > 1e-6:
            raise AssertionError(f"{name}: batched output changed with {workers} query workers")
    print(
        name,
        "baseline",
        round(float(np.median(baseline_ms)), 1),
        "batched",
        {n: round(float(np.median(v["ms"])), 1) for n, v in report["batched"].items()},
        flush=True,
    )
    for threads in (2, 4, 8):
        with ThreadPoolExecutor(max_workers=threads) as pool:
            threaded = ThreadedResidualFit.from_cpu(cpu, pool)
            (pose, detail), milliseconds = timed(lambda fitter=threaded: fitter.correct(proposal), repeats)
        position_delta_mm = float(np.linalg.norm(pose[:3, 3] - baseline[0][:3, 3]) * 1000)
        angle_delta_deg = float(
            np.degrees(Rotation.from_matrix(pose[:3, :3] @ baseline[0][:3, :3].T).magnitude())
        )
        report["threaded_jacobian"][str(threads)] = {
            "ms": milliseconds,
            "pose_delta_mm": position_delta_mm,
            "pose_delta_deg": angle_delta_deg,
            "evaluations": detail["evaluations"],
            "cost": detail["cost"],
        }
        if position_delta_mm > 1e-6 or angle_delta_deg > 1e-6:
            raise AssertionError(f"{name}: threaded output changed with {threads} workers")
    print(
        name,
        "threaded Jacobian",
        {n: round(float(np.median(v["ms"])), 1) for n, v in report["threaded_jacobian"].items()},
        flush=True,
    )
    return report


def measure_objects(chosen: list, repeats: int) -> dict:
    prepared = [
        (name, fit_input(ref, depth, mask, k), proposal) for name, ref, depth, mask, k, proposal in chosen
    ]
    single_query_prepared = [
        (
            name,
            WholeSurfaceFit(
                cpu.reference,
                cpu.observation,
                cpu.k,
                replace(cpu.config, nn_queries_per_worker=10**9),
            ),
            proposal,
        )
        for name, cpu, proposal in prepared
    ]
    report = {}
    baseline_poses = None
    for mode in (
        "baseline",
        "single_query",
        "batch_4",
        "batch_8",
        "jac_threads_2",
        "jac_threads_4",
        "jac_threads_8",
        "single_query_jac_threads_4",
    ):
        jac_threads = int(mode.rsplit("_", 1)[1]) if mode.startswith("jac_threads_") else 0
        if mode == "single_query_jac_threads_4":
            jac_threads = 4
        jac_pool = ThreadPoolExecutor(max_workers=jac_threads) if jac_threads else None
        try:
            if mode == "baseline":
                fits = [(name, cpu, proposal) for name, cpu, proposal in prepared]
            elif mode == "single_query":
                fits = single_query_prepared
            elif mode.startswith("batch_"):
                workers = int(mode.rsplit("_", 1)[1])
                fits = [
                    (name, BatchedNearestFit.from_cpu(cpu, workers), proposal)
                    for name, cpu, proposal in prepared
                ]
            else:
                fits = [
                    (name, ThreadedResidualFit.from_cpu(cpu, jac_pool), proposal)
                    for name, cpu, proposal in (
                        single_query_prepared if mode == "single_query_jac_threads_4" else prepared
                    )
                ]
            for object_workers in (1, 2, 3):
                with ThreadPoolExecutor(max_workers=object_workers) as pool:
                    results, durations = timed(
                        lambda selected=fits: list(pool.map(lambda item: item[1].correct(item[2]), selected)),
                        repeats,
                    )
                poses = [result[0] for result in results]
                if baseline_poses is None:
                    baseline_poses = poses
                for pose, baseline in zip(poses, baseline_poses, strict=True):
                    np.testing.assert_array_equal(pose, baseline)
                report[f"{mode}_objects_{object_workers}"] = durations
                print(
                    "three objects, mode",
                    mode,
                    "object workers",
                    object_workers,
                    "median_ms",
                    round(float(np.median(durations)), 1),
                    flush=True,
                )
        finally:
            if jac_pool:
                jac_pool.shutdown()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    selected = [
        case for case in cases() if case[0] in {"ring_visible_motion", "dowel_held_315", "dowel_static_13"}
    ]
    report = {
        "scipy": scipy.__version__,
        "cpu_count": os.cpu_count(),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "cases": {case[0]: measure_case(case, args.repeats) for case in selected},
        "three_object_makespan_ms": measure_objects(selected, args.repeats),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
