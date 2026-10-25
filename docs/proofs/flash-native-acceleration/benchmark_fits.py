"""Measure experimental native kernels and pooled searches on recorded fits."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from candidates import SearchPool, compiled_kernels, fitter_type
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flash-gpu-surface-fit"))
from compare import cases, fit_input  # noqa: E402
from compare_same_poses import candidate_poses  # noqa: E402


def compare_poses(actual, expected):
    return {
        "translation_mm": float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]) * 1000),
        "rotation_deg": float(
            np.degrees(Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude())
        ),
        "matrix_max_abs": float(np.max(np.abs(actual - expected))),
    }


def run(output, repeats):
    result = {}
    kernels = compiled_kernels()
    with ExitStack() as stack:
        pools = {n: stack.enter_context(SearchPool(n)) for n in (2, 4, 8)}
        candidates = stack.enter_context(ThreadPoolExecutor(max_workers=4))
        modes = {
            "baseline": None,
            **{f"pool_{n}": fitter_type(search_pool=pools[n]) for n in pools},
            "compiled": fitter_type(kernels=kernels),
            "compiled_pool_8": fitter_type(kernels=kernels, search_pool=pools[8]),
            "compiled_candidates_4": fitter_type(kernels=kernels, candidate_pool=candidates),
        }
        for name, reference, depth, mask, k, proposal in cases():
            baseline = fit_input(reference, depth, mask, k)
            expected, expected_detail = baseline.correct(proposal)
            poses = candidate_poses(baseline, proposal, 20)
            residuals = [baseline.residual(p) for p in poses]
            row = {}
            for label, factory in modes.items():
                fit = (
                    baseline
                    if factory is None
                    else factory(reference, baseline.observation, k, baseline.config)
                )
                # Compilation and thread startup are excluded from steady timing.
                errors = [
                    float(np.max(np.abs(fit.residual(p) - r))) for p, r in zip(poses, residuals, strict=True)
                ]
                fit.correct(proposal)
                milliseconds, cpu_seconds = [], []
                for _ in range(repeats):
                    start, cpu = time.perf_counter(), time.process_time()
                    actual, detail = fit.correct(proposal)
                    milliseconds.append((time.perf_counter() - start) * 1000)
                    cpu_seconds.append(time.process_time() - cpu)
                row[label] = {
                    "fit_ms": milliseconds,
                    "cpu_seconds": cpu_seconds,
                    "residual_max_abs": max(errors),
                    "pose_difference": compare_poses(actual, expected),
                    "cost": detail["cost"],
                    "baseline_cost": expected_detail["cost"],
                    "evaluations": detail["evaluations"],
                    "baseline_evaluations": expected_detail["evaluations"],
                    "converged": detail["converged"],
                    "baseline_converged": expected_detail["converged"],
                    "used_coarse": detail["used_coarse"],
                    "baseline_used_coarse": expected_detail["used_coarse"],
                }
                print(name, label, json.dumps(row[label]), flush=True)
            result[name] = row
            output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run(args.output, args.repeats)
