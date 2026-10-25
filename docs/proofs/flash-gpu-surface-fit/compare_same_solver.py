"""Compare CPU and CUDA residuals under the identical production optimizer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from compare import cases, fit_input
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_gpu_fit import GpuResidualScipyFit, GpuSolverConfig


def timed_fit(fitter, proposal, repeats):
    times = []
    pose, detail = None, None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        pose, detail = fitter.correct(proposal)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return pose, detail, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nearest-method", choices=("brute_force", "tree"), default="brute_force")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {}
    for name, reference, depth, mask, intrinsics, proposal in cases():
        cpu = fit_input(reference, depth, mask, intrinsics)
        start = time.perf_counter()
        gpu = GpuResidualScipyFit(cpu, GpuSolverConfig(nearest_method=args.nearest_method))
        gpu.residual(proposal)
        torch.cuda.synchronize()
        upload_and_warmup_ms = (time.perf_counter() - start) * 1000
        cpu_pose, cpu_detail, cpu_times = timed_fit(cpu, proposal, args.repeats)
        gpu_pose, gpu_detail, gpu_times = timed_fit(gpu, proposal, args.repeats)
        position_delta = np.linalg.norm(cpu_pose[:3, 3] - gpu_pose[:3, 3]) * 1000
        rotation_delta = np.degrees(Rotation.from_matrix(gpu_pose[:3, :3] @ cpu_pose[:3, :3].T).magnitude())
        report[name] = {
            "nearest_method": args.nearest_method,
            "cpu_ms": cpu_times,
            "gpu_same_solver_ms": gpu_times,
            "gpu_upload_and_warmup_ms": upload_and_warmup_ms,
            "cpu_cost": cpu_detail["cost"],
            "gpu_pose_cpu_cost": cpu.score(gpu_pose),
            "cpu_evaluations": cpu_detail["evaluations"],
            "gpu_evaluations": gpu_detail["evaluations"],
            "cpu_used_coarse": cpu_detail["used_coarse"],
            "gpu_used_coarse": gpu_detail["used_coarse"],
            "position_delta_mm": position_delta,
            "rotation_delta_deg": rotation_delta,
        }
        print(
            f"{name}: CPU {np.median(cpu_times):.0f} ms; GPU same solver {np.median(gpu_times):.0f} ms; "
            f"delta {position_delta:.3f} mm, {rotation_delta:.3f} deg; "
            f"evals {cpu_detail['evaluations']}/{gpu_detail['evaluations']}",
            flush=True,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
