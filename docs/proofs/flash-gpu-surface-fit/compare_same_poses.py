"""Time CPU and CUDA residuals on exactly the same solver candidate poses."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from compare import cases, fit_input

from lerobot.policies.flash.experimental_gpu_fit import GpuWholeSurfaceFit


def candidate_poses(cpu, proposal, limit):
    poses = []
    original = cpu.residual

    def record(pose):
        poses.append(np.asarray(pose).copy())
        return original(pose)

    cpu.residual = record
    try:
        cpu.correct(proposal)
    finally:
        cpu.residual = original
    indices = np.linspace(0, len(poses) - 1, min(limit, len(poses))).astype(int)
    return [poses[index] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {}
    for name, reference, depth, mask, intrinsics, proposal in cases():
        cpu = fit_input(reference, depth, mask, intrinsics)
        poses = candidate_poses(cpu, proposal, args.poses)
        gpu = GpuWholeSurfaceFit(cpu)
        gpu.residuals(gpu.tensor(poses[0])[None], gpu._hull_indices(poses[0]))
        torch.cuda.synchronize()
        cpu_ms, gpu_ms = [], []
        maximum_difference = 0.0
        for _ in range(args.repeats):
            start = time.perf_counter()
            cpu_values = [cpu.residual(pose) for pose in poses]
            cpu_ms.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            gpu_values = []
            for pose in poses:
                residual = gpu.residuals(gpu.tensor(pose)[None], gpu._hull_indices(pose))[0]
                gpu_values.append(residual.cpu().numpy())
            torch.cuda.synchronize()
            gpu_ms.append((time.perf_counter() - start) * 1000)
            maximum_difference = max(
                maximum_difference,
                max(
                    float(np.max(np.abs(cpu_r - gpu_r)))
                    for cpu_r, gpu_r in zip(cpu_values, gpu_values, strict=True)
                ),
            )
        report[name] = {
            "candidate_poses": len(poses),
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms,
            "residual_max_abs_difference": maximum_difference,
        }
        print(
            f"{name}: {len(poses)} identical poses; CPU {np.median(cpu_ms):.0f} ms; "
            f"GPU {np.median(gpu_ms):.0f} ms; max residual difference {maximum_difference:.2e}",
            flush=True,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
