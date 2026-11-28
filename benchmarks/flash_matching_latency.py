"""Alternate LK/UFM updates from identical recorded tracker states; no live service calls."""

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from benchmarks.flash_motion_recovery import full_recording
from lerobot.policies.flash.shared_image_tracking import SharedFrameTracker, SharedMatchingConfig
from lerobot.policies.flash.surface_tracking import load_ufm

REVISION = "980825b59331413bb51e5773f9272bfce525bce2"
TARGETS = (5381, 5387, 5395, 5401, 5410, 5467, 5475, 5512)


def gpu_state():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
        text=True,
    ).strip()


def summary(rows):
    return {
        field: {"median": float(np.median(a)), "p95": float(np.percentile(a, 95))}
        for field in ("matching_ms", "shared_update_ms", "remaining_update_ms")
        for a in [np.asarray([r[field] for r in rows])]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 2 or args.warmup_rounds < 1:
        parser.error("need at least two measured rounds and one warmup round")
    cv2.setNumThreads(1)
    meta, frames = full_recording(args.recording)
    config = replace(SharedMatchingConfig.from_dict(meta["settings"]["shared_matching"]), backend="lk")
    tracker = SharedFrameTracker(config)
    snapshots, inputs = {}, {}
    for n in meta["cases"]["cube_again"]:
        if n > max(TARGETS):
            break
        frame, masks = frames[n]
        if n in TARGETS:
            snapshots[n] = copy.deepcopy(tracker)
            inputs[n] = (frame, masks)
        tracker.update(frame, masks)
    assert set(snapshots) == set(TARGETS)
    print("Prepared identical source states:", {n: len(s.ids) for n, s in snapshots.items()}, flush=True)
    before = gpu_state()
    start = time.perf_counter()
    model = load_ufm(revision=REVISION)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    print(f"UFM loaded in {load_s:.2f}s; warming up", flush=True)
    rows = []
    for iteration in range(-args.warmup_rounds, args.rounds):
        for i, n in enumerate(TARGETS):
            order = ("lk", "ufm") if (iteration + i) % 2 == 0 else ("ufm", "lk")
            for backend in order:
                # Restoring source state is setup, not part of the timed update.
                trial = copy.deepcopy(snapshots[n])
                trial.config = replace(config, backend=backend)
                trial.model = model if backend == "ufm" else None
                query_count = len(trial.ids)
                torch.cuda.synchronize()
                start = time.perf_counter()
                trial.update(*inputs[n])
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if iteration >= 0:
                    match_ms = trial.diagnostics["matching_ms"]
                    rows.append(
                        {
                            "round": iteration,
                            "source_frame": snapshots[n].previous.frame_id,
                            "target_frame": n,
                            "backend": backend,
                            "query_count": query_count,
                            "accepted_2d_matches": int(trial.raw[-1].sum()),
                            "output_points": len(trial.ids),
                            "matching_ms": match_ms,
                            "shared_update_ms": elapsed,
                            "remaining_update_ms": elapsed - match_ms,
                        }
                    )
        print(f"Completed {'warmup' if iteration < 0 else 'measured'} round {iteration + 1}", flush=True)
    assert len(rows) == args.rounds * len(TARGETS) * 2
    samples = args.output.with_suffix(".samples.jsonl")
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    report = {
        "recording": args.recording.name,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Warmed one-frame shared tracker updates from identical LK-produced source snapshots. Both backends receive the same source/target RGB, query coordinates, masks and depth. Alternating backend order per pair and round; source restoration and disk reads excluded. CUDA synchronized before/after each timed update. Matching includes RGB-to-gray conversion for LK, or model inference, CPU/GPU transfers and point sampling for UFM. Remaining update includes resizing, depth support, point replenishment and depth lifting. Excludes scene groups, object fitting, SAM, depth inference, camera IO and model startup. This is latency, not a tracking-quality or full-pipeline FPS benchmark. Other processes are not stopped.",
        "hardware": {
            "cpu": cpu,
            "gpu": torch.cuda.get_device_name(),
            "gpu_before": before,
            "gpu_after": gpu_state(),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "torch_threads": torch.get_num_threads(),
        },
        "ufm": {
            "checkpoint": "infinity1096/UFM-Base-DINOv2L-init",
            "revision": REVISION,
            "inference_resolution_wh": model.inference_resolution,
            "model_load_s": load_s,
            "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        },
        "matching_image_wh": list(snapshots[TARGETS[0]].image.shape[1::-1]),
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "pairs": [
            {"source": s.previous.frame_id, "target": n, "queries": len(s.ids)} for n, s in snapshots.items()
        ],
        "summary_ms": {b: summary([r for r in rows if r["backend"] == b]) for b in ("lk", "ufm")},
        "per_pair_ms": {
            str(n): {
                b: summary([r for r in rows if r["backend"] == b and r["target_frame"] == n])
                for b in ("lk", "ufm")
            }
            for n in TARGETS
        },
        "samples": samples.name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary_ms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
