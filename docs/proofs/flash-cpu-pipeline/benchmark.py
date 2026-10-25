"""Paired ordered RGB-D replay, with actual UFM and recorded SAM masks.

Preloads a bounded sequence so image decoding does not confound scheduling.
Each mode sees identical source frames. This measures throughput, not live
capture age, and cached masks exclude SAM cost.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import MethodType

import cv2
import numpy as np
import torch

from lerobot.policies.flash import vision_pipeline
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.perception_step import PerceptionPipeline, PerceptionRunner
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.tracking_debug_worker import debug_reference
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend


def original_update():
    # Preserve the pre-change per-object overlap, rather than calling the new
    # two-stage serial scheduler "before". Geometry/math stay scalar in this mode.
    source = subprocess.check_output(
        ["git", "show", "8c32aee5b:src/lerobot/policies/flash/vision_pipeline.py"], text=True
    )
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "SurfacePoseBackend"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update")
    namespace = dict(vars(vision_pipeline))
    # Execute only our fixed historical commit, never a supplied expression or revision.
    exec(compile(ast.Module(body=[method], type_ignores=[]), "original_surface_backend", "exec"), namespace)  # nosec B102
    return namespace["update"]


def run(args):
    cv2.setNumThreads(1)
    rows = [json.loads(line) for line in (args.trace / "frames.jsonl").read_text().splitlines()]
    rows = rows[args.start : args.start + args.frames]
    sequence, masks = [], {}
    for row in rows:
        n = row["frame_id"]
        stem = f"{n:06d}.png"
        sequence.append(
            RGBDFrame(
                n,
                cv2.cvtColor(cv2.imread(str(args.trace / "rgb" / stem)), cv2.COLOR_BGR2RGB),
                cv2.imread(str(args.trace / "depth" / stem), -1),
                np.asarray(row["intrinsics"]),
                row["depth_m_per_unit"],
                np.eye(4),
                row["source_time_s"],
                row["source_time_s"],
                row["calibration_version"],
            )
        )
        masks[n] = {
            role: cv2.imread(str(args.trace / "masks" / role / stem), 0) > 0 for role in row["objects"]
        }

    class Masks:
        def masks(self, frame):
            return masks[frame.frame_id]

    config = VisionPipelineConfig.from_dict(
        json.loads((args.trace / "config.json").read_text())["vision_pipeline"]
    )
    roles = list(masks[sequence[0].frame_id])
    refs = tuple(debug_reference(role, sequence[0]) for role in roles)
    model = load_ufm()
    report = {
        "trace": str(args.trace),
        "frames": [f.frame_id for f in sequence],
        "roles": roles,
        "sam": "recorded masks; excluded from timing",
        "modes": {},
    }
    baseline = None
    for name, batch, pipeline, workers in [
        ("original", False, False, 2),
        ("scalar", False, False, 2),
        ("batch", True, False, 2),
        ("batch_pipeline", True, True, 2),
        ("batch_pipeline_3workers", True, True, 3),
    ]:
        if args.modes and name not in args.modes:
            continue
        runs = []
        for repeat in range(args.repeats):
            settings = replace(
                config,
                pipeline_surface_frames=pipeline,
                surface_object_workers=workers,
                shape=replace(config.shape, batch_jacobian=batch),
            )
            backend = build_vision_backend(settings, Masks(), matcher=model, local_reference=True)
            if name == "original":
                backend.update = MethodType(original_update(), backend)
            tracker = PerceptionPipeline(backend, refs)
            source = iter(sequence)
            results, times = [], []
            torch.cuda.synchronize()
            start = time.perf_counter()
            with PerceptionRunner(tracker) as runner:
                while (item := runner.process_next(lambda source=source: next(source, None))) is not None:
                    frame, result = item
                    times.append(time.perf_counter() - start)
                    results.append(result)
            elapsed = time.perf_counter() - start
            backend.close()
            if baseline is None:
                baseline = results
            max_error = 0.0
            for expected, current in zip(baseline, results, strict=True):
                assert (expected.frame_id, expected.source_time_s, expected.failures) == (
                    current.frame_id,
                    current.source_time_s,
                    current.failures,
                )
                for role in roles:
                    np.testing.assert_array_equal(expected.masks[role], current.masks[role])
                    a, b = expected.details[role], current.details[role]
                    assert (a["status"], a["accepted"]) == (b["status"], b["accepted"])
                    if "camera_from_reference" in a:
                        pa, pb = np.array(a["camera_from_reference"]), np.array(b["camera_from_reference"])
                        max_error = max(max_error, float(np.max(np.abs(pa - pb))))
                        np.testing.assert_allclose(pa, pb, atol=1e-10, rtol=0)
            # First pair includes model warmup: report steady intervals separately.
            intervals = np.diff(times)[2:]
            value = {
                "elapsed_s": elapsed,
                "steady_fps": 1 / float(np.mean(intervals)),
                "interval_median_ms": float(np.median(intervals) * 1000),
                "interval_p95_ms": float(np.percentile(intervals, 95) * 1000),
                "max_pose_matrix_difference": max_error,
                "accepted": {role: sum(r.details[role]["accepted"] for r in results) for role in roles},
                "timings": {
                    key: float(
                        np.median(
                            [
                                r.details[role]["diagnostics"][key]
                                for r in results[3:]
                                for role in roles
                                if key in r.details[role]["diagnostics"]
                            ]
                        )
                    )
                    for key in ("preprocess_ms", "match_ms", "fit_ms")
                },
            }
            runs.append(value)
            print(name, repeat, value, flush=True)
        report["modes"][name] = runs
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--modes", nargs="+")
    run(p.parse_args())
