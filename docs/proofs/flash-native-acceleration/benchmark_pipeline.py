"""Full sequential trajectories through the shared, pipelined backend."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from benchmark_fits import compare_poses
from candidates import SearchPool, compiled_kernels, fitter_type

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.perception_step import PerceptionPipeline, PerceptionRunner
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.tracking_debug_worker import debug_reference
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend


def inputs(args):
    sequence, masks = [], {}
    if args.trace:
        rows = [json.loads(line) for line in (args.trace / "frames.jsonl").read_text().splitlines()]
        rows = rows[args.start : args.start + args.frames]
        config = VisionPipelineConfig.from_dict(
            json.loads((args.trace / "config.json").read_text())["vision_pipeline"]
        )
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
    else:
        config = VisionPipelineConfig()
        recording = Path.home() / ".cache/huggingface/lerobot/demos/insertion/recording"
        mask_root = Path.home() / ".cache/flash-mask-depth-experiment/sam3-full"
        k = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
        for path in sorted((recording / "rgb").glob("*.jpg"))[args.start : args.start + args.frames]:
            n = int(path.stem)
            sequence.append(
                RGBDFrame(
                    n,
                    cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB),
                    cv2.imread(str(recording / "depth" / f"{n:06d}.png"), -1),
                    k,
                    0.001,
                    np.eye(4),
                    n / 30,
                    n / 30,
                    "recorded-demo",
                )
            )
            masks[n] = {"object": cv2.imread(str(mask_root / f"{n:06d}.png"), 0) > 0}
    return sequence, masks, config


def run(args):
    if not args.modes or args.modes[0] != "baseline":
        raise ValueError("the independent baseline must run first")
    cv2.setNumThreads(1)
    sequence, masks, config = inputs(args)
    roles = list(masks[sequence[0].frame_id])
    references = tuple(debug_reference(role, sequence[0]) for role in roles)

    class Masks:
        def masks(self, frame):
            return masks[frame.frame_id]

    model = load_ufm()
    kernels = compiled_kernels()
    report = {
        "trace": str(args.trace) if args.trace else "insertion demo",
        "frames": [f.frame_id for f in sequence],
        "roles": roles,
        "sam": "recorded masks",
        "config": config.to_dict(),
        "runs": [],
    }
    baseline = None
    with ExitStack() as stack:
        pool = stack.enter_context(SearchPool(8))
        factories = {
            "baseline": None,
            "pool": fitter_type(search_pool=pool),
            "compiled": fitter_type(kernels=kernels),
            "compiled_pool": fitter_type(kernels=kernels, search_pool=pool),
        }
        for repeat in range(args.repeats):
            for mode in args.modes:
                backend = build_vision_backend(config, Masks(), matcher=model, local_reference=True)
                pipeline = PerceptionPipeline(backend, references)
                source = iter(sequence)
                results, times = [], []
                start, cpu = time.perf_counter(), time.process_time()
                context = ExitStack()
                if factories[mode] is not None:
                    context.enter_context(
                        patch("lerobot.policies.flash.surface_tracking.WholeSurfaceFit", factories[mode])
                    )
                with context, PerceptionRunner(pipeline) as runner:
                    while (item := runner.process_next(lambda source=source: next(source, None))) is not None:
                        results.append(item[1])
                        times.append(time.perf_counter() - start)
                        if len(results) % 100 == 0:
                            print(mode, "frames", len(results), flush=True)
                elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
                if baseline is None:
                    baseline = results
                errors, mismatches = [], []
                for old, new in zip(baseline, results, strict=True):
                    assert old.frame_id == new.frame_id and old.source_time_s == new.source_time_s
                    for role in roles:
                        np.testing.assert_array_equal(old.masks[role], new.masks[role])
                        a, b = old.details[role], new.details[role]
                        assert ("camera_from_reference" in a) == ("camera_from_reference" in b)
                        if (a["status"], a["accepted"]) != (b["status"], b["accepted"]):
                            mismatches.append(
                                {
                                    "frame": new.frame_id,
                                    "role": role,
                                    "old": [a["status"], a["accepted"]],
                                    "new": [b["status"], b["accepted"]],
                                }
                            )
                        if "camera_from_reference" in a and "camera_from_reference" in b:
                            error = compare_poses(
                                np.asarray(b["camera_from_reference"]), np.asarray(a["camera_from_reference"])
                            )
                            errors.append({"frame": new.frame_id, "role": role, **error})
                row = {
                    "mode": mode,
                    "repeat": repeat,
                    "elapsed_s": elapsed,
                    "steady_fps": 1 / float(np.mean(np.diff(times)[2:])),
                    "busy_cpu_equivalents": cpu_elapsed / elapsed,
                    "accepted": {role: sum(r.details[role]["accepted"] for r in results) for role in roles},
                    "decision_mismatches": mismatches,
                    "max_pose_difference": {
                        key: max((e[key] for e in errors), default=0)
                        for key in ("translation_mm", "rotation_deg", "matrix_max_abs")
                    },
                    "pose_differences": errors,
                }
                report["runs"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print({key: value for key, value in row.items() if key != "pose_differences"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--modes", nargs="+", default=["baseline", "compiled", "pool", "compiled_pool"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
