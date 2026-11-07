"""Time the live consumer on recorded RGB-D, including its scene encoder.

Saved masks isolate matching changes. --sam runs the actual shared SAM3 source,
initialized by a click inside each saved first-frame mask. It is a separate
end-to-end measurement because the resulting masks can differ. No frame is
skipped, and no robot/camera hardware is accessed. Network and browser rendering
are outside the measurement.
"""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_runtime_benchmark import FitProfile
from benchmarks.flash_surface_score import save_surface_version
from benchmarks.flash_tracker_experiment import code_snapshot, load_ufm, score_case
from lerobot.gui.flash_group_overlay import GroupOverlay
from lerobot.gui.flash_live_scene import live_scene_packet
from lerobot.policies.flash.appearance import Sam3MaskSource
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_session_config import validate
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--sam", action="store_true")
    parser.add_argument(
        "--sam-prompts", type=Path, help="Per-case prompts_by_role, as in flash_runtime_benchmark"
    )
    parser.add_argument("--profile", action="store_true", help="Serial attribution; not a speed measurement")
    args = parser.parse_args()
    if args.frames < 2 or (args.sam_prompts and not args.sam):
        parser.error("Use at least two frames; --sam-prompts requires --sam")
    args.output.mkdir(parents=True, exist_ok=False)
    case = json.loads(args.case.read_text())
    settings = validate(json.loads(args.settings.read_text()))
    if args.profile:
        settings["config"]["parallel_surface_objects"] = False
    config = VisionPipelineConfig.from_dict(settings["config"])
    shared = settings["shared_matching"]
    model = load_ufm() if shared is None or shared["backend"] == "ufm" else None
    prompts = json.loads(args.sam_prompts.read_text())[case["id"]] if args.sam_prompts else None
    if prompts and set(prompts["prompts_by_role"]) != set(case["roles"]):
        parser.error("SAM prompts must cover every tracked role")
    code_snapshot(args.output)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))
    (args.output / "case.json").write_text(json.dumps(case, indent=2))
    cv2.setNumThreads(1)

    class SavedMasks:
        prompts = {role: role for role in case["roles"]}

        def masks(self, frame):
            return current_masks

    source = SavedMasks()
    overlays, statuses = defaultdict(GroupOverlay), defaultdict(Counter)
    rows, predictions, digest = [], [], hashlib.sha256()
    kernels = compiled_kernels() if config.compiled_surface_fit else None
    pool = SearchPool(config.shape.batch_query_workers) if config.reuse_surface_search else None
    live = None
    try:
        with FitProfile() if args.profile else nullcontext() as profile:
            for frame, current_masks in islice(samples(case), args.frames):
                if live is None:
                    if args.sam:
                        clicks = {}
                        for role, mask in current_masks.items() if prompts is None else ():
                            if not mask.any():
                                raise ValueError(f"Cannot initialize {role}: empty first-frame mask")
                            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
                            y, x = np.unravel_index(distance.argmax(), distance.shape)
                            clicks[role] = (float(x), float(y))
                        source = Sam3MaskSource(
                            prompts["prompts_by_role"] if prompts else source.prompts,
                            camera_id="recorded-rgbd",
                            resolution=config.sam_resolution,
                            seed_points_by_role=clicks,
                            detection_correction=config.detection_correction.control(offline=False),
                        )
                    live = LiveSurfaceSession(
                        settings, source, model=model, kernels=kernels, search_pool=pool
                    )
                started = time.perf_counter()
                row, _, masks = live.step(frame)
                matched = time.perf_counter()
                packet = live_scene_packet(frame, live, row, masks, overlays)
                ended = time.perf_counter()
                # Evaluation/archival is outside the live latency measurement.
                # Never feed labels or the labelled model into the session.
                # Compare the masks actually consumed, including fresh SAM output.
                for value in (frame.rgb, frame.depth, frame.intrinsics, *masks.values()):
                    value = np.asarray(value)
                    digest.update(str((value.shape, value.dtype)).encode() + value.tobytes())
                predictions.append(row)
                for role in case.get("evaluation_surfaces", {}):
                    track = live.engine.candidates[live.method][role]
                    save_surface_version(
                        args.output,
                        live.method,
                        role,
                        track,
                        row["objects"][role][live.method + "_details"],
                    )
                timing = {
                    "frame": frame.frame_id,
                    "pipeline_ms": 1000 * (matched - started),
                    "segmentation_ms": row["mask_ms"],
                    "tracking_ms": 1000 * (matched - started) - row["mask_ms"],
                    "display_ms": 1000 * (ended - matched),
                    "total_ms": 1000 * (ended - started),
                    "packet_bytes": len(packet),
                    "mask_pixels": {role: int(mask.sum()) for role, mask in masks.items()},
                }
                for role, result in row["objects"].items():
                    statuses[role][result["group_growing_status"]] += 1
                rows.append(timing)
                if len(rows) == 1:
                    np.savez_compressed(args.output / "first_masks.npz", rgb=frame.rgb, **masks)
                if len(rows) % 30 == 0:
                    print(f"{case['id']}: {len(rows)} frames, {timing['total_ms']:.1f} ms", flush=True)
            attribution = profile.stats if args.profile else None
    finally:
        if live is not None:
            live.close()
        if pool is not None:
            pool.close()
    steady = {}
    for name in ("segmentation_ms", "tracking_ms", "pipeline_ms", "display_ms", "total_ms"):
        values = np.asarray([row[name] for row in rows[1:]])
        steady[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    report = {
        "scope": "LiveSurfaceSession + scene encoding; excludes file decode, network and browser; initialization reported separately",
        "sam": args.sam,
        "sam_prompts": prompts,
        "profiled_serial": args.profile,
        "input_sha256": digest.hexdigest(),
        "input_hash_includes_tracked_masks": True,
        "frames": len(rows),
        "objects": dict(statuses),
        "initialization_ms": rows[0]["total_ms"],
        "steady": steady,
        "profile": attribution,
        "quality": score_case(
            case, predictions, live.engine.references, live.engine.reference_frames, run=args.output
        ),
    }
    with (args.output / "frames.jsonl").open("w") as stream:
        for row in predictions:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "reference_points.npz", **live.engine.references)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    (args.output / "timing.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
