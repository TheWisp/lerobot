"""Compare covariance selection and temporal Kalman fusion on occlusion AND pickup.

Labels are consumed only after tracking. No patches, mocks or robot commands.
"""

import argparse
import gzip
import json
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_tracker_experiment import code_snapshot
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig
from tests.policies.flash.test_gamepad_stationary_integration import (
    ANNOTATION,
    stationary_errors,
    track_sequence,
)


def summarize(report, annotation):
    stationary = report["rows"][: annotation["stationary_last_frame"] + 1]
    missing = [r["frame_id"] for r in report["rows"] if r["camera_from_reference"] is None]
    result = {
        "missing_frames": missing,
        "elapsed_s": report["elapsed_s"],
        "input_sha256": report["input_sha256"],
    }
    if all(r["camera_from_reference"] is not None for r in stationary):
        center, angle = stationary_errors(report, annotation["stationary_last_frame"])
        result.update(
            max_center_mm=float(center.max()),
            max_rotation_deg=float(angle.max()),
            rotation_frame=int(angle.argmax()),
            stationary_passed=bool(
                center.max() <= annotation["max_center_displacement_mm"]
                and angle.max() <= annotation["max_rotation_deg"]
            ),
        )
    else:
        result["stationary_passed"] = False
    pose = report["rows"][-1]["camera_from_reference"]
    if pose is not None:
        ref = np.asarray(report["reference"])
        result["final_center_travel_mm"] = float(
            np.linalg.norm(move(ref, np.asarray(pose)).mean(0) - ref.mean(0)) * 1000
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["lk", "ufm"], required=True)
    parser.add_argument("--mode", choices=["baseline", "covariance", "kalman"], default="covariance")
    parser.add_argument("--uncertainty-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotation = json.loads(ANNOTATION.read_text())
    config = (
        PoseUncertaintyConfig(**json.loads(args.uncertainty_config.read_text()))
        if args.uncertainty_config
        else PoseUncertaintyConfig(
            enabled=args.mode != "baseline", mode="kalman" if args.mode == "kalman" else "selection"
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = code_snapshot(args.output)
    snapshot.update(uncertainty=asdict(config), annotation=annotation)
    (args.output / "provenance.json").write_text(json.dumps(snapshot, indent=2))
    model = load_ufm(revision=annotation["ufm_revision"]) if args.backend == "ufm" else None
    cv2.setNumThreads(1)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        report = track_sequence(
            islice(samples(annotation["case"]), annotation["run_last_frame"] + 1),
            annotation["role"],
            model,
            matching_config=SharedMatchingConfig(backend=args.backend),
            surface_config=GrowingSurfaceConfig(retain_unobserved_pose=True, uncertainty=config),
        )
    report["elapsed_s"] = time.perf_counter() - start
    report["summary"] = summarize(report, annotation)
    with gzip.open(args.output / "run.json.gz", "wt") as stream:
        json.dump(report, stream)
    (args.output / "summary.json").write_text(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
