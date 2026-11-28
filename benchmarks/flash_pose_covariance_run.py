"""Run the covariance ablation through the real recorded integration consumer."""

import argparse
import json
import time
from dataclasses import asdict
from functools import partial
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_pose_covariance import CovarianceConfig, correct_with_covariance
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, ImageConditionedFit
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig
from tests.policies.flash.test_gamepad_stationary_integration import (
    ANNOTATION,
    stationary_errors,
    track_sequence,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ufm", "lk"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=281)
    parser.add_argument(
        "--variants", nargs="+", default=["baseline", "independent", "correlated2", "correlated5"]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    annotation = json.loads(ANNOTATION.read_text())
    model = load_ufm(revision=annotation["ufm_revision"]) if args.backend == "ufm" else None
    variants = {
        "baseline": None,
        "independent": CovarianceConfig(correlated_error_m=0),
        "correlated2": CovarianceConfig(correlated_error_m=0.002),
        "correlated5": CovarianceConfig(correlated_error_m=0.005),
        "gated2": CovarianceConfig(correlated_error_m=0.002, compatibility_probability=0.99),
        "gated5": CovarianceConfig(correlated_error_m=0.005, compatibility_probability=0.99),
    }
    original = ImageConditionedFit.correct_surface_first
    cv2.setNumThreads(1)
    summaries = {}
    for name in args.variants:
        config = variants[name]
        method = (
            partial(correct_with_covariance, baseline_correct=original, config=config) if config else original
        )

        # A wrapper preserves method binding. Only the fit hook differs; SAM,
        # depth cleaning, matching, groups, map growth and acceptance are shared.
        def replacement(self, proposal, previous, _method=method):
            return _method(self, proposal, previous)

        start = time.perf_counter()
        with (
            threadpool_limits(limits=1),
            patch.object(ImageConditionedFit, "correct_surface_first", replacement),
        ):
            report = track_sequence(
                islice(samples(annotation["case"]), args.frames),
                annotation["role"],
                model,
                matching_config=SharedMatchingConfig(backend=args.backend),
                surface_config=GrowingSurfaceConfig(
                    retain_unobserved_pose=True, uncertainty=PoseUncertaintyConfig(enabled=False)
                ),
            )
        report["experiment"] = {"variant": name, "covariance_config": asdict(config) if config else None}
        report["elapsed_s"] = time.perf_counter() - start
        # Labels are first used here, after every causal estimate is complete.
        center, angle = stationary_errors(report, min(annotation["stationary_last_frame"], args.frames - 1))
        travel, _ = stationary_errors(report, args.frames - 1)
        cov = [r["diagnostics"].get("fit", {}).get("covariance_experiment", {}) for r in report["rows"]]
        summary = {
            "max_center_mm": float(center.max()),
            "max_rotation_deg": float(angle.max()),
            "rotation_frame": int(angle.argmax()),
            "center_frame": int(center.argmax()),
            "rotation_p95_deg": float(np.percentile(angle, 95)),
            "final_center_travel_mm": float(travel[-1]),
            "elapsed_s": report["elapsed_s"],
            "covariance_updates": sum(c.get("applied", False) for c in cov),
            "support_rejections": sum(c.get("consistency", {}).get("support_rejected", False) for c in cov),
            "input_sha256": report["input_sha256"],
            "matching_calls": report["matching_calls"],
        }
        report["summary"] = summary
        (args.output / f"{args.backend}-{name}.json").write_text(json.dumps(report) + "\n")
        summaries[name] = summary
        print("RESULT", args.backend, name, json.dumps(summary), flush=True)
        (args.output / f"{args.backend}-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
