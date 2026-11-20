"""Known-truth and independent recorded controls for constrained corrections.

Truth is passed exclusively to the shadow evaluator, never to the fitter,
filter, constraint construction or projection. No method updates live state.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_constrained_kalman import shadow_experiment, summary
from benchmarks.flash_kalman_recovery_control import WARMUP, run_scene, scene
from benchmarks.flash_partial_surface_control import box_observation
from benchmarks.flash_recorded_recovery import FIXTURES, load, replay as replay_checkpoint
from benchmarks.flash_recovery_matrix import scenarios
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter


def no_reset(pose, center, time_s, config):
    return PoseKalmanFilter(
        pose,
        center,
        time_s,
        replace(config, reset_on_geometry_conflict=False, unmeasured_rotation_std_deg_sqrt_s=10),
    )


def changed_scene(angle, *, strip=None, gap=0.25, translation=0.0, noise=0.0):
    frames = scene(0, occluded=False)[: WARMUP + 8]
    initial = box_observation(20)[2]
    rng = np.random.default_rng(20261120)
    for frame in frames:
        index = frame["index"]
        frame["time_s"] = index / 30 if index < WARMUP else (WARMUP - 1) / 30 + (index - WARMUP + 1) * gap
        if index < WARMUP:
            continue
        current, _, pose = box_observation(angle, displacement_m=[translation, 0, 0])
        depth, mask = current["depth"].copy(), current["mask"].copy()
        if noise:
            depth[mask] += rng.normal(0, noise, mask.sum())
        if strip is not None:
            left = np.where(mask)[1].min()
            screen = np.indices(mask.shape)[1] >= left + strip
            depth[screen] = 0.5
            mask[screen] = False
        frame.update(depth=depth, mask=mask, truth=pose @ np.linalg.inv(initial), hidden=not mask.any())
        frame["rgb"] = np.full((*mask.shape, 3), 40, np.uint8)
        frame["rgb"][mask] = (190, 170, 120)
    return frames


def controls():
    return {
        **scenarios(),
        "sudden_25deg_250ms": changed_scene(45),
        "sudden_60deg_250ms": changed_scene(80),
        "translation_50mm_250ms": changed_scene(20, translation=0.05),
        "moving_strip_8px": changed_scene(45, strip=8),
        "moving_strip_20px": changed_scene(45, strip=20),
        "noisy_rotation_1mm": changed_scene(45, noise=0.001),
        "symmetric_half_turn": changed_scene(200),
    }


def accurate(error):
    return error["median_point_error_mm"] < 5 and error["rotation_error_deg"] < 1


def evaluate_records(records):
    result = summary(records)
    for name in result["methods"]:
        proposed = [r["methods"][name][0] for r in records if r["methods"][name][0]["pose"] is not None]
        labelled = [p for p in proposed if "error" in p]
        result["methods"][name].update(
            labelled_proposals=len(labelled),
            wrong_proposals=sum(not accurate(p["error"]) for p in labelled),
            wrong_after_repair=sum(
                p["reason"] != "already_compatible" and not accurate(p["error"]) for p in labelled
            ),
            max_rotation_error_deg=max((p["error"]["rotation_error_deg"] for p in labelled), default=None),
            max_point_error_mm=max((p["error"]["median_point_error_mm"] for p in labelled), default=None),
        )
    return result


def run(output, *, names=None):
    cv2.setNumThreads(1)
    results = {}
    for name, frames in controls().items():
        if names and name not in names:
            continue
        truth = {f["index"]: f["truth"] for f in frames}
        with shadow_experiment(truth=truth) as records:
            baseline = run_scene(
                frames, "strict_broader_motion", grow=name == "corrupted_then_clear", filter_factory=no_reset
            )
        metrics = evaluate_records(records)
        metrics.update(
            processed_updates=len(frames) - 1,
            baseline_observed=sum(r["status"] == "observed_candidate" for r in baseline["rows"]),
        )
        results[name] = {"summary": metrics, "records": records, "baseline": baseline}
        print(name, json.dumps(metrics), flush=True)
    for directory in sorted(FIXTURES.iterdir()):
        if not directory.is_dir() or names:
            continue
        data, meta = load(directory.name)
        truth = {meta["frame"]: data["label_pose"]} if "label_pose" in data else None
        with shadow_experiment(truth=truth) as records:
            baseline = replay_checkpoint(directory.name)
        metrics = evaluate_records(records)
        results[directory.name] = {
            "summary": metrics,
            "records": records,
            "label_source": meta["label_source"],
            "baseline_status": baseline["result"]["status"],
            "baseline_reason": baseline["result"]["diagnostics"].get("reason"),
        }
        print(directory.name, json.dumps(metrics), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="*")
    args = parser.parse_args()
    run(args.output, names=args.names)
