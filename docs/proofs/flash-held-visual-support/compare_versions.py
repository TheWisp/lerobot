"""Separate historical FLASH pose-trace changes from held-decision changes.

Run with PYTHONPATH=src and one or more --pair OLD_BUNDLE NEW_BUNDLE pairs.
Each pair must come from the same raw demo with identical annotations and
reference geometry. The old bundle is evaluated by today's updater, while
its recorded trace preserves the historical updater's result. This is a
software replay, not independent pose ground truth.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ablate import evaluate


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    prepared = path.resolve().parent
    root = prepared.parents[3]
    report = json.loads((prepared / "preparation-report.json").read_text())
    role = json.loads((prepared / "annotation.json").read_text())["held_role"]
    with np.load(root / report["pose_trace_cache"], allow_pickle=False) as cache:
        return (
            cache[f"{role}_poses"].copy(),
            cache[f"{role}_evidence"].copy(),
            cache[f"{role}_estimator"].copy(),
            {
                int(index): mask.copy()
                for index, mask in zip(cache["held_mask_indices"], cache["held_masks"], strict=True)
            },
        )


def compare(old: Path, new: Path) -> dict:
    if not old.is_file() or not new.is_file():
        raise ValueError("both prepared bundles must exist")
    old_prepared, new_prepared = old.resolve().parent, new.resolve().parent
    if old_prepared.parents[3] != new_prepared.parents[3]:
        raise ValueError("prepared versions must belong to the same demo")
    for name in ("annotation.json", "keypoints.json", "models.npz"):
        if _digest(old_prepared / name) != _digest(new_prepared / name):
            raise ValueError(f"{name} differs; this is not an isolated version comparison")
    old_pose, old_evidence, old_estimator, old_masks = _trace(old)
    new_pose, new_evidence, new_estimator, new_masks = _trace(new)
    if old_pose.shape != new_pose.shape:
        raise ValueError("pose traces have different lengths")
    both_observed = (old_evidence == "observed") & (new_evidence == "observed")
    pose_changed = ~np.all(np.isclose(old_pose, new_pose, atol=1e-8), axis=(1, 2))
    changed_common_masks = [
        index
        for index in old_masks.keys() & new_masks.keys()
        if not np.array_equal(old_masks[index], new_masks[index])
    ]
    if np.flatnonzero(both_observed & pose_changed).size or changed_common_masks:
        raise ValueError("jointly observed poses or masks changed; updater change is not isolated")
    old_result = evaluate(old, "baseline", require_recorded_parity=False)
    new_result = evaluate(new, "baseline")
    return {
        "demo": old_prepared.parents[3].name,
        "old_preparation": old_prepared.name,
        "new_preparation": new_prepared.name,
        "old_recorded_conflicted_camera_frames": old_result["recorded_conflicted_camera_frames"],
        "current_rule_on_old_pose_trace_conflicted_camera_frames": old_result["conflicted_camera_frames"],
        "new_recorded_conflicted_camera_frames": new_result["recorded_conflicted_camera_frames"],
        "current_rule_on_new_pose_trace_conflicted_camera_frames": new_result["conflicted_camera_frames"],
        "both_observed_pose_difference_frames": np.flatnonzero(both_observed & pose_changed).tolist(),
        "evidence_difference_frames": np.flatnonzero(old_evidence != new_evidence).tolist(),
        "estimator_difference_frames": np.flatnonzero(old_estimator != new_estimator).tolist(),
        "common_observation_masks": len(old_masks.keys() & new_masks.keys()),
        "changed_common_observation_masks": changed_common_masks,
        "new_recorded_parity": new_result["recorded_parity"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True, nargs=2, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [compare(old, new) for old, new in args.pair]
    args.output.write_text(json.dumps({"comparisons": results}, indent=2) + "\n")
