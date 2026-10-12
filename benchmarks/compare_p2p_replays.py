# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Compare complete, paired ground-truth replay runs; never infer accuracy from unit tests.

Input: suite.json (sequences), <sequence>/{baseline,fixed}/{manifest.json,
errors.npz}. Errors contain per-frame ADD/ADD-S in metres and P2P lost flags.
The manifest records input/configuration hashes and the random seed. A replay
must score every frame, including lost frames using the pose it actually emits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INPUT_KEYS = ("rgb", "depth", "annotated_poses", "initial_mask", "intrinsics", "config", "seed")


def auc(errors: np.ndarray, limit: float = 0.1) -> float:
    """ADD recall area over 0..10 cm, as used by the existing YCB evaluation."""
    ordered = np.sort(errors)
    recall = np.arange(1, len(ordered) + 1) / len(ordered)
    keep = ordered < limit
    if not keep.any():
        return 0.0
    x = np.r_[0.0, ordered[keep], limit]
    y = np.r_[0.0, recall[keep], recall[keep][-1]]
    return float(np.sum(np.diff(x) * y[1:]) / limit * 100)


def load_run(path: Path, expected_frames: int) -> tuple[dict, dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    if any(manifest.get(key) is None for key in INPUT_KEYS):
        raise ValueError(f"{path}: incomplete input identity")
    with np.load(path / "errors.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ("add", "adds", "lost")}
    if any(value.shape != (expected_frames,) for value in arrays.values()):
        raise ValueError(f"{path}: missing or extra frames")
    if arrays["lost"].dtype != np.bool_:
        raise ValueError(f"{path}: lost flags must be boolean")
    for key in ("add", "adds"):
        if not np.isfinite(arrays[key]).all() or (arrays[key] < 0).any():
            raise ValueError(f"{path}: invalid {key} errors; do not omit failed frames")
    return manifest, arrays


def metrics(data: dict, bad_add_m: float) -> dict:
    return {
        "frames": len(data["add"]),
        "add_auc_percent": auc(data["add"]),
        "adds_auc_percent": auc(data["adds"]),
        "add_p95_mm": float(np.percentile(data["add"], 95) * 1000),
        "add_max_mm": float(np.max(data["add"]) * 1000),
        "lost_frames": int(data["lost"].sum()),
        # This is P2P's lost flag, not the robot controller's acceptance decision.
        "not_lost_above_limit": int(((data["add"] > bad_add_m) & ~data["lost"]).sum()),
    }


def compare_suite(
    root: Path, *, bad_add_mm: float, auc_drop_pp: float, p95_rise_mm: float, extra_lost_frames: int
) -> dict:
    if not np.isfinite([bad_add_mm, auc_drop_pp, p95_rise_mm]).all():
        raise ValueError("comparison limits must be finite")
    if bad_add_mm <= 0 or min(auc_drop_pp, p95_rise_mm, extra_lost_frames) < 0:
        raise ValueError("invalid comparison limits")
    suite = json.loads((root / "suite.json").read_text())
    sequences = suite["sequences"]
    if not sequences:
        raise ValueError("empty suite")
    rows = []
    for name, frames in sequences.items():
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
            raise ValueError(f"{name}: invalid expected frame count")
        before_manifest, before = load_run(root / name / "baseline", frames)
        after_manifest, after = load_run(root / name / "fixed", frames)
        if any(before_manifest[key] != after_manifest[key] for key in INPUT_KEYS):
            raise ValueError(f"{name}: runs use different inputs, configuration or seed")
        a, b = metrics(before, bad_add_mm / 1000), metrics(after, bad_add_mm / 1000)
        failures = []
        for key in ("add_auc_percent", "adds_auc_percent"):
            if a[key] - b[key] > auc_drop_pp:
                failures.append(f"{key} decreased beyond allowance")
        if b["add_p95_mm"] - a["add_p95_mm"] > p95_rise_mm:
            failures.append("95th-percentile ADD increased beyond allowance")
        if b["not_lost_above_limit"] > a["not_lost_above_limit"]:
            failures.append("more high-error frames reported as not lost")
        if b["lost_frames"] - a["lost_frames"] > extra_lost_frames:
            failures.append("more lost frames than allowed")
        rows.append({"sequence": name, "baseline": a, "fixed": b, "regressions": failures})
    return {
        "passed": not any(row["regressions"] for row in rows),
        "limits": {
            "bad_add_mm": bad_add_mm,
            "auc_drop_pp": auc_drop_pp,
            "p95_rise_mm": p95_rise_mm,
            "extra_lost_frames": extra_lost_frames,
        },
        "sequences": rows,
        "scope": "Paired recorded-sequence comparison only; not proof of physical task reliability.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bad-add-mm", type=float, required=True, help="explicit high-error reporting limit")
    parser.add_argument(
        "--auc-drop-pp", type=float, required=True, help="allowed decrease in AUC percentage points"
    )
    parser.add_argument(
        "--p95-rise-mm", type=float, required=True, help="allowed rise in 95th-percentile ADD"
    )
    parser.add_argument(
        "--extra-lost-frames", type=int, required=True, help="allowed extra lost frames per sequence"
    )
    args = parser.parse_args()
    result = compare_suite(
        args.root,
        bad_add_mm=args.bad_add_mm,
        auc_drop_pp=args.auc_drop_pp,
        p95_rise_mm=args.p95_rise_mm,
        extra_lost_frames=args.extra_lost_frames,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
