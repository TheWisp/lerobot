"""Compare complete recovery replays using identical, held-out pose labels.

Separate within-limit output, outside-limit output, and missing output. A
withheld wrong pose is useful, but must not count as an extra correct pose.
Use the baseline's measured reference points for every arm of the comparison.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from benchmarks.flash_experiment_labels import load_pose_labels
from benchmarks.flash_full_pose_score import score_transforms, serializable_trace


def classify(available, cloud, angle, *, distance_mm=20, rotation_deg=20):
    state = np.full(len(available), "missing", dtype=object)
    state[available] = "outside"
    state[available & (cloud <= distance_mm) & (angle <= rotation_deg)] = "within"
    return state


def rows_from(run):
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    case = json.loads((run / "case.json").read_text())
    assert len(rows) == case["expected_frames"]
    assert [r["frame"] for r in rows] == list(range(len(rows)))
    return rows


def evaluate(run, role, source, truth, first, baseline_rows):
    rows = rows_from(run)
    assert [r["time_s"] for r in rows] == [r["time_s"] for r in baseline_rows]
    objects = [r["objects"][role] for r in rows]
    poses = [o["group_growing"] for o in objects]
    available = np.asarray([p is not None for p in poses])
    # All runs use the same fixed camera-space coordinates at initialization.
    summary = json.loads((run / "summary.json").read_text())
    assert summary["roles"][role]["reference_frame"] == first
    valid = available.copy()
    valid[: first + 1] = False
    result = {"run": str(run.resolve()), "available": int(valid.sum()), "total": len(rows) - first - 1}
    decisions = [o["group_growing_details"].get("kalman", {}) for o in objects]
    result["filter_decisions"] = dict(Counter(k.get("filter_reason", k.get("reason")) for k in decisions))
    result["accepted_over_innovation_limit"] = [
        i
        for i, k in enumerate(decisions)
        if k.get("accepted") and k.get("innovation_squared", 0) > k.get("innovation_limit", float("inf"))
    ]
    result["median_step_ms"] = float(np.median([o["group_growing_ms"] for o in objects[first + 1 :]]))
    if truth is None:
        result["states"] = [
            "initial" if i <= first else "unlabelled" if a else "missing" for i, a in enumerate(available)
        ]
        return result
    transforms = np.asarray([np.eye(4) if p is None else p for p in poses])
    cloud, angle, center = score_transforms(transforms, valid, truth @ np.linalg.inv(truth[first]), source)
    states = classify(valid, cloud, angle)
    states[: first + 1] = "initial"
    measured = [k.get("observation_pose") for k in decisions]
    measured_valid = np.asarray([p is not None for p in measured])
    measured_valid[: first + 1] = False
    candidate_cloud, candidate_angle, _ = score_transforms(
        np.asarray([np.eye(4) if p is None else p for p in measured]),
        measured_valid,
        truth @ np.linalg.inv(truth[first]),
        source,
    )
    candidate_states = classify(measured_valid, candidate_cloud, candidate_angle)
    candidate_states[: first + 1] = "initial"
    accepted = np.asarray([bool(k.get("accepted")) for k in decisions])
    result.update(
        counts=dict(Counter(states[first + 1 :])),
        severe_count=int(np.sum(valid & ((cloud > 50) | (angle > 45)))),
        cloud_mm_median=float(np.median(cloud[valid])) if valid.any() else None,
        angle_deg_median=float(np.median(angle[valid])) if valid.any() else None,
        cloud_mm=serializable_trace(cloud),
        angle_deg=serializable_trace(angle),
        center_mm=serializable_trace(center),
        states=states.tolist(),
        candidate_states=candidate_states.tolist(),
        correct_candidate_rejected=np.flatnonzero((candidate_states == "within") & ~accepted).tolist(),
        wrong_candidate_accepted=np.flatnonzero((candidate_states == "outside") & accepted).tolist(),
        correct_candidate_wrong_output=np.flatnonzero(
            (candidate_states == "within") & (states == "outside")
        ).tolist(),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--variant", nargs=2, action="append", metavar=("NAME", "ROOT"), default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = {
        "limits": {"cloud_mm": 20, "rotation_deg": 20},
        "meaning": "Agreement with dataset labels; not a calibrated probability of correctness",
        "initialization_excluded": True,
        "cases": {},
    }
    for name in args.cases:
        entry = manifest[name]
        baseline = Path(entry["runs"]["baseline25"])
        case = json.loads((baseline / "case.json").read_text())
        role = entry["role"]
        rows = rows_from(baseline)
        labels, label_sources = load_pose_labels(case, rows)
        first = json.loads((baseline / "summary.json").read_text())["roles"][role]["reference_frame"]
        with np.load(baseline / "reference_points.npz") as points:
            source = points[role]
        variants = {"baseline": baseline}
        variants.update({label: Path(root) / case["id"] for label, root in args.variant})
        result = {
            "case": case["id"],
            "role": role,
            "labels": label_sources,
            "evaluation_reference_points": len(source),
            "evaluation_reference_extent_m": np.ptp(source, axis=0).tolist(),
            "time_s": [r["time_s"] for r in rows],
            "variants": {},
        }
        baseline_hash = json.loads((baseline / "summary.json").read_text())["input_sha256"]
        result["input_sha256"] = baseline_hash
        for variant, run in variants.items():
            if not (run / "summary.json").exists():
                continue
            assert json.loads((run / "summary.json").read_text())["input_sha256"] == baseline_hash
            result["variants"][variant] = evaluate(run, role, source, labels.get(role), first, rows)
        original = result["variants"]["baseline"]["states"]
        for outcome in result["variants"].values():
            outcome["changes_from_baseline"] = dict(
                Counter(
                    f"{a}->{b}"
                    for a, b in zip(original[first + 1 :], outcome["states"][first + 1 :], strict=True)
                )
            )
        report["cases"][name] = result
        print(
            name, {v: r.get("counts", {"available": r["available"]}) for v, r in result["variants"].items()}
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
