"""Check recorded FLASH replay quality and latency without treating fit cost as accuracy.

Run flash_runtime_benchmark with the five pinned source runs first. Saved masks
keep the matcher comparison identical; this gate reads published poses and labels.
"""

import argparse
import json
from pathlib import Path

from benchmarks.flash_runtime_benchmark import latency_distribution

CORE_CASES = (
    "ycb-mustard_easy_00_02",  # easy, labelled
    "dexycb-blue-mug",  # intermittent occlusion, labelled
    "laval-clock-interaction-hard",  # difficult rotation, labelled
    "flash-demo-dowel-ring",  # two interacting objects
    "flash-demo-gamepad-cube",  # two interacting objects, severe occlusion
)


# Deliberately modest floors for the imperfect current tracker. The labelled
# counts are actual pose accuracy; the demos have no independent pose labels,
# so their clear intervals can only guard availability, not correctness.
LABELLED_FLOORS = {
    "ycb-mustard_easy_00_02": ("object", 177),
    "dexycb-blue-mug": ("object", 30),
    "laval-clock-interaction-hard": ("clock", 27),
}
CLEAR_AVAILABILITY = {
    ("flash-demo-dowel-ring", "dowel"): (1, 180, 170),
    ("flash-demo-dowel-ring", "ring"): (1, 400, 390),
    ("flash-demo-gamepad-cube", "gamepad"): (1, 190, 180),
    ("flash-demo-gamepad-cube", "cube"): (1, 400, 340),
}


def read_suite(root: Path) -> dict:
    reports = json.loads((root / "summary.json").read_text())
    cases = {report["case"]: report for report in reports}
    if set(cases) != set(CORE_CASES):
        raise ValueError(f"core suite must contain exactly {CORE_CASES}; found {tuple(cases)}")
    result = {}
    for index, report in enumerate(reports):
        case_id = report["case"]
        with (root / f"{index:03d}" / case_id / "frames.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != report["frames"]:
            raise ValueError(f"incomplete frame trace: {case_id}")
        method = report["processing"]["method"]
        if method != "group_growing":
            raise ValueError(f"core suite expects group_growing: {case_id}")
        result[case_id] = (report, rows, method)
    return result


def evaluate(
    root: Path,
    *,
    baseline: Path | None = None,
    max_p90_ms=None,
    max_frame_ms=None,
    max_stddev_ms=None,
):
    cases = read_suite(root)
    earlier = read_suite(baseline) if baseline else None
    failures, metrics = [], {}
    for case_id in CORE_CASES:
        report, rows, method = cases[case_id]
        latency = latency_distribution([row["processing_ms"] for row in rows[1:]])
        metrics[case_id] = {"latency_ms": latency, "roles": {}}
        if max_p90_ms is not None and latency["p90"] > max_p90_ms:
            failures.append(f"{case_id}: p90 {latency['p90']:.1f} > {max_p90_ms} ms")
        if max_frame_ms is not None and latency["max"] > max_frame_ms:
            failures.append(f"{case_id}: max {latency['max']:.1f} > {max_frame_ms} ms")
        if max_stddev_ms is not None and latency["stddev"] > max_stddev_ms:
            failures.append(f"{case_id}: stddev {latency['stddev']:.1f} > {max_stddev_ms} ms")
        if earlier and report["input_sha256"] != earlier[case_id][0]["input_sha256"]:
            failures.append(f"{case_id}: inputs differ from baseline")
        for role, data in report["roles"].items():
            quality = data["methods"][method]
            pose = quality.get("pose_error", {})
            present = sum(row["objects"][role][method] is not None for row in rows[1:])
            observed = sum(
                row["objects"][role][method + "_status"] == "observed_candidate" for row in rows[1:]
            )
            metrics[case_id]["roles"][role] = {
                "available": present,
                "observed": observed,
                "correct_20mm_20deg": pose.get("within_20mm_20deg"),
                "wrong_or_outside_20mm_20deg": (
                    pose["available"] - pose["within_20mm_20deg"] if pose else None
                ),
                "labelled_frames": pose.get("total"),
                "surface_error_median_mm": quality.get("surface_error", {}).get("median_mm"),
            }
            # An observed pose cannot be published after failing its own
            # geometry gate. A predicted pose is a different, weaker claim.
            for row in rows:
                item = row["objects"][role]
                if item[method + "_status"] == "observed_candidate":
                    geometry = item[method + "_details"].get("fit", {}).get("geometry_check")
                    if geometry is None or not geometry["accepted"]:
                        failures.append(f"{case_id}/{role} frame {row['frame']}: unvalidated pose published")
            if case_id in LABELLED_FLOORS and role == LABELLED_FLOORS[case_id][0]:
                floor = LABELLED_FLOORS[case_id][1]
                if pose.get("within_20mm_20deg", 0) < floor:
                    failures.append(f"{case_id}/{role}: fewer than {floor} correct labelled poses")
            window = CLEAR_AVAILABILITY.get((case_id, role))
            if window and len(rows) < window[1]:
                failures.append(f"{case_id}/{role}: missing clear-window frames")
            if window and len(rows) >= window[1]:
                start, stop, floor = window
                count = sum(row["objects"][role][method] is not None for row in rows[start:stop])
                metrics[case_id]["roles"][role]["clear_window_available"] = count
                if count < floor:
                    failures.append(f"{case_id}/{role}: only {count}/{stop - start} clear poses")
            if earlier:
                old_quality = earlier[case_id][0]["roles"][role]["methods"][method]
                old_pose = old_quality.get("pose_error", {})
                if pose and pose["within_20mm_20deg"] < old_pose["within_20mm_20deg"]:
                    failures.append(f"{case_id}/{role}: labelled pose accuracy regressed")
                if pose and (
                    pose["available"] - pose["within_20mm_20deg"]
                    > old_pose["available"] - old_pose["within_20mm_20deg"]
                ):
                    failures.append(f"{case_id}/{role}: additional wrong labelled poses")
                old_rows = earlier[case_id][1]
                old_present = sum(row["objects"][role][method] is not None for row in old_rows[1:])
                if not pose and present < old_present:
                    failures.append(f"{case_id}/{role}: pose availability regressed")
    return {"pass": not failures, "failures": failures, "cases": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-p90-ms", type=float)
    parser.add_argument("--max-frame-ms", type=float)
    parser.add_argument("--max-stddev-ms", type=float)
    args = parser.parse_args()
    report = evaluate(
        args.run,
        baseline=args.baseline,
        max_p90_ms=args.max_p90_ms,
        max_frame_ms=args.max_frame_ms,
        max_stddev_ms=args.max_stddev_ms,
    )
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
