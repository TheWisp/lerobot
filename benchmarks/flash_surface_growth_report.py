"""Compare fixed and growing point cohorts without rewarding short tracks.

Every added cohort is scored only on frames *after* its birth. Original points
are scored separately in both runs, so growth cannot hide regressions by
adding easy, young points to a pooled denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.flash_surface_benchmark import _load, score


def _cohort(reference: dict, prediction: dict, ids: np.ndarray, after_birth: bool) -> dict:
    ref = reference.copy()
    pred = prediction.copy()
    for key in ("queries", "initial_depth_m"):
        if key in ref:
            ref[key] = ref[key][ids]
    for key in ("truth_xy", "truth_xyz", "truth_visible", "reference_depth_gap_m"):
        if key in ref:
            ref[key] = ref[key][:, ids]
    for key in ("queries",):
        pred[key] = pred[key][ids]
    for key in ("xy", "xyz", "visible"):
        if key in pred:
            pred[key] = pred[key][:, ids]
    if after_birth:
        ref["truth_visible"] = ref["truth_visible"].copy()
        ref["truth_visible"] &= np.arange(len(ref["frames"]))[:, None] > ref["queries"][:, 0]
    return score(ref, pred)


def _coverage(result: dict) -> dict:
    return {
        "eligible": result["eligible_point_frames"],
        "coverage_2px": result["coverage_px"]["2"],
        "coverage_5mm": result["coverage_mm"]["5"],
        "available_fraction": result["available_fraction"],
    }


def collect(base_root: Path, growth_root: Path) -> dict:
    rows = []
    schedules = set()
    spacings = set()
    for grown in sorted(growth_root.iterdir()):
        if not (grown / "queries.npz").is_file():
            continue
        base = base_root / grown.name
        growth_manifest = json.loads((grown / "manifest.json").read_text())
        schedules.add(tuple((int(k), v) for k, v in growth_manifest["query_birth_schedule"].items()))
        spacings.add(growth_manifest["minimum_old_query_spacing_px"])
        fixed_ref, grown_ref = _load(base / "queries.npz"), _load(grown / "queries.npz")
        if not np.array_equal(fixed_ref["queries"], grown_ref["queries"][: len(fixed_ref["queries"])]):
            raise ValueError(f"initial query identities differ in {grown.name}")
        born = grown_ref["queries"][:, 0].astype(int)
        for result_path in sorted(grown.glob("*.npz")):
            if result_path.name == "queries.npz":
                continue
            name = result_path.stem
            fixed_path = base / result_path.name
            if not fixed_path.is_file():
                continue
            fixed_pred, grown_pred = _load(fixed_path), _load(result_path)
            fixed_time = json.loads(fixed_path.with_suffix(".timing.json").read_text())
            grown_time = json.loads(result_path.with_suffix(".timing.json").read_text())
            cohorts = {
                str(birth): _coverage(_cohort(grown_ref, grown_pred, born == birth, after_birth=True))
                for birth in sorted(set(born))
            }
            initial_ids = born == 0
            initial_grown = _coverage(_cohort(grown_ref, grown_pred, initial_ids, after_birth=False))
            initial_fixed = _coverage(score(fixed_ref, fixed_pred))
            row = {
                "clip": grown.name,
                "model": name,
                "mode": grown_time["mode"],
                "point_schedule": {
                    str(birth): int(np.count_nonzero(born == birth)) for birth in sorted(set(born))
                },
                "initial_fixed": initial_fixed,
                "initial_grown": initial_grown,
                "after_birth_cohorts": cohorts,
                "model_fps_fixed": fixed_time.get(
                    "model_only_fps", fixed_time.get("unique_source_frame_fps_model_only")
                ),
                "model_fps_grown": grown_time.get(
                    "model_only_fps", grown_time.get("unique_source_frame_fps_model_only")
                ),
                "adapter_fps_fixed": fixed_time.get(
                    "decode_track_lift_fps", fixed_time.get("unique_source_frame_fps_decode_plus_model")
                ),
                "adapter_fps_grown": grown_time.get(
                    "decode_track_lift_fps", grown_time.get("unique_source_frame_fps_decode_plus_model")
                ),
            }
            if "tracker_s" in grown_pred and "tracker_s" in fixed_pred:
                boundaries = sorted(set(born)) + [len(grown_ref["frames"])]
                steady_intervals = [(a + 1, b) for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
                row["steady_tracker_ms_fixed"] = [
                    float(np.median(fixed_pred["tracker_s"][a:b]) * 1000) for a, b in steady_intervals
                ]
                row["steady_tracker_ms_grown"] = [
                    float(np.median(grown_pred["tracker_s"][a:b]) * 1000) for a, b in steady_intervals
                ]
                row["birth_tracker_ms_grown"] = {
                    str(birth): float(grown_pred["tracker_s"][birth] * 1000) for birth in boundaries[1:-1]
                }
            rows.append(row)
    if not rows:
        raise ValueError("no matching fixed/growth predictions")
    if len(schedules) != 1 or len(spacings) != 1:
        raise ValueError("report requires a consistent query growth schedule")
    macro = []
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        births = sorted(set().union(*(row["after_birth_cohorts"] for row in subset)), key=int)
        summary = {
            "model": model,
            "clips": len(subset),
            "initial_fixed_2px": float(np.mean([row["initial_fixed"]["coverage_2px"] for row in subset])),
            "initial_grown_2px": float(np.mean([row["initial_grown"]["coverage_2px"] for row in subset])),
            "cohorts_2px": {
                birth: float(np.mean([row["after_birth_cohorts"][birth]["coverage_2px"] for row in subset]))
                for birth in births
            },
            "median_model_fps_fixed": float(np.median([row["model_fps_fixed"] for row in subset])),
            "median_model_fps_grown": float(np.median([row["model_fps_grown"] for row in subset])),
            "median_adapter_fps_fixed": float(np.median([row["adapter_fps_fixed"] for row in subset])),
            "median_adapter_fps_grown": float(np.median([row["adapter_fps_grown"] for row in subset])),
        }
        if "steady_tracker_ms_grown" in subset[0]:
            summary["median_steady_tracker_ms_fixed"] = np.median(
                [row["steady_tracker_ms_fixed"] for row in subset], axis=0
            ).tolist()
            summary["median_steady_tracker_ms_grown"] = np.median(
                [row["steady_tracker_ms_grown"] for row in subset], axis=0
            ).tolist()
        macro.append(summary)
    return {
        "status": "development clips; labels are pose-derived with measured-depth uncertainty; no held-out test",
        "cohort_rule": "new queries are seeded on GT mask plus measured depth, spaced from old projected material points; only post-birth frames are scored",
        "timing_rule": "full-clip methods may use future frames; causal stage time excludes segmentation/fit/robot work",
        "schedule": list(next(iter(schedules))),
        "minimum_old_query_spacing_px": next(iter(spacings)),
        "rows": rows,
        "macro_equal_clip_weight": macro,
    }


def markdown(report: dict) -> str:
    schedule = report["schedule"]
    if len(schedule) != 2:
        raise ValueError("two-birth report renderer requires exactly two scheduled birth cohorts")
    birth_a, birth_b = (birth for birth, _ in schedule)
    count_a, count_b = (count for _, count in schedule)
    births_description = (
        f"{count_a} new points at each of frames {birth_a} and {birth_b}"
        if count_a == count_b
        else f"{count_a} and {count_b} new points at frames {birth_a} and {birth_b}"
    )
    lines = [
        "# FLASH growing-point pilot",
        "",
        report["status"],
        "",
        f"The same initial queries run with and without {births_description}. "
        "New points are chosen from the object's labelled mask and measured depth, "
        f"at least {report['minimum_old_query_spacing_px']:g} pixels "
        "from projected existing material points. These are supplied queries, not autonomous point discovery. "
        "Each cohort's 2-pixel score starts on the frame **after** birth, so a model does not earn "
        "credit merely for repeating its supplied birth coordinate. The original 64 points are scored "
        "separately to expose interference from adding points.",
        "",
        f"| Model | Clips | Original 2 px fixed → grown | Frame {birth_a} cohort | Frame {birth_b} cohort | Model FPS fixed → grown | Adapter FPS fixed → grown |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["macro_equal_clip_weight"]:
        cohorts = row["cohorts_2px"]
        lines.append(
            f"| {row['model']} | {row['clips']} | {row['initial_fixed_2px']:.3f} → {row['initial_grown_2px']:.3f} | "
            f"{cohorts.get(str(birth_a), float('nan')):.3f} | {cohorts.get(str(birth_b), float('nan')):.3f} | "
            f"{row['median_model_fps_fixed']:.1f} → {row['median_model_fps_grown']:.1f} | "
            f"{row['median_adapter_fps_fixed']:.1f} → {row['median_adapter_fps_grown']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Full-clip CoTracker3, TrackEverything and TAPIP3D throughput is not causal publication rate. The other "
            "frontends run on every source frame. Later cohorts have shorter lifetimes and different visible "
            "surface, so their scores cannot be read as proof that growth repaired older tracks. "
            "The points were provided using ground-truth masks for this isolation test; a live discovery "
            "policy and segmentation overhead remain unmeasured.",
            "",
            f"| Causal model | Median per-frame tracker ms: fixed points, frames 1–{birth_a - 1} / {birth_a + 1}–{birth_b - 1} / {birth_b + 1}–end | "
            "Median per-frame tracker ms: growing points, same intervals |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in report["macro_equal_clip_weight"]:
        if "median_steady_tracker_ms_grown" not in row:
            continue
        fixed = " / ".join(f"{x:.2f}" for x in row["median_steady_tracker_ms_fixed"])
        grown = " / ".join(f"{x:.2f}" for x in row["median_steady_tracker_ms_grown"])
        lines.append(f"| {row['model']} | {fixed} | {grown} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--growth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.base_root, args.growth_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(report))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
