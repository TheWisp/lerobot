"""Summarize a frozen visible-surface pilot without pooling correlated point frames.

The table keeps labelled accuracy apart from diagnostics and separates offline
full-clip throughput from causal per-frame throughput. It does not rank a model
whose mode or reference quality differs from another model's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.flash_surface_benchmark import _load, score


def collect(root: Path) -> dict:
    rows = []
    for experiment in sorted(root.iterdir()):
        manifest_path, query_path = experiment / "manifest.json", experiment / "queries.npz"
        if not manifest_path.is_file() or not query_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        reference = _load(query_path)
        if "truth_visible" not in reference:
            continue
        for prediction_path in sorted(experiment.glob("*.npz")):
            if prediction_path.name == "queries.npz" or prediction_path.stem.endswith("_paced"):
                continue
            model = prediction_path.stem
            timing_path = prediction_path.with_suffix(".timing.json")
            if not timing_path.is_file():
                continue
            prediction = _load(prediction_path)
            accuracy = score(reference, prediction)
            timing = json.loads(timing_path.read_text())
            row = {
                "clip": experiment.name,
                "model": model,
                "mode": timing["mode"],
                "eligible_point_frames": accuracy["eligible_point_frames"],
                "labelled_visible_fraction": manifest.get("labelled_visible_fraction"),
                "reference_depth_gap_median_mm": (
                    manifest.get("reference_depth_gap_m_median_on_visible", float("nan")) * 1000
                ),
                "coverage_2px": accuracy["coverage_px"]["2"],
                "coverage_5mm": accuracy.get("coverage_mm", {}).get("5"),
                "coverage_10mm": accuracy.get("coverage_mm", {}).get("10"),
                "predicted_visible_fraction": accuracy["available_fraction"],
                "available_error_2d_median_px": (accuracy["error_px_when_available"] or {}).get("median"),
                "available_error_3d_median_mm": (accuracy.get("error_mm_when_available") or {}).get("median"),
                "model_only_fps": timing.get(
                    "model_only_fps", timing.get("unique_source_frame_fps_model_only")
                ),
                "decode_plus_model_fps": timing.get(
                    "decode_track_lift_fps", timing.get("unique_source_frame_fps_decode_plus_model")
                ),
                "gpu_peak_gib": timing.get("cuda_peak_allocated_bytes", 0) / (1024**3),
                "geometry_source": str(prediction.get("geometry_source", "unknown")),
            }
            rows.append(row)
    if not rows:
        raise ValueError(f"no labelled benchmark predictions under {root}")
    models = sorted({row["model"] for row in rows})
    macro = []
    for model in models:
        subset = [row for row in rows if row["model"] == model]
        macro.append(
            {
                "model": model,
                "clips": len(subset),
                "clip_mean_coverage_2px": float(np.mean([row["coverage_2px"] for row in subset])),
                "clip_mean_coverage_5mm": float(np.mean([row["coverage_5mm"] for row in subset])),
                "median_model_only_fps": float(np.median([row["model_only_fps"] for row in subset])),
                "median_decode_plus_model_fps": float(
                    np.median([row["decode_plus_model_fps"] for row in subset])
                ),
                "peak_gpu_gib": max(row["gpu_peak_gib"] for row in subset),
            }
        )
    return {
        "status": "development pilot; clips are not a held-out or independent-source result",
        "reference": "YCBInEOAT object pose plus source depth; visibility gated by GT mask/raw depth",
        "modes_not_interchangeable": "full-clip windowed inference may use future frames; causal throughput excludes SAM/fit/consumer",
        "rows": rows,
        "macro_equal_clip_weight": macro,
    }


def collect_paced(root: Path) -> dict:
    rows = []
    for experiment in sorted(root.iterdir()):
        manifest_path, query_path = experiment / "manifest.json", experiment / "queries.npz"
        if not manifest_path.is_file() or not query_path.is_file():
            continue
        reference = _load(query_path)
        capture = reference["capture_s"]
        for prediction_path in sorted(experiment.glob("*_paced.npz")):
            timing_path = prediction_path.with_suffix(".timing.json")
            if not timing_path.is_file():
                continue
            prediction = _load(prediction_path)
            measured = json.loads(timing_path.read_text())
            pacing = measured["pacing"]
            row = {
                "clip": experiment.name,
                "model": prediction_path.stem.removesuffix("_paced"),
                "source_fps": float((len(capture) - 1) / capture[-1]),
                "source_frames": len(capture),
                "processed": pacing["processed_source_frames"],
                "skipped": pacing["skipped_source_frames"],
                "first_publication_ms": pacing["publication_s"][0] * 1000,
                "consumer_deadline_misses": pacing["consumer_deadline_misses"],
                "consumer_ticks_without_result": pacing["consumer_ticks_without_result"],
                "age_median_ms": pacing["observed_pose_age_ms"]["median"],
                "age_p90_ms": pacing["observed_pose_age_ms"]["p90"],
                "publication_gap_p90_ms": pacing["publication_gap_ms"]["p90"],
                "model_only_fps_on_processed": measured["model_only_fps"],
            }
            if "truth_visible" in reference:
                accuracy = score(reference, prediction)
                row["coverage_2px"] = accuracy["coverage_px"]["2"]
                row["coverage_5mm"] = accuracy["coverage_mm"]["5"]
            rows.append(row)
    if not rows:
        raise ValueError(f"no paced predictions under {root}")
    return {
        "status": "paced development pilot, one run per model; not steady-state or live arm FPS",
        "rows": rows,
    }


def markdown(report: dict) -> str:
    lines = [
        "# FLASH visible-surface pilot",
        "",
        report["status"],
        "",
        "Coverage includes missing point predictions in its denominator. The reference transports source-depth "
        "points through object-pose annotations; its own measured-depth disagreement is shown alongside scores. "
        "Every number below is a development result, not a tracker selection.",
        "",
        "| Clip | Model | 2 px coverage | 5 mm coverage | Median reference depth gap | Model FPS | Full decode FPS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['clip']} | {row['model']} | {row['coverage_2px']:.3f} | "
            f"{row['coverage_5mm']:.3f} | {row['reference_depth_gap_median_mm']:.1f} mm | "
            f"{row['model_only_fps']:.1f} | {row['decode_plus_model_fps']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Model FPS is full-clip throughput for TrackEverything and causal per-frame tracker throughput for "
            "the other adapters. Full decode FPS includes recorded RGB-D loading; neither includes segmentation, "
            "pose fitting, network transfer, recording, or a 30 Hz consumer. They are not measured live-loop FPS.",
            "",
            "| Model | Clips | Mean 2 px coverage | Mean 5 mm coverage | Median model FPS | Peak GPU memory |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["macro_equal_clip_weight"]:
        lines.append(
            f"| {row['model']} | {row['clips']} | {row['clip_mean_coverage_2px']:.3f} | "
            f"{row['clip_mean_coverage_5mm']:.3f} | {row['median_model_only_fps']:.1f} | "
            f"{row['peak_gpu_gib']:.2f} GiB |"
        )
    return "\n".join(lines) + "\n"


def paced_markdown(report: dict) -> str:
    lines = [
        "# FLASH paced perception pilot",
        "",
        report["status"],
        "",
        "The producer releases recorded frames at their source timestamps and processes the newest available "
        "frame. The independent consumer ticks at 30 Hz and holds the last published result. Unprocessed "
        "frames remain missing in accuracy scores. Model loading is excluded, but first-frame model warmup is included.",
        "",
        "| Clip | Model | Source FPS | Processed / source | First result | Consumer age median / p90 | Gap p90 | 30 Hz misses |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['clip']} | {row['model']} | {row['source_fps']:.1f} | "
            f"{row['processed']}/{row['source_frames']} | {row['first_publication_ms']:.0f} ms | "
            f"{row['age_median_ms']:.0f} / {row['age_p90_ms']:.0f} ms | "
            f"{row['publication_gap_p90_ms']:.0f} ms | {row['consumer_deadline_misses']} |"
        )
    lines.extend(
        [
            "",
            "This is a minimal Python scheduling probe. It does not run SAM, the rigid fit, FLASH planning, motor IO, "
            "or display work. Consumer deadline misses therefore do not certify the robot loop. The public "
            "YCBInEOAT clip's source rate is about 16 Hz; a 30 Hz consumer naturally repeats estimates there. "
            "The local dowel source is about 30 Hz.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paced", action="store_true")
    args = parser.parse_args()
    result = collect_paced(args.root) if args.paced else collect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(paced_markdown(result) if args.paced else markdown(result))
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
