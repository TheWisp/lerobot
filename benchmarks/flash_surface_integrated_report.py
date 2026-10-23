"""Summarize repeated paced tracker/mask/fit timing runs on one source clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODES = ("none", "cached-mask-fit", "sam3-mask-fit")
STAGES = ("decode_s", "mask_s", "tracker_s", "lift_s", "fit_s")


def summarize(root: Path, output: Path) -> dict:
    records: dict[str, list[dict]] = {}
    checkpoints = set()
    for mode in MODES:
        paths = sorted(root.glob(f"tapnext_integrated_{mode}_*.timing.json"))
        if len(paths) < 3:
            raise ValueError(f"{mode}: need at least three repeated runs")
        records[mode] = []
        for path in paths:
            report = json.loads(path.read_text())
            if report["integration"] != mode or report["model"] != "tapnext":
                raise ValueError(f"{path}: mismatched timing mode/model")
            checkpoints.add(report["checkpoint_sha256"])
            prediction_path = path.with_name(path.name.removesuffix(".timing.json") + ".npz")
            with np.load(prediction_path, allow_pickle=False) as trace:
                indices = np.asarray(report["pacing"]["publication_source_index"], int)
                if len(indices) != report["pacing"]["processed_source_frames"]:
                    raise ValueError(f"{path}: publication/processed count differs")
                if len(indices) < 2:
                    raise ValueError(f"{path}: no steady frames")
                stages = {key: float(np.median(trace[key][indices[1:]]) * 1000) for key in STAGES}
            pace = report["pacing"]
            records[mode].append(
                {
                    "source_frames": report["frames"],
                    "processed": pace["processed_source_frames"],
                    "age_median_ms": pace["observed_pose_age_ms"]["median"],
                    "age_p90_ms": pace["observed_pose_age_ms"]["p90"],
                    "gap_p90_ms": pace["publication_gap_ms"]["p90"],
                    "deadline_misses": pace["consumer_deadline_misses"],
                    "ticks_without_result": pace["consumer_ticks_without_result"],
                    "stages_ms": stages,
                    "fit_available": report["fit_available_frames"],
                    "mask_nonempty": report["mask_nonempty_frames"],
                    "gpu_peak_gib": report["cuda_peak_allocated_bytes"] / 1024**3,
                }
            )
    if len(checkpoints) != 1:
        raise ValueError("runs use different tracker checkpoints")
    summary = {
        "clip": root.name,
        "checkpoint_sha256": checkpoints.pop(),
        "hardware": report["hardware"],
        "repeats": {mode: len(records[mode]) for mode in MODES},
        "modes": {},
    }
    for mode, runs in records.items():
        summary["modes"][mode] = {
            key: float(np.median([run[key] for run in runs]))
            for key in (
                "source_frames",
                "processed",
                "age_median_ms",
                "age_p90_ms",
                "gap_p90_ms",
                "deadline_misses",
                "ticks_without_result",
                "fit_available",
                "mask_nonempty",
                "gpu_peak_gib",
            )
        }
        summary["modes"][mode]["stages_ms"] = {
            stage: float(np.median([run["stages_ms"][stage] for run in runs])) for stage in STAGES
        }
        summary["modes"][mode]["per_repeat"] = runs
    output.mkdir(parents=True, exist_ok=True)
    (output / "INTEGRATED.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, output / "INTEGRATED.md")
    _plot(summary, output / "integrated_stage_breakdown.png")
    return summary


def _write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Paced tracker, mask and fit overhead",
        "",
        "Three independent process runs per mode on the same 91-frame, approximately 30 Hz local dowel clip. "
        "All modes used the same TAPNext checkpoint and 64 first-frame queries. The consumer is a separate "
        "30 Hz Python thread; the producer processes the latest available source frame and publishes only "
        "after the selected stages finish. Live SAM3 is point-seeded on the first frame at 504 px and then "
        "propagated. This measures one serialized perception path, not the full FLASH robot or concurrent "
        "segmentation and point-tracking processes. The common rigid fit uses measured depth and 10 mm "
        "RANSAC residuals; its availability is not a pose-accuracy score.",
        "",
        "| Mask/fit mode | Fresh frames | Fits available | Mask ms | Tracker ms | Fit ms, all frames | Age p50/p90 ms | Gap p90 ms | 30 Hz misses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        x = summary["modes"][mode]
        s = x["stages_ms"]
        lines.append(
            f"| {mode} | {x['processed']:.0f}/{x['source_frames']:.0f} | {x['fit_available']:.0f} | "
            f"{s['mask_s']:.1f} | {s['tracker_s']:.1f} | {s['fit_s']:.1f} | "
            f"{x['age_median_ms']:.1f}/{x['age_p90_ms']:.1f} | {x['gap_p90_ms']:.1f} | "
            f"{x['deadline_misses']:.0f} |"
        )
    lines += [
        "",
        "[Stage and result-age chart](integrated_stage_breakdown.png). "
        "The mask/tracker/fit times exclude model loading but include per-frame work; first-frame model "
        "warmup is included in skipped-frame and age counts. All fit and depth stages run after tracking. "
        "A ticking consumer can still act on an old estimate: the live-mask mode's p90 age is much larger "
        "than its 33 ms tick period. No motor IO, camera capture, recording, display, planner or independent "
        "point-identity truth is part of this timing probe. Mask correctness and fit correctness must be "
        "evaluated separately; nonempty masks and available fits are not accuracy evidence. "
        "The cached path's fit cost is near zero on many frames because the point frontend has fewer "
        "than three visible points with valid measured depth there, so no 3D fit is possible. "
        "The slower live-mask schedule skips source frames and therefore feeds a different sequence into "
        "the temporal tracker; its higher fit availability must not be attributed to better masks.",
        "",
    ]
    path.write_text("\n".join(lines))


def _plot(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Tracker only", "Cached mask + fit", "Live SAM3 + fit"]
    colors = {
        "decode_s": "#a0a0a0",
        "mask_s": "#d56543",
        "tracker_s": "#256eaa",
        "lift_s": "#7fb899",
        "fit_s": "#8c60ad",
    }
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2))
    left = np.zeros(len(MODES))
    for stage in STAGES:
        values = np.array([summary["modes"][mode]["stages_ms"][stage] for mode in MODES])
        axes[0].barh(labels, values, left=left, color=colors[stage], label=stage.removesuffix("_s"))
        left += values
    axes[0].axvline(1000 / 30, color="#333333", linestyle="--", linewidth=1.2, label="30 Hz period")
    axes[0].set_xlabel("Median work per processed frame (ms)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.6, 0.985))
    axes[0].invert_yaxis()
    x = np.arange(len(MODES))
    mid = [summary["modes"][mode]["age_median_ms"] for mode in MODES]
    p90 = [summary["modes"][mode]["age_p90_ms"] for mode in MODES]
    axes[1].bar(x - 0.17, mid, width=0.34, label="median age", color="#256eaa")
    axes[1].bar(x + 0.17, p90, width=0.34, label="p90 age", color="#d56543")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Age of estimate seen by 30 Hz consumer (ms)")
    axes[1].legend(fontsize=8)
    fig.subplots_adjust(left=0.29, right=0.97, top=0.91, bottom=0.10, hspace=0.43)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
