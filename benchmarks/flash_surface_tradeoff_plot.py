"""Plot visible-point quality against isolated adapter throughput by temporal mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def render(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text())
    if report["cohort"] != "motion-holdout":
        raise ValueError("trade-off plot requires the frozen moving holdout")
    names = {
        "bootstapir": "BootsTAPIR",
        "cotracker3": "CoTracker3",
        "litetracker": "LiteTracker",
        "lk": "Optical flow",
        "point4d": "Point4D (2D)",
        "spatracker2": "SpaTrackerV2",
        "tapip3d": "TAPIP3D filtered",
        "tapip3d_unfiltered": "TAPIP3D unfiltered",
        "tapnext": "TAPNext++",
        "trackeverything": "TrackEverything",
        "trackon_r": "Track-On-R",
    }
    modes = {row["model"]: row["mode"] for row in report["rows"]}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for model in report["macro_equal_clip_weight"]:
        key = model["model"]
        if key == "spatracker2_sensor":
            continue  # Same 2D track and model time; only its 3D lifting changes.
        clip_rows = [row for row in report["rows"] if row["model"] == key]
        causal = modes[key].startswith("causal")
        axis = axes[0 if causal else 1]
        x = model["median_decode_plus_model_fps"]
        y = model["clip_mean_coverage_2px"]
        lower = min(row["coverage_2px"] for row in clip_rows)
        upper = max(row["coverage_2px"] for row in clip_rows)
        axis.plot([x, x], [lower, upper], color="#94a3b8", linewidth=2, zorder=1)
        axis.scatter(
            x, y, s=58, marker="o" if causal else "s", color="#2563eb" if causal else "#ea580c", zorder=2
        )
        axis.annotate(names[key], (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8)
    for axis, title in zip(axes, ("Causal frame-by-frame", "Full clip / future context"), strict=True):
        axis.set_title(title)
        axis.set_xscale("log")
        axis.set_xlabel("Isolated adapter throughput (source frames/s)")
        axis.set_xlim(3, 400)
        axis.set_ylim(0, 1)
        axis.axvline(30, color="#64748b", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Correct visible point coverage within 2 px")
    figure.suptitle(
        "FLASH: moving RGB-D holdout, three clips\nRange shows clip variation; throughput excludes segmentation and robot loop"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.report, args.output)


if __name__ == "__main__":
    main()
