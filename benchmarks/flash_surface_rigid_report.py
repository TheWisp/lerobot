"""Summarize common rigid fits while counting wrong available poses as failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def collect(root: Path, residual_mm: float) -> dict:
    fit_dir = f"fits-{residual_mm:g}mm"
    rows = []
    configurations = set()
    for clip in sorted(root.iterdir()):
        folder = clip / fit_dir
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.npz")):
            settings = json.loads(path.with_suffix(".json").read_text())
            configurations.add(
                (
                    settings["success_probability"],
                    settings["prior_inlier_fraction"],
                    settings["ransac_trials_per_frame"],
                )
            )
            with np.load(path, allow_pickle=False) as saved:
                available = saved["available"][1:]
                error = saved["cloud_error_mm"][1:]
                angle = saved["angular_error_deg"][1:]
                fit_ms = saved["fit_s"][1:] * 1000
                inliers = saved["inlier_count"][1:]
            row = {
                "clip": clip.name,
                "model": path.stem,
                "evaluated_frames": len(error),
                "available_fraction": float(available.mean()),
                "transported_cloud_coverage": {
                    str(limit): float(np.mean(available & (error <= limit))) for limit in (5, 10, 20)
                },
                "cloud_error_mm_median_when_available": float(np.median(error[available]))
                if available.any()
                else None,
                "cloud_error_mm_p90_when_available": float(np.quantile(error[available], 0.9))
                if available.any()
                else None,
                "rotation_error_deg_median_when_available": float(np.median(angle[available]))
                if available.any()
                else None,
                "median_inliers": float(np.median(inliers)),
                "fit_ms_median": float(np.median(fit_ms)),
                "fit_ms_p90": float(np.quantile(fit_ms, 0.9)),
            }
            rows.append(row)
    if not rows:
        raise ValueError(f"no fitted results in {root} at residual {residual_mm:g} mm")
    if len(configurations) != 1:
        raise ValueError("mixed robust-fit settings in one report")
    probability, inlier_prior, trials = next(iter(configurations))
    macro = []
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        macro.append(
            {
                "model": model,
                "clips": len(subset),
                "available_fraction": float(np.mean([row["available_fraction"] for row in subset])),
                "cloud_coverage_5mm": float(
                    np.mean([row["transported_cloud_coverage"]["5"] for row in subset])
                ),
                "cloud_coverage_10mm": float(
                    np.mean([row["transported_cloud_coverage"]["10"] for row in subset])
                ),
                "cloud_coverage_20mm": float(
                    np.mean([row["transported_cloud_coverage"]["20"] for row in subset])
                ),
                "median_fit_ms": float(np.median([row["fit_ms_median"] for row in subset])),
            }
        )
    return {
        "status": "same estimator, pose-derived reference, no robot validation or symmetry-aware task score",
        "residual_mm": residual_mm,
        "success_probability": probability,
        "prior_inlier_fraction": inlier_prior,
        "ransac_trials_per_frame": trials,
        "anchor_frame_excluded": True,
        "rows": rows,
        "macro_equal_clip_weight": macro,
    }


def markdown(report: dict) -> str:
    lines = [
        "# FLASH common rigid-fit pilot",
        "",
        report["status"],
        "",
        "Every model's first-frame depth seeds and predicted 3D positions enter the same 3-point RANSAC/Kabsch "
        f"fit with a {report['residual_mm']:g} mm residual threshold. The "
        f"{report['ransac_trials_per_frame']}-trial budget follows a "
        f"{report['success_probability']:.0%} success target with a declared "
        f"{report['prior_inlier_fraction']:.0%} inlier prior. The fit receives no evaluation pose or mask. "
        "After fitting, the initial point cloud is transported by the estimated transform and compared with "
        "the dataset's annotated object motion. The source frame is excluded; missing poses count as failures. "
        "The 5/10/20 mm columns are coverage over **all later frames**, not error only when a pose was emitted.",
        "",
        "| Model | Clips | Pose available | Cloud within 5 mm | Within 10 mm | Within 20 mm | Median fit time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["macro_equal_clip_weight"]:
        lines.append(
            f"| {row['model']} | {row['clips']} | {row['available_fraction']:.3f} | "
            f"{row['cloud_coverage_5mm']:.3f} | {row['cloud_coverage_10mm']:.3f} | "
            f"{row['cloud_coverage_20mm']:.3f} | {row['median_fit_ms']:.2f} ms |"
        )
    lines.extend(
        [
            "",
            "A fit can be available and still wrong: a coherent set of drifting or background points may satisfy "
            "the inlier threshold. This estimator adds about 2 ms of CPU work on these short clips; it is not "
            "an integrated segmentation-to-robot latency measurement. Native model 3D outputs and RGB tracks "
            "lifted through measured depth use the same fitter but have different depth-error mechanisms. "
            "The raw point report remains the primary visible-surface identity test.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--residual-mm", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.root, args.residual_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(report))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
