"""Score nested point budgets without changing the baseline material points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.flash_surface_benchmark import _load, score

MODELS = ("tapnext", "litetracker", "trackon_r")
BUDGETS = (64, 256, 1024)


def _first_points(values: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    point_keys = {
        "queries": 0,
        "initial_depth_m": 0,
        "truth_xy": 1,
        "truth_xyz": 1,
        "truth_visible": 1,
        "reference_depth_gap_m": 1,
        "xy": 1,
        "xyz": 1,
        "visible": 1,
    }
    return {
        key: np.take(value, np.arange(count), axis=point_keys[key]) if key in point_keys else value
        for key, value in values.items()
    }


def summarize(root: Path, output: Path, base_count: int = 64) -> dict:
    baseline = _load(root / f"budget-{base_count}" / "queries.npz")
    rows = []
    for model in MODELS:
        checkpoint = None
        for budget in BUDGETS:
            experiment = root / f"budget-{budget}"
            reference = _load(experiment / "queries.npz")
            prediction = _load(experiment / f"{model}.npz")
            if not np.array_equal(reference["queries"][:base_count], baseline["queries"]):
                raise ValueError(f"{model}/{budget}: baseline query identities changed")
            timing = json.loads((experiment / f"{model}.timing.json").read_text())
            if checkpoint is not None and checkpoint != timing["checkpoint_sha256"]:
                raise ValueError(f"{model}: checkpoint changed across budgets")
            checkpoint = timing["checkpoint_sha256"]
            common = score(_first_points(reference, base_count), _first_points(prediction, base_count))
            all_points = score(reference, prediction)
            rows.append(
                {
                    "model": model,
                    "budget": budget,
                    "common_eligible": common["eligible_point_frames"],
                    "common_2px": common["coverage_px"]["2"],
                    "common_5mm": common["coverage_mm"]["5"],
                    "all_eligible": all_points["eligible_point_frames"],
                    "all_2px": all_points["coverage_px"]["2"],
                    "all_5mm": all_points["coverage_mm"]["5"],
                    "model_fps": timing["model_only_fps"],
                    "adapter_fps": timing["decode_track_lift_fps"],
                    "gpu_gib": timing["cuda_peak_allocated_bytes"] / 1024**3,
                    "checkpoint_sha256": checkpoint,
                }
            )
    result = {
        "source": str(root.resolve()),
        "common_point_count": base_count,
        "clip": "YCBInEOAT sugar-box moving holdout, frames 619–649",
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "NESTED_BUDGET.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Nested point-budget sweep",
        "",
        "One labelled 31-frame moving sugar-box clip, with 64, 256 and 1,024 queries selected as nested "
        "prefixes of a farthest-point ordering. The same initial 64 material points are scored in every "
        "row; the all-point score is secondary because adding queries changes which surfaces enter the "
        "denominator. These are isolated, causal all-frame runs, with no SAM, shared fit, robot IO, "
        "recording or 30 Hz scheduler. Model FPS excludes RGB-D decoding and depth lifting; adapter FPS "
        "includes them. GPU peak is allocated memory, not process-resident memory.",
        "",
        "| Model | Queries | First 64: 2 px | First 64: 5 mm | All: 2 px | Model FPS | Adapter FPS | Peak GPU GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['budget']} | {row['common_2px']:.3f} | "
            f"{row['common_5mm']:.3f} | {row['all_2px']:.3f} | {row['model_fps']:.1f} | "
            f"{row['adapter_fps']:.1f} | {row['gpu_gib']:.2f} |"
        )
    lines += [
        "",
        "The sweep isolates query-count scaling on one fairly visible object. It does not establish "
        "quality on the low-texture dowel or determine a final point budget. If first-64 accuracy changes, "
        "that is a real interaction with added queries, while changes in the all-point column can also "
        "reflect the newly sampled surface locations. Three models are shown because they provide causal "
        "per-frame updates; full-clip models need a separate offline budget sweep.",
        "",
    ]
    (output / "NESTED_BUDGET.md").write_text("\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
