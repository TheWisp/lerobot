"""Report visible-point quality on a second labelled RGB-D source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.flash_surface_benchmark import score

MODELS = (
    "bootstapir",
    "tapnext",
    "litetracker",
    "trackon_r",
    "lk",
    "cotracker3",
    "tapip3d_unfiltered",
    "trackeverything",
    "spatracker2",
    "point4d",
)
FULL_CLIP = {"cotracker3", "tapip3d_unfiltered", "trackeverything", "spatracker2", "point4d"}
SOURCE_REVISION = "e8a226a40a33432dc901e995a93230e1e3cf4c89"


def _fps(timing: dict) -> float:
    for key in ("decode_track_lift_fps", "unique_source_frame_fps_decode_plus_model"):
        if key in timing:
            return float(timing[key])
    raise ValueError("adapter has no comparable decode-plus-inference throughput")


def summarize(experiments: tuple[Path, ...], output: Path) -> dict:
    rows = []
    clips = []
    for experiment in experiments:
        manifest = json.loads((experiment / "manifest.json").read_text())
        if "LSFOdyssey supplied point trajectories" not in manifest["ground_truth"]:
            raise ValueError(f"{experiment}: wrong reference source")
        with np.load(experiment / "queries.npz", allow_pickle=False) as source:
            query = source["queries"]
            truth_visible = source["truth_visible"]
            truth_xy = source["truth_xy"]
            reference = {key: source[key] for key in source.files}
        visible_pair = truth_visible[1:] & truth_visible[:-1]
        step_px = np.linalg.norm(np.diff(truth_xy, axis=0), axis=-1)
        travel_px = np.sum(np.where(visible_pair, step_px, 0), axis=0)
        high_motion_ids = np.argsort(-travel_px, kind="stable")[: max(1, len(query) // 4)]
        high_motion_eligible = truth_visible[:, high_motion_ids]
        clips.append(
            {
                "clip": experiment.name,
                "frames": manifest["last_frame"] + 1,
                "queries": len(query),
                "reference_depth_gap_mm": manifest["reference_depth_gap_m_median_visible"] * 1000,
                "eligible_point_frames": int(truth_visible.sum()),
                "high_motion_query_ids": high_motion_ids.tolist(),
                "high_motion_minimum_travel_px": float(travel_px[high_motion_ids[-1]]),
                "source_track_ids": manifest["source_track_ids"],
                "source_hashes_sha256": {
                    name: manifest[f"source_{name}_sha256"]
                    for name in ("rgb", "depth", "tracks", "intrinsics")
                },
                "queries_sha256": manifest["queries_sha256"],
            }
        )
        for model in MODELS:
            evaluation_path = experiment / f"{model}.eval.json"
            if not evaluation_path.is_file():
                continue
            accuracy = json.loads(evaluation_path.read_text())["accuracy"]
            timing = json.loads((experiment / f"{model}.timing.json").read_text())
            with np.load(experiment / f"{model}.npz", allow_pickle=False) as prediction:
                if not np.array_equal(prediction["queries"], query):
                    raise ValueError(f"{experiment}/{model}: query identities changed")
                rescored = score(reference, {key: prediction[key] for key in prediction.files})
                if rescored["coverage_px"] != accuracy["coverage_px"]:
                    raise ValueError(f"{experiment}/{model}: saved score disagrees with source arrays")
                initial_error = np.linalg.norm(prediction["xy"][0] - query[:, 1:3], axis=1)
                error_px = np.linalg.norm(
                    prediction["xy"][:, high_motion_ids] - truth_xy[:, high_motion_ids], axis=-1
                )
                high_motion_correct = (
                    high_motion_eligible
                    & prediction["visible"][:, high_motion_ids]
                    & np.isfinite(error_px)
                    & (error_px <= 2)
                )
            rows.append(
                {
                    "clip": experiment.name,
                    "model": model,
                    "temporal_mode": "full clip" if model in FULL_CLIP else "causal frame by frame",
                    "first_frame_median_px": float(np.nanmedian(initial_error)),
                    "eligible_point_frames": accuracy["eligible_point_frames"],
                    "coverage_2px": accuracy["coverage_px"]["2"],
                    "high_motion_quartile_coverage_2px": float(
                        high_motion_correct.sum() / high_motion_eligible.sum()
                    ),
                    "coverage_4px": accuracy["coverage_px"]["4"],
                    "available_fraction": accuracy["available_fraction"],
                    "adapter_fps": _fps(timing),
                    "checkpoint_sha256": timing.get("checkpoint_sha256"),
                }
            )
    result = {
        "source_dataset": "https://huggingface.co/datasets/wwcreator/LSFOdyssey",
        "source_revision": SOURCE_REVISION,
        "clips": clips,
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "LSFODYSSEY.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# LSFOdyssey labelled RGB-D clips",
        "",
        "An exploratory second labelled source checks material-point identity on two 40-frame "
        "synthetic scene videos. "
        "The clips are `cab_e_3rd/000000` and `dancingroom3_3rd/000000` from "
        "[LSFOdyssey](https://huggingface.co/datasets/wwcreator/LSFOdyssey), revision "
        f"`{SOURCE_REVISION}`. Each uses 128 first-frame visible points "
        "selected spatially from the supplied 2D/3D trajectories. Ground-truth coordinates and later "
        "visibility are never adapter inputs; no object mask is provided. The points span both moving "
        "people and static room surfaces. This is a frontend identity check, not a rigid-object pose fit.",
        "",
        "The source's depth differs from its labelled point depth by about 33–35 mm median at visible "
        "pixels, so a 5 mm metric score would mostly test that disagreement. The table ranks 2D identity "
        "only. The video reports 24 fps, but no independent camera timestamps are supplied; adapter FPS "
        "below is isolated compute throughput. Full-clip models may see future frames and have no causal "
        "publication rate here. The original RGB frames are kept in the local cache rather than copied "
        "into this repository. Model-training overlap with these public scenes was not audited.",
        "",
        "The high-motion column separately scores the quarter of initial queries with the greatest "
        "sum of visible consecutive-frame pixel travel within each clip. The ranking uses source labels "
        "only for evaluation, never as model input. It is image motion, not a rigid-object label. "
        "Missing predicted visible points remain in both coverage denominators.",
        "",
        "| Clip | Model | Mode | 2 px coverage | High-motion 2 px | 4 px coverage | Initial px error | Adapter FPS |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['clip']} | {row['model']} | {row['temporal_mode']} | "
            f"{row['coverage_2px']:.3f} | {row['high_motion_quartile_coverage_2px']:.3f} | "
            f"{row['coverage_4px']:.3f} | "
            f"{row['first_frame_median_px']:.2f} | {row['adapter_fps']:.1f} |"
        )
    lines += [
        "",
        "These two scenes provide independent point labels but are synthetic, distant (metre-scale) and "
        "mostly textured. The YCBInEOAT moving holdout and local dowel/gamepad clips remain necessary for "
        "small, weak-texture manipulation surfaces. One source or one accuracy threshold cannot select "
        "the FLASH tracker.",
        "",
    ]
    (output / "LSFODYSSEY.md").write_text("\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(tuple(args.experiments), args.output), indent=2))


if __name__ == "__main__":
    main()
