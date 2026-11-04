"""Offline alignment search from a recorded initial surface; not a tracking run.

No labels select the pose. Save every candidate so rejection and recovery can
be reviewed separately. The optional scene-depth rule is not a new default.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.flash_experiment_inputs import sample_at
from lerobot.policies.flash.surface_acceleration import compiled_kernels
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit, pca_candidates
from lerobot.policies.flash.surface_observation import voxel_sample
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


class OverlapSeekingFit(WholeSurfaceFit):
    """Historical objective retained as an independent equivalence-test oracle.

    The current CLI uses WholeSurfaceFit's prefer_observed_overlap setting.

    A missing/hidden point contributes a constant search cost. Moving measured
    surface into measured agreement reduces that cost. This preference is not
    a contradiction or an assertion that unobserved shape should be present.
    The ordinary depth contradiction checks still decide admissibility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverse_fits = [
            (
                camera,
                OverlapSeekingFit(
                    reverse.reference,
                    reverse.observation,
                    reverse.k,
                    reverse.config,
                    fit_kernels=reverse.fit_kernels,
                ),
            )
            for camera, reverse in self.reverse_fits
        ]

    def _one_view_residual(self, pose, predicted, model_distance, target_distance):
        residual = super()._one_view_residual(pose, predicted, model_distance, target_distance)
        visible, _, _ = self._projection_evidence(predicted)
        agreement = visible * np.exp(-((model_distance / self.config.depth_tolerance_m) ** 2))
        residual[: len(predicted)] = np.sqrt(np.clip(1 - agreement, 0, 1) / len(predicted))
        return residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-objective", action="store_true")
    parser.add_argument("--keep-all-depth-ranges", action="store_true")
    parser.add_argument(
        "--visible-outline",
        action="store_true",
        help="Prefer covering more of the visible outline; uncovered areas alone do not reject a partial fit",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case = json.loads((args.run / "case.json").read_text())
    provenance = json.loads((args.run.parent / "provenance.json").read_text())
    vision = VisionPipelineConfig.from_dict(provenance["config"])
    config = replace(
        vision.shape,
        scene_depth_constraints=True,
        prefer_observed_overlap=args.overlap_objective,
        require_full_mask_coverage=args.visible_outline,
    )
    observation = replace(
        vision.observation,
        component_gap_m=None,
        depth_outlier_sigma=None if args.keep_all_depth_ranges else 3.0,
        filtered_depth_for_fitting=True,
    )
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "case": case,
                "shape": asdict(config),
                "observation": asdict(observation),
                "overlap_objective": args.overlap_objective,
                "visible_outline": args.visible_outline,
                "role": args.role,
                "source_frame": 0,
                "source_code": Path(__file__).read_text(),
            },
            indent=2,
        )
        + "\n"
    )

    def prepare(index):
        frame, masks = sample_at(case, index)
        builder = SurfaceTrackingSession(None, shape_config=config, observation_config=observation)
        return builder.prepare(frame.rgb, frame.metric_depth(), masks[args.role], frame.intrinsics, index)

    source = prepare(0)
    reference = source["current"]["all_surface_points"]
    reference = reference[voxel_sample(reference, config.surface_spacing_m)]
    np.savez_compressed(args.output / "reference.npz", points=reference)
    rows = [json.loads(line) for line in (args.run / "frames.jsonl").open()]
    uncertainty = PoseUncertaintyConfig(max_visible_outlier_fraction=0.25)
    results = []
    for index in args.frames:
        started = perf_counter()
        current = prepare(index)
        fit = WholeSurfaceFit(
            reference,
            current["current"],
            current["intrinsics"],
            config,
            reference_observation=source["current"],
            reference_intrinsics=source["intrinsics"],
            fit_kernels=compiled_kernels(),
        )
        seeds = pca_candidates(reference, current["current"]["all_surface_points"], all_axis_orders=True)
        # Include the old output as a search seed, never as ground truth.
        old = rows[index]["objects"][args.role].get("group_growing")
        if old is not None:
            seeds.insert(0, np.asarray(old))

        def check(pose, geometry=fit):
            return check_surface_contradiction(geometry, pose, uncertainty)["accepted"]

        candidates = []
        for i, seed in enumerate(seeds):
            pose, detail = fit.refine(seed, admissible=check)
            candidates.append(
                {
                    "seed": i,
                    "pose": pose.tolist(),
                    "detail": detail,
                    "check": check_surface_contradiction(fit, pose, uncertainty),
                }
            )
        valid = [i for i, candidate in enumerate(candidates) if candidate["check"]["accepted"]]
        selected = min(valid, key=lambda i: candidates[i]["detail"]["cost"]) if valid else None
        row = {
            "frame": index,
            "outline_coverage_active": fit.coverage_enabled,
            "selected": selected,
            "candidates": candidates,
            "wall_s": perf_counter() - started,
        }
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            index,
            "accepted",
            len(valid),
            "of",
            len(candidates),
            "selected",
            selected,
            "seconds",
            round(row["wall_s"], 2),
            flush=True,
        )


if __name__ == "__main__":
    main()
