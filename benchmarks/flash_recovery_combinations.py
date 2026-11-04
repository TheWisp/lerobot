"""Combine the parked no-reset filters with the new surface recovery experiment.

Every case starts at frame zero through the real UFM/group/surface pipeline.
Use the same filter and surface settings for all cases. Never publish these
outputs over the GUI results; labels are consumed only by the scorer.
"""

import argparse
import json
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from benchmarks.flash_velocity_filter import VelocityConfig, VelocityPoseFilter
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter

FILTERS = {
    "reset": PoseKalmanFilter,
    "reset_broader": PoseKalmanFilter,
    "strict": PoseKalmanFilter,
    "strict_broader": PoseKalmanFilter,
    "velocity": VelocityPoseFilter,
}


def configurations(provenance, fitting, filter_name):
    from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    if filter_name not in FILTERS or fitting not in {"baseline", "partial", "recovery"}:
        raise ValueError("Unknown recovery comparison configuration")
    vision = VisionPipelineConfig.from_dict(provenance["config"])
    surface = GrowingSurfaceConfig(**provenance["surface_experiment"])
    # Store the actual policy in the same settings consumed by normal reruns.
    # A factory override used to make saved "strict" runs claim reset=True.
    surface = replace(
        surface,
        uncertainty=replace(
            surface.uncertainty, reset_on_geometry_conflict=filter_name in {"reset", "reset_broader"}
        ),
    )
    if fitting == "partial":
        # Isolate the corrected partial-overlap rule from the additional
        # depth cleanup, reduced source views and coarse recovery guesses.
        surface = replace(surface, partial_surface_matching=True)
    if fitting == "recovery":
        vision = replace(
            vision,
            observation=replace(
                vision.observation,
                component_gap_m=None,
                depth_outlier_sigma=3.0,
                filtered_depth_for_fitting=True,
            ),
        )
        surface = replace(surface, partial_surface_matching=True, recovery_starts=4, fit_view_limit=1)
    if filter_name in {"reset_broader", "strict_broader"}:
        surface = replace(
            surface, uncertainty=replace(surface.uncertainty, position_process_std_m_sqrt_s=0.03)
        )
    return vision, surface


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    import cv2
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    from benchmarks.flash_experiment_groups import GroupConfig
    from benchmarks.flash_tracker_experiment import code_snapshot, run_case
    from lerobot.policies.flash import experimental_surface_session as session
    from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
    from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
    from lerobot.policies.flash.surface_tracking import load_ufm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter", choices=FILTERS, required=True)
    parser.add_argument("--fitting", choices=["baseline", "partial", "recovery"], required=True)
    parser.add_argument("--rotation-process-std", type=float)
    parser.add_argument("--scene-depth", action="store_true")
    parser.add_argument("--prefer-observed-overlap", action="store_true")
    parser.add_argument("--visible-outline", action="store_true")
    parser.add_argument(
        "--outline-weight",
        type=float,
        default=1.0,
        help="Complete-reference outline cost; recorded partial RGB-D surfaces do not use it",
    )
    parser.add_argument("--cases", nargs="+", default=["mustard", "yellow", "drill", "clock", "gamepad"])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    source = code_snapshot(args.output)
    # code_snapshot covers src and the shared runner; include the filter ablation.
    for name in (
        "flash_recovery_combinations.py",
        "flash_velocity_filter.py",
        "flash_kalman_recovery_control.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model, kernels = None, None
    results = []
    for name in args.cases:
        previous = Path(manifest[name]["runs"]["baseline25"])
        old = json.loads((previous.parent / "provenance.json").read_text())
        case = json.loads((previous / "case.json").read_text())
        vision, surface = configurations(old, args.fitting, args.filter)
        vision = replace(
            vision,
            shape=replace(
                vision.shape,
                scene_depth_constraints=args.scene_depth,
                prefer_observed_overlap=args.prefer_observed_overlap,
                require_full_mask_coverage=args.visible_outline,
                mask_coverage_weight=args.outline_weight,
            ),
        )
        if args.rotation_process_std is not None:
            surface = replace(
                surface,
                uncertainty=replace(
                    surface.uncertainty, rotation_process_std_deg_sqrt_s=args.rotation_process_std
                ),
            )
        shared = SharedMatchingConfig.from_dict(old["shared_matching"])
        group = GroupConfig.from_dict(old["group_experiment"])
        provenance = {
            **source,
            "config": asdict(vision),
            "surface_experiment": asdict(surface),
            "shared_matching": asdict(shared),
            "group_experiment": asdict(group),
            "filter": args.filter,
            "fitting": args.fitting,
            "velocity_config": asdict(VelocityConfig()) if args.filter == "velocity" else None,
            "previous": str(previous),
        }
        write(args.output / f"{case['id']}.provenance.json", provenance)
        if model is None and shared.backend == "ufm":
            revision = json.loads(Path("tests/policies/flash/fixtures/gamepad_stationary.json").read_text())[
                "ufm_revision"
            ]
            model = load_ufm(revision=revision)
        if kernels is None and vision.compiled_surface_fit:
            kernels = compiled_kernels()
        started = perf_counter()

        def progress(frame, total, case_name=name):
            if shutil.disk_usage(args.output).free < 700_000_000:
                raise RuntimeError("Disk reserve reached")
            if (args.output / "cancel").exists():
                raise RuntimeError("Cancelled")
            if frame % 25 == 0 or frame == total - 1:
                write(args.output / "progress.json", {"case": case_name, "frame": frame, "total": total})

        print("START", name, args.filter, args.fitting, flush=True)
        with (
            threadpool_limits(limits=1),
            SearchPool(vision.shape.batch_query_workers) as pool,
            patch.object(session, "PoseKalmanFilter", FILTERS[args.filter]),
        ):
            summary = run_case(
                case,
                args.output / case["id"],
                model,
                vision,
                kernels if vision.compiled_surface_fit else None,
                pool if vision.reuse_surface_search else None,
                retain_motion_reference=old.get("retain_motion_reference", False),
                surface_experiment=surface,
                group_experiment=group,
                shared_matching_config=shared,
                surface_methods=("group_growing",),
                progress=progress,
            )
        baseline = json.loads((previous / "summary.json").read_text())
        assert summary["input_sha256"] == baseline["input_sha256"], "Different input is not a paired trial"
        assert summary["frames"] == case["expected_frames"]
        results.append({"name": name, "wall_s": perf_counter() - started, "summary": summary})
        write(args.output / "results.json", results)
        print("DONE", name, round(perf_counter() - started, 1), flush=True)


if __name__ == "__main__":
    main()
