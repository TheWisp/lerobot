"""Complete serializable settings for the restart experiment (not robot control)."""

import ast
import inspect
from dataclasses import asdict, fields, is_dataclass

from benchmarks.flash_experiment_groups import GroupConfig
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

OPTION_HELP = {
    "backend": "Backend selection (surface/appearance for vision; lk/ufm for shared matching).",
    "sam_resolution": "SAM input resolution in pixels; unused when replaying saved masks.",
    "min_similarity": "Minimum appearance-match similarity; appearance backend only.",
    "registration_threshold_m": "Appearance registration inlier distance, metres.",
    "registration_min_inliers": "Required appearance registration correspondences.",
    "registration_trials": "Appearance registration sampling trials.",
    "visible_shape_fallback": "Allow visible-shape fallback in the appearance backend.",
    "parallel_surface_objects": "Live vision: allow concurrent object fitting.",
    "pipeline_surface_frames": "Live vision: overlap image matching and surface fitting.",
    "surface_object_workers": "Live vision: maximum concurrent fitting workers.",
    "surface_reference_max_points_per_view": "Live vision: stored points per reference view.",
    "compiled_surface_fit": "Use compiled CPU fitting kernels.",
    "reuse_surface_search": "Reuse a bounded nearest-neighbour search worker pool.",
    "offline_interval_frames": "Text-detection correction interval during offline processing.",
    "runtime_interval_frames": "Text-detection correction interval during live processing.",
    "min_detector_score": "Minimum score for a fresh text detection to repair the mask.",
    "min_tracker_score": "Tracker-score threshold used by detector correction.",
    "min_mask_overlap": "Required overlap when associating detection with tracked mask.",
    "mask_margin_px": "Exclude this many pixels inside mask boundaries from measured geometry.",
    "depth_span_m": "Reject neighbourhoods whose depth range exceeds this value, metres.",
    "depth_neighborhood_px": "Depth-validity neighbourhood width, pixels.",
    "normal_bilateral_diameter_px": "Bilateral-filter diameter for surface normals, pixels.",
    "normal_bilateral_sigma_px": "Spatial smoothing scale for normals, pixels.",
    "huber_m": "Depth smoothing scale for normal estimation (bilateral depth sigma), metres.",
    "surface_spacing_m": "Voxel spacing for sampling measured/fitted surface, metres.",
    "samples_across_object": "Adaptive sampling density across the object's extent.",
    "min_surface_spacing_m": "Finest allowed adaptive surface spacing, metres.",
    "component_gap_m": "3D connectivity distance used during observation cleanup, metres.",
    "validation_tile_px": "Pixel tile size for depth/mask validation sampling.",
    "depth_tolerance_m": "Allowed geometric depth disagreement, metres; also the covariance mode's absolute contradiction distance.",
    "mask_boundary_tolerance_px": "Allowed projected silhouette-boundary discrepancy, pixels.",
    "require_full_mask_coverage": "Require stored surface to cover the current mask; only appropriate for a complete reference.",
    "max_surface_distance_m": "Maximum nearest-surface distance considered by fitting, metres.",
    "max_evaluations": "Maximum least-squares objective evaluations per local fit.",
    "solver_tolerance": "Numerical least-squares stopping tolerance; legacy surface-priority cost tie tolerance.",
    "max_fit_points": "Maximum sampled points in each direction of the fitting objective.",
    "nn_queries_per_worker": "Minimum query work per nearest-neighbour worker.",
    "batch_jacobian": "Batch finite-difference pose candidates for shared spatial queries.",
    "batch_query_workers": "Maximum workers for batched nearest-neighbour queries.",
    "crop_min_px": "Minimum object-centred UFM crop size; legacy matching path.",
    "crop_max_px": "Maximum object-centred UFM crop size; legacy matching path.",
    "crop_extent_multiplier": "Crop extent relative to object bounds; legacy matching path.",
    "input_size_px": "UFM image size for legacy object-crop matching.",
    "grid_step_px": "Spacing of legacy UFM correspondence queries, pixels.",
    "covisibility_min": "Minimum UFM co-visibility score; not calibrated tracking accuracy.",
    "min_ransac_pairs": "Minimum measured 3D correspondence pairs for a rigid motion.",
    "ransac_distance_m": "Rigid-motion inlier distance, metres.",
    "ransac_confidence": "RANSAC sampling success target, not pose confidence.",
    "refinement_steps": "Inlier/refit iterations for measured rigid motion.",
    "connected_dilation_px": "Neighbourhood for connected correspondence support, pixels.",
    "spacing_m": "Voxel spacing for the stored object surface, metres.",
    "agreement_m": "Tracking overlap and image residual distance scale, metres; does not authorize shape growth.",
    "min_overlap": "Tracking: required fraction of observed points within agreement_m of the fitted stored surface.",
    "growth_agreement_m": "Growth: tighter overlap/confirmation distance, metres; also caps the retained-view depth tolerance.",
    "growth_min_overlap": "Growth: required fraction of current observations within growth_agreement_m of the active surface before adding OR confirming patches.",
    "patch_consistency": "Growth: required fraction of a proposed addition not contradicted by earlier retained depth images and masks.",
    "patch_overlap": "Growth: required fraction of a pending patch measured again within growth_agreement_m.",
    "image_weight": "Secondary image residual weight; shared object point evidence replaces duplicate image pairs.",
    "surface_priority": "Prefer geometry before image/group evidence; false selects the archived weighted objective.",
    "confirmations": "Number of distinct observation frames needed to admit a pending patch.",
    "max_patches": "Maximum retained active measurement patches.",
    "max_pending": "Maximum pending, unconfirmed measurement patches.",
    "points_per_patch": "Maximum raw measured points stored per patch.",
    "max_points": "Maximum points in the combined surface.",
    "grow": "Enable measured surface additions. Benchmark fixed/growing methods override this per variant.",
    "retain_unobserved_pose": "Keep last world pose when all observations/predictions disappear. Group benchmark variants set this true.",
    "context_radius_m": "Search radius for surrounding point support, metres.",
    "min_inliers": "Minimum point correspondences for a group prediction.",
    "fit_threshold_m": "Group rigid-motion inlier and separation tolerance, metres.",
    "object_max_points": "Maximum object points sampled into group evidence; legacy seed budget.",
    "object_max_frame_gap_s": "Maximum gap for legacy dedicated object LK tracking, seconds; unused with shared matching.",
    "object_fb_threshold_px": "Legacy dedicated object LK forward/backward threshold, pixels; unused with shared matching.",
    "object_image_scale": "Legacy dedicated object LK image scale; unused with shared matching.",
    "depth_support_radius_px": "Neighbourhood radius for valid depth at tracked pixels.",
    "support_confirmation_s": "Continuous measured co-movement required before joining a group, seconds.",
    "point_evidence_weight": "Weight per independent point source, normalized by point count.",
    "max_image_size_px": "Maximum shared matching image dimension, pixels.",
    "scene_points": "Budget for shared scene point identities.",
    "object_points": "Point identity budget per segmented object.",
    "spacing_px": "Minimum query point spacing, pixels.",
    "history_frames": "Number of frames retaining point identities for reconnection.",
    "window_size_px": "LK image patch width, pixels; must be odd.",
    "pyramid_levels": "LK optical-flow pyramid levels.",
    "forward_backward_limit_px": "LK forward/backward consistency threshold, pixels.",
    "config.detection_correction.enabled": "Enable periodic text-detection mask correction; unused with saved masks.",
    "surface_experiment.uncertainty.enabled": "Enable temporal uncertainty; new surface experiments default to Kalman filtering.",
    "surface_experiment.uncertainty.mode": "selection: compare candidates; kalman: predict/correct pose and covariance over time.",
    "initial_position_std_m": "Initial position standard deviation, metres; also used after a directly contradicted prediction is replaced.",
    "initial_rotation_std_deg": "Initial rotation standard deviation, degrees; an assumption, not measured accuracy.",
    "position_process_std_m_sqrt_s": "Unmodelled position-motion standard deviation per square-root second. Kalman process variance grows with elapsed time.",
    "rotation_process_std_deg_sqrt_s": "Unmodelled rotation-motion standard deviation in degrees per square-root second.",
}


def defaults():
    return {
        "config": asdict(VisionPipelineConfig()),
        "surface_experiment": asdict(GrowingSurfaceConfig()),
        "group_experiment": asdict(GroupConfig()),
        "shared_matching": asdict(SharedMatchingConfig()),
        "retain_motion_reference": True,
    }


def validate(values):
    """Return what the real benchmark consumers accept; reject unknown keys."""
    unknown = values.keys() - defaults().keys()
    if unknown:
        raise ValueError(f"Unknown pipeline sections: {sorted(unknown)}")
    if set(values) != set(defaults()):
        raise ValueError("Supply every pipeline section; use null to disable an optional section")
    result = {"config": asdict(VisionPipelineConfig.from_dict(values["config"]))}
    for key, parse in (
        ("surface_experiment", lambda v: GrowingSurfaceConfig(**v)),
        ("group_experiment", GroupConfig.from_dict),
        ("shared_matching", SharedMatchingConfig.from_dict),
    ):
        result[key] = asdict(parse(values[key])) if values[key] is not None else None
    if result["group_experiment"] is not None and result["surface_experiment"] is None:
        raise ValueError("Groups require surface_experiment")
    if type(values["retain_motion_reference"]) is not bool:
        raise ValueError("retain_motion_reference must be boolean")
    result["retain_motion_reference"] = values["retain_motion_reference"]
    return result


def from_recording(provenance, methods):
    # Match the existing rerun behavior, including adding the group ablation
    # to older surface experiments. Runtime/detection options remain visible
    # but saved-mask benchmarks do not invoke SAM or the robot loop.
    surface = provenance.get("surface_experiment")
    if surface is not None and "uncertainty" not in surface:
        # Old recordings predate uncertainty. Do not silently change their recipe.
        surface = {**surface, "uncertainty": asdict(PoseUncertaintyConfig())}
    return validate(
        {
            "config": provenance["config"],
            "surface_experiment": surface,
            "group_experiment": (provenance.get("group_experiment") or {}) if surface else None,
            "shared_matching": provenance.get("shared_matching"),
            "retain_motion_reference": "ufm_retained_reference" in methods,
        }
    )


def catalog():
    """Every dataclass field, including nested ones; no parallel default table."""
    result = []

    def visit(obj, prefix):
        source = inspect.getsource(type(obj))
        lines = source.splitlines()
        declarations = {
            n.target.id: n.lineno
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        }
        for field in fields(obj):
            value = getattr(obj, field.name)
            path = f"{prefix}.{field.name}"
            if is_dataclass(value):
                visit(value, path)
                continue
            line = declarations.get(field.name, 1) - 2
            comments = []
            while line >= 0 and lines[line].strip().startswith("#"):
                comments.insert(0, lines[line].strip()[1:].strip())
                line -= 1
            result.append(
                {
                    "path": path,
                    "default": value,
                    "description": OPTION_HELP.get(path, OPTION_HELP.get(field.name, " ".join(comments))),
                }
            )

    for prefix, obj in (
        ("config", VisionPipelineConfig()),
        ("surface_experiment", GrowingSurfaceConfig()),
        ("group_experiment", GroupConfig()),
        ("shared_matching", SharedMatchingConfig()),
    ):
        visit(obj, prefix)
    result.append(
        {
            "path": "retain_motion_reference",
            "default": True,
            "description": "Include the last-matched-frame comparison method.",
        }
    )
    return result


def reference_markdown():
    """Generate the option table from the same catalogue shown in the GUI."""
    import json

    lines = [
        "# FLASH experiment pipeline settings",
        "",
        "Open **Run → FLASH → Pipeline settings** beside the selected video. New surface experiments use **Kalman filter**. Earlier recordings retain their saved settings; choose **Kalman filter** and set `max_visible_outlier_fraction` to `0.25` to compare them with the new default. The JSON contains the recording's settings with current defaults filled in. **Rerun this scene with these settings** validates and runs it. Playback itself never recomputes poses.",
        "",
        "This is the restart experiment pipeline. It does not configure robot control. Saved-mask benchmarks reuse segmentation, so SAM/detection options are listed but not executed. Shared matching replaces the legacy object-crop matcher. Live-only scheduling options have no effect in the sequential benchmark.",
        "",
        "Each optional section may be `null`. Groups require the surface experiment. Fixed/growing and group/non-group comparison methods select `grow` and `retain_unobserved_pose` themselves. Every effective setting is saved with results and included in cache identity. A changed configuration reruns the scene; it does not reinterpret an old result.",
        "",
        "Tracking and growth have separate acceptance thresholds. Tracking defaults to 25% overlap within 5 mm; growth requires 75% within 3 mm before adding or confirming patches. An addition also needs 95% consistency with earlier retained depth/masks and 80% of its points measured again within 3 mm. These are experimental settings, not calibrated confidence: a geometrically ambiguous fit can still pass. Rejecting growth leaves pose tracking active. Initializing the first partial surface has no previous surface to compare against.",
        "",
        "New surface experiments default to **Kalman filtering with a 25% geometry rejection allowance**. This is an empirical research default, not calibrated confidence. The allowance applies to both unmatched observed points and contradicted stored points; it also affects when the filter restarts. It is separate from the stricter growth thresholds above. Candidate selection and filter-off remain available. Noise values are assumptions, not measured confidence. `depth_tolerance_m` and `max_visible_outlier_fraction` test direct contradictions independently of covariance. Missing observations still allow group prediction. The mode does not estimate the probability that an object remains attached to its group.",
        "Kalman mode propagates a 6D covariance with camera changes, consecutive measured group motion and time-scaled process noise. Current visible-surface information corrects the prediction in constrained directions. Innovation checks use both prediction and observation uncertainty. A directly contradicted prediction is reinitialized from a valid measured fit. Rejected updates never grow the surface. The uncertainty calculation currently includes depth-normal constraints but omits outline and image constraints used by the fitter, so sideways motion on a flat visible patch can lag. Group and surface errors are treated as conditionally independent; shared sensor biases, temporal correlations, shape errors and mistaken group membership remain limitations. `prefer_supported_prediction` applies only to candidate selection.",
        "",
        "For CLI use, `flash_tracker_experiment` accepts `--config` (a JSON object containing `vision_pipeline`), `--surface-experiment-config`, `--group-experiment-config`, and `--shared-matching-config`. The GUI uses the complete sectioned JSON below. `retain_motion_reference` corresponds to `--retain-motion-reference`.",
        "",
        "Regenerate this reference with `python -m benchmarks.flash_pipeline_settings --write-reference <path>`. Defaults come from the actual configuration classes. Units appear in names: `_m` metres, `_px` pixels, `_s` seconds.",
        "",
        "| Option | Default | Meaning / scope |",
        "| --- | --- | --- |",
    ]
    for row in catalog():
        lines.append(f"| `{row['path']}` | `{json.dumps(row['default'])}` | {row['description']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-reference", type=Path)
    args = parser.parse_args()
    if args.write_reference:
        args.write_reference.write_text(reference_markdown())
    else:
        print(json.dumps(defaults(), indent=2))
