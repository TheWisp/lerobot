"""Complete serializable settings for the restart experiment (not robot control)."""

import ast
import inspect
from dataclasses import asdict, fields, is_dataclass

from benchmarks.flash_experiment_groups import GroupConfig
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_session_config import defaults as defaults, validate as validate
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

OPTION_HELP = {
    "config.shape.registration.backend": "3D surface solver: visible_icp, small_gicp, open3d_plane or open3d_color. Does not change UFM/LK image matching or depth rejection.",
    "voxel_sizes_m": "Coarse-to-fine point sampling for library solvers, metres.",
    "iterations_per_level": "Maximum library iterations per resolution, including point-to-point initialization.",
    "correspondence_voxels": "Maximum correspondence distance as a multiple of the current voxel size.",
    "threads": "CPU threads used by small_gicp.",
    "translation_epsilon_m": "small_gicp translation convergence tolerance, metres.",
    "rotation_epsilon_rad": "small_gicp rotation convergence tolerance, radians.",
    "colored_geometry_weight": "Geometry contribution in Open3D colored ICP; library default 0.968.",
    "appearance_weight": "Measured RGB correlation used to rank library proposals; depth contradictions still reject poses.",
    "partial_surface_matching": "Experimental fitting using depth/mask visibility in current and retained measured views.",
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
    "depth_outlier_sigma": "Experimental depth-range cleanup using median absolute deviation; null disables it. May remove a genuine minority surface at another depth.",
    "filtered_depth_for_fitting": "Legacy saved-run option; rejected depth is always unknown in fitting and validation. The full mask and reliable scene depth remain available.",
    "validation_tile_px": "Pixel tile size for depth/mask validation sampling.",
    "depth_tolerance_m": "Allowed geometric depth disagreement, metres; also the covariance mode's absolute contradiction distance.",
    "mask_boundary_tolerance_px": "Allowed projected silhouette-boundary discrepancy, pixels.",
    "require_full_mask_coverage": "Enable a soft preference to cover more of the current visible mask. Applies to partial surfaces; uncovered pixels alone never reject a pose.",
    "mask_coverage_weight": "Strength of the optional outline-coverage preference; 0 disables it. Strong values can bias fits of incomplete surfaces.",
    "scene_depth_constraints": "Include scene depth in the fitting cost. Rejecting connected patches in measured empty space is mandatory regardless of this setting. Missing depth and nearer occluders remain unknown.",
    "prefer_observed_overlap": "Opt-in search preference for measured surface agreement over hiding the stored surface. Does not make missing measurements a contradiction.",
    "reset_on_geometry_conflict": "Allow a geometry-conflicting prediction to reset Kalman at the proposed pose. Disable to retain the normal innovation test on every update.",
    "max_surface_distance_m": "Maximum nearest-surface distance considered by fitting, metres.",
    "max_evaluations": "Maximum rigid-update iterations in the single local fit.",
    "solver_tolerance": "Rigid-step stopping tolerance; also used for surface-priority cost ties.",
    "max_fit_points": "Maximum sampled points in each direction of the fitting objective.",
    "nn_queries_per_worker": "Minimum query work per nearest-neighbour worker.",
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
    "spacing_m": "Fixed object-coordinate grid spacing for both initial and newly added surface, metres.",
    "agreement_m": "Tracking overlap and image residual distance scale, metres; does not authorize shape growth.",
    "min_overlap": "Tracking: required fraction of observed points within agreement_m of the fitted stored surface.",
    "growth_agreement_m": "Growth: tighter overlap/confirmation distance, metres; also caps the retained-view depth tolerance.",
    "growth_min_overlap": "Growth: required fraction of current observations within growth_agreement_m of the active surface before adding OR confirming patches.",
    "patch_consistency": "Growth: required fraction of a proposed addition not contradicted by earlier retained depth images and masks.",
    "patch_overlap": "Growth: required fraction of a pending patch measured again within growth_agreement_m.",
    "image_weight": "Secondary image residual weight; shared object point evidence replaces duplicate image pairs.",
    "surface_priority": "Prefer geometry before image/group evidence; false selects the archived weighted objective.",
    "fit_view_limit": "Experimental: number of latest measured views used for visibility checks; null keeps all retained views. Limiting this can forget earlier empty-space evidence.",
    "confirmations": "Number of distinct observation frames needed to admit a pending patch.",
    "max_patches": "Maximum retained active measurement patches.",
    "max_pending": "Maximum pending, unconfirmed measurement patches.",
    "points_per_patch": "Maximum measured points per patch after grid sampling; additional thinning applies only above this memory limit.",
    "max_points": "Maximum points in the combined surface.",
    "grow": "Enable measured surface additions. Benchmark fixed/growing methods override this per variant.",
    "retain_unobserved_pose": "Keep last world pose when all observations/predictions disappear. Group benchmark variants set this true.",
    "group_experiment.max_points": "Maximum measured and hidden point identities retained across the whole scene.",
    "max_groups": "Maximum scene motion groups; actively measured groups take precedence over dormant groups.",
    "motion_hypotheses": "Maximum batched rigid-motion hypotheses per group solve.",
    "max_hidden_s": "Maximum time to retain a point identity without another measurement, seconds.",
    "min_inliers": "Minimum point correspondences for a group prediction.",
    "fit_threshold_m": "Inlier tolerance when fitting a group's measured rigid motion, metres.",
    "split_tolerance_m": "Measured point disagreement that starts a possible departure, metres.",
    "split_confirmation_s": "Time that disagreement must persist before a point leaves its group, seconds.",
    "merge_tolerance_m": "Measured co-motion tolerance for point joining and group merging, metres.",
    "merge_confirmation_s": "Time that co-motion must persist before point joining or group merging, seconds.",
    "object_max_points": "Maximum object points sampled into group evidence; legacy seed budget.",
    "object_max_frame_gap_s": "Maximum gap for legacy dedicated object LK tracking, seconds; unused with shared matching.",
    "object_fb_threshold_px": "Legacy dedicated object LK forward/backward threshold, pixels; unused with shared matching.",
    "object_image_scale": "Legacy dedicated object LK image scale; unused with shared matching.",
    "depth_support_radius_px": "Neighbourhood radius for valid depth at tracked pixels.",
    "support_confirmation_s": "Measured support required before a new group can predict hidden points, seconds.",
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

    values = defaults()
    for prefix, obj in (
        ("config", VisionPipelineConfig.from_dict(values["config"])),
        ("surface_experiment", GrowingSurfaceConfig(**values["surface_experiment"])),
        ("group_experiment", GroupConfig.from_dict(values["group_experiment"])),
        ("shared_matching", SharedMatchingConfig.from_dict(values["shared_matching"])),
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
        "Open **Run → FLASH → Pipeline settings** beside the selected video. Fresh live sessions use **LK + surface fit + groups + growing surface + Kalman**. Recorded experiments retain their saved settings; use the same complete configuration for a fair live/replay comparison. The JSON contains settings for the next rerun. Retired SciPy settings migrate to the new surface fitter; saved playback keeps its original matcher label. **Rerun this scene with these settings** validates and runs it. Playback itself never recomputes poses.",
        "",
        "This is the restart experiment pipeline. It does not configure robot control. Saved-mask benchmarks reuse segmentation, so SAM/detection options are listed but not executed. Shared matching replaces the legacy object-crop matcher. Live-only scheduling options have no effect in the sequential benchmark.",
        "",
        "Each optional section may be `null`. Groups require the surface experiment. Fixed/growing and group/non-group comparison methods select `grow` and `retain_unobserved_pose` themselves. Every effective setting is saved with results and included in cache identity. A changed configuration reruns the scene; it does not reinterpret an old result.",
        "",
        "Tracking and growth have separate acceptance thresholds. Tracking defaults to 25% overlap within 5 mm; growth requires 75% within 3 mm before adding or confirming patches. An addition also needs 95% consistency with earlier retained depth/masks and 80% of its points measured again within 3 mm. These are experimental settings, not calibrated confidence: a geometrically ambiguous fit can still pass. Rejecting growth leaves pose tracking active. Initializing the first partial surface has no previous surface to compare against.",
        "",
        "New surface experiments default to **Kalman filtering with a 25% geometry rejection allowance**. This is an empirical research default, not calibrated confidence. The allowance applies to aggregate point disagreement; it never permits a connected surface patch in measured empty space. It is separate from the stricter growth thresholds above. Candidate selection and filter-off remain available. Noise values are assumptions, not measured confidence. `depth_tolerance_m` and `max_visible_outlier_fraction` test direct contradictions in every mode, including with uncertainty disabled. The field remains in the uncertainty section for saved-configuration compatibility. Missing observations still allow group prediction. The mode does not estimate the probability that an object remains attached to its group.",
        "Kalman mode propagates a 6D covariance with camera changes, consecutive measured group motion and time-scaled process noise. Current visible-surface information corrects the prediction in constrained directions. Innovation checks use both prediction and observation uncertainty. The legacy `reset_on_geometry_conflict=true` policy can restart the filter at a geometry-compatible proposal; setting it to `false` keeps the innovation check even when the previous prediction conflicts with depth/mask. A later geometry rejection restores the filter state. Rejected updates never grow the surface. Depth-normal constraints currently determine observation uncertainty; outline and image constraints used by fitting are omitted, so sideways motion on a flat patch can lag. Shared sensor biases and mistaken group membership remain limitations.",
        "",
        "The shared fresh-session recipe uses one bounded surface fit from the current motion estimate, with measured depth and mask checking its output. It enables depth-range cleanup and one retained fit view. Kalman reset on geometry conflict is disabled. Outline coverage is a soft preference with weight 0.5, not a rejection threshold. Older saved playback keeps its original results; rerunning uses the current single-fit pipeline. Newly exposed sides may be absent from the stored surface, so a strong coverage preference can bias a fit. Dated comparisons and their limitations are in the [research log](../../../../../docs/proofs/flash-tracker-restart/RESEARCH.md).",
        "",
        "For CLI use, `flash_tracker_experiment` accepts `--config` (a JSON object containing `vision_pipeline`), `--surface-experiment-config`, `--group-experiment-config`, and `--shared-matching-config`. The GUI uses the complete sectioned JSON below. `retain_motion_reference` corresponds to `--retain-motion-reference`.",
        "",
        "Regenerate this reference with `python -m benchmarks.flash_pipeline_settings --write-reference <path>`. Defaults come from `surface_session_config.defaults()`, shared with the live debugger; they include the current research recipe over the compatibility dataclass defaults. Units appear in names: `_m` metres, `_px` pixels, `_s` seconds.",
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
