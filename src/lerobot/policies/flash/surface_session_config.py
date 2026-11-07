"""One settings schema for the experiment runner and live inspector."""

from dataclasses import asdict

from .experimental_surface_session import GrowingSurfaceConfig
from .shared_image_tracking import SharedMatchingConfig
from .surface_group_hints import GroupConfig
from .vision_pipeline import VisionPipelineConfig


def defaults():
    # The validated research recipe, shared by fresh live and recorded sessions.
    # Stored playback retains its original metadata. Reruns normalize retired
    # solver settings to the maintained surface fitter via VisionPipelineConfig.
    config = asdict(VisionPipelineConfig())
    config["observation"].update(component_gap_m=None, depth_outlier_sigma=3, filtered_depth_for_fitting=True)
    config["shape"].update(
        require_full_mask_coverage=True,
        mask_coverage_weight=0.5,
        scene_depth_constraints=True,
        prefer_observed_overlap=True,
    )
    surface = asdict(GrowingSurfaceConfig())
    surface.update(partial_surface_matching=True, fit_view_limit=1)
    surface["uncertainty"].update(position_process_std_m_sqrt_s=0.03, reset_on_geometry_conflict=False)
    return {
        "config": config,
        "surface_experiment": surface,
        "group_experiment": asdict(GroupConfig()),
        "shared_matching": asdict(SharedMatchingConfig(backend="lk")),
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
