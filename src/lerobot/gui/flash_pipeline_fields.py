"""Editable controls consumed by the shared frame pipeline, not legacy config fields."""

from lerobot.policies.flash.surface_session_config import defaults


def pipeline_fields():
    # Like training's field catalog: backend keys, typed controls, plain labels
    # and help. Values/defaults still come from the actual pipeline schema.
    sections = [
        (
            "Observation",
            [
                (
                    "config.sam_resolution",
                    "SAM image size",
                    "int",
                    "Larger images cost more segmentation time.",
                    {"min": 224, "step": 14, "live_only": True},
                ),
                (
                    "config.observation.mask_margin_px",
                    "Ignore boundary depth (pixels)",
                    "int",
                    "The outline still counts; only nearby depth readings are withheld.",
                    {"min": 0, "step": 1},
                ),
                (
                    "config.observation.depth_span_m",
                    "Allowed local depth jump (mm)",
                    "float",
                    "Withhold depth near larger jumps.",
                    {"min": 0.1, "step": 1, "scale": 1000},
                ),
                (
                    "config.detection_correction.enabled",
                    "Correct masks using fresh detections",
                    "bool",
                    "Only applies when the object was selected by name.",
                    {"live_only": True},
                ),
                (
                    "config.detection_correction.runtime_interval_frames",
                    "Detection interval (frames)",
                    "int",
                    "Used when detection correction is enabled.",
                    {"min": 1, "step": 1, "live_only": True},
                ),
            ],
        ),
        (
            "Surface fitting",
            [
                (
                    "config.shape.max_evaluations",
                    "Total fitting iteration limit",
                    "int",
                    "One solve per frame. Applies to every surface matcher.",
                    {"min": 1, "step": 1},
                ),
                (
                    "config.shape.registration.iterations_per_level",
                    "Iterations per resolution",
                    "int",
                    "Library matchers only; also bounded by the total limit.",
                    {"min": 1, "step": 1, "exclude_matcher": "visible_icp"},
                ),
                (
                    "config.shape.depth_tolerance_m",
                    "Depth agreement tolerance (mm)",
                    "float",
                    "Distance used to judge matching depth. Coherent protrusions are rejected.",
                    {"min": 0.1, "step": 0.5, "scale": 1000},
                ),
                (
                    "config.shape.mask_coverage_weight",
                    "Outline coverage preference",
                    "float",
                    "Prefer filling the observed outline among geometrically valid fits.",
                    {"min": 0, "step": 0.1},
                ),
                (
                    "config.shape.registration.appearance_weight",
                    "Texture preference",
                    "float",
                    "Image appearance influences ranking, never overrides geometry rejection.",
                    {"min": 0, "step": 0.1},
                ),
            ],
        ),
        (
            "Image points and groups",
            [
                (
                    "shared_matching.scene_points",
                    "Surrounding point budget",
                    "int",
                    "Image points available to support nearby objects.",
                    {"min": 1, "step": 1},
                ),
                (
                    "shared_matching.object_points",
                    "Point budget per object",
                    "int",
                    "Image points sampled inside each object.",
                    {"min": 1, "step": 1},
                ),
                (
                    "shared_matching.lk.forward_backward_limit_px",
                    "LK return-match tolerance (pixels)",
                    "float",
                    "Reject points that do not return near their start when matched backward.",
                    {"min": 0.1, "step": 0.1, "image_backend": "lk"},
                ),
                (
                    "shared_matching.ufm.covisibility_min",
                    "UFM visibility threshold",
                    "float",
                    "Minimum model visibility for image matches.",
                    {"min": 0, "max": 1, "step": 0.05, "image_backend": "ufm"},
                ),
                (
                    "group_experiment.support_confirmation_s",
                    "New group confirmation (seconds)",
                    "float",
                    "How long a newly formed group must retain measured support before it can predict hidden points.",
                    {"min": 0, "step": 0.1},
                ),
                (
                    "group_experiment.fit_threshold_m",
                    "Rigid motion fit tolerance (mm)",
                    "float",
                    "Point error allowed when estimating one group's motion.",
                    {"min": 0.1, "step": 0.5, "scale": 1000},
                ),
                (
                    "group_experiment.split_tolerance_m",
                    "Split difference (mm)",
                    "float",
                    "Measured point error that starts a possible split; tune above the sensor's usual depth noise.",
                    {"min": 0.1, "step": 0.5, "scale": 1000},
                ),
                (
                    "group_experiment.split_confirmation_s",
                    "Split duration (seconds)",
                    "float",
                    "How long that difference must persist before points leave a group.",
                    {"min": 0, "step": 0.1},
                ),
                (
                    "group_experiment.merge_tolerance_m",
                    "Join difference (mm)",
                    "float",
                    "Maximum measured co-motion error for a point to join a group or groups to merge.",
                    {"min": 0.1, "step": 0.5, "scale": 1000},
                ),
                (
                    "group_experiment.merge_confirmation_s",
                    "Join duration (seconds)",
                    "float",
                    "How long co-motion must persist before a point joins or two groups merge.",
                    {"min": 0, "step": 0.1},
                ),
            ],
        ),
        (
            "Growing surface",
            [
                (
                    "surface_experiment.spacing_m",
                    "Stored surface spacing (mm)",
                    "float",
                    "Smaller spacing preserves more detail and uses more points.",
                    {"min": 0.1, "step": 0.5, "scale": 1000},
                ),
                (
                    "surface_experiment.growth_min_overlap",
                    "Overlap required to add surface (%)",
                    "float",
                    "Growth is stricter than accepting a pose.",
                    {"min": 0, "max": 100, "step": 1, "scale": 100},
                ),
                (
                    "surface_experiment.growth_agreement_m",
                    "Growth agreement tolerance (mm)",
                    "float",
                    "New measurements must agree with existing surface within this distance.",
                    {"min": 0.1, "step": 0.5, "scale": 1000},
                ),
                (
                    "surface_experiment.max_points",
                    "Stored point limit",
                    "int",
                    "Bounds the accumulated surface size.",
                    {"min": 3, "step": 1},
                ),
            ],
        ),
        (
            "Pose uncertainty",
            [
                (
                    "surface_experiment.uncertainty.enabled",
                    "Use pose uncertainty",
                    "bool",
                    "Filter pose updates using estimated observation and motion uncertainty.",
                    {},
                ),
                (
                    "surface_experiment.uncertainty.position_process_std_m_sqrt_s",
                    "Position uncertainty growth (mm / √s)",
                    "float",
                    "Larger values allow faster position corrections.",
                    {"min": 0.01, "step": 1, "scale": 1000},
                ),
                (
                    "surface_experiment.uncertainty.rotation_process_std_deg_sqrt_s",
                    "Rotation uncertainty growth (° / √s)",
                    "float",
                    "Larger values allow faster orientation corrections.",
                    {"min": 0.01, "step": 0.5},
                ),
                (
                    "surface_experiment.uncertainty.independent_error_m",
                    "Independent depth noise (mm)",
                    "float",
                    "Assumed measurement noise, not a measured accuracy score.",
                    {"min": 0.01, "step": 0.1, "scale": 1000},
                ),
                (
                    "surface_experiment.uncertainty.correlated_error_m",
                    "Shared depth noise (mm)",
                    "float",
                    "Accounts for nearby measurements having similar errors.",
                    {"min": 0, "step": 0.1, "scale": 1000},
                ),
            ],
        ),
    ]
    values = defaults()
    result = []
    for section, entries in sections:
        for key, label, kind, description, options in entries:
            value = values
            for part in key.split("."):
                value = value[part]
            result.append(
                {
                    "key": key,
                    "label": label,
                    "type": kind,
                    "description": description,
                    "section": section,
                    "default": value,
                    **options,
                }
            )
    return result
