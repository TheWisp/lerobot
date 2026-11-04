"""Outline coverage is a configurable preference; depth contradictions stay hard."""

from dataclasses import replace

import numpy as np
import pytest

from lerobot.policies.flash.surface_acceleration import compiled_kernels
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig
from tests.policies.flash.test_scene_depth_constraints import scene


@pytest.mark.parametrize("weight", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
def test_outline_weight_preserves_depth_rejection_and_scalar_compiled_parity(weight):
    source, current, k = scene()
    config = ShapeConfig(
        scene_depth_constraints=True, require_full_mask_coverage=True, mask_coverage_weight=weight
    )
    wrong = np.eye(4)
    wrong[0, 3] = 8 * 0.4 / k[0, 0]
    scalar = WholeSurfaceFit(source["obs"]["points"], current, k, config)
    accelerated = WholeSurfaceFit(source["obs"]["points"], current, k, config, fit_kernels=compiled_kernels())
    for fit in (scalar, accelerated):
        assert not check_surface_contradiction(fit, wrong, PoseUncertaintyConfig())["accepted"]
        assert check_surface_contradiction(fit, np.eye(4), PoseUncertaintyConfig())["accepted"]
    np.testing.assert_allclose(scalar.residual(wrong), accelerated.residual(wrong), atol=1e-12)
    np.testing.assert_allclose(
        scalar.batch_residuals([wrong, np.eye(4)]),
        [scalar.residual(wrong), scalar.residual(np.eye(4))],
        atol=1e-12,
    )


def test_zero_weight_equals_disabled_outline_and_weight_scales_cost_not_other_evidence():
    source, current, k = scene()
    config = ShapeConfig(scene_depth_constraints=True, require_full_mask_coverage=True)
    # Move the reference left, leaving a piece of the measured outline uncovered.
    pose = np.eye(4)
    pose[0, 3] = -0.012
    fits = {
        weight: WholeSurfaceFit(
            source["obs"]["points"],
            current,
            k,
            replace(config, mask_coverage_weight=weight),
        )
        for weight in (0, 0.25, 1)
    }
    disabled = WholeSurfaceFit(
        source["obs"]["points"],
        current,
        k,
        replace(config, require_full_mask_coverage=False),
    )
    np.testing.assert_array_equal(fits[0].residual(pose), disabled.residual(pose))
    assert fits[1].score(pose) > fits[0].score(pose)
    assert fits[0.25].score(pose) - fits[0].score(pose) == pytest.approx(
        0.25 * (fits[1].score(pose) - fits[0].score(pose))
    )
    # Current outline may be a preference; an earlier partial view cannot demand
    # that every newly measured point existed in its own visible silhouette.
    partial = WholeSurfaceFit(source["obs"]["points"], current, k, config, reference_observation=source)
    assert partial.coverage_enabled
    assert not partial.reverse_fits[0][1].config.require_full_mask_coverage


def test_pipeline_configuration_reaches_partial_fitter_and_changes_coverage_preference():
    from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit

    source, current, k = scene()
    pose = np.eye(4)
    pose[0, 3] = -0.012
    for kernels in (None, compiled_kernels()):
        fits = []
        for weight in (0, 0.25, 1):
            config = VisionPipelineConfig.from_dict(
                {
                    "shape": {
                        "require_full_mask_coverage": True,
                        "mask_coverage_weight": weight,
                        "scene_depth_constraints": True,
                    }
                }
            )
            fits.append(
                ImageConditionedFit(
                    source["obs"]["points"],
                    current,
                    k,
                    config.shape,
                    reference_observation=source,
                    fit_kernels=kernels,
                    image_reference=np.empty((0, 3)),
                    image_current=np.empty((0, 3)),
                    image_scale_m=0.005,
                    image_weight=0,
                )
            )
        # Use the actual image-assisted fitter with a partial measured reference;
        # no test-only model flag can enable a different path from real runs.
        assert fits[1].score(pose) > fits[0].score(pose)
        assert fits[2].score(pose) > fits[1].score(pose)
        assert fits[2].score(np.eye(4)) < fits[2].score(pose)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_pipeline_rejects_invalid_outline_weight(weight):
    with pytest.raises(ValueError, match="mask_coverage_weight"):
        VisionPipelineConfig.from_dict({"shape": {"mask_coverage_weight": weight}})


def test_pipeline_round_trip_preserves_zero_and_nonzero_outline_weights():
    for weight in (0, 0.25, 1):
        config = VisionPipelineConfig.from_dict(
            {"shape": {"require_full_mask_coverage": True, "mask_coverage_weight": weight}}
        )
        assert VisionPipelineConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("weight", [0, 0.1, 0.25, 0.5, 1, 2])
def test_newly_visible_side_is_not_rejected_for_incomplete_outline(weight):
    from benchmarks.flash_partial_surface_control import box_observation
    from lerobot.policies.flash.surface_grounding import project
    from lerobot.policies.flash.surface_observation import move

    source, k, _ = box_observation(0)
    current, _, truth = box_observation(40)
    fit = WholeSurfaceFit(
        source["obs"]["points"],
        current,
        k,
        ShapeConfig(
            require_full_mask_coverage=True, mask_coverage_weight=weight, scene_depth_constraints=True
        ),
        reference_observation=source,
    )
    assert fit.coverage_enabled
    assert fit.uncovered_mask(project(move(fit.envelope, truth), k)).max() > 10
    assert check_surface_contradiction(fit, truth, PoseUncertaintyConfig())["accepted"]


@pytest.mark.parametrize("weight", [0, 0.25, 1, 2])
def test_recorded_gamepad_wrong_pose_still_fails_geometry_for_any_outline_weight(weight):
    from benchmarks.flash_recorded_recovery import load
    from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession

    data, metadata = load("gamepad_271")
    vision = VisionPipelineConfig.from_dict(metadata["vision"])
    shape = replace(
        vision.shape,
        scene_depth_constraints=True,
        require_full_mask_coverage=True,
        mask_coverage_weight=weight,
    )
    builder = SurfaceTrackingSession(None, shape_config=shape, observation_config=vision.observation)
    current = builder.prepare(
        data["rgb_271"], data["depth_271"], data["mask_271"], data["intrinsics_271"], 271
    )["current"]
    fit = WholeSurfaceFit(data["map_points"], current, data["intrinsics_271"], shape)
    assert not check_surface_contradiction(fit, data["recorded_pose"], PoseUncertaintyConfig())["accepted"]
