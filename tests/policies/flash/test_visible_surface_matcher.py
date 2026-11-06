"""The replacement must solve recorded placements and retain geometry guards."""

import numpy as np
import pytest

from benchmarks.flash_gamepad_placement import METHODS, replay
from benchmarks.flash_partial_surface_control import box_observation, perturbed_pose, pose_error
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_registration import LibraryConfig
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


@pytest.mark.parametrize("history", METHODS)
@pytest.mark.parametrize("frame", (540, 627))
@pytest.mark.parametrize("clean", (False, True))
def test_recorded_clear_gamepad_placement(history, frame, clean):
    result = replay(history, frame, clean_start=clean, matcher="visible_icp")
    assert result["solver"]["library"] == "visible_icp"
    assert result["geometry"]["accepted"], result
    assert result["result"]["mean_feature_error_mm"] <= 5, result


@pytest.mark.parametrize("maximum", (1, 50))
def test_saved_iteration_budget_reaches_solver_without_weakening_geometry(maximum):
    config = VisionPipelineConfig.from_dict(
        {"shape": {"registration": {"backend": "visible_icp"}, "max_evaluations": maximum}}
    )
    config = VisionPipelineConfig.from_dict(config.to_dict())
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(40)
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(reference, current, k, config.shape, reference_observation=source)
    pose, details = fit.refine(perturbed_pose(truth))
    assert details["evaluations"] <= maximum
    if pose is not None:
        assert fit.check_pose(pose)["accepted"]
    if maximum == 50:
        assert pose is not None
        assert pose_error(reference, pose, truth)["median_point_error_mm"] < 5
    wrong = truth.copy()
    wrong[0, 3] += 0.04
    assert not fit.check_pose(wrong)["accepted"]


def test_a_different_visible_face_remains_a_partial_observation():
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(60)
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(
        reference,
        current,
        k,
        ShapeConfig(registration=LibraryConfig("visible_icp")),
        reference_observation=source,
    )
    result, _ = fit.refine(perturbed_pose(truth))
    assert result is not None
    assert pose_error(reference, result, truth)["rotation_error_deg"] < 1
    assert fit.check_pose(result)["accepted"]
    assert not np.array_equal(result, perturbed_pose(truth))


@pytest.mark.parametrize("overlap", (False, True))
def test_convergence_means_a_stable_step_not_merely_an_early_exit(overlap):
    current, k, _ = box_observation(0)
    fit = WholeSurfaceFit(
        current["obs"]["points"],
        current,
        k,
        ShapeConfig(registration=LibraryConfig("visible_icp")),
        reference_observation=current,
    )
    initial = np.eye(4)
    if not overlap:
        initial[0, 3] = 2
    _, details, _ = fit._solve(initial, 1)
    assert details["converged"] is overlap
    assert details["evaluations"] == int(overlap)


def test_retired_settings_migrate_only_when_starting_a_new_session():
    import copy
    from dataclasses import asdict

    from lerobot.policies.flash.surface_registration import MATCHERS, LibraryConfig
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    saved = {
        "shape": {
            "registration": {"backend": "scipy"},
            "max_evaluations": 17,
            "max_correction_iterations": 23,
            "batch_jacobian": True,
        }
    }
    original = copy.deepcopy(saved)
    current = VisionPipelineConfig.from_dict(saved)
    assert saved == original  # Saved playback provenance must never change.
    assert current.shape.registration.backend == "visible_icp"
    assert current.shape.max_evaluations == 17
    assert "max_correction_iterations" not in asdict(current.shape)
    assert VisionPipelineConfig.from_dict(asdict(current)) == current
    assert "scipy" not in MATCHERS
    with pytest.raises(ValueError):
        LibraryConfig(backend="scipy")
