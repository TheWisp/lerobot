"""Coarse proposals cannot replace measured inputs or detailed validation."""

import numpy as np
import pytest

from benchmarks.flash_coarse_to_fine import PointToPlaneTrial, RefinementTrial, coarse_fitter, problems
from benchmarks.flash_partial_surface_control import box_observation, perturbed_pose, pose_error
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


@pytest.mark.parametrize("simplify_trees", [False, True])
def test_coarse_fit_keeps_measured_geometry_and_full_validation(simplify_trees):
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(20)
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(reference, current, k, ShapeConfig(), reference_observation=source)
    initial = perturbed_pose(truth)
    residual = fit.residual(initial)
    mask, depth = current["mask"].copy(), current["depth"].copy()
    queries, target_queries = fit.reference_queries, fit.target_queries
    trees = fit.reference_tree, fit.target_tree
    coarse = coarse_fitter(fit, 256, 4, simplify_trees=simplify_trees)
    assert len(coarse.reference_queries) <= 256 < len(queries)
    assert coarse.reference is fit.reference
    assert coarse.observation is fit.observation
    assert coarse.reverse_fits[0][1].observation is source
    original_refine = WholeSurfaceFit._solve
    with RefinementTrial(256, 4, simplify_trees=simplify_trees) as trial:
        pose, detail = fit.correct(initial)
    assert WholeSurfaceFit._solve is original_refine
    assert trial.calls
    assert detail["fit_reference_points"] == len(queries)
    assert detail["fit_target_points"] == len(target_queries)
    assert detail["cost"] == fit.score(pose)
    assert fit.reference_queries is queries and fit.target_queries is target_queries
    assert (fit.reference_tree, fit.target_tree) == trees
    np.testing.assert_array_equal(fit.residual(initial), residual)
    np.testing.assert_array_equal(current["mask"], mask)
    np.testing.assert_array_equal(current["depth"], depth)
    # Known rigid motion, independently evaluated by the hidden rendering model.
    error = pose_error(reference, pose, truth)
    assert error["median_point_error_mm"] < 5
    assert error["rotation_error_deg"] < 1
    assert check_surface_contradiction(fit, pose, PoseUncertaintyConfig())["accepted"]
    wrong = truth.copy()
    wrong[0, 3] += 0.04
    assert not check_surface_contradiction(fit, wrong, PoseUncertaintyConfig())["accepted"]


def test_experimental_adapter_is_removed_even_when_run_fails():
    original = WholeSurfaceFit._solve
    with pytest.raises(RuntimeError, match="interrupted trial"):
        with RefinementTrial(256, 4):
            raise RuntimeError("interrupted trial")
    assert WholeSurfaceFit._solve is original


def test_coarse_plane_proposal_cannot_replace_a_better_recorded_ring_seed():
    pytest.importorskip("open3d")
    _, reference, current, k, initial, _, source = next(
        problem for problem in problems() if problem[0] == "ring_visible_motion_20261016"
    )
    config = ShapeConfig()
    fit = WholeSurfaceFit(reference, current, k, config, reference_observation=source)
    with PointToPlaneTrial((0.012, 0.006), 12, 8, keep_better_seed=True) as trial:
        pose, detail = fit.correct(initial)
    assert trial.calls
    assert fit.config is config
    assert detail["cost"] == fit.score(pose)
    assert check_surface_contradiction(fit, pose, PoseUncertaintyConfig())["accepted"]
    # Final evaluation still has the complete measured surface, not the ICP
    # pyramid. This is a geometry check, not independent pose ground truth.
    assert detail["fit_reference_points"] == len(fit.reference_queries)
    assert len(fit.reference_queries) > trial.calls[0]["coarse_reference_queries"]
