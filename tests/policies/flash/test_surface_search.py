"""Search owns validity, including every solver and retry path."""

from collections import Counter

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig
from tests.policies.flash.test_surface_publication import frame


def fitter(cls=WholeSurfaceFit, **kwargs):
    prepared = frame(0)
    observation = prepared["current"]
    return cls(observation["obs"]["points"], observation, prepared["intrinsics"], ShapeConfig(), **kwargs)


def protruding():
    pose = np.eye(4)
    pose[0, 3] = 0.015
    return pose


def assert_static(fit, pose):
    assert pose is not None
    np.testing.assert_allclose(move(fit.reference, pose), fit.reference, atol=1e-8)


def test_lower_cost_invalid_start_cannot_displace_valid_start(monkeypatch):
    fit = fitter()
    wrong = protruding()
    assert not fit.check_pose(wrong)["accepted"]
    # Numerical cost is deliberately misleading; measured depth is real.
    monkeypatch.setattr(fit, "score", lambda pose: 0.0 if np.array_equal(pose, wrong) else 1.0)
    monkeypatch.setattr("lerobot.policies.flash.surface_grounding.pca_candidates", lambda *a, **k: [wrong])
    monkeypatch.setattr(
        fit,
        "_solve",
        lambda initial, limit: (
            initial,
            {"evaluations": 1, "converged": True},
            [(fit.score(initial), initial)],
        ),
    )
    monkeypatch.setattr(fit, "_refine_visible_correspondences", lambda *a: iter([wrong]))
    result, _ = fit.correct(np.eye(4))
    assert_static(fit, result)


def test_valid_intermediate_correction_is_kept_when_last_step_is_wrong(monkeypatch):
    fit = fitter()
    wrong = protruding()
    monkeypatch.setattr(
        fit, "_solve", lambda initial, limit: (wrong, {"evaluations": 1, "converged": True}, [(0.0, wrong)])
    )
    monkeypatch.setattr(fit, "_refine_visible_correspondences", lambda *a: iter([np.eye(4), wrong]))
    result, detail = fit.refine(wrong)
    assert_static(fit, result)
    assert detail["visible_correction_iterations"] == 2


def test_no_valid_pose_returns_no_pose_not_lowest_cost_invalid_pose(monkeypatch):
    fit = fitter()
    wrong = protruding()
    monkeypatch.setattr(
        fit, "_solve", lambda initial, limit: (wrong, {"evaluations": 1, "converged": True}, [(0.0, wrong)])
    )
    monkeypatch.setattr(fit, "_refine_visible_correspondences", lambda *a: iter([wrong]))
    pose, detail = fit.correct(wrong)
    assert pose is None
    assert detail["reason"] == "no_observation_compatible_candidate"


def test_all_starts_share_a_budget_and_validate_unchanged_pose_once(monkeypatch):
    fit = fitter()
    checks = Counter()
    limits = []

    def check(pose):
        checks[pose.tobytes()] += 1
        return fit.check_pose(pose)

    def solve(initial, limit):
        limits.append(limit)
        return initial, {"evaluations": limit, "converged": False}, [(0.0, initial)] * 3

    monkeypatch.setattr(fit, "_solve", solve)
    candidates, detail = fit.search(
        np.eye(4), check=check, evaluation_budget=1, recovery_starts=24, extra=[("same", np.eye(4))]
    )
    assert limits == [1]
    assert detail["evaluations"] == 1
    assert max(checks.values()) == 1
    assert any(c["check"]["accepted"] for c in candidates)


@pytest.mark.parametrize("enabled,mode", [(False, "selection"), (True, "selection"), (True, "kalman")])
def test_uncertainty_switch_does_not_change_geometric_validity(enabled, mode):
    fit = fitter(
        ImageConditionedFit,
        image_reference=np.empty((0, 3)),
        image_current=np.empty((0, 3)),
        image_scale_m=0.005,
        image_weight=0.0,
    )
    seen = {}

    def check(pose):
        verdict = fit.check_pose(pose, max_outlier_fraction=0.25)
        seen[pose.tobytes()] = verdict
        return verdict

    pose, detail = fit.correct_surface_first(
        protruding(), np.eye(4), check=check, uncertainty=PoseUncertaintyConfig(enabled=enabled, mode=mode)
    )
    assert pose is not None
    assert check(pose)["accepted"]
    assert not check(protruding())["accepted"]
    assert detail["geometry_check"]["accepted"]


def test_geometry_validation_does_not_read_weighted_solver_residual(monkeypatch):
    fit = fitter()

    def forbidden(*a):
        raise AssertionError("Changing the objective layout must not change geometry validation")

    monkeypatch.setattr(fit, "residual", forbidden)
    assert fit.check_pose(np.eye(4), max_outlier_fraction=0.25)["accepted"]
    assert not fit.check_pose(protruding(), max_outlier_fraction=0.25)["accepted"]


@pytest.mark.parametrize("enabled,mode", [(False, "selection"), (True, "selection"), (True, "kalman")])
def test_disabling_uncertainty_cannot_admit_nonoverlapping_pose(monkeypatch, enabled, mode):
    fit = fitter(
        ImageConditionedFit,
        image_reference=np.empty((0, 3)),
        image_current=np.empty((0, 3)),
        image_scale_m=0.005,
        image_weight=0.0,
    )
    behind = np.eye(4)
    behind[2, 3] = 0.03
    # It does not protrude into measured air, but also cannot explain the
    # visible object. Both checks are needed, with and without covariance.
    assert not fit.scene_depth_check(behind)["scene_depth_coherent_protrusion"]
    monkeypatch.setattr(
        WholeSurfaceFit, "_solve", lambda *a: (behind, {"evaluations": 1, "converged": True}, [(0.0, behind)])
    )
    monkeypatch.setattr(WholeSurfaceFit, "_refine_visible_correspondences", lambda *a: iter(()))
    pose, _ = fit.correct_surface_first(
        behind, behind, uncertainty=PoseUncertaintyConfig(enabled=enabled, mode=mode)
    )
    assert pose is None


@pytest.mark.parametrize("scene_cost", [False, True])
@pytest.mark.parametrize("outline_weight", [0, 1, 100])
def test_objective_weights_do_not_change_geometry_acceptance(scene_cost, outline_weight):
    prepared = frame(0)
    observed = prepared["current"]
    fit = WholeSurfaceFit(
        observed["obs"]["points"],
        observed,
        prepared["intrinsics"],
        ShapeConfig(
            scene_depth_constraints=scene_cost,
            require_full_mask_coverage=True,
            mask_coverage_weight=outline_weight,
        ),
    )
    assert fit.check_pose(np.eye(4), max_outlier_fraction=0.25)["accepted"]
    assert not fit.check_pose(protruding(), max_outlier_fraction=0.25)["accepted"]


def test_valid_numerical_iterate_survives_lower_cost_invalid_final_step(monkeypatch):
    fit = fitter()
    wrong = protruding()
    monkeypatch.setattr(
        fit,
        "_solve",
        lambda *a: (wrong, {"evaluations": 2, "converged": True}, [(1.0, np.eye(4)), (0.0, wrong)]),
    )
    monkeypatch.setattr(fit, "_refine_visible_correspondences", lambda *a: iter(()))
    result, _ = fit.refine(wrong)
    assert_static(fit, result)


def test_stationary_frame_uses_one_numerical_fit(monkeypatch):
    """The actual stationary fit cannot quietly regain three fitting routes."""
    fit = fitter(
        ImageConditionedFit,
        image_reference=np.empty((0, 3)),
        image_current=np.empty((0, 3)),
        image_scale_m=0.005,
        image_weight=1.0,
    )
    calls = []
    original = WholeSurfaceFit._solve

    def measured(self, initial, limit):
        calls.append(initial.copy())
        return original(self, initial, limit)

    monkeypatch.setattr(WholeSurfaceFit, "_solve", measured)
    pose, _ = fit.correct_surface_first(np.eye(4), np.eye(4))
    assert_static(fit, pose)
    assert len(calls) == 1
