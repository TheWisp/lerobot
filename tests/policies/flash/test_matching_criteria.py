"""Measured geometry is mandatory; appearance and coverage rank allowed fits."""

import numpy as np
import pytest

from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_registration import LibraryConfig, LibraryMatcher
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession


def textured_face():
    mask = np.zeros((100, 140), bool)
    mask[25:75, 40:100] = True
    k = np.array([[800.0, 0, 69.5], [0, 800.0, 49.5], [0, 0, 1]])
    depth = np.where(mask, 0.8, 1.2).astype(np.float32)
    y, x = np.indices(mask.shape)
    rgb = np.stack(
        (np.where(x < 70, 240, 20), np.where(y < 50, 200, 30), np.full(mask.shape, 60)), axis=-1
    ).astype(np.uint8)
    return rgb, depth, mask, k


def measured_fit(source_rgb, current_rgb, depth, mask, k):
    source = SurfaceTrackingSession(None).prepare(source_rgb, depth, mask, k, 0)["current"]
    current = SurfaceTrackingSession(None).prepare(current_rgb, depth, mask, k, 1)["current"]
    return WholeSurfaceFit(
        source["obs"]["points"],
        current,
        k,
        # Deliberately overwhelming texture preference: it cannot buy permission
        # to occupy empty space. No optional library solve runs in these tests.
        ShapeConfig(registration=LibraryConfig("small_gicp", appearance_weight=1e6)),
        reference_observation=source,
    )


def propose(monkeypatch, fit, poses):
    """Control only the solver output; use real RGB-D checks and real scores."""
    monkeypatch.setattr(
        fit,
        "_solve",
        lambda *args: (
            poses[-1],
            {"evaluations": 1, "converged": True},
            [(fit.score(pose), pose) for pose in poses],
        ),
    )


@pytest.mark.parametrize("conflict", ["outside_mask", "in_front_of_visible_face", "behind_visible_face"])
@pytest.mark.parametrize("valid_alternative", [False, True])
def test_better_texture_score_cannot_buy_a_geometry_violation(monkeypatch, conflict, valid_alternative):
    rgb, depth, mask, k = textured_face()
    fit = measured_fit(rgb, np.roll(rgb, 15, axis=1), depth, mask, k)
    valid, wrong = np.eye(4), np.eye(4)
    if conflict == "outside_mask":
        wrong[0, 3] = 0.015  # Perfect pattern match, but protrudes past the face.
        assert LibraryMatcher.appearance_cost(fit, wrong) == pytest.approx(0)
    elif conflict == "in_front_of_visible_face":
        wrong[2, 3] = -0.02  # Inside the mask, in front of measured object depth.
    else:
        wrong[2, 3] = 0.02  # The current face contradicts the retained source view.
    assert fit.score(wrong) < fit.score(valid)

    def check(pose):
        return fit.check_pose(pose, max_outlier_fraction=0.25)

    assert check(valid)["accepted"]
    assert not check(wrong)["accepted"]
    propose(monkeypatch, fit, [valid, wrong] if valid_alternative else [wrong])
    result, _ = fit.refine(wrong, check=check)
    if valid_alternative:
        np.testing.assert_array_equal(result, valid)
    else:
        assert result is None


def test_texture_selects_between_two_geometrically_valid_orientations(monkeypatch):
    rgb, depth, mask, k = textured_face()
    fit = measured_fit(rgb, rgb[::-1, ::-1].copy(), depth, mask, k)
    initial, rotated = np.eye(4), np.eye(4)
    rotated[0, 0] = rotated[1, 1] = -1

    def check(pose):
        return fit.check_pose(pose, max_outlier_fraction=0.25)

    assert all(check(pose)["accepted"] for pose in (initial, rotated))
    assert fit.score(rotated) < fit.score(initial)
    propose(monkeypatch, fit, [initial, rotated])
    result, _ = fit.refine(initial, check=check)
    np.testing.assert_array_equal(result, rotated)


def test_outline_coverage_prefers_a_better_fit_without_rejecting_an_incomplete_one(monkeypatch):
    from tests.policies.flash.test_scene_depth_constraints import scene

    source, current, k = scene()
    fit = WholeSurfaceFit(
        source["obs"]["points"],
        current,
        k,
        ShapeConfig(require_full_mask_coverage=True),
        reference_observation=source,
    )
    initial, better = np.eye(4), np.eye(4)
    initial[0, 3] = -0.4 / k[0, 0]  # One pixel less coverage; both pass depth checks.

    def check(pose):
        return fit.check_pose(pose, max_outlier_fraction=0.25)

    assert all(check(pose)["accepted"] for pose in (initial, better))
    assert fit.score(better) < fit.score(initial)
    propose(monkeypatch, fit, [initial, better])
    result, _ = fit.refine(initial, check=check)
    np.testing.assert_array_equal(result, better)
