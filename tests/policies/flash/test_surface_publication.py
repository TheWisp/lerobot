"""Final published poses must respect the same geometry as fitted candidates.

All observations, fitting and Kalman updates run normally. A downstream source
can decline an alignment, but doing so must not resurrect a contradicted prior.
"""

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    PoseHint,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_kalman import MeasuredGroupMotion
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


@pytest.mark.parametrize("kalman,surface_priority", [(False, False), (False, True), (True, True)])
def test_any_matcher_proposal_must_pass_empty_space_check_before_publication_and_growth(
    monkeypatch, kalman, surface_priority
):
    config = GrowingSurfaceConfig(
        surface_priority=surface_priority,
        uncertainty=PoseUncertaintyConfig(enabled=kalman, mode="kalman", max_visible_outlier_fraction=0.99),
    )
    session = GrowingSurfaceSession(ShapeConfig(scene_depth_constraints=False), config)
    initial = frame(0)
    session.update(initial, 0.0)
    original = session.store.points.copy()
    wrong = np.eye(4)
    wrong[0, 3] = 0.008  # Narrow protrusion, not an obviously detached object.

    # The solver boundary may return any candidate: neither solver confidence
    # nor a very good matching score is permission to occupy measured air.
    def propose_wrong(fitter, seed, limit):
        return wrong.copy(), {"evaluations": 1, "converged": True}, [(0.0, wrong.copy())]

    monkeypatch.setattr(WholeSurfaceFit, "_solve", propose_wrong)
    monkeypatch.setattr(WholeSurfaceFit, "_refine_visible_correspondences", lambda *a: iter(()))
    result = session.update(frame(1), 0.1)
    returned = result["camera_from_reference"]
    if returned is not None:
        # Keeping the compatible previous pose is valid; publishing the
        # contradicted proposal is not. Use the independently known static scene.
        points = original @ np.asarray(returned)[:3, :3].T + np.asarray(returned)[:3, 3]
        assert np.max(np.linalg.norm(points - original, axis=1)) < 0.005
    np.testing.assert_array_equal(session.store.points, original)


def frame(index, *, extension=0):
    mask = np.zeros((100, 140), bool)
    mask[25:75, 35 : 85 + extension] = True
    mask[25:45, 85:100] = True
    depth = np.where(mask, 0.8, 1.2).astype(np.float32)
    k = np.array([[800.0, 0, 70], [0, 800.0, 50], [0, 0, 1]])
    return SurfaceTrackingSession(None).prepare(np.zeros((*mask.shape, 3), np.uint8), depth, mask, k, index)


@pytest.mark.parametrize("kalman", [False, True])
@pytest.mark.parametrize("wrong_support", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("scene_depth", [False, True])
def test_downstream_rejection_cannot_publish_a_geometrically_impossible_fallback(
    kalman, wrong_support, partial, scene_depth
):
    config = GrowingSurfaceConfig(
        partial_surface_matching=partial,
        uncertainty=PoseUncertaintyConfig(enabled=kalman, mode="kalman", max_visible_outlier_fraction=0.25),
    )
    session = GrowingSurfaceSession(ShapeConfig(scene_depth_constraints=scene_depth), config)
    session.update(frame(0), 0.0)
    original = session.store.points.copy()
    pose = np.eye(4)
    pose[0, 3] = 0.05 if wrong_support else 0
    support = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.06, 0.06) for z in (0.6, 0.7)])
    hint = PoseHint(
        pose,
        "measured_support",
        0.1,
        0.1,
        0,
        motion=MeasuredGroupMotion(support, support + pose[:3, 3], 0.0, 0.1, "measured_support"),
    )
    current = frame(1)
    comparisons = []

    def decline(points, candidate):
        comparisons.append(candidate.copy())
        return False, {"reason": "independent_source_disagreement"}

    result = session.update(current, 0.1, hint=hint, candidate_check=decline)
    assert comparisons, "The real fitter must reach the downstream acceptance boundary"
    returned = result["camera_from_reference"]
    if wrong_support:
        assert returned is None, "Rejected fit must not revive the contradicted support prediction"
    else:
        assert returned is not None, "A compatible fallback must remain available"
        geometry = WholeSurfaceFit(original, current["current"], current["intrinsics"], session.shape_config)
        assert check_surface_contradiction(geometry, np.asarray(returned), config.uncertainty)["accepted"]
    np.testing.assert_array_equal(session.store.points, original)
    assert session.connected_frame == 0
    if kalman:
        assert not result["diagnostics"]["kalman"]["accepted"], (
            "Rolled-back correction must not be reported as accepted"
        )
        np.testing.assert_array_equal(
            session.kalman.covariance, result["diagnostics"]["kalman"]["prior_covariance"]
        )
    # Refusal must not poison the reference or prevent a subsequent clear
    # observation from correcting the false support motion.
    returning = session.update(frame(2), 0.2)
    assert returning["status"] == "observed_candidate"
    returned = np.asarray(returning["camera_from_reference"])
    predicted = original @ returned[:3, :3].T + returned[:3, 3]
    assert np.max(np.linalg.norm(predicted - original, axis=1)) < 0.005


@pytest.mark.parametrize("wrong_support", [False, True])
@pytest.mark.parametrize("scene_depth", [False, True])
def test_insufficient_overlap_checks_fallback_and_rolls_back_filter(wrong_support, scene_depth):
    config = GrowingSurfaceConfig(min_overlap=0.99, growth_min_overlap=0.99)
    session = GrowingSurfaceSession(ShapeConfig(scene_depth_constraints=scene_depth), config)
    session.update(frame(0), 0.0)
    original = session.store.points.copy()
    pose = np.eye(4)
    pose[0, 3] = 0.05 if wrong_support else 0
    support = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.06, 0.06) for z in (0.6, 0.7)])
    hint = PoseHint(
        pose,
        "measured_support",
        0.1,
        0.1,
        0,
        motion=MeasuredGroupMotion(support, support + pose[:3, 3], 0.0, 0.1, "measured_support"),
    )
    result = session.update(frame(1, extension=10), 0.1, hint=hint)
    reason = result["diagnostics"]["reason"]
    if reason == "no_observation_compatible_candidate":
        # The stronger depth check may now reject a bad support-seeded fit
        # before Kalman or the later overlap check. No correction occurred.
        assert wrong_support
        assert not result["diagnostics"]["kalman"]["accepted"]
        np.testing.assert_array_equal(
            session.kalman.covariance, result["diagnostics"]["kalman"]["prior_covariance"]
        )
    else:
        assert reason == "insufficient_surface_overlap"
        assert not result["diagnostics"]["kalman"]["accepted"]
        np.testing.assert_array_equal(
            session.kalman.covariance, result["diagnostics"]["kalman"]["prior_covariance"]
        )
    assert (result["camera_from_reference"] is None) == wrong_support
    np.testing.assert_array_equal(session.store.points, original)
    assert session.connected_frame == 0
