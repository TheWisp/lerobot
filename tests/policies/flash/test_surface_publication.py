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
from lerobot.policies.flash.surface_registration import LibraryConfig
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


@pytest.mark.parametrize("kalman,surface_priority", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("backend", ["visible_icp"])
def test_any_matcher_proposal_must_pass_empty_space_check_before_publication_and_growth(
    monkeypatch, kalman, surface_priority, backend
):
    config = GrowingSurfaceConfig(
        surface_priority=surface_priority,
        uncertainty=PoseUncertaintyConfig(enabled=kalman, mode="kalman", max_visible_outlier_fraction=0.99),
    )
    session = GrowingSurfaceSession(
        ShapeConfig(scene_depth_constraints=False, registration=LibraryConfig(backend)), config
    )
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
@pytest.mark.parametrize("compatible_group", [False, True])
def test_failed_local_fit_only_publishes_depth_compatible_group_prediction(
    monkeypatch, kalman, compatible_group
):
    config = GrowingSurfaceConfig(
        retain_unobserved_pose=False,
        uncertainty=PoseUncertaintyConfig(enabled=kalman, mode="kalman"),
    )
    session = GrowingSurfaceSession(ShapeConfig(), config)
    session.update(frame(0), 0.0)
    original = session.store.points.copy()
    wrong = np.eye(4)
    wrong[0, 3] = 0.05

    def propose_wrong(fitter, seed, limit):
        return wrong.copy(), {"evaluations": 1, "converged": True}, [(0.0, wrong.copy())]

    monkeypatch.setattr(WholeSurfaceFit, "_solve", propose_wrong)
    support = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.06, 0.06) for z in (0.6, 0.7)])
    group_pose = np.eye(4)
    if not compatible_group:
        group_pose[0, 3] = 0.05
    hint = PoseHint(
        group_pose,
        "measured_support",
        0.1,
        0.1,
        0,
        motion=MeasuredGroupMotion(support, support + group_pose[:3, 3], 0.0, 0.1, "measured_support"),
    )
    result = session.update(frame(1), 0.1, hint=hint)
    assert result["status"] == ("predicted" if compatible_group else "lost")
    assert result["diagnostics"]["prediction_geometry"]["accepted"] == compatible_group
    if compatible_group:
        np.testing.assert_allclose(result["camera_from_reference"], np.eye(4), atol=1e-6)
    else:
        assert result["camera_from_reference"] is None
        # Cross the real live display transport: losing the public pose must
        # not lose the failing estimate or its geometry verdict for debugging.
        import json
        import struct
        from types import SimpleNamespace

        from lerobot.gui.flash_live_scene import live_scene_packet
        from lerobot.policies.flash.perception import RGBDFrame

        current = frame(1)["current"]
        measured = RGBDFrame(
            1, current["rgb"], current["depth"], current["intrinsics"], 1, np.eye(4), 0.1, 0.1, "fixture"
        )
        method = "group_growing"
        live = SimpleNamespace(
            method=method,
            mask_source=SimpleNamespace(prompts={"object": "test object"}),
            engine=SimpleNamespace(
                references={"object": session.reference},
                candidates={method: {"object": session}},
                sessions={"object": None},
            ),
        )
        row = {
            "objects": {
                "object": {
                    method: None,
                    method + "_status": result["status"],
                    method + "_details": result["diagnostics"],
                }
            }
        }
        packet = live_scene_packet(measured, live, row, {"object": current["mask"]})
        size = struct.unpack("<I", packet[:4])[0]
        saved = json.loads(packet[4 : 4 + size])["objects"]["object"]
        assert saved["pose"] is None  # Rejected estimates stay out of the scene.
        diagnostics = saved["diagnostics"]
        np.testing.assert_allclose(diagnostics["unpublished_camera_from_reference"], session.pose)
        np.testing.assert_allclose(diagnostics["fit"]["rejected_camera_from_reference"], wrong)
        np.testing.assert_allclose(saved["rejected_poses"]["fit"], wrong)
        np.testing.assert_allclose(saved["rejected_poses"]["prediction"], session.pose)
        assert diagnostics["prediction_geometry"]["scene_depth_coherent_protrusion"]
    assert session.accepted_frame == 0
    np.testing.assert_array_equal(session.store.points, original)


@pytest.mark.parametrize("kalman", [False, True])
@pytest.mark.parametrize("wrong_support", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("scene_depth", [False, True])
@pytest.mark.parametrize("backend", ["visible_icp"])
def test_downstream_rejection_cannot_publish_a_geometrically_impossible_fallback(
    kalman, wrong_support, partial, scene_depth, backend
):
    config = GrowingSurfaceConfig(
        partial_surface_matching=partial,
        uncertainty=PoseUncertaintyConfig(enabled=kalman, mode="kalman", max_visible_outlier_fraction=0.25),
    )
    session = GrowingSurfaceSession(
        ShapeConfig(scene_depth_constraints=scene_depth, registration=LibraryConfig(backend)), config
    )
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
    assert session.accepted_frame == 0
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
    assert session.accepted_frame == 0
