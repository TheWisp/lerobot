"""Latest accepted measurements determine veto eligibility, not ICP inputs."""

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from test_distributed_free_space import patches, scene

from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, SurfaceView
from lerobot.policies.flash.surface_grounding import ShapeConfig, SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_refresh import (
    LatestObservationFit,
    RefreshingSurfaceStore,
    refreshed_membership,
)
from lerobot.policies.flash.surface_registration import LibraryConfig, LibraryMatcher


def evaluate(fit, pose=None):
    return fit.check_pose(
        np.eye(4) if pose is None else pose, agreement_m=0.005, min_overlap=0.25, max_outlier_fraction=0.25
    )


def test_unobserved_points_keep_helping_icp_but_cannot_veto_next_pose():
    current, k = scene()
    visible, hidden = current["obs"]["points"], patches(k)
    reference = np.concatenate((visible, hidden))
    config = ShapeConfig(registration=LibraryConfig(backend="small_gicp"))
    assert not evaluate(WholeSurfaceFit(reference, current, k, config))["accepted"]
    full = LatestObservationFit(
        reference, current, k, config, accepted_view=SurfaceFitView(current, k, np.eye(4))
    )
    eligible = full._latest_rejection[0]
    assert evaluate(full)["accepted"]
    assert eligible[: len(visible)].all() and not eligible[len(visible) :].any()
    assert full.reference is reference
    levels = LibraryMatcher(config.registration).prepare(full)
    assert levels
    # Inspect the REAL library's preprocessed source, not a mocked caller.
    _, source, _, _, center, _ = levels[-1]
    points = np.asarray(source.points())[:, :3] + center
    assert points[:, 0].min() <= hidden[:, 0].min() + 0.003
    assert points[:, 0].max() >= hidden[:, 0].max() - 0.003


def test_transferred_surface_uses_current_geometry_without_an_accepted_live_view():
    current, k = scene()
    fit = LatestObservationFit(current["obs"]["points"], current, k, ShapeConfig(), accepted_view=None)
    assert evaluate(fit)["accepted"]
    wrong = np.eye(4)
    wrong[2, 3] = -0.01
    result = evaluate(fit, wrong)
    assert not result["accepted"]
    assert result["rejection_reference_frame"] is None


@pytest.mark.parametrize("offset", [-0.01, 0.01])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reliable_surface_still_rejects_real_displacements(axis, offset):
    current, k = scene()
    full = LatestObservationFit(
        current["obs"]["points"],
        current,
        k,
        ShapeConfig(),
        accepted_view=SurfaceFitView(current, k, np.eye(4)),
    )
    flags = full._latest_rejection[0]
    assert flags.all()
    assert evaluate(full)["accepted"]
    wrong = np.eye(4)
    wrong[axis, 3] = offset
    assert not evaluate(full, wrong)["accepted"]


@pytest.mark.parametrize("angle", [45, 70])
def test_reliable_surface_rejects_rotations_that_protrude(angle):
    current, k = scene()
    full = LatestObservationFit(
        current["obs"]["points"],
        current,
        k,
        ShapeConfig(),
        accepted_view=SurfaceFitView(current, k, np.eye(4)),
    )
    wrong = np.eye(4)
    wrong[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
    assert not evaluate(full, wrong)["accepted"]


@pytest.mark.parametrize("reliable_count", [0, 1, 2])
def test_missing_veto_evidence_does_not_erase_current_surface_agreement(reliable_count):
    current, k = scene()
    reference = current["obs"]["points"] + [0, 0, 0.001]
    reference[:reliable_count] = current["obs"]["points"][:reliable_count]
    full = LatestObservationFit(
        reference, current, k, ShapeConfig(), accepted_view=SurfaceFitView(current, k, np.eye(4))
    )
    flags = full._latest_rejection[0]
    assert flags.sum() == reliable_count
    result = evaluate(full)
    assert result["accepted"] and result["has_visible_overlap"]
    # With no trusted veto points, a pose still needs actual surface support.
    wrong = np.eye(4)
    wrong[0, 3] = 0.1
    assert not evaluate(full, wrong)["accepted"]


def test_reliability_expires_when_hidden_and_returns_only_after_replacement():
    current, k = scene()
    points = current["obs"]["points"]
    store = RefreshingSurfaceStore(GrowingSurfaceConfig())
    store.add(points, np.eye(4), 0, active=True)
    assert refreshed_membership(store.points, current, np.eye(4)).all()
    hidden_observation = {**current, "obs": {**current["obs"], "points": points[:0]}}
    assert not refreshed_membership(store.points, hidden_observation, np.eye(4)).any()
    measured = points + [0, 0, 0.002]
    reappeared = {**current, "obs": {**current["obs"], "points": measured}}
    # Merely nearby new evidence cannot promote the OLD coordinates.
    assert not refreshed_membership(store.points, reappeared, np.eye(4)).any()
    store.extend(
        measured,
        np.eye(4),
        1,
        view=SurfaceView.from_observation(current, k, ShapeConfig()),
        geometry_tolerance_m=0.005,
    )
    flags = refreshed_membership(store.points, reappeared, np.eye(4))
    assert flags.any()
    # The boundary is deliberately not replaced; it stays in ICP but cannot vote.
    old = np.isclose(store.points[:, 2], 0.4, atol=1e-7)
    assert old.any() and not flags[old].any()
    assert flags[~old].all()


def test_acceptance_refresh_removes_untrusted_contradictions_and_preserves_hidden_shape():
    current, k = scene()
    points = current["obs"]["points"]
    back = points + [0, 0, 0.03]
    stale = patches(k)
    store = RefreshingSurfaceStore(GrowingSurfaceConfig())
    store.add(np.concatenate((points, back, stale)), np.eye(4), 0, active=True)
    # add() uses the ordinary storage voxel spacing: compare the stored points,
    # not every dense sensor point supplied to it.
    stored_back = store.points[store.points[:, 2] > 0.42].copy()
    full = LatestObservationFit(
        store.points, current, k, ShapeConfig(), accepted_view=SurfaceFitView(current, k, np.eye(4))
    )
    assert evaluate(full)["accepted"]
    report = store.extend(
        points,
        np.eye(4),
        1,
        view=SurfaceView.from_observation(current, k, ShapeConfig()),
        geometry_tolerance_m=0.005,
    )
    assert report["retired_points"] == len(stale)
    actual = store.patches[0].points()
    assert np.all(cKDTree(actual).query(stored_back)[0] < 1e-12)
    flags = refreshed_membership(store.points, current, np.eye(4))
    assert flags.any() and not flags.all()


def test_reliability_is_frozen_before_candidate_and_is_independent_of_rotation():
    current, k = scene()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [28, -16, 91], degrees=True).as_matrix()
    pose[:3, 3] = [0.1, -0.03, 0.2]
    reference = move(current["obs"]["points"], np.linalg.inv(pose))
    full = LatestObservationFit(
        reference, current, k, ShapeConfig(), accepted_view=SurfaceFitView(current, k, pose)
    )
    flags = full._latest_rejection[0]
    assert flags.all() and evaluate(full, pose)["accepted"]
    wrong = pose.copy()
    wrong[2, 3] -= 0.01
    assert not evaluate(full, wrong)["accepted"]
    assert evaluate(full, pose)["accepted"]
    np.testing.assert_array_equal(full._latest_rejection[0], flags)


def refreshed_fixture(reference, source, source_pose, current, k, config):
    store = RefreshingSurfaceStore(GrowingSurfaceConfig())
    store.add(reference, np.eye(4), source["frame_id"] - 1, active=True)
    store.extend(
        source["obs"]["points"],
        source_pose,
        source["frame_id"],
        view=SurfaceView.from_observation(source, k, config),
        geometry_tolerance_m=0.005,
    )
    fit = LatestObservationFit(
        store.points, current, k, config, accepted_view=SurfaceFitView(source, k, source_pose)
    )
    flags = fit._latest_rejection[0]
    assert flags.any()  # A negative result with zero usable evidence proves nothing.
    return fit


def test_recorded_static_controller_and_displaced_controls_after_refresh():
    from test_retained_view_boundary_depth import clip

    saved = clip.__wrapped__()
    old = saved["two"]
    source_view = old.reference_views[0]
    fit = refreshed_fixture(
        old.reference,
        source_view.observation,
        source_view.camera_from_reference,
        old.observation,
        old.k,
        old.config,
    )
    pose = saved["arrays"]["prediction_pose"]
    assert evaluate(fit, pose)["accepted"]
    for delta in [(0, 0, -0.01), (0, 0, 0.01), (0.008, 0, 0), (0, 0.008, 0)]:
        wrong = pose.copy()
        wrong[:3, 3] += delta
        assert not evaluate(fit, wrong)["accepted"]
