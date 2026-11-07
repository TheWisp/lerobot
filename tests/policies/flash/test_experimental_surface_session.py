"""Measured-patch lineage and real fitting boundaries, with only UFM stubbed."""

from dataclasses import replace

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    ImageConditionedFit,
    PoseHint,
    SurfaceStore,
    SurfaceView,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig


def transform(x=0, y=0, z=0):
    p = np.eye(4)
    p[:3, 3] = [x, y, z]
    return p


def points():
    x, y = np.meshgrid(np.linspace(-0.03, 0.03, 8), np.linspace(-0.02, 0.02, 7))
    return np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))


def test_initial_sampling_keeps_each_cell_when_only_duplicate_measurements_exceed_budget():
    spacing = 0.003
    x, y = np.meshgrid(np.arange(40), np.arange(24))
    cells = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 100)))
    dense = np.concatenate([(cells + offset) * spacing for offset in (0.2, 0.4, 0.6)])
    store = SurfaceStore(GrowingSurfaceConfig(spacing_m=spacing, points_per_patch=1024))
    store.add(dense, np.eye(4), 0, active=True)
    # 2,880 readings describe only 960 cells; the 1,024-point budget must not
    # discard unique surface just because the camera oversampled those cells.
    assert len(store.points) == len(cells)
    assert {tuple(c) for c in np.floor(store.points / spacing).astype(int)} == {tuple(c) for c in cells}


@pytest.mark.parametrize("rotated", [False, True])
def test_growth_fills_adjacent_cells_at_same_density_as_initial_surface(rotated):
    from scipy.spatial.transform import Rotation

    spacing = 0.003
    x, y = np.meshgrid(np.arange(24), np.arange(10))
    cells = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 100)))
    old = cells[:, 0] < 20
    cloud = (cells + 0.1) * spacing
    cloud[old, 0] += 0.8 * spacing
    store = SurfaceStore(GrowingSurfaceConfig(spacing_m=spacing))
    store.add(cloud[old], np.eye(4), 0, active=True)
    initial = store.points.copy()
    pose = transform(0.021, -0.034, 0.012)
    if rotated:
        pose[:3, :3] = Rotation.from_euler("xyz", [24, -37, 16], degrees=True).as_matrix()
    observed = move(cloud, pose)
    store.extend(observed, pose, 1, view=None)
    np.testing.assert_array_equal(store.points, initial)  # Still needs confirmation.
    store.extend(observed, pose, 2, view=None)
    actual = {tuple(c) for c in np.floor(store.points / spacing).astype(int)}
    assert actual == {tuple(c) for c in cells}
    assert len(store.points) == len(cells)
    # Repeated overlap neither adds duplicates nor inflates the surface.
    count = len(store.patches)
    store.extend(observed, pose, 3, view=None)
    assert len(store.patches) == count
    assert len(store.points) == len(cells)


def test_patch_confirmation_is_later_and_does_not_contaminate_the_fit_target():
    store = SurfaceStore(GrowingSurfaceConfig())
    original = points()
    store.add(original, np.eye(4), 0, active=True)
    initial = store.points.copy()
    new = original + [0, 0, 0.025]
    pose = transform(0.01)
    store.add(move(new, pose), pose, 1)
    np.testing.assert_array_equal(store.points, initial)
    store.confirm(move(new, pose), pose, 1)  # Same frame cannot confirm itself.
    np.testing.assert_array_equal(store.points, initial)
    assert not store.patches[1].active
    store.confirm(move(new, pose), pose, 2)
    assert store.patches[1].active
    np.testing.assert_allclose(store.patches[1].points(), new)
    assert store.version == 2
    assert store.patches[1].support_frames == {1, 2}


def test_wrong_patch_can_be_repositioned_and_removed_without_destroying_original_measurements():
    store = SurfaceStore(GrowingSurfaceConfig())
    original = points()
    store.add(original, np.eye(4), 0, active=True)
    initial = store.points.copy()
    store.add(original + [0, 0, 0.02], transform(0.1), 1, active=True)
    assert store.points[:, 0].min() < -0.1
    measured = store.patches[1].measured.copy()
    store.reposition(1, np.eye(4))
    np.testing.assert_array_equal(store.patches[1].measured, measured)
    assert store.points[:, 0].min() >= -0.03
    store.exclude(1)
    np.testing.assert_array_equal(store.points, initial)
    with pytest.raises(KeyError):
        store.reposition(1, np.eye(4))  # No silent promise of unlimited undo.


def test_surface_and_pending_history_are_bounded():
    config = GrowingSurfaceConfig(max_patches=3, max_pending=2, max_points=50, points_per_patch=20)
    store = SurfaceStore(config)
    for i in range(40):
        store.add(points() + [i * 0.01, 0, 0], np.eye(4), i, active=i % 2 == 0)
        assert len(store.patches) <= 5
        assert len(store.points) <= 50
        assert all(len(p.measured) <= 20 for p in store.patches.values())
        store.events.clear()
    assert 0 not in store.patches


def test_memory_limit_keeps_broad_old_view_over_redundant_recent_patches():
    store = SurfaceStore(GrowingSurfaceConfig(max_patches=2))
    initial = points()
    store.add(initial, np.eye(4), 0, active=True)
    novel = points()[:5] + [0, 0, 0.02]
    store.add(novel, np.eye(4), 1, active=True)
    store.add(novel, np.eye(4), 2, active=True)
    assert 0 in store.patches
    assert len(store.patches) == 2
    assert len(store.points) >= len(initial)


def prepared_frames(monkeypatch):
    # Exact synthetic correspondence at the NN boundary; observation cleanup,
    # projection, fitting, missing-mask handling and history all run for real.
    from lerobot.policies.flash import surface_tracking

    cloud = points()
    monkeypatch.setattr(surface_tracking, "ufm_pairs", lambda *_args, **_kw: (cloud, cloud.copy()))
    prepare = SurfaceTrackingSession(None, shape_config=ShapeConfig())
    rgb = np.zeros((80, 100, 3), np.uint8)
    depth = np.full((80, 100), 1.0, np.float32)
    mask = np.zeros((80, 100), bool)
    mask[20:61, 25:76] = True
    k = np.array([[500.0, 0, 50], [0, 500.0, 40], [0, 0, 1]])
    return prepare, rgb, depth, mask, k


def test_recent_view_budget_keeps_map_and_uses_latest_accepted_measurement(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(
        prepare.shape_config, GrowingSurfaceConfig(partial_surface_matching=True, fit_view_limit=1)
    )
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    original = tracker.store.points.copy()
    # A later accepted measurement need not grow the surface to be a useful
    # source view. Keep its actual pixels/calibration and its mapped pose.
    latest = prepare.prepare(rgb, depth, mask, k, 1)
    tracker.accepted_frame = 1
    tracker.accepted_observation = latest["current"]
    tracker.accepted_intrinsics = latest["intrinsics"]
    tracker.accepted_pose = transform(x=0.01)
    views = tracker._fit_views()
    assert len(views) == 1
    assert views[0].observation is latest["current"]
    np.testing.assert_array_equal(views[0].camera_from_reference, tracker.accepted_pose)
    np.testing.assert_array_equal(tracker.store.points, original)
    assert 0 in tracker.store.patches
    tracker.config = replace(tracker.config, fit_view_limit=None)
    assert len(tracker._fit_views()) == 2


@pytest.mark.parametrize("prior_x", [0.0, 0.1])
def test_rejected_filter_update_keeps_its_reason_in_session_diagnostics(monkeypatch, prior_x):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(prepare.shape_config, GrowingSurfaceConfig())
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    # Isolate the filter/session boundary: a candidate can pass geometric
    # checks but provide no directional information for the Kalman update.
    # This is what the recorded clock's back view exposed.
    tracker.pose = tracker.kalman.pose = transform(x=prior_x)
    initial = tracker.store.points.copy()
    correct = tracker.kalman.correct

    def no_information(pose, information, pivot, **kwargs):
        return correct(pose, np.zeros_like(information), pivot, **kwargs)

    monkeypatch.setattr(tracker.kalman, "correct", no_information)
    result = tracker.update(prepare.prepare(rgb, depth, mask, k, 1), 0.1)
    if prior_x:
        assert result["camera_from_reference"] is None
        assert result["status"] == "lost"
    else:
        assert result["status"] == "predicted"
        np.testing.assert_allclose(result["camera_from_reference"], np.eye(4))
    assert result["diagnostics"]["prediction_geometry"]["accepted"] == (prior_x == 0)
    assert result["diagnostics"]["kalman"]["observable_rank"] == 0
    assert result["diagnostics"]["reason"] == "no_observation_information"
    np.testing.assert_array_equal(tracker.pose, transform(x=prior_x))
    np.testing.assert_array_equal(tracker.store.points, initial)
    np.testing.assert_array_equal(
        tracker.kalman.covariance, result["diagnostics"]["kalman"]["prior_covariance"]
    )
    assert tracker.accepted_frame == 0


def test_hidden_without_prediction_support_publishes_lost_not_empty_predicted(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(prepare.shape_config, GrowingSurfaceConfig())
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    initial = tracker.store.points.copy()
    covariance = tracker.kalman.covariance.copy()
    result = tracker.update(prepare.prepare(rgb, depth, mask & False, k, 1), 0.1)
    # The filter still predicts internally, but retention is disabled and no
    # group supports publication. Both consumers must see the same outcome.
    assert result["status"] == "lost"
    assert result["camera_from_reference"] is None
    assert result["diagnostics"]["reason"] == "no_mask"
    assert np.trace(tracker.kalman.covariance) > np.trace(covariance)
    np.testing.assert_array_equal(tracker.store.points, initial)
    assert tracker.accepted_frame == 0


def test_prediction_during_empty_mask_never_grows_or_reseeds_surface(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(
        prepare.shape_config, GrowingSurfaceConfig(uncertainty=PoseUncertaintyConfig(enabled=False))
    )
    empty = prepare.prepare(rgb, depth, mask & False, k, 0)
    assert tracker.update(empty, 0.0)["status"] == "lost"
    first = tracker.update(prepare.prepare(rgb, depth, mask, k, 1), 0.1)
    assert first["status"] == "initialized"
    initial = tracker.reference.copy()
    model = tracker.store.points.copy()
    hint = PoseHint(transform(0.04), "test co-moving group", 0.2, 0.1, 1)
    result = tracker.update(prepare.prepare(rgb, depth, mask & False, k, 2), 0.2, hint=hint)
    assert result["status"] == "predicted"
    np.testing.assert_allclose(result["camera_from_reference"], hint.camera_from_reference)
    np.testing.assert_array_equal(tracker.store.points, model)
    np.testing.assert_array_equal(tracker.reference, initial)
    assert tracker.accepted_frame == 1
    assert result["diagnostics"]["prediction_age_s"] == pytest.approx(0.1)
    assert tracker.update(prepare.prepare(rgb, depth, mask & False, k, 3), 0.3)["status"] == "lost"
    # Returning pixels reconnect to frame 1, not to the prediction at frame 2.
    returning = prepare.prepare(rgb, depth, mask, k, 4)
    assert returning["match_source_frame"] == 1
    result = tracker.update(returning, 0.4)
    assert result["status"] == "observed_candidate"
    assert np.linalg.norm(np.asarray(result["camera_from_reference"])[:3, 3]) < 0.005
    np.testing.assert_array_equal(tracker.reference, initial)
    assert tracker.reference_frame == 1
    assert result["diagnostics"]["covariance"] is None


def test_persistent_prediction_follows_group_then_world_camera_and_reacquires(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(
        prepare.shape_config,
        GrowingSurfaceConfig(uncertainty=PoseUncertaintyConfig(enabled=False), retain_unobserved_pose=True),
    )
    empty = prepare.prepare(rgb, depth, mask & False, k, 0)
    assert tracker.update(empty, 0.0)["camera_from_reference"] is None
    tracker.update(prepare.prepare(rgb, depth, mask, k, 1), 0.1)
    model = tracker.store.points.copy()
    # Surrounding measurements continue moving the hidden object.
    for i, x in [(2, 0.02), (3, 0.04)]:
        hint = PoseHint(transform(x), "environment_point_group", i / 10, 0.1, 1)
        result = tracker.update(prepare.prepare(rgb, depth, mask & False, k, i), i / 10, hint=hint)
        np.testing.assert_allclose(result["camera_from_reference"], transform(x))
    # Then all support disappears. Hold world state, including with a moving camera.
    for i, camera_x in [(4, 0.01), (5, 0.03), (6, 0.05)]:
        result = tracker.update(
            prepare.prepare(rgb, depth, mask & False, k, i), i / 10, base_from_camera=transform(camera_x)
        )
        assert result["status"] == "predicted"
        np.testing.assert_allclose(result["camera_from_reference"], transform(0.04 - camera_x), atol=1e-12)
        assert result["diagnostics"]["prediction_source"] == "last_world_pose"
        assert result["diagnostics"]["prediction_age_s"] == pytest.approx(i / 10 - 0.3)
        assert result["diagnostics"]["observation_age_s"] == pytest.approx(i / 10 - 0.1)
        np.testing.assert_array_equal(tracker.store.points, model)
        assert tracker.accepted_frame == 1
    # Real surface fitting corrects the stale prediction; the coordinate origin stays fixed.
    result = tracker.update(prepare.prepare(rgb, depth, mask, k, 7), 0.7)
    assert result["status"] == "observed_candidate"
    assert np.linalg.norm(np.asarray(result["camera_from_reference"])[:3, 3]) < 0.005
    assert tracker.reference_frame == 1
    assert result["diagnostics"]["observation_age_s"] == 0


def test_growth_checks_previous_rays_without_removing_hidden_back_surface(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    first = prepare.prepare(rgb, depth, mask, k, 0)
    view = SurfaceView.from_observation(first["current"], k, prepare.shape_config)
    store = SurfaceStore(GrowingSurfaceConfig())
    store.add(first["current"]["obs"]["points"], np.eye(4), 0, active=True, view=view)
    local = np.array([[0, 0, 0.97], [0, 0, 1.03], [0.08, 0, 1.0], [0, 0, 1.002]])
    # In front of observed depth / behind it / outside silhouette / within noise.
    np.testing.assert_array_equal(store.compatible_additions(local, np.eye(4)), [False, True, False, True])
    pose = transform(0.05, 0.02)
    np.testing.assert_array_equal(
        store.compatible_additions(move(local, pose), pose), [False, True, False, True]
    )
    # No depth means unknown, not a reason to remove the back or complete it.
    unknown = dict(first["current"], depth=np.zeros_like(depth), scene_depth=np.zeros_like(depth))
    empty_view = SurfaceView.from_observation(unknown, k, prepare.shape_config)
    assert not empty_view.contradicts(local).any()


@pytest.mark.parametrize("surface_priority", [False, True])
def test_repeated_wrong_fit_cannot_confirm_geometry_in_previously_empty_space(monkeypatch, surface_priority):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    # Even a mocked solver cannot publish a pose contradicted by scene depth,
    # or use repeated failures to confirm new geometry.
    tracker = GrowingSurfaceSession(
        prepare.shape_config,
        GrowingSurfaceConfig(
            uncertainty=PoseUncertaintyConfig(enabled=False),
            surface_priority=surface_priority,
            growth_min_overlap=0.25,
        ),
    )
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    initial = tracker.store.points.copy()
    # A fit can overlap the old plane while placing a new strip 20 mm in front
    # of a depth view that already ruled that space out. Repeat the bad fit.
    monkeypatch.setattr(ImageConditionedFit, "correct", lambda self, seed, **_: (np.eye(4), {}))
    monkeypatch.setattr(
        ImageConditionedFit, "correct_surface_first", lambda self, seed, previous, **_: (np.eye(4), {})
    )
    for i in (1, 2, 3):
        new_depth = depth.copy()
        new_depth[20:35, 25:76] = np.linspace(0.98, 1.0, 15)[:, None]
        # A small part of the same bad placement lands in space that was
        # previously hidden. It must not survive a mostly contradicted patch.
        new_depth[49:61, 30:40] = np.linspace(1.0, 1.012, 12)[:, None]
        result = tracker.update(prepare.prepare(rgb, new_depth, mask, k, i), i / 10)
        assert result["status"] == "lost"
        assert result["camera_from_reference"] is None
        assert result["diagnostics"]["fit"]["reason"] == "measured_empty_space_conflict"
        np.testing.assert_array_equal(tracker.store.points, initial)
        assert tracker.store.version == 1
        assert tracker.accepted_frame == 0
    # A valid observation can recover; rejection is not a permanent latch.
    result = tracker.update(prepare.prepare(rgb, depth, mask, k, 4), 0.4)
    assert result["status"] == "observed_candidate"
    assert tracker.accepted_frame == 4


@pytest.mark.parametrize("surface_priority", [False, True])
def test_no_overlap_does_not_reset_coordinate_origin(monkeypatch, surface_priority):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(
        prepare.shape_config,
        GrowingSurfaceConfig(
            uncertainty=PoseUncertaintyConfig(enabled=False), surface_priority=surface_priority
        ),
    )
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    old = tracker.reference.copy()
    # Real solver receives a distant observation. Make its iteration budget one
    # and prohibit the coarse-start branch only for this fit-rejection boundary.
    from lerobot.policies.flash import experimental_surface_session as module

    monkeypatch.setattr(module.ImageConditionedFit, "correct", lambda self, seed, **_: (seed, {}))
    monkeypatch.setattr(
        module.ImageConditionedFit, "correct_surface_first", lambda self, seed, previous, **_: (seed, {})
    )
    new = prepare.prepare(rgb, depth + 0.5, mask, k, 1)
    new["step"] = None
    result = tracker.update(new, 0.1)
    assert result["status"] == "lost"
    assert result["diagnostics"]["reason"] == "no_observation_compatible_candidate"
    assert result["diagnostics"]["fit"]["reason"] == "measured_empty_space_conflict"
    np.testing.assert_array_equal(tracker.reference, old)
    assert tracker.accepted_frame == 0
    assert tracker.store.version == 1


def test_joint_image_term_is_identical_in_scalar_and_batched_derivatives(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    current = prepare.prepare(rgb, depth, mask, k, 0)["current"]
    ref = current["obs"]["points"]
    fitter = ImageConditionedFit(
        ref,
        current,
        k,
        prepare.shape_config,
        image_reference=points(),
        image_current=points() + [0.001, 0, 0],
        image_scale_m=0.005,
        image_weight=1.0,
    )
    poses = [np.eye(4), transform(0.002), transform(0, 0, 0.004)]
    for actual, pose in zip(fitter.batch_residuals(poses), poses, strict=True):
        np.testing.assert_allclose(actual, fitter.residual(pose), atol=1e-12)
    # Zero image weight reproduces the old objective exactly.
    from lerobot.policies.flash.surface_grounding import WholeSurfaceFit

    fitter.image_weight = 0
    baseline = WholeSurfaceFit(ref, current, k, prepare.shape_config)
    np.testing.assert_array_equal(fitter.residual(poses[1]), baseline.residual(poses[1]))


def test_invalid_config_and_prediction_do_not_silently_enter_session():
    with pytest.raises(ValueError):
        replace(GrowingSurfaceConfig(), max_points=0)
    with pytest.raises(ValueError):
        replace(GrowingSurfaceConfig(), image_weight=float("nan"))
    with pytest.raises(ValueError):
        PoseHint(np.zeros((4, 4)), "test", 0.0, 0.0, 0)
    with pytest.raises(ValueError):
        PoseHint(np.eye(4), "test", 0.0, 1.0, 0)


def test_invalid_hint_can_be_retried_and_removing_model_does_not_reinitialize(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(
        prepare.shape_config, GrowingSurfaceConfig(uncertainty=PoseUncertaintyConfig(enabled=False))
    )
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    current = prepare.prepare(rgb, depth, mask, k, 1)
    with pytest.raises(ValueError, match="another time"):
        tracker.update(current, 0.1, hint=PoseHint(np.eye(4), "test", 0.2, 0.0, 0))
    assert tracker.update(current, 0.1)["status"] == "observed_candidate"
    for frame in list(tracker.store.patches):
        tracker.store.exclude(frame)
    result = tracker.update(prepare.prepare(rgb, depth, mask, k, 2), 0.2)
    assert result["status"] == "lost"
    assert result["diagnostics"]["reason"] == "surface_removed"
    assert tracker.reference_frame == 0


def test_addition_records_model_used_for_fit_before_other_promotions():
    store = SurfaceStore(GrowingSurfaceConfig())
    store.add(points(), np.eye(4), 0, active=True)
    before = store.version
    store.add(points() + [0, 0, 0.03], np.eye(4), 1)
    store.confirm(points() + [0, 0, 0.03], np.eye(4), 2)
    assert store.version > before
    store.add(points() + [0, 0, 0.06], np.eye(4), 2, fitted_map_version=before)
    assert store.patches[2].fitted_map_version == before


def test_visible_surface_corrects_wrong_group_prediction(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(prepare.shape_config)
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    from lerobot.policies.flash.experimental_surface_session import PointEvidence

    evidence = PointEvidence(tracker.reference, tracker.reference + [0, 0, 0.02], "group", "predicted", 0.005)
    hint = PoseHint(transform(0, 0, 0.02), "environment_point_group", 0.1, 0.0, 0, (evidence,))
    prepared = prepare.prepare(rgb, depth, mask, k, 1)
    result = tracker.update(prepared, 0.1, hint=hint)
    assert result["status"] == "observed_candidate"
    # Full RGB-D surface remains at the original depth; do not copy the prior.
    pose = np.asarray(result["camera_from_reference"])
    assert np.median(np.linalg.norm(move(tracker.reference, pose) - tracker.reference, axis=1)) < 0.005
    assert result["diagnostics"]["hint_disagreement_mm"] > 15


@pytest.mark.parametrize(
    "shared,kind,weight", [(True, "measured", 0), (True, "predicted", 1), (False, "measured", 1)]
)
def test_shared_object_matches_contribute_once_to_real_fit_residual(monkeypatch, shared, kind, weight):
    from lerobot.policies.flash.experimental_surface_session import PointEvidence

    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(prepare.shape_config)
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    current = prepare.prepare(rgb, depth, mask, k, 1)
    current["shared_matches"] = shared
    evidence = PointEvidence(points(), points(), "arbitrary_source", kind, 0.005, reuses_image_matches=True)
    hint = PoseHint(np.eye(4), "object_point_group", 0.1, 0.0, 0, (evidence,))
    original = ImageConditionedFit._solve
    checked = []

    def check(self, seed, limit):
        # Cross the caller -> actual objective boundary. Both evidence sets
        # would otherwise append a residual for the SAME point identities.
        assert self.image_weight == weight
        assert self.point_evidence == (evidence,)
        checked.append(True)
        return original(self, seed, limit)

    monkeypatch.setattr(ImageConditionedFit, "_solve", check)
    tracker.update(current, 0.1, hint=hint)
    assert checked


def test_point_sources_constrain_pose_and_remain_normalized_and_batch_equivalent(monkeypatch):
    from lerobot.policies.flash.experimental_surface_session import PointEvidence

    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    current = prepare.prepare(rgb, depth, mask, k, 0)["current"]
    ref = current["obs"]["points"]
    predicted = ref + [0.003, 0, 0]
    source = PointEvidence(ref, predicted, "group", "predicted", 0.005)
    fitter = ImageConditionedFit(
        ref,
        current,
        k,
        prepare.shape_config,
        image_reference=np.empty((0, 3)),
        image_current=np.empty((0, 3)),
        image_scale_m=0.005,
        image_weight=0.0,
        point_evidence=(source,),
    )
    plain = ImageConditionedFit(
        ref,
        current,
        k,
        prepare.shape_config,
        image_reference=np.empty((0, 3)),
        image_current=np.empty((0, 3)),
        image_scale_m=0.005,
        image_weight=0.0,
    )
    base_pose, _ = plain.correct(np.eye(4))
    pose, _ = fitter.correct(np.eye(4))
    # Evidence influences the real optimizer, not only its initialization.
    assert pose[0, 3] > base_pose[0, 3] + 0.0001
    for p, residual in zip(
        [pose, transform(0.002)], fitter.batch_residuals([pose, transform(0.002)]), strict=True
    ):
        np.testing.assert_allclose(residual, fitter.residual(p), atol=1e-12)
    value = np.sum(fitter.residual(pose) ** 2)
    fitter.point_evidence = (
        PointEvidence(np.tile(ref, (4, 1)), np.tile(predicted, (4, 1)), "group", "predicted", 0.005),
    )
    assert np.sum(fitter.residual(pose) ** 2) == pytest.approx(value)


def test_growth_requires_stronger_overlap_before_addition_and_confirmation():
    original = points()
    new = original + [0, 0, 0.02]
    measured = np.concatenate((original, new))
    strict = SurfaceStore(GrowingSurfaceConfig())
    legacy = SurfaceStore(
        GrowingSurfaceConfig(
            growth_agreement_m=0.005, growth_min_overlap=0.25, patch_consistency=0.5, patch_overlap=0.5
        )
    )
    for store in (strict, legacy):
        store.add(original, np.eye(4), 0, active=True)
        store.add(new, np.eye(4), 1)
    initial = strict.points.copy()
    weak = strict.extend(measured, np.eye(4), 2, view=None)
    assert weak["observed_overlap"] == 0.5
    assert not weak["alignment_accepted"]
    np.testing.assert_array_equal(strict.points, initial)
    assert not strict.patches[1].active
    assert strict.patches[1].support_frames == {1}
    assert 2 not in strict.patches  # Neither confirmation nor a new patch.
    legacy.extend(measured, np.eye(4), 2, view=None)
    assert legacy.patches[1].active
    assert legacy.points[:, 2].max() > 1.01

    # Clear overlap must still grow, including after rigid movement of the object.
    store = SurfaceStore(GrowingSurfaceConfig())
    store.add(original, np.eye(4), 0, active=True)
    measured = np.concatenate((original, new[:8]))
    for i in (1, 2):
        pose = transform(x=i * 0.01)
        result = store.extend(move(measured, pose), pose, i, view=None)
        assert result["alignment_accepted"]
    assert store.patches[1].active
    np.testing.assert_allclose(store.patches[1].points(), new[:8], atol=1e-12)
    assert store.points[:, 2].max() > 1.01


@pytest.mark.parametrize("distance,fraction", [(0.004, 1.0), (0.0, 0.6)])
def test_patch_confirmation_has_its_own_tighter_distance_and_fraction(distance, fraction):
    patch = points()
    count = int(len(patch) * fraction)
    measured = patch.copy()
    measured[:count, 2] += distance
    measured[count:, 2] += 0.02
    strict = SurfaceStore(GrowingSurfaceConfig())
    legacy = SurfaceStore(GrowingSurfaceConfig(growth_agreement_m=0.005, patch_overlap=0.5))
    for store in (strict, legacy):
        store.add(patch, np.eye(4), 1)
        store.confirm(measured, np.eye(4), 2)
    assert not strict.patches[1].active
    assert legacy.patches[1].active


def test_growth_thresholds_cannot_be_weaker_than_tracking():
    for settings in ({"growth_agreement_m": 0.006}, {"growth_min_overlap": 0.2}):
        with pytest.raises(ValueError, match="growth must require"):
            GrowingSurfaceConfig(**settings)
    for name in ("growth_agreement_m", "growth_min_overlap", "patch_consistency"):
        with pytest.raises(ValueError):
            GrowingSurfaceConfig(**{name: float("nan")})


def test_tracking_can_accept_a_pose_without_accepting_surface_growth(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(
        prepare.shape_config, GrowingSurfaceConfig(uncertainty=PoseUncertaintyConfig(enabled=False))
    )
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    initial = tracker.store.points.copy()
    # Accept a known pose at the fitter boundary. Actual RGB-D cleanup and both
    # downstream acceptance paths run. Disagreement is below the pose tolerance
    # but above the stricter growth tolerance, not the old 12 mm protrusion.
    pose = transform(x=0.001)
    monkeypatch.setattr(ImageConditionedFit, "correct_surface_first", lambda *_, **__: (pose.copy(), {}))
    changed = depth.copy()
    offset = (tracker.config.growth_agreement_m + prepare.shape_config.depth_tolerance_m) / 2
    changed[mask] += offset
    for i in (1, 2, 3):
        result = tracker.update(prepare.prepare(rgb, changed, mask, k, i), i / 10)
        assert result["status"] == "observed_candidate"
        np.testing.assert_allclose(result["camera_from_reference"], pose)
        assert tracker.accepted_frame == i
        assert result["diagnostics"]["growth"]["reason"] == "insufficient_growth_overlap"
        np.testing.assert_array_equal(tracker.store.points, initial)
        assert tracker.store.version == 1


def test_addition_consistency_is_distinct_from_later_patch_confirmation(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    first = prepare.prepare(rgb, depth, mask, k, 0)
    original = first["current"]["obs"]["points"]
    view = SurfaceView.from_observation(first["current"], k, prepare.shape_config)
    novel = points()[:10] + [0, 0, 0.025]
    novel[:2, 2] = 0.975  # Two points contradict previously measured empty space.
    measured = np.concatenate((original, novel))
    for consistency, expected in [(0.95, False), (0.5, True)]:
        store = SurfaceStore(GrowingSurfaceConfig(patch_consistency=consistency))
        store.add(original, np.eye(4), 0, active=True, view=view)
        result = store.extend(measured, np.eye(4), 1, view=None)
        assert result["alignment_accepted"]
        assert result["contradicted_by_retained_views"] == 2
        assert result["patch_consistent"] is expected
        assert (1 in store.patches) is expected
        if expected:
            np.testing.assert_allclose(store.patches[1].points(), novel[2:])
