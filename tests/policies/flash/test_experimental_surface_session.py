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
)
from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession


def transform(x=0, y=0, z=0):
    p = np.eye(4)
    p[:3, 3] = [x, y, z]
    return p


def points():
    x, y = np.meshgrid(np.linspace(-0.03, 0.03, 8), np.linspace(-0.02, 0.02, 7))
    return np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))


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
    prepare = SurfaceTrackingSession(None, shape_config=ShapeConfig(batch_jacobian=False))
    rgb = np.zeros((80, 100, 3), np.uint8)
    depth = np.full((80, 100), 1.0, np.float32)
    mask = np.zeros((80, 100), bool)
    mask[20:61, 25:76] = True
    k = np.array([[500.0, 0, 50], [0, 500.0, 40], [0, 0, 1]])
    return prepare, rgb, depth, mask, k


def test_prediction_during_empty_mask_never_grows_or_reseeds_surface(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(prepare.shape_config)
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
    assert tracker.connected_frame == 1
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


def test_no_overlap_does_not_reset_coordinate_origin(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    tracker = GrowingSurfaceSession(prepare.shape_config)
    tracker.update(prepare.prepare(rgb, depth, mask, k, 0), 0.0)
    old = tracker.reference.copy()
    # Real solver receives a distant observation. Make its iteration budget one
    # and prohibit the coarse-start branch only for this fit-rejection boundary.
    from lerobot.policies.flash import experimental_surface_session as module

    monkeypatch.setattr(module.ImageConditionedFit, "correct", lambda self, seed: (seed, {}))
    new = prepare.prepare(rgb, depth + 0.5, mask, k, 1)
    new["step"] = None
    result = tracker.update(new, 0.1)
    assert result["status"] == "lost"
    assert result["diagnostics"]["reason"] == "insufficient_surface_overlap"
    np.testing.assert_array_equal(tracker.reference, old)
    assert tracker.connected_frame == 0
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
    tracker = GrowingSurfaceSession(prepare.shape_config)
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
