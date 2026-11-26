"""Newest measured overlap replaces old samples; unknown geometry survives."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    SurfaceStore,
    SurfaceView,
)
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_refresh import RefreshingSurfaceStore


def plane(z=0.5012):
    x, y = np.meshgrid(np.arange(8) * 0.006 + 0.0005, np.arange(7) * 0.006 + 0.0005)
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def view(z=0.5032):
    return SurfaceView(
        np.full((120, 160), z),
        np.zeros((120, 160), bool),
        np.array([[500.0, 0, 60], [0, 500.0, 40], [0, 0, 1]]),
        0.003,
    )


def make_store(points, cls=RefreshingSurfaceStore):
    store = cls(GrowingSurfaceConfig())
    store.add(points, np.eye(4), 0, active=True)
    return store


@pytest.mark.parametrize("shift", [0.002, 0.004, 0.012, -0.004])
def test_current_observation_replaces_overlap_in_one_frame_even_beyond_growth_radius(shift):
    old = plane()
    measured = old + [0, 0, shift]
    store = make_store(old)
    report = store.extend(measured, np.eye(4), 1, view=view(measured[0, 2]), geometry_tolerance_m=0.005)
    np.testing.assert_array_equal(store.patches[0].points(), measured)
    np.testing.assert_array_equal(store.patches[0].measured, old)
    assert report["refined_points"] == len(old)
    assert report["retired_points"] == 0
    assert len(store.points) == len(old)


def test_replacement_oracle_distinguishes_append_only_control():
    old, measured = plane(), plane(0.5032)
    for cls, equal in [(RefreshingSurfaceStore, True), (SurfaceStore, False)]:
        store = make_store(old, cls)
        store.extend(measured, np.eye(4), 1, view=view())
        assert np.array_equal(store.patches[0].points(), measured) == equal


def test_refresh_replaces_region_with_measured_set_not_one_sample_per_old_point():
    old = plane()
    measured = np.concatenate((old + [0, 0, 0.002], old + [0.002, 0, 0.002]))
    store = make_store(old)
    report = store.extend(measured, np.eye(4), 1, view=view())
    np.testing.assert_array_equal(store.patches[0].points(), measured)
    assert report["refreshed_measured_points"] == len(measured)
    assert report["admitted_points"] == 0  # Refresh, not a pending growth patch.
    assert len(store.patches) == 1


def test_replacement_does_not_accumulate_a_weight_that_resists_new_measurements():
    old = plane()
    store = make_store(old)
    for frame in range(1, 21):
        store.extend(old, np.eye(4), frame, view=view(old[0, 2]))
    measured = old + [0, 0, 0.004]
    store.extend(measured, np.eye(4), 21, view=view(measured[0, 2]))
    np.testing.assert_array_equal(store.patches[0].points(), measured)


@pytest.mark.parametrize("unobserved", ["hole", "foreground", "offscreen", "mask_boundary"])
def test_refresh_changes_visible_surface_and_keeps_unobserved_points_exactly(unobserved):
    old, measured = plane(), plane(0.5032)
    current = view()
    hidden = old[:8] + [0, 0, 0.020]
    if unobserved == "hole":
        hidden = old[:8] + [0, -0.002, 0]
        current.nearest_depth[37:40, :] = 0
    elif unobserved == "offscreen":
        hidden = old[:8] + [-1, 0, 0]
    elif unobserved == "mask_boundary":
        hidden = old[:8] + [0, -0.002, 0]
        current.mask = np.ones(current.nearest_depth.shape, dtype=bool)
        current.mask[37:40, :] = False
    store = make_store(np.concatenate((old, hidden)))
    store.extend(measured, np.eye(4), 1, view=current)
    np.testing.assert_array_equal(store.patches[0].points()[-8:], hidden)
    np.testing.assert_array_equal(store.patches[0].points()[:-8], measured)


def test_refresh_removes_contradictions_and_keeps_accumulated_back_surface():
    old, measured = plane(), plane(0.5032)
    hidden = old + [0, 0, 0.030]
    protruding = np.array([[0.0505, 0.0385, 0.40]])
    store = make_store(np.concatenate((old, hidden, protruding)))
    report = store.extend(measured, np.eye(4), 1, view=view(), geometry_tolerance_m=0.005)
    np.testing.assert_array_equal(store.patches[0].points(), np.concatenate((measured, hidden)))
    assert report["retired_points"] == 1


def test_refresh_respects_fixed_object_coordinates_under_arbitrary_camera_pose():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [37, -23, 61], degrees=True).as_matrix()
    pose[:3, 3] = [0.1, -0.04, 0.08]
    old, measured = plane(), plane(0.5052)
    store = make_store(move(old, np.linalg.inv(pose)))
    store.extend(measured, pose, 1, view=view(measured[0, 2]))
    np.testing.assert_allclose(store.patches[0].points(), move(measured, np.linalg.inv(pose)), atol=1e-12)


def test_confirmed_point_refresh_is_idempotent_for_same_frame():
    store = make_store(plane())
    store.extend(plane(0.5032), np.eye(4), 1, view=view())
    before = store.points.copy()
    store.extend(plane(0.5042), np.eye(4), 1, view=view(0.5042))
    np.testing.assert_array_equal(store.points, before)


def test_refresh_keeps_working_patch_memory_bounded_without_inventing_points():
    from scipy.spatial import cKDTree

    old = plane()
    # Denser observations must not grow an unbounded retained patch even though
    # the combined fitting map already has its own sampling limit.
    store = RefreshingSurfaceStore(GrowingSurfaceConfig(max_points=100))
    store.add(old, np.eye(4), 0, active=True)
    measured = np.concatenate([old + [offset, 0, 0.002] for offset in np.linspace(0, 0.002, 10)])
    for frame in range(1, 11):
        store.extend(measured, np.eye(4), frame, view=view())
        assert len(store.patches[0].points()) <= store.config.max_points
        assert len(store.points) <= store.config.max_points
        assert np.all(cKDTree(measured).query(store.patches[0].points())[0] < 1e-12)
    np.testing.assert_array_equal(store.patches[0].measured, old)


def test_reposition_rebuilds_original_measurement_and_exclude_removes_refreshed_patch():
    old = plane()
    store = make_store(old)
    store.extend(plane(0.5032), np.eye(4), 1, view=view())
    pose = np.eye(4)
    pose[0, 3] = 0.01
    store.reposition(0, pose)
    np.testing.assert_array_equal(store.patches[0].points(), move(old, np.linalg.inv(pose)))
    store.exclude(0)
    assert len(store.points) == 0


def test_pending_contradictions_are_removed_before_confirmation():
    old = plane()
    store = make_store(old)
    stale = np.array([[0.0505, 0.0385, 0.40]])
    store.add(stale, np.eye(4), 1)
    report = store.extend(old, np.eye(4), 2, view=view(old[0, 2]), geometry_tolerance_m=0.005)
    assert report["retired_points"] == 1
    assert len(store.patches[1].points()) == 0
    assert not store.patches[1].active
    assert store.patches[1].support_frames == {1}
