"""Measured depth rejection of carried scene group points."""

import json
from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.motion_groups import GroupConfig, SceneMotionGroups
from lerobot.policies.flash.observations import RGBDFrame
from lerobot.policies.flash.scene_depth import SceneDepth, project
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, prepare_surface_frame


def points():
    xyz = np.random.default_rng(42).uniform([-0.04, -0.04, 0.98], [0.04, 0.04, 1.02], (40, 3))
    xyz[0] = [0, 0, 1]
    return xyz


def scene(index, xyz, ids=None):
    ids = np.arange(len(xyz)) if ids is None else np.asarray(ids)
    return SceneTracks(index, index / 10, ids, np.column_stack((ids + 2, np.full(len(ids), 2))), xyz)


def frame(index, depth=0.0, camera=None):
    return RGBDFrame(
        index,
        np.zeros((128, 128, 3), np.uint8),
        np.full((128, 128), depth, np.float32),
        np.array([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]]),
        1.0,
        np.eye(4) if camera is None else camera,
        index / 10,
        index / 10,
        "synthetic",
    )


def evidence(f):
    filters = prepare_surface_frame(f.metric_depth(), asdict(SurfaceObservationConfig()))
    return SceneDepth.from_frame(f, filters, tolerance_m=0.005, radius_px=2)


def established():
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    for index in (0, 1):
        tracker.update(scene(index, points()))
    assert len(tracker.frame.groups) == 1
    return tracker


@pytest.mark.parametrize(
    "observed,removed",
    [(1.1, True), (0.9, False), (1.0, False), (1.004, False), (0.0, False), (np.nan, False)],
)
@pytest.mark.parametrize("measurement", ["missing", "outlier"])
def test_depth_rejects_only_contradicted_carried_members(observed, removed, measurement):
    tracker = established()
    xyz = points()
    if measurement == "missing":
        current = scene(2, xyz[1:], np.arange(1, 40))
    else:
        xyz[0, 0] += 0.05
        # A stray measurement keeps its place for its split window; one point cannot split.
        for index in (2, 3):
            waiting = tracker.update(scene(index, xyz), depth=evidence(frame(index, observed)))
            assert 0 in next(iter(waiting.groups.values())).ids and not waiting.depth_rejected_ids
        current = scene(4, xyz)
    result = tracker.update(current, depth=evidence(frame(current.frame_id, observed)))
    group = next(iter(result.groups.values()))
    assert (0 not in group.ids) == removed
    assert (0 in result.depth_rejected_ids) == removed
    # Accepted measurements and the group's measured motion remain intact.
    assert group.measured.sum() == 39
    np.testing.assert_allclose(group.base_xyz[group.measured], points()[1:], atol=1e-10)
    if measurement == "outlier":
        assert 0 in current.ids  # Rejection does not kill the current image track.
    # Unknown depth next frame must not resurrect a rejected prediction.
    later = current.frame_id + 1
    again = tracker.update(scene(later, points()[1:], np.arange(1, 40)), depth=evidence(frame(later)))
    assert (0 not in next(iter(again.groups.values())).ids) == removed


def test_depth_check_keeps_a_departing_cohort_until_it_splits():
    """Points leaving together are not erased at their old spot while their split is pending."""
    tracker = established()
    xyz = points()
    cohort = np.argsort(xyz[:, 0])[:16]
    xyz[cohort, 0] += 0.05
    # The camera sees past every carried position, including the cohort's old spot.
    for index in range(2, 5):
        result = tracker.update(scene(index, xyz), depth=evidence(frame(index, 1.1)))
        assert not result.depth_rejected_ids
    assert len(result.groups) == 2
    branch = next(g for g in result.groups.values() if set(g.ids.tolist()) & set(cohort.tolist()))
    assert set(branch.ids.tolist()) == set(cohort.tolist()) and branch.confirmed


def test_compare_current_carried_pixel_and_camera_z_after_group_and_camera_motion():
    tracker = established()
    movement = np.eye(4)
    movement[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    center = np.array([0.0, 0.0, 1.0])
    movement[:3, 3] = center - movement[:3, :3] @ center + [0.13, 0.02, 0.1]
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("y", -5, degrees=True).as_matrix()
    camera[:3, 3] = [0.05, 0, 0.3]
    actual = move(points(), movement)
    f = frame(2, 0.1, camera)
    predicted_camera = move(actual[:1], np.linalg.inv(camera))
    x, y = np.rint(project(predicted_camera, f.intrinsics)[0]).astype(int)
    assert abs(x - 64) > 15 and predicted_camera[0, 2] + 0.05 < 1
    f.depth[y - 6 : y + 7, x - 6 : x + 7] = predicted_camera[0, 2] + 0.05
    current = scene(2, actual[1:], np.arange(1, 40))
    result = tracker.update(current, depth=evidence(f))
    assert result.depth_rejected_ids == {0}
    np.testing.assert_allclose(next(iter(result.groups.values())).base_xyz, actual[1:], atol=1e-10)


@pytest.mark.parametrize("kind", ["hole", "occluder", "depth_edge", "offscreen"])
def test_uncertain_projection_neighborhood_preserves_missing_point(kind):
    tracker = established()
    f = frame(2, 1.1)
    if kind == "hole":
        f.depth[64, 65] = 0
    elif kind == "occluder":
        f.depth[60:69, 60:69] = 0.8
    elif kind == "depth_edge":
        f.depth[:, 65:] = 0.8
    else:
        f.base_from_camera[0, 3] = 2
    result = tracker.update(scene(2, points()[1:], np.arange(1, 40)), depth=evidence(f))
    assert not result.depth_rejected_ids
    assert 0 in next(iter(result.groups.values())).ids


def _reject_depth_per_group(tracker, depth, current):
    """Pre-batching implementation, retained as an equivalence oracle."""
    rejected = set()
    for group in list(tracker._groups.values()):
        ids = sorted(group.anchors.keys() - group.inliers)
        if not ids:
            continue
        predicted = move(np.array([group.anchors[i] for i in ids]), group.pose)
        for i, bad in zip(ids, depth.contradicted(predicted), strict=True):
            if not bad:
                continue
            del group.anchors[i]
            tracker._departing.pop(i, None)
            tracker._departed_from.pop(i, None)
            if i not in current:
                tracker._seen.pop(i, None)
            rejected.add(i)
        if len(group.anchors) < tracker.config.min_motion_points:
            del tracker._groups[group.id]
    return frozenset(rejected)


@pytest.mark.parametrize("depth_kind", ["recorded", "unknown", "farther"])
def test_batched_depth_preserves_per_group_results_and_cleanup(depth_kind, monkeypatch):
    root = Path(__file__).parent / "fixtures/group_depth_20261119"
    meta = json.loads((root / "manifest.json").read_text())
    tracker = SceneMotionGroups(GroupConfig.from_dict(meta["settings"]["group_experiment"]))
    with np.load(root / "measurements.npz", allow_pickle=False) as saved:
        for n, t in zip(meta["frames"], meta["times_s"], strict=True):
            measured = SceneTracks(n, t, saved[f"ids_{n}"], saved[f"xy_{n}"], saved[f"xyz_{n}"])
            tracker.update(measured)
        depth = saved["depth"].copy()
        if depth_kind == "unknown":
            depth.fill(0)
        elif depth_kind == "farther":
            depth.fill(10)
        f = RGBDFrame(
            n,
            np.zeros((*depth.shape, 3), np.uint8),
            depth,
            saved["intrinsics"],
            1.0,
            saved["base_from_camera"],
            t,
            t,
            "recorded",
        )
        observed = evidence(f)
    # Include complete group retirement, as well as mixed measured/carried groups.
    if depth_kind == "farther":
        for group in tracker._groups.values():
            group.inliers.clear()
    legacy = copy(tracker)
    previous_group_count = len(tracker._groups)
    for name in ("_groups", "_seen", "_departing", "_departed_from"):
        setattr(legacy, name, deepcopy(getattr(tracker, name)))
    expected = _reject_depth_per_group(legacy, observed, set(measured.ids))
    original, calls = SceneDepth.contradicted, []

    def counted(self, xyz):
        calls.append(len(xyz))
        return original(self, xyz)

    monkeypatch.setattr(SceneDepth, "contradicted", counted)
    assert sum(bool(g.anchors.keys() - g.inliers) for g in tracker._groups.values()) > 1
    assert tracker._reject_depth(observed, set(measured.ids)) == expected
    assert len(calls) == 1
    np.testing.assert_equal(
        {i: asdict(g) for i, g in tracker._groups.items()},
        {i: asdict(g) for i, g in legacy._groups.items()},
    )
    for name in ("_seen", "_departing", "_departed_from"):
        np.testing.assert_equal(getattr(tracker, name), getattr(legacy, name))
    if depth_kind == "recorded":
        assert set(meta["old_ids"]) <= expected
    elif depth_kind == "unknown":
        assert not expected
    else:
        # Offscreen points can remain; groups wholly contradicted in view retire.
        assert len(tracker._groups) < previous_group_count


def test_depth_query_is_skipped_without_carried_members():
    tracker = established()

    def unexpected_query(xyz):
        pytest.fail("Accepted measured members must not require a depth query")

    tracker.update(scene(2, points()), depth=SimpleNamespace(contradicted=unexpected_query))
