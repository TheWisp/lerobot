"""Retire impossible carried positions without treating occlusion as a lost point."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.motion_groups import GroupConfig, SceneMotionGroups
from lerobot.policies.flash.object_group_association import ObjectGroupAssociation
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.scene_depth import SceneDepth, project
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.surface_frame_session import SurfaceFrameSession
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, prepare_surface_frame
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


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
        current = scene(2, xyz)
    result = tracker.update(current, depth=evidence(frame(2, observed)))
    group = next(iter(result.groups.values()))
    assert (0 not in group.ids) == removed
    assert (0 in result.depth_rejected_ids) == removed
    # Accepted measurements and the group's measured motion remain intact.
    assert group.measured.sum() == 39
    np.testing.assert_allclose(group.base_xyz[group.measured], points()[1:], atol=1e-10)
    if measurement == "outlier":
        assert 0 in current.ids  # Rejection does not kill the current image track.
    # Unknown depth next frame must not resurrect a rejected prediction.
    again = tracker.update(scene(3, points()[1:], np.arange(1, 40)), depth=evidence(frame(3)))
    assert (0 not in next(iter(again.groups.values())).ids) == removed


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


@pytest.mark.parametrize("background_survives", [True, False])
def test_object_consumer_drops_connection_only_after_its_group_members_are_rejected(background_survives):
    tracker = established()
    source = ObjectGroupAssociation(tracker.config)
    initial = scene(1, points())
    mask = np.zeros((128, 128), bool)
    mask[2, 2:18] = True
    source.record_observation(
        frame(1),
        initial,
        mask,
        {
            "status": "initialized",
            "camera_from_reference": np.eye(4),
            "diagnostics": {"reference_frame": 1},
        },
        points()[:16],
    )
    assert source.query(tracker.frame, frame(1), initial, mask, 1)[0] is not None
    for index, depth in ((2, 0.8), (3, 1.2), (4, 0)):
        ids = np.arange(16, 40) if background_survives else np.empty(0, dtype=int)
        current = scene(index, points()[ids], ids)
        snapshot = tracker.update(current, depth=evidence(frame(index, depth)))
        hint, _ = source.query(snapshot, frame(index, depth), current, np.zeros_like(mask), 1)
        assert (hint is not None) == (background_survives and index == 2)
        assert len(snapshot.groups) == (1 if background_survives or index == 2 else 0)
        if index > 2:
            assert source._group is None  # The relation cannot outlive removal of its whole group.
        if background_survives:
            assert next(iter(snapshot.groups.values())).measured.sum() == 24


@pytest.mark.parametrize("units", [1.0, 0.001])
def test_recorded_stale_members_are_removed_through_frame_owner(units):
    root = Path(__file__).parent / "fixtures/group_depth_20261119"
    meta = json.loads((root / "manifest.json").read_text())
    path = root / "measurements.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
    config = VisionPipelineConfig.from_dict(meta["settings"]["config"])
    engine = SurfaceFrameSession(
        (),
        None,
        config,
        None,
        None,
        surface_experiment=GrowingSurfaceConfig(),
        group_experiment=GroupConfig.from_dict(meta["settings"]["group_experiment"]),
        single_method="group_growing",
    )
    old = set(meta["old_ids"])
    try:
        with np.load(path, allow_pickle=False) as saved:
            for n, t in zip(meta["frames"], meta["times_s"], strict=True):
                measured = SceneTracks(n, t, saved[f"ids_{n}"], saved[f"xy_{n}"], saved[f"xyz_{n}"])
                # The fixture boundary freezes measured image/depth tracks, not groups.
                engine.scene_tracker = SimpleNamespace(update=lambda frame, measured=measured: measured)
                depth = saved["depth"] if n == meta["frames"][-1] else np.zeros_like(saved["depth"])
                if units == 0.001:
                    depth = np.rint(depth / units).astype(np.uint16)
                f = RGBDFrame(
                    n,
                    np.zeros((*depth.shape, 3), np.uint8),
                    depth,
                    saved["intrinsics"],
                    units,
                    saved["base_from_camera"],
                    t,
                    t,
                    "recorded",
                )
                if n == meta["frames"][-1]:
                    before = {int(i) for g in engine.scene_groups.frame.groups.values() for i in g.ids}
                    assert old <= before, "Fixture must reconstruct the actual stale membership"
                row, _ = engine.step(f, {})
            after = engine.scene_groups.frame
            assert old <= after.depth_rejected_ids
            assert not old & {int(i) for g in after.groups.values() for i in g.ids}
            assert {6, 591, 592} <= set(measured.ids)  # These measurements still exist elsewhere.
            assert len(row["scene_groups"]["depth_rejected_ids"]) >= 15
            assert sum(g.measured.sum() for g in after.groups.values()) >= 400
    finally:
        engine.close()
