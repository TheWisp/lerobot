"""Scene groups from physical point motion, independent of selected objects."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.motion_groups import GroupConfig, SceneMotionGroups
from lerobot.policies.flash.object_group_association import ObjectGroupAssociation
from lerobot.policies.flash.observations import RGBDFrame
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.surface_frame_session import scene_group_record
from lerobot.policies.flash.surface_observation import move


def cloud(n=80):
    return np.random.default_rng(42).uniform([-0.08, -0.06, 0.95], [0.08, 0.06, 1.05], (n, 3))


def scene(i, xyz, ids=None, rate=10):
    ids = np.arange(len(xyz)) if ids is None else np.asarray(ids)
    return SceneTracks(i, i / rate, ids, np.column_stack((ids % 60 + 2, ids // 60 + 2)), np.asarray(xyz))


def delta(angle=0, translation=(0, 0, 0)):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    center = np.array([0, 0, 1.0])
    pose[:3, 3] = center - pose[:3, :3] @ center + translation
    return pose


def memberships(result):
    return {int(i): g.id for g in result.groups.values() for i in g.ids}


def frame(i, camera=None):
    return RGBDFrame(
        i,
        np.zeros((64, 64, 3), np.uint8),
        np.ones((64, 64), np.float32),
        np.array([[500.0, 0, 32], [0, 500.0, 32], [0, 0, 1]]),
        1.0,
        np.eye(4) if camera is None else camera,
        i / 10,
        i / 10,
        "synthetic",
    )


def mask_for(s, ids):
    mask = np.zeros((64, 64), bool)
    keep = np.isin(s.ids, ids)
    xy = s.xy[keep].astype(int)
    mask[xy[:, 1], xy[:, 0]] = True
    return mask


def initialize(source, s, reference):
    source.record_observation(
        frame(s.frame_id),
        s,
        mask_for(s, range(16)),
        {"status": "initialized", "camera_from_reference": np.eye(4), "diagnostics": {"reference_frame": 0}},
        reference,
    )


@pytest.mark.parametrize(
    ("body_step_m", "camera_step_m"),
    [(0.0, 0.0), (0.01, 0.0), (0.0, 0.004), (0.01, 0.004)],
)
def test_hidden_object_points_follow_stationary_or_moving_support_with_camera_motion(
    body_step_m, camera_step_m
):
    """The point-group layer carries hidden landmarks through the four motion combinations."""
    rng = np.random.default_rng(91)
    ring = rng.uniform([-0.06, -0.06, 0.96], [0, 0, 1.0], (16, 3))
    tray = rng.uniform([-0.12, -0.12, 0.95], [0.12, 0.12, 1.05], (48, 3))
    background = rng.uniform([0.18, -0.12, 0.95], [0.35, 0.12, 1.05], (20, 3))
    reference = np.vstack((ring, tray, background))
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0, split_confirmation_s=0.1))
    for index in range(9):
        actual = reference.copy()
        actual[:64, 0] += index * body_step_m
        camera = np.eye(4)
        camera[1, 3] = index * camera_step_m
        camera_xyz = actual - camera[:3, 3]
        uv = np.rint(camera_xyz[:, :2] / camera_xyz[:, 2:3] * 200 + 64).astype(int)
        ids = np.arange(84) if index < 4 or index == 8 else np.arange(16, 84)
        measured = SceneTracks(index, index / 10, ids, uv[ids], actual[ids])
        snapshot = tracker.update(measured)
        if index == 0:
            continue  # The group core needs a measured pair to establish motion.
        owners = memberships(snapshot)
        assert len({owners[i] for i in range(64)}) == 1
        if body_step_m:
            assert owners[0] != owners[64]
        carrier = snapshot.groups[owners[0]]
        by_id = {int(point_id): row for row, point_id in enumerate(carrier.ids)}
        np.testing.assert_allclose(carrier.base_xyz[[by_id[i] for i in range(16)]], actual[:16], atol=1e-9)
        if 4 <= index <= 7:
            assert not carrier.measured[[by_id[i] for i in range(16)]].any()
            rgbd = RGBDFrame(
                index,
                np.zeros((128, 128, 3), np.uint8),
                np.ones((128, 128), np.float32),
                np.array([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]]),
                1.0,
                camera,
                index / 10,
                index / 10,
                "synthetic",
            )
            display = scene_group_record(snapshot, measured, rgbd)
            dots = next(group["predicted_pixels"] for group in display if group["id"] == carrier.id)
            assert all(uv[i].tolist() in dots for i in range(16))


def test_object_queries_and_rejected_pose_cannot_change_future_scene_groups():
    points = cloud()
    config = GroupConfig(support_confirmation_s=0)
    alone = SceneMotionGroups(config)
    queried = SceneMotionGroups(config)
    source = ObjectGroupAssociation(config)
    for index in range(5):
        measured = scene(index, move(points, delta(index * 2, (index * 0.002, 0, 0))))
        baseline = alone.update(measured)
        actual = queried.update(measured)
        if index == 0:
            initialize(source, measured, points[:16])
        source.query(actual, frame(index), measured, mask_for(measured, range(16)), 0)
        source.record_observation(
            frame(index),
            measured,
            mask_for(measured, range(16)),
            {"status": "lost", "camera_from_reference": None, "diagnostics": {}},
            points[:16],
        )
        assert memberships(actual) == memberships(baseline)
        for group_id in baseline.groups:
            expected, observed = baseline.groups[group_id], actual.groups[group_id]
            np.testing.assert_allclose(observed.base_from_group, expected.base_from_group)
            np.testing.assert_allclose(observed.base_xyz, expected.base_xyz)
            np.testing.assert_array_equal(observed.measured, expected.measured)


def test_new_object_association_threshold_does_not_change_scene_group_birth():
    points = cloud(16)
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    tracker.update(scene(0, points))
    current = scene(1, points)
    tracker.update(current)
    assert len(tracker.frame.groups) == 1
    strict = ObjectGroupAssociation(tracker.config, min_association_points=20)
    permissive = ObjectGroupAssociation(tracker.config, min_association_points=4)
    for source in (strict, permissive):
        initialize(source, current, points)
    mask = mask_for(current, range(16))
    assert strict.query(tracker.frame, frame(1), current, mask, 0)[0] is None
    assert permissive.query(tracker.frame, frame(1), current, mask, 0)[0] is not None
    assert len(tracker.frame.groups) == 1


def test_stationary_and_moving_bodies_form_separate_groups():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    tracker.update(scene(0, points))
    moved = points.copy()
    moved[:24] = move(moved[:24], delta(12, (0.02, 0, 0)))
    result = tracker.update(scene(1, moved))
    assigned = memberships(result)
    assert len({assigned[i] for i in range(24)}) == 1
    assert len({assigned[i] for i in range(24, 80)}) == 1
    assert assigned[0] != assigned[30]
    display = scene_group_record(result, scene(1, moved), frame(1))
    assert {group["id"] for group in display} == set(result.groups)
    assert all(group["pixels"] and group["camera_xyz"] for group in display)
    # This display path takes no object pose, mask or Kalman result. It must
    # still show both independent groups when a downstream pose is rejected.
    hidden_scene = scene(2, moved[24:], np.arange(24, 80))
    hidden_snapshot = tracker.update(hidden_scene)
    hidden_display = scene_group_record(hidden_snapshot, hidden_scene, frame(2))
    assert any(group["predicted_pixels"] for group in hidden_display)


def test_group_display_retains_every_point_id_when_membership_grows_or_points_hide():
    """The display must not sparsify the tracker output and redraw false links."""
    from types import SimpleNamespace

    points = cloud(100)

    def display(ids, visible_ids):
        group = SimpleNamespace(
            id=4,
            ids=np.asarray(ids),
            base_xyz=points[np.asarray(ids)],
            measured=np.isin(ids, visible_ids),
            confirmed=True,
            supported=True,
            residual_m=0.0,
        )
        snapshot = SimpleNamespace(groups={4: group})
        return scene_group_record(snapshot, scene(0, points[np.asarray(visible_ids)], visible_ids), frame(0))[
            0
        ]

    first = display(range(60), range(60))
    assert first["ids"] == list(range(60))
    grown = display(range(100), range(100))
    assert grown["ids"] == list(range(100))
    partially_hidden = display(range(100), range(20, 100))
    assert partially_hidden["ids"] == list(range(20, 100))
    assert len(partially_hidden["predicted_pixels"]) == 20
    returned = display(range(100), range(100))
    assert returned["ids"] == list(range(100))


def test_rejected_depth_is_displayed_as_a_group_prediction():
    from types import SimpleNamespace

    points = cloud(16)
    group = SimpleNamespace(
        id=4,
        ids=np.arange(16),
        base_xyz=points,
        measured=np.array([False] + [True] * 15),
        confirmed=True,
        supported=True,
        residual_m=0.0,
    )
    measured = scene(0, points)
    measured.base_xyz[0] += [0.03, 0, 0]
    record = scene_group_record(SimpleNamespace(groups={4: group}), measured, frame(0))[0]
    assert 0 not in record["ids"]
    assert 0 in record["predicted_ids"]
    assert len(record["predicted_pixels"]) == 1


def test_merge_carries_object_relation_through_scene_group_coordinates():
    points = cloud()
    config = GroupConfig(support_confirmation_s=0, merge_confirmation_s=0.4)
    tracker = SceneMotionGroups(config)
    tracker.update(scene(0, points))
    initial = scene(1, points)
    snapshot = tracker.update(initial)
    parent_id = memberships(snapshot)[0]
    source = ObjectGroupAssociation(config)
    initialize(source, initial, points[:16])
    source.query(snapshot, frame(1), initial, mask_for(initial, range(16)), 0)
    moved = points.copy()
    moved[:24, 0] += 0.04
    split_id = None
    split_relation = None
    for index in range(2, 16):
        current = scene(index, moved)
        snapshot = tracker.update(current)
        if split_id is not None and split_id in snapshot.merges:
            surviving_id, surviving_from_split = snapshot.merges[split_id]
            hint, details = source.query(snapshot, frame(index), current, mask_for(current, range(16)), 0)
            assert surviving_id == parent_id == details["group_id"]
            assert details["merged_from_group"] == split_id
            assert split_relation is not None
            expected = snapshot.groups[surviving_id].base_from_group @ surviving_from_split @ split_relation
            np.testing.assert_allclose(hint.camera_from_reference, expected, atol=1e-10)
            break
        source.query(snapshot, frame(index), current, mask_for(current, range(16)), 0)
        if source._group != parent_id and split_id is None:
            split_id = source._group
            split_relation = source._group_from_reference.copy()
            # Later accepted points can have noisy reference coordinates. The
            # merge must use scene lineage, not re-fit this changed point set.
            for point_id in source.anchors:
                source.anchors[point_id] += [0.012, 0, 0]
    else:
        pytest.fail("expected the measured split to merge after resting")


def test_object_association_memory_is_bounded_even_without_a_group_prediction():
    points = cloud()
    groups = SceneMotionGroups(GroupConfig(max_points=80))
    source = ObjectGroupAssociation(groups.config)
    first = scene(0, points)
    groups.update(first)
    initialize(source, first, points[:16])
    full_mask = np.ones((64, 64), bool)
    for i in range(1, 20):
        current = scene(i, points, np.arange(80) + i * 80)
        # Use valid image coordinates; IDs may grow without allocating by ID.
        current.xy[:] = first.xy
        groups.update(current)
        hint, _ = source.query(groups.frame, frame(i), current, full_mask, 0)
        assert hint is None  # No identities survive to measure co-movement.
        source.record_observation(
            frame(i),
            current,
            full_mask,
            {
                "status": "observed_candidate",
                "camera_from_reference": np.eye(4),
                "diagnostics": {"reference_frame": 0},
            },
            points[:16],
        )
        assert len(source.anchors) <= 80


def test_object_queries_cannot_change_groups_and_camera_transform_is_applied():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    s = scene(0, points)
    tracker.update(s)
    first, second = ObjectGroupAssociation(tracker.config), ObjectGroupAssociation(tracker.config)
    initialize(first, s, points[:16])
    initialize(second, s, points[:16])
    motion = delta(15, (0.02, 0, 0))
    s = scene(1, move(points, motion))
    tracker.update(s)
    camera = delta(0, (0.2, -0.1, 0))
    before = memberships(tracker.frame)
    hint, _ = first.query(tracker.frame, frame(1, camera), s, mask_for(s, range(16)), 0)
    np.testing.assert_allclose(hint.camera_from_reference, np.linalg.inv(camera) @ motion, atol=1e-10)
    # Arbitrary rejected surface outputs cannot enter scene-group state.
    second.record_observation(
        frame(1),
        s,
        mask_for(s, range(16)),
        {"status": "lost", "camera_from_reference": None, "diagnostics": {"reference_frame": 0}},
        points[:16],
    )
    assert memberships(tracker.frame) == before
    other, _ = second.query(tracker.frame, frame(1, camera), s, mask_for(s, range(16)), 0)
    np.testing.assert_allclose(other.camera_from_reference, hint.camera_from_reference)


def test_associated_group_remains_visible_when_it_has_no_new_motion_hint():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(tracker.config, point_evidence_weight=0.25)
    first = scene(0, points)
    tracker.update(first)
    initialize(source, first, points[:16])
    measured = scene(1, points)
    tracker.update(measured)
    hint, details = source.query(tracker.frame, frame(1), measured, mask_for(measured, range(16)), 0)
    assert hint is not None
    assert hint.points[0].weight == 0.25
    associated = details["group_id"]
    hidden = scene(2, np.empty((0, 3)), np.empty(0, dtype=int))
    tracker.update(hidden)
    hint, details = source.query(tracker.frame, frame(2), hidden, np.zeros((64, 64), bool), 0)
    assert hint is None  # No fresh group motion is measurable.
    assert details["group_id"] == associated  # The view is not tied to pose publication.
    assert scene_group_record(tracker.frame, hidden, frame(2))[0]["predicted_pixels"]


def test_object_hidden_prediction_uses_surroundings_and_only_measured_kalman_pairs():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(tracker.config)
    s = scene(0, points)
    tracker.update(s)
    initialize(source, s, points[:16])
    s = scene(1, points)
    tracker.update(s)
    source.query(tracker.frame, frame(1), s, mask_for(s, range(16)), 0)
    motion = delta(10, (0.03, 0, 0))
    s = scene(2, move(points[16:], motion), np.arange(16, 80))
    tracker.update(s)
    hint, details = source.query(tracker.frame, frame(2), s, np.zeros((64, 64), bool), 0)
    np.testing.assert_allclose(hint.camera_from_reference, motion, atol=1e-10)
    assert len(hint.motion.current) == 64
    assert all(p.kind == "predicted" for p in hint.points)
    assert details["object_inliers"] == 0 and details["environment_inliers"] == 64
    assert details["hidden_members"] == 16
    assert list(
        tracker.frame.groups[details["group_id"]].ids[~tracker.frame.groups[details["group_id"]].measured]
    ) == list(range(16))


def test_visible_known_object_points_separate_from_old_environment_group():
    """A confirmed scene split invalidates an object-side relation to the old body."""
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig())
    source = ObjectGroupAssociation(tracker.config)
    for i in range(11):
        current = scene(i, points)
        tracker.update(current)
    initialize(source, current, points[:16])
    old_hint, old = source.query(tracker.frame, frame(10), current, mask_for(current, range(16)), 0)
    assert old_hint is not None
    old_group = old["group_id"]

    moved = points.copy()
    moved[:16, 0] += 0.03
    for i in range(11, 14):
        current = scene(i, moved)
        tracker.update(current)
    assert all(i not in tracker.frame.groups[old_group].ids for i in range(16))
    visible_hint, visible = source.query(tracker.frame, frame(13), current, mask_for(current, range(16)), 0)
    assert visible_hint is not None
    assert visible["group_id"] != old_group
    np.testing.assert_allclose(
        visible_hint.camera_from_reference, delta(translation=(0.03, 0, 0)), atol=1e-10
    )

    # Changing the current mask cannot undo a measured membership transition.
    # Both associations follow the same persistent IDs into the measured
    # branch, which inherits the confirmed parent's trust.
    hidden_source = ObjectGroupAssociation(tracker.config)
    hidden_source.anchors = {i: p.copy() for i, p in source.anchors.items()}
    hidden_source.reference = source.reference.copy()
    hidden_source.reference_frame = source.reference_frame
    hidden_source._group = old_group
    hidden_source._group_from_reference = old_hint.camera_from_reference.copy()
    hidden_hint, _ = hidden_source.query(tracker.frame, frame(13), current, np.zeros((64, 64), bool), 0)
    assert hidden_hint is not None
    np.testing.assert_allclose(
        hidden_hint.camera_from_reference, visible_hint.camera_from_reference, atol=1e-10
    )
    assert hidden_source._group == source._group != old_group
    stale_hint, _ = source.query(tracker.frame, frame(13), current, np.zeros((64, 64), bool), 0)
    assert stale_hint is not None
    np.testing.assert_allclose(
        stale_hint.camera_from_reference, visible_hint.camera_from_reference, atol=1e-10
    )

    for i in range(14, 24):
        shifted = moved.copy()
        shifted[:16, 0] += (i - 13) * 0.003
        current = scene(i, shifted)
        tracker.update(current)
    new_hint, new_details = source.query(tracker.frame, frame(23), current, mask_for(current, range(16)), 0)
    assert new_hint is not None
    assert new_details["group_id"] != old_group
    assert new_details["object_inliers"] >= 12


def test_never_joined_object_ids_do_not_break_existing_group_association():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(tracker.config)
    first = scene(0, points)
    tracker.update(first)
    first = scene(1, points)
    tracker.update(first)
    initialize(source, first, points[:16])
    old_hint, old_details = source.query(tracker.frame, frame(1), first, mask_for(first, range(16)), 0)
    assert old_hint is not None

    # Fresh IDs can be recorded from an accepted fit, including a bad fit.
    # Their coordinates alone do not prove that established members departed.
    new_ids = np.arange(80, 96)
    source.anchors.update({int(i): p.copy() for i, p in zip(new_ids, points[:16], strict=True)})
    current = scene(2, np.concatenate((points, points[:16] + [0.04, 0, 0])), np.arange(96))
    tracker.update(current)
    retained, details = source.query(tracker.frame, frame(2), current, mask_for(current, new_ids), 0)
    assert retained is not None
    np.testing.assert_allclose(retained.camera_from_reference, old_hint.camera_from_reference)
    assert details["association_check"]["unassigned_points"] == 16
    assert details["association_check"]["persistent_members"] == 16

    hidden = scene(3, points)
    tracker.update(hidden)
    retained, hidden_details = source.query(tracker.frame, frame(3), hidden, np.zeros((64, 64), bool), 0)
    assert retained is not None
    assert hidden_details["group_id"] == old_details["group_id"]
    assert old_details["group_id"] in tracker.frame.groups


def test_one_bad_measurement_does_not_disconnect_established_object_members():
    points = cloud(16)
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(tracker.config)
    tracker.update(scene(0, points))
    first = scene(1, points)
    tracker.update(first)
    initialize(source, first, points)
    old_hint, old = source.query(tracker.frame, frame(1), first, mask_for(first, range(16)), 0)
    assert old_hint is not None
    moved = points.copy()
    moved[0, 0] += 0.03
    for index in range(2, 5):
        current = scene(index, moved)
        tracker.update(current)
    assert 0 in tracker.frame.groups[old["group_id"]].ids
    assert not tracker.frame.groups[old["group_id"]].measured[0]
    hint, details = source.query(tracker.frame, frame(4), current, mask_for(current, [0]), 0)
    assert hint is not None
    assert details["group_id"] == old["group_id"]
    assert details["association_check"]["persistent_members"] == 16
    assert "association_contradiction" not in details


def test_repeated_single_point_disagreement_never_becomes_a_departure():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(tracker.config)
    tracker.update(scene(0, points))
    current = scene(1, points)
    tracker.update(current)
    initialize(source, current, points[:16])
    _, initial = source.query(tracker.frame, frame(1), current, mask_for(current, range(16)), 0)
    gid = initial["group_id"]
    shifted = points.copy()
    shifted[0, 0] += 0.03
    # A recovered reading resets the split timer; repeated later disagreement
    # still cannot make one point into another rigid body.
    for index, measured in [(2, shifted), (3, points), (4, shifted), (5, shifted), (6, shifted)]:
        current = scene(index, measured)
        snapshot = tracker.update(current)
        hint, details = source.query(snapshot, frame(index), current, mask_for(current, [0]), 0)
        assert hint is not None
        assert not snapshot.groups[gid].departed_ids
        assert details["association_check"]["persistent_members"] == 16
        assert details["association_check"]["unassigned_points"] == 0
        assert "association_contradiction" not in details


def test_capacity_eviction_is_not_a_motion_departure():
    points = cloud()
    config = GroupConfig(support_confirmation_s=0, max_points=80)
    tracker = SceneMotionGroups(config)
    source = ObjectGroupAssociation(config)
    tracker.update(scene(0, points))
    current = scene(1, points)
    tracker.update(current)
    initialize(source, current, points[:16])
    old, details = source.query(tracker.frame, frame(1), current, mask_for(current, range(16)), 0)
    gid = details["group_id"]
    for index in (2, 3):
        current = scene(index, np.concatenate((points[16:], points[:16])), np.arange(16, 96))
        snapshot = tracker.update(current)
        hint, _ = source.query(snapshot, frame(index), current, np.zeros((64, 64), bool), 0)
        assert hint is not None
        np.testing.assert_allclose(hint.camera_from_reference, old.camera_from_reference, atol=1e-12)
    assert not set(range(16)) & set(snapshot.groups[gid].ids)
    assert not snapshot.groups[gid].departed_ids


def test_visible_object_ids_omitted_at_group_capacity_are_not_departures():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0, max_points=80))
    source = ObjectGroupAssociation(tracker.config)
    for index in (0, 1):
        current = scene(index, points, np.arange(16, 96))
        tracker.update(current)
    own_ids = np.arange(80, 96)
    source.record_observation(
        frame(1),
        current,
        mask_for(current, own_ids),
        {"status": "initialized", "camera_from_reference": np.eye(4), "diagnostics": {"reference_frame": 0}},
        points[-16:],
    )
    _, old = source.query(tracker.frame, frame(1), current, mask_for(current, own_ids), 0)
    gid = old["group_id"]
    # The group input cap chooses the lowest 80 IDs. All 96 remain measured in
    # the shared scene, including the 16 known object IDs excluded by the cap.
    current = scene(2, np.concatenate((cloud(16), points)), np.arange(96))
    snapshot = tracker.update(current)
    assert not set(own_ids) & set(snapshot.groups[gid].ids)
    assert not snapshot.groups[gid].departed_ids
    hint, details = source.query(snapshot, frame(2), current, mask_for(current, own_ids), 0)
    assert hint is not None
    assert details["group_id"] == gid
    assert details["association_check"]["unassigned_points"] == 16


def test_new_measurements_can_reassociate_after_identity_turnover_without_moving_old_anchors():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(tracker.config)
    s = scene(0, points)
    tracker.update(s)
    initialize(source, s, points[:16])
    old = {i: p.copy() for i, p in source.anchors.items()}
    s = scene(1, points)
    tracker.update(s)
    source.query(tracker.frame, frame(1), s, mask_for(s, range(16)), 0)
    # An accepted noisy pose cannot rewrite the existing point identities.
    source.record_observation(
        frame(1),
        s,
        mask_for(s, range(16)),
        {
            "status": "observed_candidate",
            "camera_from_reference": delta(25),
            "diagnostics": {"reference_frame": 0},
        },
        points[:16],
    )
    for i, p in old.items():
        np.testing.assert_array_equal(source.anchors[i], p)
    # Actual new identities are grounded from an accepted observation, while
    # the original group memberships remain independent of this association.
    changed = np.arange(80)
    changed[:16] += 1000
    s = scene(2, points, changed)
    tracker.update(s)
    before = memberships(tracker.frame)
    source.record_observation(
        frame(2),
        s,
        mask_for(s, changed[:16]),
        {
            "status": "observed_candidate",
            "camera_from_reference": np.eye(4),
            "diagnostics": {"reference_frame": 0},
        },
        points[:16],
    )
    assert set(changed[:16]) <= source.anchors.keys()
    assert memberships(tracker.frame) == before


def test_one_bad_object_association_does_not_discard_a_measured_group():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(tracker.config)
    s = scene(0, points)
    tracker.update(s)
    initialize(source, s, points[:16])
    # One stale/incorrect object-coordinate binding; all other measured IDs
    # still establish the same rigid relationship. This must not erase support.
    source.anchors[0] = source.anchors[0] + [0.1, 0.03, 0]
    s = scene(1, move(points, delta(15, (0.02, 0, 0))))
    tracker.update(s)
    hint, _ = source.query(tracker.frame, frame(1), s, mask_for(s, range(16)), 0)
    assert hint is not None
    np.testing.assert_allclose(hint.camera_from_reference, delta(15, (0.02, 0, 0)), atol=1e-10)
    assert len(tracker.frame.groups) == 1


def test_changing_visible_subset_cannot_move_a_fixed_object_group_relationship():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(tracker.config)
    s = scene(0, points)
    tracker.update(s)
    initialize(source, s, points[:16])
    s = scene(1, points)
    tracker.update(s)
    first, _ = source.query(tracker.frame, frame(1), s, mask_for(s, range(16)), 0)
    assert first is not None
    # Newly identified surface points have a small observation-coordinate
    # error. Switching visibility to them must not redefine either frame.
    for i in range(16, 40):
        source.anchors[i] = points[i] + [0.003, 0, 0]
    s = scene(2, points[16:], np.arange(16, 80))
    tracker.update(s)
    second, _ = source.query(tracker.frame, frame(2), s, mask_for(s, range(16, 40)), 0)
    assert second is not None
    np.testing.assert_allclose(second.camera_from_reference, first.camera_from_reference, atol=1e-10)


def test_hidden_points_follow_nearest_split_neighbors_without_objects():
    """Two spatial neighborhoods split; only measured neighbors establish motion."""
    rng = np.random.default_rng(7)
    points = np.concatenate(
        (
            rng.uniform([-0.22, -0.02, 0.98], [-0.18, 0.02, 1.02], (40, 3)),
            rng.uniform([0.25, -0.05, 0.95], [0.35, 0.05, 1.05], (40, 3)),
        )
    )
    config = GroupConfig(support_confirmation_s=0)
    tracker, untouched = SceneMotionGroups(config), SceneMotionGroups(config)
    for index in range(11):
        current = scene(index, points)
        tracker.update(current)
        untouched.update(current)
    old_gid = memberships(tracker.frame)[0]
    source = ObjectGroupAssociation(config)
    initialize(source, current, points[:16])
    source.query(tracker.frame, frame(10), current, mask_for(current, range(16)), 0)
    visible = np.arange(16, 72)
    motion = delta(15, (0.06, 0.01, 0))
    for index in range(11, 16):
        xyz = points.copy()
        xyz[:40] = move(points[:40], motion)
        current = scene(index, xyz[visible], visible)
        snapshot = tracker.update(current)
        baseline = untouched.update(current)
        hint, details = source.query(snapshot, frame(index), current, np.zeros((64, 64), bool), 0)
        assert memberships(snapshot) == memberships(baseline)
        for gid in snapshot.groups:
            np.testing.assert_array_equal(snapshot.groups[gid].base_xyz, baseline.groups[gid].base_xyz)
    mapping = memberships(snapshot)
    assert mapping[0] == mapping[16] != old_gid
    assert mapping[72] == mapping[40] == old_gid
    carried = snapshot.groups[mapping[0]]
    assert not carried.measured[np.isin(carried.ids, range(16))].any()
    np.testing.assert_allclose(
        carried.base_xyz[np.isin(carried.ids, range(16))], move(points[:16], motion), atol=1e-10
    )
    assert details["group_id"] == mapping[0]
    np.testing.assert_allclose(hint.camera_from_reference, motion, atol=1e-10)

    # Once the branches rest and merge, hidden points and the object follow the
    # merged coordinate frame without an object-specific regrouping operation.
    for index in range(16, 30):
        current = scene(index, xyz[visible], visible)
        snapshot = tracker.update(current)
        hint, details = source.query(snapshot, frame(index), current, np.zeros((64, 64), bool), 0)
    assert len(snapshot.groups) == 1
    assert len(snapshot.groups[details["group_id"]].ids) == 80
    np.testing.assert_allclose(hint.camera_from_reference, motion, atol=1e-10)
