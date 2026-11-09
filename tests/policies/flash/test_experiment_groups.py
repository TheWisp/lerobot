"""Scene groups from physical point motion, independent of selected objects."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.motion_groups import GroupConfig, SceneMotionGroups
from lerobot.policies.flash.object_group_association import ObjectGroupAssociation
from lerobot.policies.flash.perception import RGBDFrame
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


def test_scene_has_groups_without_any_selected_object():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0.3))
    for i in range(6):
        result = tracker.update(scene(i, move(points, delta(i * 4, (i * 0.01, 0, 0)))))
    assert len(result.groups) == 1
    group = next(iter(result.groups.values()))
    assert group.confirmed and group.supported
    np.testing.assert_allclose(group.base_xyz, move(points, delta(20, (0.05, 0, 0))), atol=1e-10)


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


def test_real_separation_uses_short_confirmation_and_not_occlusion():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0.3))
    for i in range(6):
        tracker.update(scene(i, points))
    original = memberships(tracker.frame)[0]
    moved = points.copy()
    moved[:24, 0] += 0.03
    tracker.update(scene(6, moved))
    result = tracker.update(scene(7, moved))
    assert memberships(result)[0] == original
    result = tracker.update(scene(8, moved))
    assigned = memberships(result)
    assert assigned[0] != assigned[30]
    # Hiding points is not a measured contradiction and preserves membership.
    result = tracker.update(scene(9, moved[24:], np.arange(24, 80)))
    assert memberships(result) == assigned
    assert result.groups[original].supported


def test_hidden_members_follow_visible_group_without_becoming_measurements():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    tracker.update(scene(0, points))
    tracker.update(scene(1, points))
    motion = delta(24, (0.03, -0.02, 0.01))
    result = tracker.update(scene(2, move(points[16:], motion), np.arange(16, 80)))
    group = next(iter(result.groups.values()))
    assert not group.measured[:16].any()
    np.testing.assert_allclose(group.base_xyz[:16], move(points[:16], motion), atol=1e-10)
    assert len(group.current_measured) == 64
    result = tracker.update(scene(3, np.empty((0, 3)), np.array([], int)))
    group = next(iter(result.groups.values()))
    assert not group.supported and not group.measured.any()
    assert len(group.current_measured) == 0


def test_merge_requires_sustained_relative_agreement():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(merge_confirmation_s=0.4))
    tracker.update(scene(0, points))
    moved = points.copy()
    moved[:24, 0] += 0.04
    assert len(tracker.update(scene(1, moved)).groups) == 2
    assert len(tracker.update(scene(2, moved)).groups) == 2
    for i in range(3, 8):
        result = tracker.update(scene(i, moved))
    assert len(result.groups) == 1
    np.testing.assert_allclose(next(iter(result.groups.values())).base_xyz, moved, atol=1e-10)


@pytest.mark.parametrize("rate", [15, 30, 60])
def test_confirmation_uses_elapsed_time_not_frame_count(rate):
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=1.0))
    for i in range(rate + 1):
        result = tracker.update(scene(i, points, rate=rate))
        assert not any(g.confirmed for g in result.groups.values())
    result = tracker.update(scene(rate + 2, points, rate=rate))
    assert len(result.groups) == 1
    assert next(iter(result.groups.values())).confirmed


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


def test_returning_to_same_endpoint_does_not_count_as_sustained_comovement():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0.4))
    tracker.update(scene(0, points))
    for i in range(1, 12):
        moved = points.copy()
        moved[:24, 0] += 0.04 if i % 2 else -0.04
        result = tracker.update(scene(i, moved))
    assert len(result.groups) >= 2
    assigned = memberships(result)
    assert assigned[0] != assigned[30]


def test_new_ids_join_but_do_not_reuse_a_disappeared_points_identity():
    points = cloud(96)
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0.2))
    for i in range(5):
        tracker.update(scene(i, points[:80]))
    for i in range(5, 16):
        result = tracker.update(scene(i, points))
    assert set(memberships(result)) == set(range(96))
    assert len(result.groups) == 1


def test_memory_is_bounded_and_expiration_is_same_for_all_points():
    tracker = SceneMotionGroups(
        GroupConfig(max_points=100, max_groups=4, max_hidden_s=0.4, support_confirmation_s=0)
    )
    points = cloud(40)
    for i in range(24):
        result = tracker.update(scene(i, points, np.arange(40) + 40 * (i // 3)))
        assert result.point_count <= 100 and len(result.groups) <= 4
    assert all(i >= 240 for i in memberships(result))


def test_snapshot_is_immutable_and_duplicate_frames_rejected():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    tracker.update(scene(0, points))
    result = tracker.update(scene(1, points))
    group = next(iter(result.groups.values()))
    with pytest.raises(ValueError):
        group.base_xyz[0] = 0
    with pytest.raises(TypeError):
        result.groups[100] = group
    with pytest.raises(ValueError, match="advance"):
        tracker.update(scene(1, points))


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
    source = ObjectGroupAssociation(tracker.config)
    first = scene(0, points)
    tracker.update(first)
    initialize(source, first, points[:16])
    measured = scene(1, points)
    tracker.update(measured)
    hint, details = source.query(tracker.frame, frame(1), measured, mask_for(measured, range(16)), 0)
    assert hint is not None
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
    assert visible_hint is None
    assert visible["object_inliers"] == 0

    # With no visible object measurement, an established co-moving group is
    # still allowed to carry a hidden object. The scene groups remain unchanged.
    hidden_source = ObjectGroupAssociation(tracker.config)
    hidden_source.anchors = {i: p.copy() for i, p in source.anchors.items()}
    hidden_source.reference = source.reference.copy()
    hidden_source.reference_frame = source.reference_frame
    hidden_source._group = old_group
    hidden_source._group_from_reference = old_hint.camera_from_reference.copy()
    hidden_hint, _ = hidden_source.query(tracker.frame, frame(13), current, np.zeros((64, 64), bool), 0)
    assert hidden_hint is not None
    stale_hint, _ = source.query(tracker.frame, frame(13), current, np.zeros((64, 64), bool), 0)
    assert stale_hint is None  # An observed split cannot silently regain its old relation.

    for i in range(14, 24):
        shifted = moved.copy()
        shifted[:16, 0] += (i - 13) * 0.003
        current = scene(i, shifted)
        tracker.update(current)
    new_hint, new_details = source.query(tracker.frame, frame(23), current, mask_for(current, range(16)), 0)
    assert new_hint is not None
    assert new_details["group_id"] != old_group
    assert new_details["object_inliers"] >= 12


@pytest.mark.parametrize(
    "key,value",
    [
        ("min_inliers", 2),
        ("max_points", 0),
        ("max_groups", 0),
        ("motion_hypotheses", 0),
        ("max_hidden_s", 0),
        ("support_confirmation_s", -1),
        ("split_confirmation_s", -1),
        ("merge_confirmation_s", -1),
        ("split_tolerance_m", 0),
        ("merge_tolerance_m", 0),
    ],
)
def test_bad_config_rejected(key, value):
    with pytest.raises(ValueError):
        GroupConfig(**{key: value})


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


def test_dormant_groups_cannot_starve_new_measured_motion_at_capacity():
    points = cloud(48)
    tracker = SceneMotionGroups(GroupConfig(max_groups=1, support_confirmation_s=0))
    tracker.update(scene(0, points))
    tracker.update(scene(1, points))
    old = next(iter(tracker.frame.groups))
    tracker.update(scene(2, points, np.arange(100, 148)))
    for i in range(3, 6):
        result = tracker.update(scene(i, points + 0.03, np.arange(100, 148)))
    assert len(result.groups) == 1
    group = next(iter(result.groups.values()))
    assert group.id != old and group.supported
    assert set(group.ids) == set(range(100, 148))


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


def test_recorded_controller_and_tray_do_not_create_runaway_groups():
    """Real LK/depth point tracks: one controller and a briefly moved tray.

    The saved 2026-11-07 live RGB-D sequence was reduced to the exact
    SceneTracks input of the group core. It contains no object poses, masks or
    display output, so this fails only if grouping itself fragments the scene.
    """
    path = Path(__file__).parent / "fixtures/controller_tray_group_run_20261107.npz"
    tracker = SceneMotionGroups(GroupConfig())
    counts = []
    measured_support = []
    with np.load(path) as saved:
        for row, frame_id in enumerate(saved["frame_ids"]):
            start, stop = saved["offsets"][row : row + 2]
            measured = SceneTracks(
                int(frame_id),
                float(saved["times"][row]),
                saved["ids"][start:stop].astype(np.int64),
                saved["xy"][start:stop],
                saved["xyz"][start:stop].astype(np.float64),
            )
            snapshot = tracker.update(measured)
            counts.append(len(snapshot.groups))
            measured_support.append(
                max(
                    (int(group.measured.sum()) for group in snapshot.groups.values() if group.confirmed),
                    default=0,
                )
            )
    assert max(counts) <= 5, counts
    assert counts[-1] <= 3, counts
    assert max(measured_support) >= 500  # The tracker cannot pass by returning no groups.
    assert any(group.confirmed and group.supported for group in tracker.frame.groups.values())


def test_small_noisy_rigid_group_rejoins_stationary_scene():
    """A resting small body must not stay split because its rotation is noisy.

    The two clouds share no object label. Once the body stops, its measured
    points move with the large scene, even though its compact geometry makes
    the fitted rotation noisy under submillimetre depth perturbations.
    """
    rng = np.random.default_rng(712)
    scene_points = rng.uniform((-0.3, -0.2, 0.37), (0.3, 0.2, 0.55), (500, 3))
    body = rng.uniform((-0.025, -0.015, 0.39), (0.025, 0.015, 0.41), (28, 3))
    points = np.concatenate((scene_points, body))
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=1.0))
    saw_split = False
    for index in range(180):
        measured = points.copy()
        if index:
            measured[500:, 0] += 0.04
        if index > 1:
            measured[500:] += rng.normal(0, 0.0006, (28, 3))
        saw_split |= len(tracker.update(scene(index, measured, rate=30)).groups) >= 2
    assert saw_split
    assert len(tracker.frame.groups) == 1


def test_split_tolerance_and_duration_are_independent_of_join_duration():
    points = cloud()
    slow_split = SceneMotionGroups(
        GroupConfig(split_tolerance_m=0.003, split_confirmation_s=0.5, merge_confirmation_s=0.8)
    )
    fast_split = SceneMotionGroups(
        GroupConfig(split_tolerance_m=0.003, split_confirmation_s=0.2, merge_confirmation_s=0.8)
    )
    for index in range(6):
        for tracker in (slow_split, fast_split):
            tracker.update(scene(index, points))
    # A one-frame discrepancy is not enough to split either tracker.
    moved = points.copy()
    moved[:24, 0] += 0.006
    for tracker in (slow_split, fast_split):
        assert len(tracker.update(scene(6, moved)).groups) == 1
        assert len(tracker.update(scene(7, points)).groups) == 1
    # Six millimetres of sustained measured motion is above the configured
    # three-millimetre split tolerance, but only the shorter timer has elapsed.
    for index in range(8, 11):
        for tracker in (slow_split, fast_split):
            tracker.update(scene(index, moved))
    assert len(fast_split.frame.groups) == 2
    assert len(slow_split.frame.groups) == 1
    for index in range(11, 15):
        slow_split.update(scene(index, moved))
    assert len(slow_split.frame.groups) == 2


def test_merge_waits_for_measured_co_motion_and_is_independently_configurable():
    points = cloud()
    short_join = SceneMotionGroups(GroupConfig(split_confirmation_s=0.2, merge_confirmation_s=0.4))
    long_join = SceneMotionGroups(GroupConfig(split_confirmation_s=0.2, merge_confirmation_s=0.8))
    for index in range(6):
        for tracker in (short_join, long_join):
            tracker.update(scene(index, points))
    moved = points.copy()
    moved[:24, 0] += 0.03
    for index in range(6, 14):
        for tracker in (short_join, long_join):
            tracker.update(scene(index, moved))
    assert len(short_join.frame.groups) == 1
    assert len(long_join.frame.groups) == 2
    for index in range(14, 18):
        long_join.update(scene(index, moved))
    assert len(long_join.frame.groups) == 1


def test_split_tolerance_controls_whether_sensor_scale_motion_separates():
    points = cloud()
    strict = SceneMotionGroups(GroupConfig(split_tolerance_m=0.003))
    permissive = SceneMotionGroups(GroupConfig(split_tolerance_m=0.010))
    for index in range(6):
        for tracker in (strict, permissive):
            tracker.update(scene(index, points))
    moved = points.copy()
    moved[:24, 0] += 0.006
    for index in range(6, 10):
        for tracker in (strict, permissive):
            tracker.update(scene(index, moved))
    assert len(strict.frame.groups) == 2
    assert len(permissive.frame.groups) == 1


def test_merge_tolerance_controls_when_two_measured_motions_count_as_one():
    points = cloud()
    strict = SceneMotionGroups(GroupConfig(merge_tolerance_m=0.003))
    permissive = SceneMotionGroups(GroupConfig(merge_tolerance_m=0.006))
    for index in range(20):
        measured = points.copy()
        if index >= 6:
            # The smaller body first moves away, then oscillates 4 mm between
            # consecutive frames. A 3 mm join limit must detect that motion.
            measured[:24, 0] += 0.03 if index <= 8 else (0.028 if index % 2 else 0.032)
        for tracker in (strict, permissive):
            tracker.update(scene(index, measured))
    assert len(strict.frame.groups) == 2
    assert len(permissive.frame.groups) == 1
