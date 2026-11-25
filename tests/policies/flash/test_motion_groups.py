"""Scene motion group membership and measured-motion invariants."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.motion_groups import GroupConfig, SceneMotionGroups
from lerobot.policies.flash.scene_groups import SceneTracks
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


def test_scene_has_groups_without_any_selected_object():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0.3))
    for i in range(6):
        result = tracker.update(scene(i, move(points, delta(i * 4, (i * 0.01, 0, 0)))))
    assert len(result.groups) == 1
    group = next(iter(result.groups.values()))
    assert group.confirmed and group.supported
    np.testing.assert_allclose(group.base_xyz, move(points, delta(20, (0.05, 0, 0))), atol=1e-10)


def test_hidden_landmarks_survive_more_than_thirty_seconds_and_rejoin():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    tracker.update(scene(0, points))
    first = tracker.update(scene(1, points))
    gid = memberships(first)[0]

    displacement = np.array([0.02, 0.0, 0.0])
    hidden_ids = np.arange(16)
    for index, fraction in ((2, 0.5), (310, 1.0)):
        snapshot = tracker.update(scene(index, points[16:] + fraction * displacement, np.arange(16, 80)))
        group = snapshot.groups[gid]
        rows = {int(point_id): row for row, point_id in enumerate(group.ids)}
        assert all(int(point_id) in rows for point_id in hidden_ids)
        assert not group.measured[[rows[int(point_id)] for point_id in hidden_ids]].any()
        np.testing.assert_allclose(
            group.base_xyz[[rows[int(point_id)] for point_id in hidden_ids]],
            points[:16] + fraction * displacement,
            atol=1e-9,
        )

    returned = tracker.update(scene(311, points + displacement))
    assert all(memberships(returned)[int(point_id)] == gid for point_id in hidden_ids)
    assert returned.groups[gid].measured.sum() == len(points)


def test_balanced_scene_keeps_all_co_moving_ring_and_tray_points_together():
    """Equal-sized bodies form separate groups, even when the ring later hides."""
    rng = np.random.default_rng(91)
    ring = rng.uniform([-0.06, -0.06, 0.96], [0, 0, 1.0], (16, 3))
    tray = rng.uniform([-0.12, -0.12, 0.95], [0.12, 0.12, 1.05], (32, 3))
    background = rng.uniform([0.18, -0.12, 0.95], [0.35, 0.12, 1.05], (48, 3))
    initial = np.vstack((ring, tray, background))
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0, split_confirmation_s=0.1))
    tracker.update(scene(0, initial))
    for index in range(1, 8):
        moved = initial.copy()
        moved[:48, 0] += index * 0.01
        visible = np.arange(16, 96) if 3 <= index <= 6 else np.arange(96)
        result = tracker.update(scene(index, moved[visible], visible))
        owners = memberships(result)
        assert len(result.groups) == 2
        assert len({owners[i] for i in range(48)}) == 1
        assert len({owners[i] for i in range(48, 96)}) == 1
        assert owners[0] != owners[48]
        moving = result.groups[owners[0]]
        ring_rows = np.searchsorted(moving.ids, np.arange(16))
        np.testing.assert_allclose(moving.base_xyz[ring_rows], moved[:16], atol=1e-9)
        if 3 <= index <= 6:
            assert not moving.measured[ring_rows].any()


def test_group_settings_reject_pose_estimator_parameters():
    with pytest.raises(TypeError):
        GroupConfig.from_dict({"point_evidence_weight": 1.0})


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


def test_one_discrepant_measurement_stays_as_a_predicted_group_point():
    """A single outlier cannot establish a separately moving body."""
    points = cloud(12)
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    tracker.update(scene(0, points))
    tracker.update(scene(1, points))
    group_id = memberships(tracker.frame)[0]
    moved = points.copy()
    moved[0, 0] += 0.03
    for index in range(2, 5):
        snapshot = tracker.update(scene(index, moved))
    assert group_id in snapshot.groups
    assert snapshot.groups[group_id].supported
    assert memberships(snapshot)[0] == group_id
    assert all(i in memberships(snapshot) for i in range(1, 12))
    assert len(snapshot.groups) == 1
    assert not snapshot.groups[group_id].measured[0]
    assert not snapshot.groups[group_id].departed_ids
    for index in range(5, 15):
        snapshot = tracker.update(scene(index, moved[1:], np.arange(1, 12)))
        assert memberships(snapshot)[0] == group_id
        assert not snapshot.groups[group_id].measured[0]


def test_hidden_members_cannot_vote_or_form_group_with_four_visible_members():
    points = cloud(12)
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    tracker.update(scene(0, points))
    tracker.update(scene(1, points))
    group_id = memberships(tracker.frame)[0]
    for index in range(2, 6):
        visible = points[8:] + [0.002 * index, 0, 0]
        snapshot = tracker.update(scene(index, visible, np.arange(8, 12)))
        assert len(snapshot.groups) == 1
        group = snapshot.groups[group_id]
        assert group.supported
        assert set(group.ids) == set(range(12))
        assert group.measured.sum() == 4
        assert set(group.ids[~group.measured]) == set(range(8))


def test_three_visible_members_preserve_but_cannot_update_existing_group_motion():
    points = cloud(12)
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    tracker.update(scene(0, points))
    tracker.update(scene(1, points))
    group_id = memberships(tracker.frame)[0]
    shifted = points + [0.02, 0, 0]
    snapshot = tracker.update(scene(2, shifted[9:], np.arange(9, 12)))
    assert len(snapshot.groups) == 1
    assert not snapshot.groups[group_id].supported
    assert set(snapshot.groups[group_id].ids) == set(range(12))
    snapshot = tracker.update(scene(3, shifted[8:], np.arange(8, 12)))
    assert snapshot.groups[group_id].supported
    assert snapshot.groups[group_id].measured.sum() == 4
    np.testing.assert_allclose(snapshot.groups[group_id].base_xyz, shifted, atol=1e-10)


def test_new_group_birth_needs_twelve_measured_pairs_not_hidden_id_history():
    points = cloud(12)
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    for index in range(2):
        assert not tracker.update(scene(index, points[:11], np.arange(11))).groups
    assert not tracker.update(scene(2, points, np.arange(12))).groups
    # Birth uses a measured window, so the twelfth ID must persist until the
    # earlier eleven-only frame ages out of that window.
    assert not tracker.update(scene(3, points, np.arange(12))).groups
    snapshot = tracker.update(scene(4, points, np.arange(12)))
    assert len(snapshot.groups) == 1
    assert next(iter(snapshot.groups.values())).measured.sum() == 12


def test_hidden_point_cannot_complete_join_confirmation():
    points = cloud(13)
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0, merge_confirmation_s=0.4))
    tracker.update(scene(0, points[:12], np.arange(12)))
    tracker.update(scene(1, points[:12], np.arange(12)))
    for index in range(2, 5):
        snapshot = tracker.update(scene(index, points, np.arange(13)))
        assert 12 not in memberships(snapshot)
    snapshot = tracker.update(scene(5, points[:12], np.arange(12)))
    assert 12 not in memberships(snapshot)
    for index in range(6, 11):
        snapshot = tracker.update(scene(index, points, np.arange(13)))
        assert 12 not in memberships(snapshot)
    snapshot = tracker.update(scene(11, points, np.arange(13)))
    assert 12 in memberships(snapshot)


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


def test_memory_is_bounded_by_capacity():
    tracker = SceneMotionGroups(GroupConfig(max_points=100, max_groups=4, support_confirmation_s=0))
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


@pytest.mark.parametrize("ending", ["rejoin", "capacity"])
def test_departure_snapshot_is_immutable_and_history_clears(ending):
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig(support_confirmation_s=0, max_points=100))
    tracker.update(scene(0, points))
    tracker.update(scene(1, points))
    gid = next(iter(tracker.frame.groups))
    moved = points.copy()
    moved[:24, 0] += 0.03
    for index in range(2, 5):
        snapshot = tracker.update(scene(index, moved))
    old = snapshot.groups[gid]
    departed = old.departed_ids
    assert set(range(24)) <= departed
    for index in range(5, 25):
        if ending == "rejoin":
            measured = scene(index, points)
        else:
            ids = np.concatenate((np.arange(24, 80), np.arange(100, 144)))
            xyz = np.concatenate((points[24:], points[:44]))
            measured = scene(index, xyz, ids)
        snapshot = tracker.update(measured)
    assert old.departed_ids == departed
    assert not snapshot.groups[gid].departed_ids
    assert (0 in snapshot.groups[gid].ids) == (ending == "rejoin")


@pytest.mark.parametrize(
    "key,value",
    [
        ("min_inliers", 2),
        ("min_motion_points", 2),
        ("max_points", 0),
        ("max_groups", 0),
        ("motion_hypotheses", 0),
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


def test_old_hidden_expiry_setting_is_ignored_when_loading_recorded_settings():
    assert GroupConfig.from_dict({"max_hidden_s": 0.01}) == GroupConfig()


def test_group_motion_support_cannot_exceed_group_birth_quorum():
    with pytest.raises(ValueError):
        GroupConfig(min_inliers=3, min_motion_points=4)


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


def test_measured_branch_of_confirmed_group_keeps_confirmation():
    points = cloud()
    tracker = SceneMotionGroups(GroupConfig())
    for index in range(10):
        snapshot = tracker.update(scene(index, points))
    assert snapshot.groups[0].confirmed

    moved = points.copy()
    moved[:24, 0] += 0.03
    for index in range(10, 14):
        snapshot = tracker.update(scene(index, moved))
    assert len(snapshot.groups) == 2
    branch_id = memberships(snapshot)[0]
    assert branch_id != memberships(snapshot)[24]
    assert snapshot.groups[branch_id].confirmed


def test_independent_new_group_does_not_inherit_confirmation():
    points = cloud()
    new_points = cloud(24) + [0.5, 0, 0]
    tracker = SceneMotionGroups(GroupConfig())
    for index in range(10):
        tracker.update(scene(index, points))
    for index in range(10, 14):
        moved = new_points + [(index - 10) * 0.01, 0, 0]
        snapshot = tracker.update(scene(index, np.concatenate((points, moved))))
    new_id = memberships(snapshot)[80]
    assert new_id != memberships(snapshot)[0]
    assert not snapshot.groups[new_id].confirmed


def test_points_near_rotation_pivot_follow_rigid_branch():
    rng = np.random.default_rng(19)
    background = rng.uniform((-0.3, -0.2, 0.98), (0.3, 0.2, 1.02), (200, 3))
    near_pivot = rng.uniform((-0.003, -0.003, 0.99), (0.003, 0.003, 1.01), (12, 3))
    far_from_pivot = rng.uniform((0.14, -0.02, 0.99), (0.18, 0.02, 1.01), (28, 3))
    points = np.concatenate((background, near_pivot, far_from_pivot))
    tracker = SceneMotionGroups(GroupConfig())
    for index in range(11):
        tracker.update(scene(index, points))

    moved = points.copy()
    rotation = Rotation.from_euler("z", 5, degrees=True).as_matrix()
    moved[200:] = (points[200:] - [0, 0, 1]) @ rotation.T + [0, 0, 1]
    for index in range(11, 15):
        snapshot = tracker.update(scene(index, moved))
    mapping = memberships(snapshot)
    assert mapping[200] == mapping[212] != mapping[0]
    assert snapshot.groups[mapping[200]].confirmed


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
