"""Surrounding support includes stationary agreement, and ends after separation."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_experiment_groups import GroupConfig, GroupHints
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.surface_observation import move


def fixture():
    rng = np.random.default_rng(42)
    points = rng.uniform([-0.07, -0.07, 0.96], [0.07, 0.07, 1.04], (48, 3))
    pixels = np.column_stack((np.arange(48) + 3, np.full(48, 10)))
    mask = np.zeros((64, 64), bool)
    mask[pixels[:16, 1], pixels[:16, 0]] = True
    return points, pixels, mask


def frame(index, camera=None, *, rate=30):
    return RGBDFrame(
        index,
        np.zeros((64, 64, 3), np.uint8),
        np.ones((64, 64), np.float32),
        np.array([[500.0, 0, 32.0], [0, 500.0, 32.0], [0, 0, 1.0]]),
        1.0,
        np.eye(4) if camera is None else camera,
        index / rate,
        index / rate,
        "fixture",
    )


def scene(index, points, pixels, *, rate=30):
    return SceneTracks(index, index / rate, np.arange(len(points)), pixels, points)


def observed(pose, status="observed_candidate"):
    return {"status": status, "camera_from_reference": pose, "diagnostics": {"reference_frame": 0}}


def test_own_points_measure_rotation_about_center_and_convert_camera_coordinates():
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    hints.observe(frame(0), scene(0, points, pixels), {"item": mask}, "item", observed(np.eye(4)), points)
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler("z", 22, degrees=True).as_matrix()
    center = points[:16].mean(axis=0)
    delta[:3, 3] = center - delta[:3, :3] @ center
    camera = np.eye(4)
    camera[:3, 3] = [0.2, -0.3, 0.1]
    hint, details = hints.predict(
        frame(1, camera), scene(1, move(points, delta), pixels), {"item": mask}, "item", 0
    )
    np.testing.assert_allclose(hint.camera_from_reference, np.linalg.inv(camera) @ delta, atol=1e-10)
    assert details["source"] == "object_point_group"
    assert details["object_inliers"] == 16
    assert details["ids"] == list(range(16))
    np.testing.assert_array_equal(details["pixels"], pixels[:16])
    np.testing.assert_allclose(details["camera_xyz"], move(move(points[:16], delta), np.linalg.inv(camera)))


def test_surrounding_relationship_can_be_learned_from_rotation_without_center_translation():
    points, pixels, mask = fixture()
    reference = points[:16]
    center = reference.mean(axis=0)
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    for index, angle in enumerate((0, 30, 45)):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        pose[:3, 3] = center - pose[:3, :3] @ center
        current = scene(index, move(points, pose), pixels)
        visible = mask if index < 2 else mask & False
        hint, details = hints.predict(frame(index), current, {"item": visible}, "item", 0)
        if index < 2:
            hints.observe(frame(index), current, {"item": mask}, "item", observed(pose), reference)
        else:
            # The object is hidden and only surrounding points remain. Their
            # rotation must carry its orientation even though its centre stays put.
            assert details["source"] == "environment_point_group"
            np.testing.assert_allclose(hint.camera_from_reference, pose, atol=1e-10)


@pytest.mark.parametrize("support_moves", [True, False])
def test_only_observed_comovement_can_bridge_an_empty_mask(support_moves):
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    for i in (0, 1):
        pose = np.eye(4)
        pose[0, 3] = 0.02 * i
        shifted = points.copy()
        shifted[: 32 if support_moves else 16, 0] += 0.02 * i
        current = scene(i, shifted, pixels)
        hints.predict(frame(i), current, {"item": mask}, "item", 0)
        hints.observe(frame(i), current, {"item": mask}, "item", observed(pose), points)
    shifted[: 32 if support_moves else 16, 0] += 0.02
    hidden = scene(2, shifted, pixels)
    hint, details = hints.predict(frame(2), hidden, {"item": mask & False}, "item", 0)
    if not support_moves:
        assert hint is None
        assert hints.environment is None
        return
    assert details["source"] == "environment_point_group"
    np.testing.assert_allclose(hint.camera_from_reference[:3, 3], [0.04, 0, 0], atol=1e-10)
    assert set(hints.environment.ids) == set(range(16, 32))  # not object or stationary points
    assert set(details["ids"]) == set(range(16, 32))
    old = hints.environment
    hints.observe(
        frame(2),
        hidden,
        {"item": mask & False},
        "item",
        observed(hint.camera_from_reference, "predicted"),
        points,
    )
    assert hints.environment is old
    assert hints.environment.anchor_time_s == 1 / 30
    # Separation is established by returning measured points, not a fitted pose.
    returned = shifted.copy()
    returned[:16] = points[:16]
    hints.predict(frame(3), scene(3, returned, pixels), {"item": mask}, "item", 0)
    assert hints.environment is None


def test_absent_or_inconsistent_points_do_not_return_a_stale_prediction():
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    hints.observe(frame(0), scene(0, points, pixels), {"item": mask}, "item", observed(np.eye(4)), points)
    for cloud, xy in [(points[:2], pixels[:2]), (points * 3, pixels)]:
        hint, details = hints.predict(frame(1), scene(1, cloud, xy), {"item": mask}, "item", 0)
        assert hint is None
        assert details["object_inliers"] == 0


def test_stationary_surroundings_bridge_occlusion_then_follow_the_group():
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    for i in (0, 1):
        current = scene(i, points, pixels)
        hints.predict(frame(i), current, {"item": mask}, "item", 0)
        hints.observe(frame(i), current, {"item": mask}, "item", observed(np.eye(4)), points)
    assert hints.environment is not None
    for i, translation in [(2, 0.0), (3, 0.02)]:
        delta = np.eye(4)
        delta[0, 3] = translation
        current = scene(i, move(points, delta), pixels)
        hint, details = hints.predict(frame(i), current, {"item": mask & False}, "item", 0)
        np.testing.assert_allclose(hint.camera_from_reference, delta, atol=1e-10)
        assert details["source"] == "environment_point_group"
        assert set(details["ids"]) == set(range(16, 48))
        # Hidden-object predictions do not count as a direct confirmation.
        hints.observe(frame(i), current, {"item": mask & False}, "item", observed(delta, "predicted"), points)
        assert hints.environment.anchor_time_s == 1 / 30
    # The object now moves independently of the surroundings. The old group
    # must not keep carrying it after that separation is directly observed.
    independent = delta.copy()
    independent[1, 3] = 0.03
    separated = move(points, delta)
    separated[:16] = move(points[:16], independent)
    current = scene(4, separated, pixels)
    hints.predict(frame(4), current, {"item": mask}, "item", 0)
    hints.observe(frame(4), current, {"item": mask}, "item", observed(independent), points)
    assert hints.environment is None
    hint, _ = hints.predict(frame(5), scene(5, separated, pixels), {"item": mask & False}, "item", 0)
    assert hint is None


def test_other_tracked_objects_can_supply_surrounding_points():
    points, pixels, mask = fixture()
    other = np.zeros_like(mask)
    other[pixels[16:32, 1], pixels[16:32, 0]] = True
    points[32:, 2] += 1  # Remaining background is outside the configured radius.
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    for i in (0, 1):
        delta = np.eye(4)
        delta[0, 3] = 0.02 * i
        current = scene(i, move(points, delta), pixels)
        masks = {"item": mask, "other": other}
        hints.predict(frame(i), current, masks, "item", 0)
        hints.observe(frame(i), current, masks, "item", observed(delta), points[:16])
    assert set(hints.environment.ids) == set(range(16, 32))
    delta[0, 3] = 0.04
    hint, details = hints.predict(
        frame(2), scene(2, move(points, delta), pixels), {"item": mask & False, "other": other}, "item", 0
    )
    np.testing.assert_allclose(hint.camera_from_reference, delta, atol=1e-10)
    assert set(details["ids"]) == set(range(16, 32))


@pytest.mark.parametrize("rate", [15, 30, 60])
def test_support_needs_sustained_agreement_then_separates_before_surface_fitting(rate):
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig())
    # Same physical observations at different source rates: confirmation is not a
    # frame counter or wall-clock processing delay.
    for i in range(rate + 2):
        current = scene(i, points, pixels, rate=rate)
        hints.predict(frame(i, rate=rate), current, {"item": mask}, "item", 0)
        hints.observe(frame(i, rate=rate), current, {"item": mask}, "item", observed(np.eye(4)), points)
        if i < rate:
            assert hints.environment is None
    assert hints.environment is not None
    # Object points alone show separation while all surrounding points stay
    # still. Invalidation happens before observe()/any new surface-fit result.
    shifted = points.copy()
    shifted[:16, 0] += 0.02
    index = rate + 2
    hint, details = hints.predict(
        frame(index, rate=rate), scene(index, shifted, pixels, rate=rate), {"item": mask}, "item", 0
    )
    assert hint.source == "object_point_group"
    assert details["environment_separated"]
    assert details["environment_ids"] == []
    assert hints.environment is None
    hint, _ = hints.predict(
        frame(index + 1, rate=rate),
        scene(index + 1, shifted, pixels, rate=rate),
        {"item": mask & False},
        "item",
        0,
    )
    assert hint is None


@pytest.mark.parametrize("tolerance,separates", [(0.005, True), (0.010, False)])
def test_separation_respects_configured_tolerance(tolerance, separates):
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig(support_confirmation_s=0, fit_threshold_m=tolerance))
    for i in (0, 1):
        current = scene(i, points, pixels)
        hints.predict(frame(i), current, {"item": mask}, "item", 0)
        hints.observe(frame(i), current, {"item": mask}, "item", observed(np.eye(4)), points)
    points[:16, 0] += 0.008
    _, details = hints.predict(frame(2), scene(2, points, pixels), {"item": mask}, "item", 0)
    assert details["environment_separated"] == separates
    assert (hints.environment is None) == separates


def test_slow_separation_cannot_merge_by_accumulating_small_steps():
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig())
    for i in range(60):
        pose = np.eye(4)
        pose[0, 3] = 0.001 * i  # Each step is smaller than the 5 mm fit tolerance.
        cloud = points.copy()
        cloud[:16] = move(points[:16], pose)
        current = scene(i, cloud, pixels)
        hints.predict(frame(i), current, {"item": mask}, "item", 0)
        hints.observe(frame(i), current, {"item": mask}, "item", observed(pose), points)
        assert hints.environment is None


def test_prediction_gap_cannot_mature_pending_support():
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig())
    for i in range(50):
        visible = i not in range(10, 30)
        current_mask = mask if visible else mask & False
        current = scene(i, points, pixels)
        hints.predict(frame(i), current, {"item": current_mask}, "item", 0)
        hints.observe(
            frame(i),
            current,
            {"item": current_mask},
            "item",
            observed(np.eye(4), "observed_candidate" if visible else "predicted"),
            points,
        )
        assert hints.environment is None


def test_old_group_config_migrates_without_retaining_obsolete_motion_threshold():
    config = GroupConfig.from_dict({"min_motion_m": 0.008, "min_inliers": 18})
    assert config.min_inliers == 18
    assert config.support_confirmation_s == 1
    assert not hasattr(config, "min_motion_m")


def test_rounded_recorded_camera_rotations_do_not_invalidate_rigid_group_pose():
    # First two recorded BOP clamp camera matrices. Each is rounded; composing
    # them produces a 1.01e-6 off-diagonal error, formerly rejected by PoseHint.
    before = np.array(
        [
            [0.77503856, 0.14269981, -0.6155912, 0.486281],
            [0.63056283, -0.23832217, 0.73864234, -0.613736],
            [-0.04130453, -0.96064553, -0.27469001, 0.294521],
            [0, 0, 0, 1.0],
        ]
    )
    after = np.array(
        [
            [0.77453051, 0.14194422, -0.6164044, 0.486914],
            [0.63113792, -0.23820976, 0.73818805, -0.613309],
            [-0.04205249, -0.96078511, -0.274087, 0.293853],
            [0, 0, 0, 1.0],
        ]
    )
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    hints.observe(
        frame(0, before),
        scene(0, move(points, before), pixels),
        {"item": mask},
        "item",
        observed(np.eye(4)),
        points,
    )
    hint, _ = hints.predict(frame(1, after), scene(1, move(points, after), pixels), {"item": mask}, "item", 0)
    np.testing.assert_allclose(hint.camera_from_reference, np.eye(4), atol=1e-6)
    np.testing.assert_allclose(
        hint.camera_from_reference[:3, :3].T @ hint.camera_from_reference[:3, :3], np.eye(3), atol=1e-12
    )
    bad = after.copy()
    bad[:3, :3] *= 1.1
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        hints.predict(frame(1, bad), scene(1, move(points, after), pixels), {"item": mask}, "item", 0)


def test_recorded_group_pixels_project_to_the_same_camera_points():
    import cv2

    from lerobot.policies.flash.perception import RGBDFrame
    from lerobot.policies.flash.scene_groups import ScenePointTracker

    rgb = np.random.default_rng(10).integers(0, 256, (128, 160, 3), dtype=np.uint8)
    moved = cv2.warpAffine(rgb, np.array([[1.0, 0, 4], [0, 1.0, 0]]), (160, 128))
    k = np.array([[500.0, 0, 80.0], [0, 500.0, 64.0], [0, 0, 1.0]])
    mask = np.zeros((128, 160), bool)
    mask[10:-10, 10:-10] = True
    tracker = ScenePointTracker()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    for i, image in enumerate([rgb, moved]):
        frame = RGBDFrame(
            i, image, np.full(mask.shape, 1000, np.uint16), k, 0.001, np.eye(4), i / 30, i / 30, "fixture"
        )
        scene = tracker.update(frame)
        if not i:
            hints.observe(frame, scene, {"item": mask}, "item", observed(np.eye(4)), scene.base_xyz)
        else:
            hint, details = hints.predict(frame, scene, {"item": mask}, "item", 0)
    assert details["object_inliers"] >= 12
    xyz = np.array(details["camera_xyz"])
    projection = xyz @ k.T
    np.testing.assert_allclose(projection[:, :2] / projection[:, 2, None], details["pixels"], atol=1e-8)
    before = hints.object_points.anchor_xyz
    shift = move(before, hint.camera_from_reference) - before
    np.testing.assert_allclose(np.median(shift, axis=0), [0.008, 0, 0], atol=0.002)


def test_dedicated_object_points_work_without_any_global_points_and_keep_ids():
    import cv2

    from lerobot.policies.flash.registration import unproject_pixels

    rng = np.random.default_rng(29)
    rgb = rng.integers(0, 256, (128, 160, 3), dtype=np.uint8)
    k = np.array([[500.0, 0, 80.0], [0, 500.0, 64.0], [0, 0, 1.0]])
    mask = np.zeros((128, 160), bool)
    mask[20:108, 20:140] = True
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    old_ids = None
    for i in range(3):
        image = cv2.warpAffine(rgb, np.array([[1.0, 0, 2 * i], [0, 1.0, 0]]), (160, 128))
        f = RGBDFrame(i, image, np.ones(mask.shape, np.float32), k, 1.0, np.eye(4), i / 30, i / 30, "test")
        empty = SceneTracks(i, i / 30, np.empty(0, int), np.empty((0, 2)), np.empty((0, 3)))
        reference = unproject_pixels(f.depth, k, np.array([[30, 30], [100, 30], [30, 90]]), 1.0)
        pose = np.eye(4)
        pose[0, 3] = 0.004 * i
        hint, detail = hints.predict(f, empty, {"item": mask}, "item", 0 if i else None)
        if i:
            assert detail["source"] == "object_point_group"
            assert len(detail["ids"]) >= 12
            assert all(point_id < 0 for point_id in detail["ids"])
            np.testing.assert_allclose(hint.camera_from_reference, pose, atol=0.002)
            xyz = np.asarray(detail["camera_xyz"])
            uv = xyz @ k.T
            np.testing.assert_allclose(uv[:, :2] / uv[:, 2, None], detail["pixels"], atol=1e-5)
            if old_ids is not None:
                assert set(detail["ids"]) <= old_ids
            old_ids = set(detail["ids"])
        hints.observe(f, empty, {"item": mask}, "item", observed(pose), reference)
    assert hints.point_generation == 1


def test_final_pose_wobble_cannot_rewrite_measured_group_motion_or_drop_support():
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig())
    for i in range(90):
        current = scene(i, points, pixels)
        hint, detail = hints.predict(frame(i), current, {"item": mask}, "item", 0)
        if i:
            np.testing.assert_allclose(hint.camera_from_reference, np.eye(4), atol=1e-9)
            assert hint.points[0].kind == "measured"
            np.testing.assert_allclose(hint.points[0].reference, hint.points[0].current, atol=1e-9)
        wrong_fit = np.eye(4)
        wrong_fit[:3, :3] = Rotation.from_euler("z", i / 3, degrees=True).as_matrix()
        center = points[:16].mean(axis=0)
        wrong_fit[:3, 3] = center - wrong_fit[:3, :3] @ center
        hints.observe(frame(i), current, {"item": mask}, "item", observed(wrong_fit), points[:16])
        if i >= 30:
            assert set(hints.environment.ids) == set(range(16, 48))
    hidden, detail = hints.predict(frame(90), scene(90, points, pixels), {"item": mask & False}, "item", 0)
    np.testing.assert_allclose(hidden.camera_from_reference, np.eye(4), atol=1e-9)
    assert hidden.points[0].kind == "predicted"


def test_real_lk_seed_coordinates_are_not_rewritten_by_fitted_pose():
    import cv2

    rgb = np.random.default_rng(8).integers(0, 256, (128, 160, 3), dtype=np.uint8)
    k = np.array([[500.0, 0, 80], [0, 500.0, 64], [0, 0, 1.0]])
    mask = np.ones((128, 160), bool)
    empty = SceneTracks(0, 0.0, np.empty(0, int), np.empty((0, 2)), np.empty((0, 3)))
    hints = GroupHints(GroupConfig())
    reference = np.array([[-0.02, -0.02, 1], [0.02, -0.02, 1], [0, 0.02, 1.0]])
    for i in range(5):
        image = cv2.warpAffine(rgb, np.array([[1.0, 0, 2 * i], [0, 1.0, 0]]), (160, 128))
        f = RGBDFrame(i, image, np.ones(mask.shape, np.float32), k, 1.0, np.eye(4), i / 30, i / 30, "test")
        hint, _ = hints.predict(f, empty, {"item": mask}, "item", 0 if i else None)
        if i:
            np.testing.assert_allclose(hint.camera_from_reference[:3, 3], [0.004 * i, 0, 0], atol=0.002)
            pairs = hint.points[0]
            np.testing.assert_allclose(
                np.median(pairs.current - pairs.reference, axis=0), [0.004 * i, 0, 0], atol=0.002
            )
        wrong_fit = np.eye(4)
        wrong_fit[0, 3] = 0.02 * i
        hints.observe(f, empty, {"item": mask}, "item", observed(wrong_fit), reference)
    assert hints.point_generation == 1


def test_measured_group_learning_does_not_wait_for_downstream_pose_acceptance():
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig())
    hints.observe(frame(0), scene(0, points, pixels), {"item": mask}, "item", observed(np.eye(4)), points)
    for i in range(1, 32):
        current = scene(i, points, pixels)
        hints.predict(frame(i), current, {"item": mask}, "item", 0)
        hints.observe(frame(i), current, {"item": mask}, "item", observed(None, "lost"), points)
    assert set(hints.environment.ids) == set(range(16, 48))


def test_surface_fits_alone_cannot_confirm_a_surrounding_relationship():
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    hints.observe(frame(0), scene(0, points, pixels), {"item": mask}, "item", observed(np.eye(4)), points)
    # Surrounding points remain visible but all object point identities are gone.
    for i in range(1, 10):
        current = SceneTracks(i, i / 30, np.arange(16, 48), pixels[16:], points[16:])
        hints.predict(frame(i), current, {"item": mask}, "item", 0)
        hints.observe(frame(i), current, {"item": mask}, "item", observed(np.eye(4)), points)
        assert hints.environment is None


@pytest.mark.parametrize("broad_reference", [False, True])
def test_small_observed_fragment_does_not_discard_stable_distant_support(broad_reference):
    points, pixels, mask = fixture()
    broad = points[:16].copy()
    center = broad.mean(0)
    points[:16] = center + (broad - center) * 0.05
    reference = broad if broad_reference else points[:16].copy()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    for i in (0, 1):
        current = scene(i, points, pixels)
        hints.predict(frame(i), current, {"item": mask}, "item", 0)
        hints.observe(frame(i), current, {"item": mask}, "item", observed(np.eye(4)), reference)
    assert set(hints.environment.ids) == set(range(16, 48))
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler("z", 25, degrees=True).as_matrix()
    rotation[:3, 3] = center - rotation[:3, :3] @ center
    jittered = points.copy()
    jittered[:16] = move(points[:16], rotation)
    assert np.max(np.linalg.norm(jittered[:16] - points[:16], axis=1)) < hints.config.fit_threshold_m
    for i in (2, 3):
        current = scene(i, jittered, pixels)
        _, details = hints.predict(frame(i), current, {"item": mask}, "item", 0)
        assert not details["environment_separated"]
        hints.observe(frame(i), current, {"item": mask}, "item", observed(rotation), reference)
        assert set(hints.environment.ids) == set(range(16, 48))
    hidden, _ = hints.predict(frame(4), scene(4, jittered, pixels), {"item": mask & False}, "item", 0)
    np.testing.assert_allclose(hidden.camera_from_reference, np.eye(4), atol=1e-9)
    # A real measured separation still wins immediately.
    separated = jittered.copy()
    separated[:16, 0] += 0.02
    _, details = hints.predict(frame(5), scene(5, separated, pixels), {"item": mask}, "item", 0)
    assert details["environment_separated"]
    assert hints.environment is None


@pytest.mark.parametrize("camera_pose_supplied", [False, True])
def test_background_group_accounts_for_camera_motion_without_separate_camera_pose(camera_pose_supplied):
    points, pixels, mask = fixture()
    hints = GroupHints(GroupConfig(support_confirmation_s=0))
    for i in (0, 1):
        current = scene(i, points, pixels)
        hints.predict(frame(i), current, {"item": mask}, "item", 0)
        hints.observe(frame(i), current, {"item": mask}, "item", observed(np.eye(4)), points[:16])
    camera_motion = np.eye(4)
    camera_motion[:3, :3] = Rotation.from_euler("y", 12, degrees=True).as_matrix()
    camera_motion[:3, 3] = [0.04, -0.02, 0]
    expected = np.linalg.inv(camera_motion)
    # Either compensate measured points with known camera extrinsics, or track
    # their apparent rigid motion directly in camera coordinates. Both carry the
    # hidden object correctly without requiring current pixels of that object.
    coordinates = points if camera_pose_supplied else move(points, expected)
    hint, details = hints.predict(
        frame(2, camera_motion if camera_pose_supplied else None),
        scene(2, coordinates, pixels),
        {"item": mask & False},
        "item",
        0,
    )
    assert details["source"] == "environment_point_group"
    np.testing.assert_allclose(hint.camera_from_reference, expected, atol=1e-9)
