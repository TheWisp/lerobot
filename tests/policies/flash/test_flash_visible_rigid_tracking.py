import numpy as np

from lerobot.policies.flash.registration import transform_points
from lerobot.policies.flash.visible_rigid_tracking import (
    VisibleRigidTracker,
    choose_visible_surface_pose,
    fit_visible_trace,
    refine_visible_surface,
    visible_surface_cloud,
)


def _pose(angle: float, translation: tuple[float, float, float]) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[:3, 3] = translation
    return pose


def test_newly_visible_points_keep_the_same_rigid_frame_after_original_points_disappear():
    rng = np.random.default_rng(9)
    material = rng.uniform(-0.03, 0.03, (14, 3)) + [0, 0, 0.4]
    tracker = VisibleRigidTracker(14, tolerance_m=0.002, min_inliers=4, trials=120)
    first = np.zeros(14, dtype=bool)
    first[:8] = True
    initial = tracker.update(material, first)
    assert initial.pose is not None

    middle = np.ones(14, dtype=bool)
    observed_middle = transform_points(_pose(0.27, (0.012, -0.006, 0.008)), material)
    second = tracker.update(observed_middle, middle)
    assert second.new_points == 6
    assert second.pose is not None

    final = np.zeros(14, dtype=bool)
    final[8:] = True
    expected = _pose(0.51, (0.025, 0.004, 0.015))
    observed_final = transform_points(expected, material)
    observed_final[13] += [0.015, 0.0, 0.0]  # A visible but sliding point ID.
    third = tracker.update(observed_final, final)
    assert third.pose is not None
    assert third.rigid_inliers == 5
    probe = np.array([[0.019, -0.011, 0.42]])
    error = np.linalg.norm(transform_points(third.pose, probe) - transform_points(expected, probe))
    assert error < 0.001


def test_visible_tracker_does_not_attach_new_points_when_the_pose_is_unknown():
    material = np.array(
        [[0, 0, 0.4], [0.01, 0, 0.4], [0, 0.01, 0.4], [0, 0, 0.41], [0.02, 0, 0.4], [0, 0.02, 0.4]]
    )
    tracker = VisibleRigidTracker(6, tolerance_m=0.001, min_inliers=3, trials=50)
    tracker.update(material, np.array([1, 1, 1, 0, 0, 0], dtype=bool))
    result = tracker.update(material, np.array([0, 0, 0, 1, 1, 1], dtype=bool))
    assert result.pose is None
    assert result.new_points == 0
    assert np.isnan(tracker.canonical[3:]).all()


def test_surface_refinement_recovers_a_known_asymmetric_visible_motion():
    rng = np.random.default_rng(20)
    source = rng.uniform(-0.035, 0.035, (150, 3)) + [0, 0, 0.4]
    expected = _pose(0.09, (0.007, -0.004, 0.005))
    target = transform_points(expected, source)
    initial = _pose(0.09, (0.009, -0.003, 0.007))
    refined = refine_visible_surface(
        source, target, initial, max_correspondence_m=0.01, iterations=30, trim_fraction=0.9
    )
    assert np.median(np.linalg.norm(transform_points(refined, source) - target, axis=1)) < 0.0001


def test_growing_visible_tracks_follow_recorded_motion_that_fixed_reference_fit_missed():
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures/dowel_visible_growth_20261021.npz"
    with np.load(fixture, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    fitted = fit_visible_trace(
        data["points"], data["supported"], tolerance_m=0.005, min_inliers=12, trials=120
    )
    first, last = 400 - 100, 489 - 100
    assert not fitted["lost"][first] and not fitted["lost"][last]
    assert fitted["new_points"].sum() > 0
    assert fitted["inliers"][last] >= 20
    motion = fitted["poses"][last] @ np.linalg.inv(fitted["poses"][first])

    def measured_projection(pose: np.ndarray) -> tuple[float, float, float]:
        xyz = transform_points(pose, data["source_surface"])
        k, mask, depth = data["intrinsics"], data["target_mask"], data["target_depth_mm"]
        x = np.rint(k[0, 0] * xyz[:, 0] / xyz[:, 2] + k[0, 2]).astype(int)
        y = np.rint(k[1, 1] * xyz[:, 1] / xyz[:, 2] + k[1, 2]).astype(int)
        inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        x, y = np.clip(x, 0, mask.shape[1] - 1), np.clip(y, 0, mask.shape[0] - 1)
        hit = inside & mask[y, x]
        valid = inside & (depth[y, x] > 0)
        delta_m = xyz[:, 2] - depth[y, x] * 0.001
        front = valid & (delta_m < -0.01)
        return float(hit.mean()), float(front.mean()), float(np.median(abs(delta_m[hit & valid])) * 1000)

    old_mask, old_front, _ = measured_projection(data["old_fixed_reference_relative"])
    mask, front, surface_mm = measured_projection(motion)
    assert old_mask < 0.2 and old_front > 0.3  # The recorded old fit fails this independent sensor check.
    assert mask > 0.75 and front < 0.1 and surface_mm < 5


def test_visible_moving_ring_remains_aligned_with_recorded_rgbd():
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures/ring_visible_motion_20261016.npz"
    with np.load(fixture, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    fitted = fit_visible_trace(
        data["points"], data["supported"], tolerance_m=0.005, min_inliers=12, trials=120
    )
    assert not fitted["lost"].any()
    assert fitted["new_points"].sum() > data["points"].shape[1] // 2

    motion = fitted["poses"][-1] @ np.linalg.inv(fitted["poses"][0])
    moved = transform_points(motion, data["source_surface"])
    k, mask, depth = data["intrinsics"], data["target_mask"], data["target_depth_mm"]
    projected = np.column_stack(
        (k[0, 0] * moved[:, 0] / moved[:, 2] + k[0, 2], k[1, 1] * moved[:, 1] / moved[:, 2] + k[1, 2])
    )
    x, y = np.rint(projected).astype(int).T
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    x, y = np.clip(x, 0, mask.shape[1] - 1), np.clip(y, 0, mask.shape[0] - 1)
    valid = inside & (depth[y, x] > 0)
    front = valid & (moved[:, 2] - depth[y, x] * 0.001 < -0.01)
    observed_y, observed_x = np.nonzero(mask)
    observed_center = np.array([observed_x.mean(), observed_y.mean()])
    assert (inside & mask[y, x]).mean() > 0.9
    assert front.mean() < 0.05
    assert np.linalg.norm(np.median(projected, axis=0) - observed_center) < 5


def test_recorded_surface_correction_keeps_point_consensus_and_removes_foreground_conflict():
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures/dowel_visible_growth_20261021.npz"
    with np.load(fixture, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    target_surface = visible_surface_cloud(
        data["target_depth_mm"] * 0.001,
        data["target_mask"],
        data["intrinsics"],
        edge_pixels=2,
        max_points=2000,
    )
    result = choose_visible_surface_pose(
        data["source_surface"],
        target_surface,
        data["fresh_point_relative"],
        data["fresh_source_tracks"],
        data["fresh_target_tracks"],
        data["target_mask"],
        data["target_depth_mm"],
        data["intrinsics"],
        depth_m_per_unit=0.001,
        track_tolerance_m=0.005,
        surface_tolerance_m=0.01,
        min_inliers=12,
        iterations=20,
        trim_fraction=0.75,
    )
    assert result.accepted
    assert result.point_inliers >= 12
    assert result.proposal.surface_supported > result.baseline.surface_supported
    assert result.proposal.in_front_of_depth < result.baseline.in_front_of_depth
