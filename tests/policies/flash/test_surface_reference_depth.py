"""A small carried patch must not certify a whole reference surface."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, observation
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession, reference_depth_support


def test_recorded_gamepad_clean_depth_separates_supported_and_unsupported_carry_frames():
    fixture = Path(__file__).parent / "fixtures" / "gamepad_reference_depth_20261028.npz"
    config = asdict(SurfaceObservationConfig())
    with np.load(fixture) as data:
        intrinsics = data["intrinsics"]
        camera_from_base = np.linalg.inv(data["base_from_camera"])
        model = data["reference_points_object_m"]
        scores = {}
        for frame_index in (290, 300, 430):
            depth = data[f"depth_{frame_index}"].astype(np.float32) / 1000
            mask = data[f"mask_{frame_index}"].astype(bool)
            clean = observation(depth, mask, intrinsics, config)["valid_image"]
            camera_from_object = camera_from_base @ data[f"pose_{frame_index}"]
            points = move(model, camera_from_object)
            scores[frame_index] = reference_depth_support(points, clean, depth, intrinsics)

    # The same saved reference is carried through all three frames. At 290 and
    # 300 a small clean patch agrees with depth. At 430 the first-view
    # reference is not supported by enough of the visible surface to certify
    # a pose, though this alone does not establish that the pose is wrong.
    for frame_index in (290, 300):
        count, p90_mm = scores[frame_index]
        assert count >= 12
        assert p90_mm is not None and p90_mm < 5
    count, p90_mm = scores[430]
    assert count >= 12
    assert p90_mm is not None and p90_mm > 100


def test_missing_reference_depth_is_unknown_not_a_perfect_fit():
    intrinsics = np.eye(3)
    points = np.array([[0.0, 0.0, 1.0]])
    count, p90_mm = reference_depth_support(
        points, np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=np.float32), intrinsics
    )
    assert count == 0
    assert p90_mm is None


def test_recorded_gamepad_grasp_sliver_has_less_lateral_extent_than_fit_error():
    fixture = Path(__file__).parent / "fixtures" / "gamepad_reference_depth_20261028.npz"
    with np.load(fixture) as data:
        intrinsics = data["intrinsics"]
        for frame_index in (223, 246, 290, 300):
            depth = data[f"depth_{frame_index}"].astype(np.float32) / 1000
            mask = data[f"mask_{frame_index}"].astype(bool)
            session = SurfaceTrackingSession(object())
            prepared = session.prepare(
                np.zeros((*depth.shape, 3), dtype=np.uint8), depth, mask, intrinsics, frame_index
            )
            assert prepared["status"] is None
            lateral_spread = prepared["diagnostics"]["visible_lateral_spread_mm"]
            fit_error = float(data[f"surface_p90_mm_{frame_index}"])
            assert (lateral_spread <= fit_error) is (frame_index in (223, 246))


def test_recorded_dowel_full_mask_retains_more_pose_leverage_than_fragment():
    fixture = Path(__file__).parent / "fixtures" / "dowel_live_visible_shape_20261020.npz"
    with np.load(fixture) as data:
        depth = data["depth"].astype(np.float32)
        if depth.max() > 10:
            depth /= 1000
        spread = {}
        for key in ("full_mask", "fragment_mask"):
            session = SurfaceTrackingSession(object())
            prepared = session.prepare(
                np.zeros((*depth.shape, 3), dtype=np.uint8),
                depth,
                data[key].astype(bool),
                data["intrinsics"],
                0,
            )
            assert prepared["status"] is None
            spread[key] = prepared["diagnostics"]["visible_lateral_spread_mm"]
    assert spread["full_mask"] > spread["fragment_mask"]


def test_reference_support_counts_depth_pixels_and_uses_front_surface():
    intrinsics = np.eye(3)
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.2], [np.nan, 0.0, 1.0]])
    count, p90_mm = reference_depth_support(
        points, np.ones((3, 3), dtype=bool), np.ones((3, 3), dtype=np.float32), intrinsics
    )
    assert count == 1
    assert p90_mm == 0
