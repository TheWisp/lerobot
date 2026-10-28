"""A small carried patch must not certify a whole reference surface."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, observation
from lerobot.policies.flash.surface_tracking import reference_depth_support


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


def test_reference_support_counts_depth_pixels_and_uses_front_surface():
    intrinsics = np.eye(3)
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.2], [np.nan, 0.0, 1.0]])
    count, p90_mm = reference_depth_support(
        points, np.ones((3, 3), dtype=bool), np.ones((3, 3), dtype=np.float32), intrinsics
    )
    assert count == 1
    assert p90_mm == 0
