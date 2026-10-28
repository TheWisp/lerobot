"""A small carried patch must not certify a whole reference surface."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_motion import measured_surface_information
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, observation
from lerobot.policies.flash.surface_reference import (
    PreparedSurfaceAgreement,
    RegisteredSurfaceModel,
    registered_surface_agreement,
)
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


def test_recorded_gamepad_surface_information_weakens_on_grasp_sliver():
    fixture = Path(__file__).parent / "fixtures" / "gamepad_reference_depth_20261028.npz"
    with np.load(fixture) as data:
        scores = {}
        for frame_index in (223, 290):
            depth = data[f"depth_{frame_index}"].astype(np.float32) / 1000
            mask = data[f"mask_{frame_index}"].astype(bool)
            measured = observation(depth, mask, data["intrinsics"], asdict(SurfaceObservationConfig()))
            camera_rotation = data["base_from_camera"][:3, :3]
            scores[frame_index] = measured_surface_information(
                data["reference_points_object_m"],
                data[f"pose_{frame_index}"],
                move(measured["points"], data["base_from_camera"]),
                measured["normals"] @ camera_rotation.T,
                correspondence_distance_m=ShapeConfig().depth_tolerance_m,
            )
    assert scores[223].matched_points < scores[290].matched_points
    assert scores[223].strength[-1] < scores[290].strength[-1]


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


def test_registered_extra_views_cannot_hide_recorded_depth_contradiction():
    fixture = Path(__file__).parent / "fixtures" / "gamepad_reference_depth_20261028.npz"
    with np.load(fixture) as data:
        intrinsics = data["intrinsics"]
        camera_from_base = np.linalg.inv(data["base_from_camera"])
        views = [data["reference_points_object_m"]]
        for frame_index in (290, 300):
            depth = data[f"depth_{frame_index}"].astype(np.float32) / 1000
            observed = observation(
                depth,
                data[f"mask_{frame_index}"].astype(bool),
                intrinsics,
                asdict(SurfaceObservationConfig()),
            )
            camera_from_object = camera_from_base @ data[f"pose_{frame_index}"]
            views.append(move(observed["points"], np.linalg.inv(camera_from_object)))
        model = RegisteredSurfaceModel("object", "recorded", tuple(views))

        def score(frame_index):
            depth = data[f"depth_{frame_index}"].astype(np.float32) / 1000
            observed = observation(
                depth,
                data[f"mask_{frame_index}"].astype(bool),
                intrinsics,
                asdict(SurfaceObservationConfig()),
            )
            camera_from_object = camera_from_base @ data[f"pose_{frame_index}"]
            return registered_surface_agreement(
                model, camera_from_object, observed["points"], observed["valid_image"], depth, intrinsics
            )

        first_223, all_223 = score(223)
        assert all_223.measured_to_reference_p90_m < first_223.measured_to_reference_p90_m
        assert all_223.reference_depth_pixels > first_223.reference_depth_pixels
        first_430, all_430 = score(430)
        # The nearest visible patch looks accurate, but the same proposed pose
        # puts saved geometry far from reliable observed depth elsewhere.
        assert all_430.measured_to_reference_p90_m < 0.005
        assert first_430.reference_to_depth_p90_m > 0.1
        assert all_430.reference_to_depth_p90_m > 0.1


def test_cached_registered_surface_distance_uses_the_moving_object_frame():
    first = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]])
    second = first + [0.0, 0.0, 0.02]
    model = RegisteredSurfaceModel("object", "v1", (first, second))
    camera_from_object = np.array(
        [[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 1.0], [0, 0, 0, 1]]
    )
    measured = move(second, camera_from_object) + [0.001, 0, 0]
    k = np.array([[100, 0, 40], [0, 100, 40], [0, 0, 1]], dtype=float)
    depth = np.ones((80, 80), np.float32)
    clean = np.ones(depth.shape, bool)
    cached = registered_surface_agreement(
        model,
        camera_from_object,
        measured,
        clean,
        depth,
        k,
        prepared=PreparedSurfaceAgreement.build(model),
    )
    direct = registered_surface_agreement(model, camera_from_object, measured, clean, depth, k)
    assert cached == direct
    expected = np.percentile(
        cKDTree(move(np.concatenate((first, second)), camera_from_object)).query(measured)[0], 90
    )
    np.testing.assert_allclose(cached[1].measured_to_reference_p90_m, expected)
    assert cached[1].measured_to_reference_p90_m < cached[0].measured_to_reference_p90_m
