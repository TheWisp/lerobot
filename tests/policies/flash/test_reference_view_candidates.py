from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.reference_view_candidates import (
    MeasuredReferenceView,
    review_reference_view_candidates,
)
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation
from lerobot.policies.flash.surface_reference import RegisteredSurfaceModel
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


@pytest.fixture
def recorded_views():
    fixture = Path(__file__).parent / "fixtures" / "gamepad_reference_depth_20261028.npz"
    with np.load(fixture) as data:
        intrinsics = data["intrinsics"]
        base_from_camera = data["base_from_camera"]
        first_points = data["reference_points_object_m"].copy()
        measured = {}
        estimates = {}
        for index in (223, 290, 300, 430):
            depth = data[f"depth_{index}"].astype(np.float32) / 1000
            mask = data[f"mask_{index}"].astype(bool)
            observed = observation(depth, mask, intrinsics, asdict(SurfaceObservationConfig()))
            measured[index] = MeasuredReferenceView(
                index, depth, observed["valid_image"], observed["points"], int(mask.sum())
            )
            estimates[index] = {
                "object": PoseEstimate(
                    "object",
                    data[f"pose_{index}"],
                    float(index),
                    float(index),
                    Evidence.OBSERVED,
                    "recorded",
                    (True, True, True),
                    (True, True, True),
                    source_frame_id=index,
                )
            }
    model = RegisteredSurfaceModel("object", "recorded", (first_points,))
    return intrinsics, base_from_camera, model, estimates, measured


def test_candidate_review_keeps_first_reference_and_exposes_reverse_depth_conflict(recorded_views):
    intrinsics, base_from_camera, model, estimates, measured = recorded_views
    original = model.view_points_object_m[0].copy()
    summaries, details = review_reference_view_candidates(
        estimates,
        (model,),
        {"object": [290, 300]},
        lambda role, index: measured[index],
        intrinsics,
        base_from_camera,
        256,
    )
    assert len(summaries) == len(details) == 2
    assert all(item["status"] == "registration_hypothesis" for item in summaries)
    assert all(item["cross_frame"]["frames"] == 3 for item in summaries)
    assert all("combined_views" in row for item in details for row in item["comparisons"])
    # A closer patch at frame 430 can coexist with a geometrically wrong
    # projected surface. Both directions must remain visible to the reviewer.
    at_430 = next(row for row in details[0]["comparisons"] if row["frame_index"] == 430)
    assert at_430["combined_views"]["reference_to_depth_p90_m"] > 0.1
    np.testing.assert_array_equal(model.view_points_object_m[0], original)
    assert len(model.view_points_object_m) == 1


def test_candidate_review_rejects_unobserved_pose_and_misaligned_frame(recorded_views):
    intrinsics, base_from_camera, model, estimates, measured = recorded_views
    observed = estimates[290]["object"]
    estimates[290]["object"] = PoseEstimate(
        observed.role,
        observed.pose,
        observed.source_time_s,
        observed.available_time_s,
        Evidence.PREDICTED,
        observed.reference_version,
        observed.translation_supported,
        observed.rotation_supported,
        source_frame_id=290,
    )
    summaries, _ = review_reference_view_candidates(
        estimates,
        (model,),
        {"object": [290]},
        lambda role, index: measured[index],
        intrinsics,
        base_from_camera,
        256,
    )
    assert summaries[0]["status"] == "unregistered"
    assert "observed" in summaries[0]["reason"]
    with pytest.raises(ValueError, match="different source frame"):
        review_reference_view_candidates(
            estimates,
            (model,),
            {"object": [300]},
            lambda role, index: measured[290],
            intrinsics,
            base_from_camera,
            256,
        )
    estimates[300]["object"] = PoseEstimate(
        observed.role,
        observed.pose,
        300.0,
        300.0,
        Evidence.OBSERVED,
        observed.reference_version,
        observed.translation_supported,
        observed.rotation_supported,
        source_frame_id=290,
    )
    with pytest.raises(ValueError, match="pose has a different source frame"):
        review_reference_view_candidates(
            estimates,
            (model,),
            {"object": [300]},
            lambda role, index: measured[index],
            intrinsics,
            base_from_camera,
            256,
        )
