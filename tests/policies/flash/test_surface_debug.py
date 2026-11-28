"""Consumer invariants for the perception-only surface debugger."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession


def test_missing_observation_never_exposes_old_pose_as_current():
    data = np.load(Path(__file__).with_name("fixtures") / "dowel_held_frame_315.npz")
    depth, mask, k = data["depth"].astype(np.float32) * 0.001, data["mask"].astype(bool), data["intrinsics"]
    rgb = np.zeros((*depth.shape, 3), np.uint8)
    session = SurfaceTrackingSession(None)
    result = session.update(rgb, depth, mask, k, 315)
    assert result["status"] == "reference"
    reference = session.reference.copy()
    origin = result["camera_from_object"]
    lost = session.update(rgb, depth, np.zeros_like(mask), k, 316)
    assert lost["status"] == "no_mask"
    assert "camera_from_object" not in lost
    np.testing.assert_array_equal(session.reference, reference)
    no_depth = session.update(rgb, np.zeros_like(depth), mask, k, 317)
    assert no_depth["status"] == "insufficient_depth"
    assert "camera_from_object" not in no_depth
    assert origin[3] == [0, 0, 0, 1]
    assert result["diagnostics"]["reference_frame_id"] == 315


def test_filters_keep_measured_depth_and_reject_boundary_and_missing_depth():
    depth = np.ones((30, 40), np.float32)
    mask = np.zeros(depth.shape, bool)
    mask[5:25, 5:35] = True
    depth[13:16, 18:21] = 0
    k = np.array([[80, 0, 20], [0, 80, 15], [0, 0, 1]], float)
    result = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    assert not result["valid_image"][5, 20]
    assert not result["valid_image"][14, 19]
    assert not result["valid_image"][12, 19]
    assert result["valid_image"][10, 10]
    pixels = result["pixels"]
    np.testing.assert_array_equal(result["points"][:, 2], depth[pixels[:, 1], pixels[:, 0]])
