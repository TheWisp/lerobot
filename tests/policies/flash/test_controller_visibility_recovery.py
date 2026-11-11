"""Recorded frame 614: a wrong seed must not discard the surface needed to recover.

Restore the accumulated map, last accepted view and Kalman checkpoint from the
Nov 10 controller clip. Use recorded RGB, selected depth and masks; no injected
fit and no SAM/depth/group inference. Repeated frozen observations test escape
from the local fit failure, not independent evidence or real-world pose accuracy.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def checkpoint():
    with np.load(Path(__file__).parent / "fixtures/controller_visibility_recovery_20261111.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    saved = json.loads(str(data["settings_json"]))
    meta = json.loads(str(data["metadata_json"]))
    vision = VisionPipelineConfig.from_dict(saved["config"])
    config = GrowingSurfaceConfig(**saved["surface_experiment"])

    def prepare(prefix, frame):
        builder = SurfaceTrackingSession(
            None,
            observation_config=vision.observation,
            shape_config=vision.shape,
            motion_config=vision.motion,
        )
        return builder.prepare(
            data[prefix + "_rgb"], data[prefix + "_depth"], data[prefix + "_mask"], data["intrinsics"], frame
        )

    source = prepare("source", meta["accepted_frame"])
    target = prepare("target", meta["frame"])
    session = GrowingSurfaceSession(vision.shape, config)
    session.store.points = data["surface"].copy()
    session.store.version = meta["map_version"]
    session.reference = session.store.points.copy()
    session.reference_frame = meta["reference_frame"]
    session.pose = data["pose"].copy()
    session.accepted_pose = data["accepted_pose"].copy()
    session.accepted_frame = meta["accepted_frame"]
    session.accepted_observation = source["current"]
    session.accepted_intrinsics = source["intrinsics"]
    session.last_frame = meta["last_frame"]
    session.last_time = meta["last_time"]
    session.last_observed_time = session.last_supported_time = meta["accepted_time"]
    session.kalman = PoseKalmanFilter(
        session.pose, data["center_object"], session.last_time, config.uncertainty
    )
    session.kalman.covariance = data["covariance"].copy()
    return session, target, meta["target_time"]


def test_recorded_stuck_controller_reaches_kalman_and_recovers_with_unchanged_pixels():
    pytest.importorskip("small_gicp")
    session, prepared, start = checkpoint()
    assert session.shape_config.registration.backend == "small_gicp"
    before = session.store.points.copy()
    for index in range(26):
        current = dict(prepared, frame_id=prepared["frame_id"] + index)
        geometry = session.prepare_geometry(current)
        result = session.update(current, start + index * 0.2, geometry=geometry)
        diagnostics = result["diagnostics"]
        # This is the regression assertion: the old visibility preselection
        # returned no geometry-compatible candidate and never reached correction.
        assert diagnostics["kalman"].get("observation_pose") is not None
        if index == 0:
            assert diagnostics["kalman"]["filter_reason"] == "corrected"
            assert diagnostics["kalman"]["reason"] == "fused_pose_contradicts_observation"
            assert result["camera_from_reference"] is None
            assert diagnostics["fit_seed_source"] == "previous_pose"
        pose = result["camera_from_reference"]
        if pose is None:
            np.testing.assert_array_equal(session.store.points, before)
            continue
        assert result["status"] == "observed_candidate"
        assert diagnostics["kalman"]["reason"] == "corrected"
        check = geometry.check_pose(
            np.asarray(pose),
            max_outlier_fraction=session.config.uncertainty.max_visible_outlier_fraction,
            agreement_m=session.config.agreement_m,
            min_overlap=session.config.min_overlap,
        )
        assert check["accepted"]
        assert check["surface_median_mm"] < 2
        assert check["observed_overlap"] > 0.95
        # Full-surface geometry still rejects a pose shifted into measured space.
        wrong = np.asarray(pose).copy()
        wrong[2, 3] -= 0.03
        assert not geometry.check_pose(wrong)["accepted"]
        break
    else:
        pytest.fail("The identical visible frame did not recover within five seconds")
