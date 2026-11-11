"""Recorded green toy, frame 928: real fit, correction and final geometry.

The fixture restores the map and pre-frame filter state. RGB, selected model
depth and masks are recorded; LK point pairs are reconstructed from saved IDs.
The gated control reproduces the recorded fit and innovation score. No fitted
pose is supplied to the solver. This does not rerun SAM, depth or grouping.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def checkpoint(gate):
    with np.load(Path(__file__).parent / "fixtures/green_toy_kalman_recovery_20261111.npz") as archive:
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
            data[prefix + "_rgb"],
            data[prefix + "_depth"],
            data[prefix + "_mask"],
            data["intrinsics"],
            frame,
        )

    source = prepare("source", meta["accepted_frame"])
    target = prepare("target", meta["frame"])
    session = GrowingSurfaceSession(vision.shape, config)
    session.store.points = data["surface"].copy()
    session.store.version = meta["map_version"]
    session.reference = session.store.points.copy()
    session.reference_frame = meta["reference_frame"]
    session.pose = data["prior_pose"].copy()
    session.accepted_pose = data["accepted_pose"].copy()
    session.accepted_frame = meta["accepted_frame"]
    session.accepted_observation = source["current"]
    session.accepted_intrinsics = data["intrinsics"]
    session.last_frame = meta["last_frame"]
    session.last_time = meta["last_time"]
    session.last_observed_time = session.last_supported_time = meta["accepted_time"]
    from benchmarks.flash_kalman_recovery_control import StrictPoseKalmanFilter

    filter_class = StrictPoseKalmanFilter if gate else PoseKalmanFilter
    session.kalman = filter_class(
        session.pose,
        data["center_object"],
        session.last_time,
        config.uncertainty,
    )
    session.kalman.covariance = data["prior_covariance"].copy()
    target.update(
        step=data["step"],
        point_pairs=(data["source_points"], data["target_points"]),
        match_source_frame=meta["accepted_frame"],
        shared_matches=True,
    )
    return session, target, meta, data


@pytest.mark.parametrize("gate", [True, False])
def test_recorded_rotation_recovers_without_innovation_veto_and_passes_geometry(gate):
    pytest.importorskip("small_gicp")
    session, prepared, meta, data = checkpoint(gate)
    geometry = session.prepare_geometry(prepared)
    result = session.update(prepared, meta["target_time"], geometry=geometry)
    k = result["diagnostics"]["kalman"]
    # Verify both arms solve the recorded alignment before measuring the change.
    np.testing.assert_allclose(k["observation_pose"], data["recorded_fitted_pose"], atol=1e-9, rtol=0)
    assert k["innovation_squared"] == pytest.approx(meta["recorded_score"], abs=1e-7)
    assert k["innovation_squared"] > k["innovation_limit"]
    if gate is True:
        assert result["status"] == "lost"
        assert k["reason"] == "innovation_conflict"
        assert session.accepted_frame == meta["accepted_frame"]
        np.testing.assert_array_equal(session.store.points, data["surface"])
        return
    assert result["status"] == "observed_candidate"
    assert k["reason"] == "corrected"
    assert session.accepted_frame == meta["frame"]
    pose = np.asarray(result["camera_from_reference"])
    check = geometry.check_pose(
        pose,
        max_outlier_fraction=session.config.uncertainty.max_visible_outlier_fraction,
        agreement_m=session.config.agreement_m,
        min_overlap=session.config.min_overlap,
    )
    assert check["accepted"] and not check["scene_depth_coherent_protrusion"]
    assert check["surface_median_mm"] < 2
    assert check["observed_overlap"] > 0.95
    # Publishing everything would not satisfy the geometric negative control.
    shifted = pose.copy()
    shifted[2, 3] -= 0.03
    assert not geometry.check_pose(shifted)["accepted"]


def test_disabling_innovation_gate_does_not_publish_a_geometry_invalid_blend():
    session, prepared, meta, data = checkpoint(False)
    # A deliberately overconfident wrong prior makes ordinary fusion stay
    # close to that prior even though the independent surface fit is valid.
    config = replace(
        session.config.uncertainty,
        unmeasured_rotation_std_deg_sqrt_s=None,
        rotation_process_std_deg_sqrt_s=0.0001,
        position_process_std_m_sqrt_s=0.000001,
    )
    session.config = replace(session.config, uncertainty=config)
    session.kalman.config = config
    session.kalman.covariance = np.eye(6) * 1e-12
    geometry = session.prepare_geometry(prepared)
    result = session.update(prepared, meta["target_time"], geometry=geometry)
    k = result["diagnostics"]["kalman"]
    assert k["filter_reason"] == "corrected"
    assert k["reason"] == "fused_pose_contradicts_observation"
    assert not k["accepted"] and result["camera_from_reference"] is None
    np.testing.assert_allclose(k["observation_pose"], data["recorded_fitted_pose"], atol=1e-9, rtol=0)
    np.testing.assert_array_equal(session.pose, data["prior_pose"])
    np.testing.assert_array_equal(session.kalman.covariance, k["prior_covariance"])
    np.testing.assert_array_equal(session.store.points, data["surface"])
    assert session.accepted_frame == meta["accepted_frame"]
