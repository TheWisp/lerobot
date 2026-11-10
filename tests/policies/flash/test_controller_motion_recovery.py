"""Real Nov 10 controller turn: fit → Kalman → final geometry → publication.

Checkpoint immediately before frame 807 (the start of a 14-second lockout).
The archive contains lossless RGB/selected Fast-FoundationStereo depth/SAM
crops, recorded LK pairs, the accumulated map and frame 800's posterior.
No proposed pose is supplied. Small_gicp and the downstream checks run here.
This isolates the pose pipeline; it does not rerun SAM or the scene groups,
and does not restore pending growth patches or assert subsequent map evolution.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.surface_correspondence import measured_pairs, robust_fit
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_session_config import defaults
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def controller_checkpoint(noise):
    with np.load(Path(__file__).parent / "fixtures/controller_motion_recovery_20261110.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    saved = json.loads(str(data["settings_json"]))
    vision = VisionPipelineConfig.from_dict(saved["config"])
    config = GrowingSurfaceConfig(**saved["surface_experiment"])
    config = replace(
        config, uncertainty=replace(config.uncertainty, unmeasured_rotation_std_deg_sqrt_s=noise)
    )

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

    initial, source, target = prepare("initial", 213), prepare("source", 800), prepare("target", 807)
    session = GrowingSurfaceSession(vision.shape, config)
    session.update(initial, float(data["initial_time"]))
    # Restore the accepted checkpoint. fit_view_limit=1 selects frame 800;
    # the initial image remains the appearance anchor, as in the live session.
    session.store.points = data["surface"].copy()
    session.pose = data["source_pose"].copy()
    session._remember_observation(source)
    session.last_frame = 800
    session.last_time = session.last_observed_time = session.last_supported_time = float(data["source_time"])
    session.kalman = PoseKalmanFilter(
        session.pose, data["center_object"], session.last_time, config.uncertainty
    )
    session.kalman.covariance = data["source_covariance"].copy()
    pairs = measured_pairs(
        source["current"], target["current"], data["source_uv"], data["target_uv"], data["intrinsics"]
    )
    step, _, _ = robust_fit(*pairs, vision.motion)
    assert step is not None
    target.update(point_pairs=pairs, step=step, match_source_frame=800, shared_matches=True)
    return session, target, float(data["target_time"])


@pytest.mark.parametrize("noise, recovers", [(None, False), (10, True)])
def test_controller_turn_publishes_only_a_geometry_compatible_correction(noise, recovers):
    session, prepared, time_s = controller_checkpoint(noise)
    geometry = session.prepare_geometry(prepared)
    old_map = session.store.points.copy()
    result = session.update(prepared, time_s, geometry=geometry)
    diagnostics = result["diagnostics"]
    kalman = diagnostics["kalman"]
    assert diagnostics["fit_seed_source"] == "object_matches"
    assert kalman["motion_source"] is None
    assert kalman["accepted"] is recovers
    if not recovers:
        assert result["status"] == "lost"
        assert kalman["reason"] == "innovation_conflict"
        assert session.accepted_frame == 800
        np.testing.assert_array_equal(session.store.points, old_map)
        return
    assert kalman["reason"] == "corrected"  # No reset, bypass or second solve.
    assert result["status"] == "observed_candidate"
    assert session.accepted_frame == 807
    check = geometry.check_pose(
        np.asarray(result["camera_from_reference"]),
        max_outlier_fraction=session.config.uncertainty.max_visible_outlier_fraction,
        agreement_m=session.config.agreement_m,
        min_overlap=session.config.min_overlap,
    )
    assert check["accepted"]
    assert check["surface_median_mm"] < 2
    assert check["observed_overlap"] > 0.95


def test_fresh_session_recipe_recovers_the_recorded_controller_turn():
    recipe = defaults()
    noise = recipe["surface_experiment"]["uncertainty"]["unmeasured_rotation_std_deg_sqrt_s"]
    session, prepared, time_s = controller_checkpoint(noise)
    result = session.update(prepared, time_s)
    assert result["status"] == "observed_candidate"
    assert result["diagnostics"]["kalman"]["reason"] == "corrected"
