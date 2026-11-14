"""Live defaults execute replacement and latest-reliable checks without patches."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.gui.api.flash_tracking import TrackingStart
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.surface_refresh import LatestObservationFit, RefreshingSurfaceStore
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def test_live_start_uses_shared_refresh_store_and_boundary_settings():
    settings = TrackingStart().model_dump()["pipeline_settings"]
    vision = VisionPipelineConfig.from_dict(settings["config"])
    session = GrowingSurfaceSession(vision.shape, GrowingSurfaceConfig(**settings["surface_experiment"]))
    assert isinstance(session.store, RefreshingSurfaceStore)
    assert vision.observation.mask_margin_px == 4
    assert vision.observation.outer_mask_margin_px == 2


@pytest.mark.parametrize("role", ["ring", "dowel"])
def test_shared_refresh_accepts_recorded_reappearance_without_eligible_old_points(role):
    with np.load(Path(__file__).parent / f"fixtures/{role}_empty_refresh_20261114.npz") as f:
        data = {key: f[key] for key in f.files}
    vision = VisionPipelineConfig.from_dict(json.loads(str(data["settings_json"]))["config"])
    observation_config = replace(vision.observation, mask_margin_px=4, outer_mask_margin_px=2)

    def prepare(prefix, frame):
        builder = SurfaceTrackingSession(
            None,
            observation_config=observation_config,
            shape_config=vision.shape,
            motion_config=vision.motion,
        )
        return builder.prepare(
            data[prefix + "rgb"], data[prefix + "depth"], data[prefix + "mask"], data["intrinsics"], frame
        )

    source = prepare("source_", int(data["source_frame"]))["current"]
    prepared = prepare("", int(data["frame"]))
    # Reconstruct a real frame-session state at the recorded failure boundary.
    session = GrowingSurfaceSession(vision.shape, GrowingSurfaceConfig(refresh_observed_surface=True))
    session.store.add(data["surface"], np.eye(4), int(data["source_frame"]), active=True)
    session.pose = data["source_pose"].copy()
    session.accepted_observation, session.accepted_intrinsics = source, data["intrinsics"]
    session.accepted_pose = data["source_pose"]
    fit = session.prepare_geometry(prepared)
    assert isinstance(fit, LatestObservationFit)
    assert not fit._latest_rejection[0].any()

    def check(pose):
        return fit.check_pose(pose, agreement_m=0.005, min_overlap=0.25, max_outlier_fraction=0.25)

    pose, detail = fit.refine(data["pose"], check=check)
    assert pose is not None, detail
    assert check(pose)["accepted"]
    wrong = pose.copy()
    wrong[2, 3] -= 0.1
    assert not check(wrong)["accepted"]


@pytest.mark.parametrize("failure", ["fitted_pose", "fused_pose"])
def test_rejection_cannot_refresh_map_or_reliability_in_shared_session(monkeypatch, failure):
    from test_experimental_surface_session import prepared_frames

    from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
    from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig

    builder, rgb, depth, mask, k = prepared_frames(monkeypatch)
    session = GrowingSurfaceSession(
        builder.shape_config,
        GrowingSurfaceConfig(
            refresh_observed_surface=True,
            uncertainty=PoseUncertaintyConfig(enabled=failure == "fused_pose", mode="kalman"),
        ),
    )
    first = builder.prepare(rgb, depth, mask, k, 0)
    assert session.update(first, 0)["status"] == "initialized"
    before, version = session.store.points.copy(), session.store.version
    accepted = session.accepted_observation
    wrong = np.eye(4)
    wrong[2, 3] = -0.02  # Measured plane is 20 mm farther away.

    def proposal(*args, **kwargs):
        return (wrong if failure == "fitted_pose" else np.eye(4)).copy(), {
            "uncertainty": {
                "information": np.eye(6).tolist(),
                "covariance": np.eye(6).tolist(),
                "pivot_m": [0, 0, 1],
            }
        }

    monkeypatch.setattr(ImageConditionedFit, "correct_surface_first", proposal)
    if failure == "fused_pose":

        def incorrect_fusion(*args, **kwargs):
            session.kalman.pose = wrong.copy()
            return {"accepted": True, "reason": "test_fusion"}

        monkeypatch.setattr(session.kalman, "correct", incorrect_fusion)
    result = session.update(builder.prepare(rgb, depth, mask, k, 1), 0.1)
    assert result["status"] != "observed_candidate"
    np.testing.assert_array_equal(session.store.points, before)
    assert session.store.version == version
    assert session.accepted_frame == 0
    assert session.accepted_observation is accepted
    assert session.store.last_refinement_frame is None
    if failure == "fused_pose":
        assert result["diagnostics"]["kalman"]["reason"] == "fused_pose_contradicts_observation"


def test_accepted_refresh_does_not_require_new_surface_growth(monkeypatch):
    from test_experimental_surface_session import prepared_frames

    from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
    from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig

    builder, rgb, depth, mask, k = prepared_frames(monkeypatch)
    session = GrowingSurfaceSession(
        builder.shape_config,
        GrowingSurfaceConfig(refresh_observed_surface=True, uncertainty=PoseUncertaintyConfig(enabled=False)),
    )
    session.update(builder.prepare(rgb, depth, mask, k, 0), 0)
    before = session.store.points.copy()
    monkeypatch.setattr(ImageConditionedFit, "correct_surface_first", lambda *a, **kw: (np.eye(4), {}))
    shifted = depth.copy()
    shifted[mask] += 0.004  # Between the 3 mm growth and 5 mm pose tolerances.
    result = session.update(builder.prepare(rgb, shifted, mask, k, 1), 0.1)
    assert result["status"] == "observed_candidate"
    growth = result["diagnostics"]["growth"]
    assert not growth["alignment_accepted"]
    assert growth["admitted_points"] == 0
    assert growth["refined_points"] > 0
    assert len(session.store.patches) == 1
    assert np.median(session.store.points[:, 2]) > np.median(before[:, 2]) + 0.003
    assert session.accepted_frame == 1


def test_discarded_disabled_averaging_setting_cannot_break_saved_live_start():
    from copy import deepcopy

    from lerobot.policies.flash.surface_session_config import defaults

    settings = defaults()
    settings["surface_experiment"]["refinement_max_weight"] = 0
    before = deepcopy(settings)
    request = TrackingStart(pipeline_settings=settings)
    assert "refinement_max_weight" not in request.pipeline_settings["surface_experiment"]
    assert request.pipeline_settings["surface_experiment"]["refresh_observed_surface"]
    assert settings == before
    settings["surface_experiment"]["refinement_max_weight"] = 8
    with pytest.raises(ValueError, match="averaging experiment was removed"):
        TrackingStart(pipeline_settings=settings)
