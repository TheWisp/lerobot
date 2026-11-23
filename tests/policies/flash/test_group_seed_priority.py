"""A visible object's measured motion must outrank a stale surrounding-group seed."""

from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    PointEvidence,
    PoseHint,
)
from lerobot.policies.flash.surface_correspondence import SurfaceMotionConfig, measured_pairs, robust_fit
from lerobot.policies.flash.surface_observation import move, spatial_sample
from lerobot.policies.flash.surface_session_config import defaults
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


@pytest.mark.parametrize(
    "fixture_name",
    ["group_seed_visible_move_20261108.npz", "group_seed_visible_move_again_20261108.npz"],
)
def test_visible_object_motion_seeds_one_fit_before_stale_scene_group(fixture_name):
    """Two recorded cube moves: the tray group stayed while LK followed the cube."""
    saved = np.load(Path(__file__).with_name("fixtures") / fixture_name)
    settings = defaults()
    vision = VisionPipelineConfig.from_dict(settings["config"])
    motion = SurfaceMotionConfig(**settings["config"]["motion"])
    observations = []
    for suffix in ("prior", "current"):
        frame_id = int(saved[f"frame_{suffix}"])
        prepare = SurfaceTrackingSession(
            None,
            observation_config=vision.observation,
            shape_config=vision.shape,
            motion_config=motion,
        )
        observations.append(
            prepare.prepare(
                saved[f"rgb_{suffix}"],
                saved[f"depth_{suffix}"],
                saved[f"mask_{suffix}"],
                saved["intrinsics"],
                frame_id,
            )
        )
    old, prepared = observations
    a, b = measured_pairs(
        old["current"],
        prepared["current"],
        saved["source_uv"],
        saved["target_uv"],
        saved["intrinsics"],
    )
    step, inliers, residual = robust_fit(a, b, motion)
    assert step is not None and inliers >= 12 and residual < 0.002
    prepared.update(
        step=step,
        point_pairs=(a, b),
        match_source_frame=int(saved["frame_prior"]),
        shared_matches=True,
    )

    config = GrowingSurfaceConfig(
        **{
            **settings["surface_experiment"],
            "grow": False,
            "uncertainty": PoseUncertaintyConfig(enabled=False),
        }
    )
    session = GrowingSurfaceSession(vision.shape, config)
    session.store.points = saved["surface"].copy()
    session.store.version = 1
    session.reference = saved["surface"].copy()
    session.reference_frame = 1919
    session.pose = saved["prior_pose"].copy()
    session.accepted_pose = saved["prior_pose"].copy()
    session.accepted_frame = int(saved["frame_prior"])
    session.accepted_observation = old["current"]
    session.accepted_intrinsics = saved["intrinsics"].copy()
    session.last_frame = int(saved["frame_prior"])
    session.last_time = session.last_observed_time = session.last_supported_time = float(saved["time_prior"])

    reference = saved["surface"][spatial_sample(saved["surface"], 256)]
    group_pose = saved["group_pose"]
    group = PointEvidence(reference, move(reference, group_pose), "scene_group:0", "predicted", 0.005)
    hint = PoseHint(
        group_pose,
        "scene_group:0",
        float(saved["time_current"]),
        float(saved["time_current"]),
        1919,
        (group,),
    )
    result = session.update(prepared, float(saved["time_current"]), hint=hint)

    assert result["status"] == "observed_candidate"
    assert result["diagnostics"]["hint_source"] == "scene_group:0"
    assert result["diagnostics"]["fit_seed_source"] == "object_matches"
    assert result["diagnostics"]["image_pairs"] >= 12
    assert result["diagnostics"]["image_weight"] == 1
    assert result["diagnostics"]["point_sources"][0]["source"] == "scene_group:0"
    assert result["diagnostics"]["fit"]["geometry_check"]["accepted"]
    assert result["diagnostics"]["fit"]["evaluations"] <= vision.shape.max_evaluations
