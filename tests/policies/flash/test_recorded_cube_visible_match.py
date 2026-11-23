"""A clearly visible cube that the live matcher lost after a tray movement.

The fixture is a cropped copy of live-history/recent frames 3004 and 3011 from
2026-11-09. It preserves the recorded Fast-FoundationStereo depth, SAM masks,
438-point accumulated surface, last accepted pose, and 22 measured LK pairs.
Both images use the same crop (original x=300:520, y=380:550); intrinsics were
shifted by that crop. No group hint was present at frame 3011. The recorded
default fit failed its current-frame free-space check despite a visible cube.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
from lerobot.policies.flash.surface_correspondence import robust_fit
from lerobot.policies.flash.surface_grounding import SurfaceFitView, WholeSurfaceFit, project
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_observation import move, spatial_sample
from lerobot.policies.flash.surface_registration import LibraryConfig, LibraryMatcher
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

FIXTURE = Path(__file__).parent / "fixtures/cube_visible_match_20261109.npz"


@pytest.fixture(scope="module")
def visible_cube():
    with np.load(FIXTURE, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    settings = json.loads(str(data["settings_json"]))
    vision = VisionPipelineConfig.from_dict(settings["config"])

    def measured_frame(name, frame_id):
        session = SurfaceTrackingSession(
            None,
            observation_config=vision.observation,
            shape_config=vision.shape,
            motion_config=vision.motion,
        )
        return session.prepare(
            data[f"{name}_rgb"],
            data[f"{name}_depth"],
            data[f"{name}_mask"],
            data["intrinsics"],
            frame_id,
        )["current"]

    source = measured_frame("source", 3004)
    target = measured_frame("target", 3011)
    source_view = SurfaceFitView(source, data["intrinsics"], data["source_pose"])
    geometry = WholeSurfaceFit(
        data["surface"],
        target,
        data["intrinsics"],
        vision.shape,
        reference_views=(source_view,),
        appearance_views=(source_view,),
    )
    surface = settings["surface_experiment"]

    def check(pose):
        return geometry.check_pose(
            pose,
            max_outlier_fraction=surface["uncertainty"]["max_visible_outlier_fraction"],
            agreement_m=surface["agreement_m"],
            min_overlap=surface["min_overlap"],
        )

    def fit(image_weight):
        matcher = ImageConditionedFit(
            data["surface"],
            target,
            data["intrinsics"],
            vision.shape,
            geometry=geometry,
            image_reference=data["image_reference"],
            image_current=data["image_current"],
            image_scale_m=surface["agreement_m"],
            image_weight=image_weight,
            reference_views=(source_view,),
            appearance_views=(source_view,),
        )
        candidates, detail = matcher.fit_once(data["seed_pose"], geometry=geometry, check=check)
        return candidates[0], detail

    return data, geometry, check, fit


def test_cube_fixture_contains_measured_motion_and_a_valid_depth_constraint(visible_cube):
    data, _, check, _ = visible_cube
    assert len(data["image_reference"]) == 22
    assert data["target_mask"].sum() > 4000
    assert (
        np.median(
            np.linalg.norm(move(data["image_reference"], data["seed_pose"]) - data["image_current"], axis=1)
        )
        < 0.002
    )
    bad = data["recorded_rejected_pose"]
    verdict = check(bad)
    assert not verdict["accepted"]
    assert verdict["scene_depth_coherent_protrusion"]


def test_fresh_measured_surface_fits_with_the_same_lk_pairs(visible_cube):
    """Distinguish a bad saved map from a bad point match or insufficient solver budget."""
    data, geometry, _, _ = visible_cube
    surface_settings = json.loads(str(data["settings_json"]))["surface_experiment"]
    source = geometry.reference_views[0].observation
    fresh = move(source["obs"]["points"], np.linalg.inv(data["source_pose"]))
    fresh = fresh[spatial_sample(fresh, len(data["surface"]))]
    fresh_geometry = WholeSurfaceFit(
        fresh,
        geometry.observation,
        data["intrinsics"],
        geometry.config,
        reference_views=geometry.reference_views,
        appearance_views=geometry.appearance_views,
    )
    fitter = ImageConditionedFit(
        fresh,
        geometry.observation,
        data["intrinsics"],
        geometry.config,
        geometry=fresh_geometry,
        image_reference=data["image_reference"],
        image_current=data["image_current"],
        image_scale_m=surface_settings["agreement_m"],
        image_weight=1.0,
        reference_views=geometry.reference_views,
        appearance_views=geometry.appearance_views,
    )

    def check(pose):
        return fresh_geometry.check_pose(
            pose,
            max_outlier_fraction=surface_settings["uncertainty"]["max_visible_outlier_fraction"],
            agreement_m=surface_settings["agreement_m"],
            min_overlap=surface_settings["min_overlap"],
        )

    candidate = fitter.fit_once(data["seed_pose"], geometry=fresh_geometry, check=check)[0][0]
    assert len(fresh) <= len(data["surface"])
    assert candidate["check"]["accepted"]
    assert candidate["check"]["scene_depth_outlier_fraction"] < 0.01
    image_error = np.linalg.norm(
        move(data["image_reference"], candidate["pose"]) - data["image_current"], axis=1
    )
    assert float(np.median(image_error)) < 0.002


def test_lk_motion_and_gicp_agree_with_consistent_source_alignment(visible_cube):
    """Do not mistake a mixed-backend absolute alignment for an LK motion error."""
    pytest.importorskip("small_gicp")
    data, geometry, _, _ = visible_cube
    settings = json.loads(str(data["settings_json"]))
    vision = VisionPipelineConfig.from_dict(settings["config"])
    shape = replace(
        geometry.config,
        registration=LibraryConfig("small_gicp", iterations_per_level=5, threads=1),
    )
    source = geometry.reference_views[0].observation
    source_fit = WholeSurfaceFit(data["surface"], source, data["intrinsics"], shape)
    source_pose, _ = LibraryMatcher(shape.registration).refine(
        source_fit, data["source_pose"], max_evaluations=40
    )

    measured_source = move(data["image_reference"], data["source_pose"])
    measured_current = data["image_current"]
    measured_motion, inliers, _ = robust_fit(measured_source, measured_current, vision.motion)
    assert inliers == len(measured_source)
    source_view = SurfaceFitView(source, data["intrinsics"], source_pose)
    target_fit = WholeSurfaceFit(
        data["surface"],
        geometry.observation,
        data["intrinsics"],
        shape,
        reference_views=(source_view,),
        appearance_views=(source_view,),
    )
    target_pose, _ = LibraryMatcher(shape.registration).refine(
        target_fit, measured_motion @ source_pose, max_evaluations=40
    )
    verdict = target_fit.check_pose(
        target_pose,
        max_outlier_fraction=settings["surface_experiment"]["uncertainty"]["max_visible_outlier_fraction"],
        agreement_m=settings["surface_experiment"]["agreement_m"],
        min_overlap=settings["surface_experiment"]["min_overlap"],
    )
    assert verdict["accepted"]

    def pixel_error(anchor):
        object_points = move(measured_source, np.linalg.inv(anchor))
        predicted = project(move(object_points, target_pose), data["intrinsics"])
        observed = project(measured_current, data["intrinsics"])
        return float(np.median(np.linalg.norm(predicted - observed, axis=1)))

    # The recorded run accepted a different 3004 pose. Its large residual is
    # an absolute frame mismatch, while the same measured LK pairs agree with
    # a GICP-to-GICP relative motion. Neither result changes the LK pixels.
    assert pixel_error(data["source_pose"]) > 10.0
    assert pixel_error(source_pose) < 5.0


def test_visible_cube_fit_recovers_when_old_pose_contradicts_current_depth(visible_cube):
    """The recorded cube's current RGB-D fit must not be fused into an impossible prior."""
    pytest.importorskip("small_gicp")
    data, geometry, _, _ = visible_cube
    settings = json.loads(str(data["settings_json"]))
    shape = replace(geometry.config, registration=LibraryConfig("small_gicp"))
    current = WholeSurfaceFit(
        data["surface"],
        geometry.observation,
        data["intrinsics"],
        shape,
        reference_views=geometry.reference_views,
        appearance_views=geometry.appearance_views,
    )
    surface = settings["surface_experiment"]
    uncertainty = PoseUncertaintyConfig(**surface["uncertainty"])

    def check(pose):
        return current.check_pose(
            pose,
            max_outlier_fraction=uncertainty.max_visible_outlier_fraction,
            agreement_m=surface["agreement_m"],
            min_overlap=surface["min_overlap"],
        )

    fitter = ImageConditionedFit(
        data["surface"],
        geometry.observation,
        data["intrinsics"],
        shape,
        geometry=current,
        image_reference=data["image_reference"],
        image_current=data["image_current"],
        image_scale_m=surface["agreement_m"],
        image_weight=1.0,
        reference_views=current.reference_views,
        appearance_views=current.appearance_views,
    )
    candidate, detail = fitter.correct_surface_first(
        data["seed_pose"], data["source_pose"], uncertainty=uncertainty, check=check
    )
    assert candidate is not None
    assert check(candidate)["accepted"]
    assert not check(data["source_pose"])["geometry_accepted"]

    information = np.asarray(detail["uncertainty"]["information"])
    pivot = np.asarray(detail["uncertainty"]["pivot_m"])
    for reset, expected in ((False, "innovation_conflict"), (True, "reinitialized_after_contradiction")):
        filter_ = PoseKalmanFilter(
            data["source_pose"],
            data["surface"].mean(0),
            0.0,
            replace(uncertainty, reset_on_geometry_conflict=reset),
        )
        filter_.covariance = np.eye(6) * 0.0001
        result = filter_.correct(candidate, information, pivot, contradicted=True)
        assert result["reason"] == expected
        assert check(filter_.pose)["accepted"] is reset


def assert_visible_match(data, candidate):
    verdict = candidate["check"]
    assert verdict["accepted"], verdict
    assert not verdict["scene_depth_coherent_protrusion"]
    assert verdict["observed_overlap"] >= 0.9
    assert verdict["surface_median_mm"] <= 2.0
    image_error = np.linalg.norm(
        move(data["image_reference"], candidate["pose"]) - data["image_current"], axis=1
    )
    assert float(np.median(image_error)) <= 0.002


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Reducing image weight accepts a pose with the same measured free-space contradiction",
)
def test_weak_image_weight_does_not_accept_the_same_depth_conflict(visible_cube):
    """A lower image weight must not turn an impossible pose into a valid one."""
    _, _, _, fit = visible_cube
    candidate, _ = fit(0.25)
    assert candidate["check"]["scene_depth_outlier_fraction"] > 0.10
    assert not candidate["check"]["accepted"], candidate["check"]


def test_recorded_matcher_recovers_the_visible_cube(visible_cube):
    data, _, _, fit = visible_cube
    candidate, detail = fit(1.0)
    assert detail["evaluations"] <= 50
    assert_visible_match(data, candidate)
