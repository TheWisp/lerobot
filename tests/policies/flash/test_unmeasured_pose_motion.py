"""Missing motion evidence must not imply certainty that an object stayed still.

The recorded checkpoint is Nov 9 recent-history frames 118, 482 and 488:
Fast-FoundationStereo, SAM masks, shared LK, small_gicp, no Kalman reset.
RGB/depth/masks are lossless crops x=130:640, y=60:395, with shifted intrinsics.
The map and posterior precede frame 488. Its recorded fit is a comparison,
not ground truth. The test reruns the observation builder, GICP, uncertainty,
Kalman and final geometry check; it does not mock the proposed pose.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
from lerobot.policies.flash.surface_correspondence import measured_pairs, robust_fit
from lerobot.policies.flash.surface_grounding import SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_kalman import MeasuredGroupMotion, PoseKalmanFilter
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, SurfacePoseUncertainty
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def settings(**kwargs):
    return PoseUncertaintyConfig(
        enabled=True,
        mode="kalman",
        reset_on_geometry_conflict=False,
        unmeasured_rotation_std_deg_sqrt_s=10,
        **kwargs,
    )


@pytest.fixture(scope="module")
def toy_fit():
    with np.load(Path(__file__).parent / "fixtures/green_toy_motion_20261109.npz") as f:
        data = {k: f[k] for k in f.files}
    saved = json.loads(str(data["settings_json"]))["pipeline_settings"]
    vision = VisionPipelineConfig.from_dict(saved["config"])
    observations = {}
    for name, fid in (("initial", 118), ("source", 482), ("target", 488)):
        prepare = SurfaceTrackingSession(
            None,
            observation_config=vision.observation,
            shape_config=vision.shape,
            motion_config=vision.motion,
        )
        observations[name] = prepare.prepare(
            data[name + "_rgb"],
            data[name + "_depth"],
            data[name + "_mask"],
            data["intrinsics"],
            fid,
        )["current"]
    view = SurfaceFitView(observations["source"], data["intrinsics"], data["source_pose"])
    geometry = WholeSurfaceFit(
        data["surface"],
        observations["target"],
        data["intrinsics"],
        vision.shape,
        reference_views=(view,),
        appearance_views=(
            SurfaceFitView(observations["initial"], data["intrinsics"], np.eye(4)),
            view,
        ),
    )
    a, b = measured_pairs(
        observations["source"],
        observations["target"],
        data["source_uv"],
        data["target_uv"],
        data["intrinsics"],
    )
    step, _, _ = robust_fit(a, b, vision.motion)
    assert step is not None
    surface = saved["surface_experiment"]
    config = PoseUncertaintyConfig(**surface["uncertainty"])
    keep = np.linalg.norm(move(a, step) - b, axis=1) <= surface["agreement_m"]
    fitter = ImageConditionedFit(
        data["surface"],
        observations["target"],
        data["intrinsics"],
        vision.shape,
        geometry=geometry,
        image_reference=move(a[keep], np.linalg.inv(data["source_pose"])),
        image_current=b[keep],
        image_scale_m=surface["agreement_m"],
        image_weight=surface["image_weight"],
        reference_views=geometry.reference_views,
        appearance_views=geometry.appearance_views,
    )

    def check(pose):
        return geometry.check_pose(
            pose,
            max_outlier_fraction=config.max_visible_outlier_fraction,
            agreement_m=surface["agreement_m"],
            min_overlap=surface["min_overlap"],
        )

    pose, detail = fitter.correct_surface_first(
        step @ data["source_pose"],
        data["source_pose"],
        uncertainty=config,
        check=check,
    )
    assert pose is not None and check(pose)["accepted"]
    assert check(pose)["surface_median_mm"] < 2
    assert not check(data["source_pose"])["accepted"]
    return data, config, pose, detail["uncertainty"], check


def test_recorded_toy_turn_corrects_normally_without_reset(toy_fit):
    data, old_config, measured, uncertainty, check = toy_fit
    for noise in (None, 10):
        f = PoseKalmanFilter(
            data["source_pose"],
            data["center_object"],
            float(data["source_time"]),
            replace(old_config, unmeasured_rotation_std_deg_sqrt_s=noise),
        )
        f.covariance = data["source_covariance"].copy()
        f.predict(float(data["target_time"]), np.eye(4), np.eye(4))
        result = f.correct(
            measured,
            np.array(uncertainty["information"]),
            np.array(uncertainty["pivot_m"]),
            contradicted=True,
        )
        assert result["accepted"]
        assert result["reason"] == "corrected"
        # A broader prior still lets the measurement move the result farther.
        # Only that configuration was established to pass final geometry here.
        if noise == 10:
            assert result["innovation_squared"] < result["innovation_limit"]
            assert check(f.pose)["accepted"]
            rotation = Rotation.from_matrix(f.pose[:3, :3] @ data["source_pose"][:3, :3].T)
            assert rotation.magnitude() > np.deg2rad(10)
            assert (
                np.median(
                    np.linalg.norm(move(data["surface"], f.pose) - move(data["surface"], measured), axis=1)
                )
                < 0.002
            )
        else:
            assert result["innovation_squared"] > result["innovation_limit"]


def test_five_planar_points_do_not_determine_yaw_or_sideways_position():
    points = np.array(
        [[-0.01, -0.01, 0.5], [-0.01, 0.01, 0.5], [0.01, -0.01, 0.5], [0.01, 0.01, 0.5], [0, 0, 0.5]]
    )
    geometry = SimpleNamespace(
        reference=points,
        observation={"obs": {"points": points, "normals": np.tile([0, 0, 1], (5, 1))}},
        config=SimpleNamespace(depth_tolerance_m=0.005),
    )
    config = settings()
    observation = SurfacePoseUncertainty(geometry, np.eye(4), config)
    assert observation.rank == 3
    f = PoseKalmanFilter(np.eye(4), points.mean(0), 0, config)
    f.predict(0.2, np.eye(4), np.eye(4))
    before = f.covariance.copy()
    proposed = np.eye(4)
    proposed[:3, :3] = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    proposed[:3, 3] = [0.05, 0, 0.002]
    result = f.correct(proposed, observation.information, f.center(proposed))
    assert result["accepted"]
    assert 0 < f.pose[2, 3] < 0.002
    np.testing.assert_allclose(f.pose[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(f.pose[:2, 3], 0, atol=1e-12)
    assert f.covariance[2, 2] == pytest.approx(before[2, 2])


def test_large_rotation_is_fused_using_covariance_instead_of_vetoed():
    f = PoseKalmanFilter(np.eye(4), np.zeros(3), 0, settings(initial_rotation_std_deg=1))
    f.predict(0.2, np.eye(4), np.eye(4))
    proposed = np.eye(4)
    proposed[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    result = f.correct(proposed, np.eye(6) * 1e6, np.zeros(3), contradicted=True)
    assert result["accepted"] and result["reason"] == "corrected"
    assert result["innovation_squared"] > result["innovation_limit"]
    prior_variance = np.asarray(f.last_prediction["prior_covariance"])[2, 2]
    gain = prior_variance / (prior_variance + 1e-6)
    assert Rotation.from_matrix(f.pose[:3, :3]).magnitude() == pytest.approx(gain * np.pi / 2)


def test_supported_group_does_not_receive_extra_rotation_noise():
    points = np.array([[x, y, z] for x in (-0.03, 0.03) for y in (-0.02, 0.02) for z in (0.4, 0.45)])
    motion = MeasuredGroupMotion(points, points + [0.001, 0, 0], 0, 0.2, "measured")
    a = PoseKalmanFilter(np.eye(4), points.mean(0), 0, settings())
    b = PoseKalmanFilter(
        np.eye(4), points.mean(0), 0, replace(settings(), unmeasured_rotation_std_deg_sqrt_s=None)
    )
    a.predict(0.2, np.eye(4), np.eye(4), motion)
    b.predict(0.2, np.eye(4), np.eye(4), motion)
    assert a.last_prediction["motion_source"] == "measured"
    np.testing.assert_array_equal(a.pose, b.pose)
    np.testing.assert_array_equal(a.covariance, b.covariance)


def test_missing_motion_uncertainty_depends_on_elapsed_time_not_frame_rate():
    a = PoseKalmanFilter(np.eye(4), np.zeros(3), 0, settings())
    b = PoseKalmanFilter(np.eye(4), np.zeros(3), 0, settings())
    a.predict(1.0, np.eye(4), np.eye(4))
    for t in np.linspace(0.1, 1, 10):
        b.predict(t, np.eye(4), np.eye(4))
    np.testing.assert_allclose(a.covariance, b.covariance)


def test_recorded_impossible_mug_fit_remains_rejected():
    from benchmarks.flash_recovery_fixture import restore

    session, prepared, meta, _ = restore("strict_broader_motion")
    session.config = replace(
        session.config,
        uncertainty=replace(
            session.config.uncertainty,
            unmeasured_rotation_std_deg_sqrt_s=10,
            reset_on_geometry_conflict=False,
        ),
    )
    session.kalman.config = session.config.uncertainty
    result = session.update(prepared, meta["time_s"])
    assert result["status"] == "lost"
    assert result["diagnostics"]["reason"] == "no_observation_compatible_candidate"
    assert result["diagnostics"]["map_events"] == []


@pytest.mark.parametrize(
    "name", ["hidden_stationary", "partial_stationary", "group_and_camera", "corrupted_then_clear"]
)
def test_rendered_controls_with_missing_motion_noise(name):
    from benchmarks.flash_kalman_recovery_control import HIDDEN, WARMUP, run_scene
    from benchmarks.flash_recovery_matrix import scenarios

    def factory(pose, center, time_s, config):
        return PoseKalmanFilter(
            pose,
            center,
            time_s,
            replace(
                config,
                unmeasured_rotation_std_deg_sqrt_s=10,
                reset_on_geometry_conflict=False,
            ),
        )

    result = run_scene(
        scenarios()[name],
        "strict_broader_motion",
        filter_factory=factory,
        grow=name == "corrupted_then_clear",
    )
    rows = result["rows"]
    if name == "corrupted_then_clear":
        bad = rows[WARMUP : WARMUP + HIDDEN]
        assert all(not r["available"] for r in bad)
        assert all(r["map_version"] == rows[WARMUP - 1]["map_version"] for r in bad)
        rows = rows[-3:]
    for r in rows:
        assert r["available"], (name, r["index"], r["reason"])
        assert r["error"]["median_point_error_mm"] < 5
        assert r["error"]["rotation_error_deg"] < 1


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_missing_motion_noise_rejected(value):
    with pytest.raises(ValueError, match="unmeasured_rotation"):
        replace(settings(), unmeasured_rotation_std_deg_sqrt_s=value)
