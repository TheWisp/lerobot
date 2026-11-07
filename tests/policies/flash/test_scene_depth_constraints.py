"""Scene depth can reject a fit without inventing geometry in missing pixels."""

import numpy as np
import pytest

from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


def scene(background=0.8):
    k = np.array([[500.0, 0, 60], [0, 500.0, 50], [0, 0, 1]])
    mask = np.zeros((100, 120), bool)
    mask[25:75, 50:75] = True
    source_depth = np.full(mask.shape, 0.8, np.float32)
    source_depth[mask] = 0.4
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    builder = SurfaceTrackingSession(None)
    source = builder.prepare(rgb, source_depth, mask, k, 0)["current"]
    visible = mask.copy()
    visible[:, :65] = False
    depth = np.full(mask.shape, background, np.float32)
    depth[:, :65] = 0.3  # A foreground occluder hides the left part.
    depth[visible] = 0.4
    current = SurfaceTrackingSession(None).prepare(rgb, depth, visible, k, 0)["current"]
    return source, current, k


def check(source, current, k, pose, *, enabled=True):
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(scene_depth_constraints=enabled))
    return fit, check_surface_contradiction(
        fit, pose, PoseUncertaintyConfig(max_visible_outlier_fraction=0.25)
    )


def test_background_contradiction_is_not_diluted_by_hidden_model_points():
    source, current, k = scene()
    wrong = np.eye(4)
    wrong[0, 3] = 8 * 0.4 / k[0, 0]
    _, without_search_penalty = check(source, current, k, wrong, enabled=False)
    fit, new = check(source, current, k, wrong)
    assert not without_search_penalty["accepted"]
    assert not new["accepted"]
    assert check(source, current, k, np.eye(4))[1]["accepted"]
    assert fit.score(wrong) > fit.score(np.eye(4))


def test_depth_diagnostics_identify_current_or_retained_image_conflict():
    source, _, k = scene()
    source["frame_id"] = 12
    depth = source["depth"].copy()
    depth[source["mask"]] = 0.39
    current = SurfaceTrackingSession(None).prepare(source["rgb"], depth, source["mask"], k, 19)["current"]
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(), reference_observation=source)
    result = fit.scene_depth_check(np.eye(4))
    # Current nearer depth can hide the old surface. But today's nearer
    # surface would occupy measured air in the retained image at this pose.
    current_check, retained_check = result["scene_depth_views"]
    assert current_check["frame_id"] == 19
    assert current_check["direction"] == "map_to_current"
    assert not current_check["coherent_protrusion"]
    assert retained_check["frame_id"] == 12
    assert retained_check["direction"] == "current_to_retained"
    assert retained_check["coherent_protrusion"]


def test_visible_live_controller_has_a_depth_compatible_local_fit():
    """Real last-live-frame evidence, not a replay of the unsaved filter/history.

    The group pose has a plausible silhouette but is too near the camera.
    The same bounded solver can correct it using this frame's geometry alone.
    Keep this distinction: passing this test does not prove live recovery.
    """
    import json
    from pathlib import Path

    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    with np.load(Path(__file__).parent / "fixtures/green_controller_live_loss_20261107.npz") as saved:
        config = VisionPipelineConfig.from_dict(json.loads(str(saved["settings_json"]))["config"])
        observation = SurfaceTrackingSession(
            None,
            observation_config=config.observation,
            shape_config=config.shape,
            motion_config=config.motion,
        ).prepare(saved["rgb"], saved["depth"], saved["mask"], saved["intrinsics"], 1695)
        fit = WholeSurfaceFit(saved["surface"], observation["current"], saved["intrinsics"], config.shape)

        def check(pose):
            return fit.check_pose(pose, max_outlier_fraction=0.25, agreement_m=0.005, min_overlap=0.25)

        assert check(saved["group_pose"])["scene_depth_coherent_protrusion"]
        pose, detail = fit.correct(saved["group_pose"], check=check)
        assert pose is not None
        verdict = check(pose)
        assert verdict["accepted"]
        assert not verdict["scene_depth_coherent_protrusion"]
        assert verdict["surface_median_mm"] < 3
        assert detail["numerical_solves"] == 1


def test_coherent_protrusion_cannot_hide_below_global_outlier_fraction():
    """A local strip in measured empty space is a contradiction, even if most points fit.

    A generous global outlier allowance must not override spatial evidence
    that this coherent strip occupies measured empty space.
    """
    source, current, k = scene()
    wrong = np.eye(4)
    wrong[0, 3] = 7 * 0.4 / k[0, 0]
    fit, _ = check(source, current, k, wrong)
    result = check_surface_contradiction(fit, wrong, PoseUncertaintyConfig(max_visible_outlier_fraction=0.99))
    assert result["scene_depth_outlier_fraction"] > 0
    assert result["scene_depth_outlier_fraction"] < 0.99
    assert not result["accepted"]
    assert result["scene_depth_coherent_protrusion"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("allowance", [0.25, 0.99])
def test_measured_empty_space_is_mandatory_even_with_generous_outlier_allowance(enabled, allowance):
    source, current, k = scene()
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(scene_depth_constraints=enabled))
    wrong = np.eye(4)
    wrong[0, 3] = 7 * 0.4 / k[0, 0]
    gate = PoseUncertaintyConfig(max_visible_outlier_fraction=allowance)
    assert not check_surface_contradiction(fit, wrong, gate)["accepted"]
    assert check_surface_contradiction(fit, np.eye(4), gate)["accepted"]


def test_surface_in_front_of_measured_depth_conflicts_even_inside_the_mask():
    source, _, k = scene()
    reference = source["obs"]["points"].copy()
    # A narrow saved patch floats in front of the observed face, entirely
    # inside the object's outline. Most of the cloud still agrees perfectly.
    # Keep the patch in reliable interior depth; a withheld edge is unknown.
    strip = np.abs(reference[:, 0] - np.median(reference[:, 0])) < 0.001
    reference[strip, 2] -= 0.008
    fit = WholeSurfaceFit(reference, source, k, ShapeConfig())
    assert not check_surface_contradiction(
        fit, np.eye(4), PoseUncertaintyConfig(max_visible_outlier_fraction=0.99)
    )["accepted"]


def test_isolated_wrong_map_point_does_not_reject_correct_pose():
    source, current, k = scene()
    # One incorrectly retained surface point lies in measured background.
    stray = np.array([[(85 - k[0, 2]) * 0.4 / k[0, 0], 0.0, 0.4]])
    reference = np.concatenate((source["obs"]["points"], stray))
    fit = WholeSurfaceFit(reference, current, k, ShapeConfig(scene_depth_constraints=True))
    result = check_surface_contradiction(
        fit, np.eye(4), PoseUncertaintyConfig(max_visible_outlier_fraction=0.25)
    )
    assert result["scene_depth_outlier_fraction"] > 0
    assert not result["scene_depth_coherent_protrusion"]
    assert result["accepted"]


@pytest.mark.parametrize("connected_strip", [False, True])
def test_scattered_conflicts_do_not_hide_or_invent_a_connected_patch(connected_strip):
    source, current, k = scene()
    fit, _ = check(source, current, k, np.eye(4))
    x, y = np.meshgrid(np.arange(8, 110, 12), np.arange(8, 90, 12))
    centers = np.c_[x.ravel(), y.ravel()]
    # Many disconnected two-point patches, each smaller than a sample
    # neighborhood. Counting their combined extent would invent a conflict.
    uv = np.concatenate((centers, centers + [0.5, 0]))
    if connected_strip:
        # One chain extends beyond a single neighborhood despite all the
        # disconnected specks. It must still trigger geometric rejection.
        uv = np.concatenate((uv, np.c_[np.arange(8, 19, 2), np.full(6, 8)]))
    points = np.c_[(uv - k[:2, 2]) / np.diag(k)[:2] * 0.4, np.full(len(uv), 0.4)]
    assert fit._coherent_protrusion(points, np.ones(len(points))) == connected_strip


@pytest.mark.parametrize("background", [0.0, 0.3])
def test_missing_depth_or_nearer_occluder_is_not_empty_space(background):
    source, current, k = scene(background)
    wrong = np.eye(4)
    wrong[0, 3] = 8 * 0.4 / k[0, 0]
    # The extension is unobservable in these controls. It must not receive the
    # background contradiction above; this does not establish the pose as true.
    fit, result = check(source, current, k, wrong)
    assert fit.scene_depth_check(wrong)["scene_depth_outlier_fraction"] == 0
    assert result["accepted"]


@pytest.mark.parametrize("enabled", [False, True])
def test_surface_filter_preserves_reliable_scene_readings_and_missing_depth(enabled):
    source, current, k = scene()
    depth = current["depth"].copy()
    depth[30:70, 76:79] = 0  # Unresolved thin occluder / sensor hole.
    from lerobot.policies.flash.surface_observation import SurfaceObservationConfig

    config = SurfaceObservationConfig(filtered_depth_for_fitting=True)
    filtered = SurfaceTrackingSession(
        None, observation_config=config, shape_config=ShapeConfig(scene_depth_constraints=enabled)
    ).prepare(np.zeros((*depth.shape, 3), np.uint8), depth, current["mask"], k, 0)["current"]
    np.testing.assert_array_equal(filtered["scene_depth"], filtered["depth"])
    assert filtered["depth"][35, 65] == 0  # Withheld everywhere, including validation.
    assert filtered["scene_depth"][35, 65] == 0
    assert filtered["scene_depth"][35, 100] == pytest.approx(0.8)  # Reliable background survives.
    assert filtered["scene_depth"][35, 77] == 0  # No hole filling.
    np.testing.assert_array_equal(filtered["mask"], current["mask"])
    fit, _ = check(source, filtered, k, np.eye(4))
    assert fit.scene_depth_check(np.eye(4))["scene_depth_outlier_fraction"] == 0


def test_scene_constraint_scalar_and_compiled_residuals_agree():
    from lerobot.policies.flash.surface_acceleration import compiled_kernels

    kernels = compiled_kernels()
    if kernels is None:
        pytest.skip("optional compiled CPU kernels unavailable")
    source, current, k = scene()
    config = ShapeConfig(scene_depth_constraints=True)
    a = WholeSurfaceFit(source["obs"]["points"], current, k, config)
    b = WholeSurfaceFit(source["obs"]["points"], current, k, config, fit_kernels=kernels)
    pose = np.eye(4)
    pose[0, 3] = 0.0064
    np.testing.assert_allclose(a.residual(pose), b.residual(pose), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shift", [-8, 8])
def test_real_translation_is_allowed_in_both_directions(shift):
    source, current, k = scene()
    depth = np.roll(current["depth"], shift, axis=1)
    mask = np.roll(current["mask"], shift, axis=1)
    moved = SurfaceTrackingSession(None).prepare(np.zeros((*depth.shape, 3), np.uint8), depth, mask, k, 1)[
        "current"
    ]
    truth = np.eye(4)
    truth[0, 3] = shift * 0.4 / k[0, 0]
    assert check(source, moved, k, truth)[1]["accepted"]


def test_recorded_gamepad_wrong_pose_is_rejected_without_growing_surface():
    from dataclasses import replace

    from benchmarks.flash_kalman_recovery_control import StrictPoseKalmanFilter
    from benchmarks.flash_recorded_recovery import load, replay
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    data, metadata = load("gamepad_271")
    vision = VisionPipelineConfig.from_dict(metadata["vision"])
    checks = []
    for enabled in (False, True):
        shape = replace(vision.shape, scene_depth_constraints=enabled)
        builder = SurfaceTrackingSession(None, shape_config=shape, observation_config=vision.observation)
        current = builder.prepare(
            data["rgb_271"], data["depth_271"], data["mask_271"], data["intrinsics_271"], 271
        )["current"]
        fit = WholeSurfaceFit(data["map_points"], current, data["intrinsics_271"], shape)
        checks.append(
            check_surface_contradiction(
                fit, data["recorded_pose"], PoseUncertaintyConfig(max_visible_outlier_fraction=0.25)
            )
        )
    assert not checks[0]["accepted"]
    assert not checks[1]["accepted"]
    report = replay(
        "gamepad_271", shape_changes={"scene_depth_constraints": True}, filter_factory=StrictPoseKalmanFilter
    )
    assert report["result"]["camera_from_reference"] is None
    assert not report["surface_changed"]


def test_search_does_not_reward_hiding_all_surface_outside_both_images():
    from benchmarks.flash_scene_depth_recovery import OverlapSeekingFit

    source, current, k = scene()
    kwargs = {"reference_observation": source}
    config = ShapeConfig(scene_depth_constraints=True)
    old = WholeSurfaceFit(source["obs"]["points"], current, k, config, **kwargs)
    new = OverlapSeekingFit(source["obs"]["points"], current, k, config, **kwargs)
    hidden = np.eye(4)
    hidden[0, 3] = 1
    # With consistent validity, the true alignment can now also have zero
    # residual. The old objective still cannot prefer evidence over no evidence.
    assert old.score(hidden) <= old.score(np.eye(4))
    assert new.score(np.eye(4)) < new.score(hidden)
    # No evidence still cannot establish a valid pose under either objective.
    assert not check_surface_contradiction(new, hidden, PoseUncertaintyConfig())["accepted"]


def test_known_surface_outline_still_constrains_pixels_with_missing_depth():
    from dataclasses import replace

    source, _, k = scene()
    depth = np.zeros(source["mask"].shape, np.float32)
    depth[35:65, 60:65] = 0.4
    current = SurfaceTrackingSession(None).prepare(
        np.zeros((*depth.shape, 3), np.uint8), depth, source["mask"], k, 1
    )["current"]
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig())
    wrong = np.eye(4)
    wrong[0, 3] = 8 * 0.4 / k[0, 0]
    assert np.all(fit.uncovered_mask(project_for_test(fit, wrong)) == 0)
    fit.config = replace(fit.config, require_full_mask_coverage=True)
    assert np.max(fit.uncovered_mask(project_for_test(fit, wrong))) > 0
    assert fit.score(wrong) > fit.score(np.eye(4))


def project_for_test(fit, pose):
    from lerobot.policies.flash.surface_grounding import project
    from lerobot.policies.flash.surface_observation import move

    return project(move(fit.envelope, pose), fit.k)


@pytest.mark.parametrize("background", [0.0, 0.3, 0.8])
def test_shared_overlap_objective_matches_previous_experiment(background):
    from dataclasses import replace

    from benchmarks.flash_scene_depth_recovery import OverlapSeekingFit
    from lerobot.policies.flash.surface_acceleration import compiled_kernels

    source, current, k = scene(background)
    old_config = ShapeConfig(scene_depth_constraints=True)
    old = OverlapSeekingFit(source["obs"]["points"], current, k, old_config, reference_observation=source)
    for kernels in (None, compiled_kernels()):
        new = WholeSurfaceFit(
            source["obs"]["points"],
            current,
            k,
            replace(old_config, prefer_observed_overlap=True),
            reference_observation=source,
            fit_kernels=kernels,
        )
        for translation in ((0, 0, 0), (0.0064, 0, 0), (0, 0, 0.02), (1, 0, 0)):
            pose = np.eye(4)
            pose[:3, 3] = translation
            np.testing.assert_allclose(new.residual(pose), old.residual(pose), atol=1e-12)


def test_current_partial_view_is_not_required_to_cover_earlier_outline():
    source, current, k = scene()
    fit = WholeSurfaceFit(
        source["obs"]["points"],
        current,
        k,
        ShapeConfig(require_full_mask_coverage=True),
        reference_observation=source,
    )
    assert fit.config.require_full_mask_coverage
    assert not fit.reverse_fits[0][1].config.require_full_mask_coverage
    assert check_surface_contradiction(fit, np.eye(4), PoseUncertaintyConfig())["accepted"]
