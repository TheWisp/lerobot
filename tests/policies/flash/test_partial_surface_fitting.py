"""Exact ray-cast observations; unseen sides are not correspondence failures."""

import numpy as np
import pytest

from benchmarks.flash_partial_surface_control import box_observation, perturbed_pose, pose_error
from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
from lerobot.policies.flash.surface_acceleration import compiled_kernels
from lerobot.policies.flash.surface_grounding import ShapeConfig, SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_registration import LibraryConfig
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


@pytest.mark.parametrize("angle", [20, 40, 60])
@pytest.mark.parametrize("perturb", [False, True])
@pytest.mark.parametrize("backend", ["visible_icp"])
def test_newly_visible_side_does_not_pull_old_surface_off_correct_alignment(angle, perturb, backend):
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(angle)
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(
        reference, current, k, ShapeConfig(registration=LibraryConfig(backend)), reference_observation=source
    )
    initial = perturbed_pose(truth) if perturb else truth
    result, _ = fit.correct(initial)
    errors = pose_error(reference, result, truth)
    assert errors["rotation_error_deg"] < 1
    assert errors["median_point_error_mm"] < (5 if perturb else 2)
    gate = PoseUncertaintyConfig(max_visible_outlier_fraction=0.25)
    assert check_surface_contradiction(fit, truth, gate)["accepted"]
    assert check_surface_contradiction(fit, result, gate)["accepted"]
    if perturb:
        assert errors["rotation_error_deg"] < pose_error(reference, initial, truth)["rotation_error_deg"] / 2


@pytest.mark.parametrize("axis,offset", [(0, 0.04), (2, -0.03), (2, 0.03)])
def test_partial_matching_still_rejects_wrong_position(axis, offset):
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(40)
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(), reference_observation=source)
    wrong = truth.copy()
    wrong[axis, 3] += offset
    gate = PoseUncertaintyConfig(max_visible_outlier_fraction=0.25)
    assert check_surface_contradiction(fit, truth, gate)["accepted"]
    assert not check_surface_contradiction(fit, wrong, gate)["accepted"]
    assert fit.score(wrong) > fit.score(truth)


@pytest.mark.parametrize("angle", [40, 60])
def test_exact_pose_is_not_rejected_for_having_a_newly_visible_side(angle):
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(angle)
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(), reference_observation=source)
    assert check_surface_contradiction(fit, truth, PoseUncertaintyConfig())["accepted"]


@pytest.mark.parametrize("source_angle,target_angle", [(40, 0), (20, -20)])
@pytest.mark.parametrize("backend", ["visible_icp"])
def test_disappearing_surface_and_different_camera_intrinsics_are_symmetric(
    source_angle, target_angle, backend
):
    source, source_k, source_pose = box_observation(source_angle)
    target_k = np.array([[540.0, 0, 117], [0, 570.0, 92], [0, 0, 1]])
    current, k, current_pose = box_observation(target_angle, k=target_k)
    truth = current_pose @ np.linalg.inv(source_pose)
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(
        reference,
        current,
        k,
        ShapeConfig(registration=LibraryConfig(backend)),
        reference_observation=source,
        reference_intrinsics=source_k,
    )
    result, _ = fit.correct(truth)
    errors = pose_error(reference, result, truth)
    assert errors["rotation_error_deg"] < 1
    assert errors["median_point_error_mm"] < 2
    gate = PoseUncertaintyConfig(max_visible_outlier_fraction=0.25)
    assert check_surface_contradiction(fit, truth, gate)["accepted"]
    assert check_surface_contradiction(fit, result, gate)["accepted"]


def test_no_visible_overlap_is_not_evidence_of_a_valid_alignment():
    source, k, _ = box_observation(0)
    current, _, _ = box_observation(40)
    # Intentionally project both clouds out of each other's cameras. A zero
    # residual caused by zero observations must not pass the acceptance check.
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(), reference_observation=source)
    wrong = np.eye(4)
    wrong[0, 3] = 1
    direct = check_surface_contradiction(fit, wrong, PoseUncertaintyConfig())
    assert not direct["has_visible_overlap"]
    assert not direct["accepted"]


@pytest.mark.parametrize("backend", ["visible_icp"])
def test_image_assisted_candidate_selection_keeps_both_measured_views(backend):
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(40)
    reference = source["obs"]["points"]
    wrong = perturbed_pose(truth)
    fit = ImageConditionedFit(
        reference,
        current,
        k,
        ShapeConfig(registration=LibraryConfig(backend)),
        reference_observation=source,
        image_reference=reference,
        image_current=move(reference, wrong),
        image_scale_m=0.005,
        image_weight=1,
    )
    result, _ = fit.correct_surface_first(
        wrong, wrong, uncertainty=PoseUncertaintyConfig(enabled=True, mode="kalman")
    )
    assert result is not None
    errors = pose_error(reference, result, truth)
    assert errors["rotation_error_deg"] < 1
    assert errors["median_point_error_mm"] < 5


def test_two_view_scalar_batched_and_compiled_paths_agree():
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(40)
    reference = source["obs"]["points"]
    config = ShapeConfig()
    scalar = WholeSurfaceFit(reference, current, k, config, reference_observation=source)
    batched = WholeSurfaceFit(reference, current, k, config, reference_observation=source)
    poses = [truth, perturbed_pose(truth)]
    for actual, expected in zip(batched.batch_residuals(poses), map(scalar.residual, poses), strict=True):
        np.testing.assert_array_equal(actual, expected)
    a, _ = scalar.correct(poses[1])
    b, _ = batched.correct(poses[1])
    np.testing.assert_array_equal(a, b)
    kernels = compiled_kernels()
    if kernels is None:
        pytest.skip("optional compiled CPU kernels unavailable")
    native = WholeSurfaceFit(reference, current, k, config, reference_observation=source, fit_kernels=kernels)
    for pose in poses:
        np.testing.assert_allclose(native.residual(pose), scalar.residual(pose), rtol=1e-12, atol=1e-12)
    c, _ = native.correct(poses[1])
    errors = pose_error(reference, c, truth)
    assert errors["rotation_error_deg"] < 1
    assert errors["median_point_error_mm"] < 5


def test_reverse_observation_uses_one_batched_query_with_identical_residuals():
    from types import SimpleNamespace

    source, k, _ = box_observation(0)
    current, _, truth = box_observation(40)
    reference = source["obs"]["points"]
    scalar = WholeSurfaceFit(reference, current, k, ShapeConfig(), reference_observation=source)
    batched = WholeSurfaceFit(reference, current, k, ShapeConfig(), reference_observation=source)
    poses = [truth, perturbed_pose(truth), np.eye(4)]
    reverse = batched.reverse_fits[0][1]
    original = reverse.target_tree.query
    queries = []

    def query(points, **kwargs):
        queries.append(len(points))
        return original(points, **kwargs)

    reverse.target_tree = SimpleNamespace(query=query)
    actual = batched.batch_residuals(poses)
    for result, pose in zip(actual, poses, strict=True):
        np.testing.assert_array_equal(result, scalar.residual(pose))
    assert queries == [len(poses) * len(reverse.reference_queries)]


def test_retained_view_pose_is_respected_and_duplicate_views_do_not_add_weight():
    source, k, _ = box_observation(0)
    current, _, truth = box_observation(40)
    points = source["obs"]["points"]
    # Change map coordinates without changing either physical observation.
    reference_from_camera = perturbed_pose(np.eye(4))
    reference = move(points, reference_from_camera)
    camera_from_reference = np.linalg.inv(reference_from_camera)
    expected = truth @ camera_from_reference
    view = SurfaceFitView(source, k, camera_from_reference)
    # Test camera transforms independently of axis-dependent spatial sampling.
    config = ShapeConfig(max_fit_points=len(points) + len(current["obs"]["points"]))
    base = WholeSurfaceFit(points, current, k, config, reference_observation=source)
    one = WholeSurfaceFit(reference, current, k, config, reference_views=(view,))
    repeated = WholeSurfaceFit(reference, current, k, config, reference_views=(view, view))
    assert one.score(expected) == pytest.approx(base.score(truth), abs=1e-10)
    assert repeated.score(expected) == pytest.approx(one.score(expected), abs=1e-10)
    gate = PoseUncertaintyConfig()
    single_check = check_surface_contradiction(one, expected, gate)
    repeated_check = check_surface_contradiction(repeated, expected, gate)
    single_counts = single_check.pop("scene_depth_exposed_weights")
    repeated_counts = repeated_check.pop("scene_depth_exposed_weights")
    assert repeated_counts == [*single_counts, single_counts[-1]]
    assert single_check == repeated_check


@pytest.mark.parametrize("backend", ["visible_icp"])
def test_partial_matching_is_used_by_sequential_session_and_its_kalman_checks(backend):
    from lerobot.policies.flash.experimental_surface_session import (
        GrowingSurfaceConfig,
        GrowingSurfaceSession,
    )

    tracker = GrowingSurfaceSession(
        ShapeConfig(registration=LibraryConfig(backend)), GrowingSurfaceConfig(partial_surface_matching=True)
    )
    previous = None
    for frame, angle in enumerate((0, 5, 10, 15, 20)):
        current, k, truth = box_observation(angle)
        prepared = {
            "frame_id": frame,
            "status": None,
            "current": current,
            "intrinsics": k,
            "diagnostics": {},
            "step": None if previous is None else truth @ np.linalg.inv(previous),
            "match_source_frame": frame - 1,
        }
        result = tracker.update(prepared, float(frame))
        assert result["camera_from_reference"] is not None
        assert tracker.connected_frame == frame
        assert len(tracker._fit_views()) <= tracker.config.max_patches + 1
        previous = truth
    assert result["diagnostics"]["kalman"]["accepted"]
    assert tracker.store.patches[0].view.depth is not None
    error = pose_error(tracker.reference, np.asarray(result["camera_from_reference"]), truth)
    assert error["rotation_error_deg"] < 2


def test_group_validation_does_not_repeat_the_old_unconditional_surface_check(monkeypatch):
    from benchmarks.flash_experiment_groups import GroupConfig, GroupHints
    from lerobot.policies.flash.experimental_surface_session import PoseHint
    from lerobot.policies.flash.perception import RGBDFrame

    source, k, _ = box_observation(0)
    current, _, truth = box_observation(40)
    frame = RGBDFrame(
        1,
        np.zeros((*current["mask"].shape, 3), np.uint8),
        current["depth"],
        k,
        1.0,
        np.eye(4),
        1.0,
        1.0,
        "controlled",
    )
    groups = GroupHints(GroupConfig(), uncertainty_config=PoseUncertaintyConfig(enabled=True))

    def old_comparison(*args):
        raise AssertionError("Geometry validation must not run a different, overwritten verdict first")

    monkeypatch.setattr(groups, "_compare", old_comparison)
    hint = PoseHint(truth, "test group", 1.0, 0.0, 0)
    accepted, details = groups.compare_candidate(
        frame,
        current["mask"],
        hint,
        source["obs"]["points"],
        truth,
        observation=current,
        reference_views=(SurfaceFitView(source, k, np.eye(4)),),
    )
    assert accepted
    assert details["observation_check"]["accepted"]


def test_retained_measurement_can_contain_surface_not_yet_added_to_map():
    source, k, _ = box_observation(0)
    seen_later, _, later_pose = box_observation(40)
    current, _, truth = box_observation(60)
    # Growth is deliberately absent. The retained 40-degree observation still
    # measured the side; it must be checked against that measurement, not the
    # incomplete stored map containing only the original front face.
    reference = source["obs"]["points"]
    fit = WholeSurfaceFit(
        reference,
        current,
        k,
        ShapeConfig(),
        reference_views=(SurfaceFitView(source, k, np.eye(4)), SurfaceFitView(seen_later, k, later_pose)),
    )
    assert check_surface_contradiction(fit, truth, PoseUncertaintyConfig())["accepted"]
    result, _ = fit.correct(truth)
    errors = pose_error(reference, result, truth)
    assert errors["rotation_error_deg"] < 1
    assert errors["median_point_error_mm"] < 2
