"""Validate the projection at real geometry consumers, including its limits."""

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

pytest.importorskip("small_gicp")

from benchmarks import flash_constrained_kalman as experiment
from benchmarks.flash_motion_recovery import load_fixture, replay
from benchmarks.flash_partial_surface_control import box_observation, pose_error
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession, ImageConditionedFit
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter


@pytest.fixture(scope="module")
def recorded():
    return load_fixture()


@pytest.fixture(scope="module", autouse=True)
def one_opencv_thread():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(previous)


def test_ring_projection_passes_real_geometry_without_modifying_tracking(recorded):
    baseline = replay("ring", fixture=recorded)
    with experiment.shadow_experiment() as records:
        shadow = replay("ring", fixture=recorded)
    for old, new in zip(baseline, shadow, strict=True):
        for key in ("status", "accepted_frame", "map_version"):
            assert old[key] == new[key], key
        for key in ("pose", "reference", "surface"):
            if old[key] is None:
                assert new[key] is None
            else:
                # Parallel native reductions vary at floating-point roundoff.
                np.testing.assert_allclose(old[key], new[key], rtol=0, atol=1e-10)
    ring = next(r for r in records if r["frame"] == 4759)
    assert not ring["fused_check"]["accepted"]
    for name in ("qp1", "qp3"):
        proposal = ring["methods"][name][0]
        assert proposal["geometry"]["accepted"]
        assert not np.allclose(proposal["pose"], ring["fitted"])
        assert not np.allclose(proposal["pose"], ring["fused"])
        assert proposal["weighted_distance_squared"] < proposal["fitted_distance_squared"]
        assert proposal["feature_calls"] <= 1 + 7 * int(name[-1])
        assert proposal["qp_iterations"] <= 20 * int(name[-1])
    for row in records:
        if row["fused_check"]["accepted"]:
            for name in ("qp1", "qp3"):
                proposal = row["methods"][name][0]
                np.testing.assert_array_equal(proposal["pose"], row["fused"])
                assert proposal["feature_calls"] == 0


def test_bad_solver_output_is_rejected_by_actual_geometry(monkeypatch, recorded):
    def ignore_constraints(fun, x0, *, jac, **kwargs):
        # Fault injection: solve only the quadratic objective, ignoring geometry.
        return OptimizeResult(x=x0 - jac(x0), nit=1, success=True, message="ignored constraints")

    monkeypatch.setattr(experiment, "minimize", ignore_constraints)
    with experiment.shadow_experiment(methods=("qp1", "qp3")) as records:
        replay("ring", fixture=recorded)
    ring = next(r for r in records if r["frame"] == 4759)
    assert not ring["fused_check"]["accepted"]
    for name in ("qp1", "qp3"):
        assert ring["methods"][name][0]["pose"] is None


def test_missing_and_wrong_object_do_not_gain_publications(recorded):
    with experiment.shadow_experiment() as records:
        rows = replay("cube", fixture=recorded, missing=True)
    assert rows[1]["status"] != "observed_candidate"
    assert rows[1]["map_version"] == rows[0]["map_version"]
    assert rows[1]["frame"] not in {r["frame"] for r in records}
    meta, frames = recorded
    meta = deepcopy(meta)
    meta["cases"]["wrong_object"] = [2735, 4759, 4765]
    with experiment.shadow_experiment() as records:
        rows = replay("wrong_object", fixture=(meta, frames))
    assert not records, "Projection cannot rescue an object rejected before Kalman"
    assert all(r["status"] == "lost" and r["pose"] is None for r in rows[1:])


def test_symmetric_geometry_does_not_establish_correct_material_rotation():
    source, k, initial = box_observation(20)
    current, _, truth = box_observation(200)
    np.testing.assert_array_equal(source["depth"], current["depth"])
    np.testing.assert_array_equal(source["mask"], current["mask"])
    reference = source["obs"]["points"]
    geometry = WholeSurfaceFit(reference, current, k, ShapeConfig(), reference_observation=source)

    def check(pose):
        return geometry.check_pose(pose, max_outlier_fraction=0.25, min_overlap=0.5)

    result, detail = experiment.constrained_proposal(
        np.eye(4), np.eye(4), np.eye(6), reference.mean(0), geometry, check
    )
    assert check(result)["accepted"]
    assert detail["reason"] == "already_compatible"
    assert pose_error(reference, result, truth @ np.linalg.inv(initial))["rotation_error_deg"] > 179
    wrong = np.eye(4)
    wrong[0, 3] = 0.4
    rejected, detail = experiment.constrained_proposal(
        wrong, wrong, np.eye(6), reference.mean(0), geometry, check
    )
    assert rejected is None and detail["reason"] == "no_compatible_fit"
    assert detail["feature_calls"] == 0


def test_experiment_restores_pipeline_on_exception():
    originals = (
        GrowingSurfaceSession.update,
        ImageConditionedFit.correct_surface_first,
        PoseKalmanFilter.correct,
    )
    with pytest.raises(RuntimeError, match="sentinel"), experiment.shadow_experiment():
        raise RuntimeError("sentinel")
    assert originals == (
        GrowingSurfaceSession.update,
        ImageConditionedFit.correct_surface_first,
        PoseKalmanFilter.correct,
    )


def test_visible_strip_can_pass_geometry_with_wrong_rotation():
    from benchmarks.flash_constrained_controls import changed_scene, no_reset
    from benchmarks.flash_kalman_recovery_control import WARMUP, run_scene

    frames = changed_scene(45, strip=20)[: WARMUP + 1]
    with experiment.shadow_experiment(truth={f["index"]: f["truth"] for f in frames}) as records:
        run_scene(frames, "strict_broader_motion", filter_factory=no_reset)
    row = records[-1]
    assert row["fitted_error"]["rotation_error_deg"] < 0.2
    assert not row["fused_check"]["accepted"]
    proposal = row["methods"]["qp1"][0]
    assert proposal["geometry"]["accepted"]
    # A research counterexample, not a desired production accuracy contract:
    # satisfying current geometry did not preserve the measured orientation.
    assert proposal["error"]["rotation_error_deg"] > 5


def test_linear_constraints_can_pass_while_actual_depth_rejects():
    from benchmarks.flash_constraint_checkpoint import load

    geometry, inputs, check = load(
        Path(__file__).parent / "fixtures/constrained_correction_20261120/pointer3997"
    )
    assert check(inputs["fitted"])["accepted"]
    assert not check(inputs["fused"])["accepted"]
    pose, detail = experiment.constrained_proposal(
        inputs["fitted"], inputs["fused"], inputs["covariance"], inputs["center"], geometry, check
    )
    assert detail["qp_success"]
    assert detail["linear_constraint_max_violation"] < 1e-6
    assert not detail["last_geometry"]["accepted"]
    assert pose is None, "The linear surrogate must not replace real depth validation"
