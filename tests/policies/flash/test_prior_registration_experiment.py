"""Opt-in controls for the compiled research prototype, through real consumers.

Build instructions live with the recorded report. No native package or live
configuration is installed by these tests.
"""

import os
from copy import deepcopy

import numpy as np
import pytest

from benchmarks.flash_motion_recovery import load_fixture, replay
from benchmarks.flash_prior_registration import controlled, experiment, native_module
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter
from lerobot.policies.flash.surface_registration import LibraryMatcher
from tests.policies.flash.test_recorded_motion_recovery import assert_visible_recovery


@pytest.fixture(scope="module")
def native():
    directory = os.environ.get("FLASH_PRIOR_NATIVE_DIR")
    if directory is None:
        pytest.skip("Opt in with FLASH_PRIOR_NATIVE_DIR after building the native experiment")
    return native_module(directory)


def test_prior_protects_strip_without_freezing_measured_depth(native):
    rows = controlled(native, count=5)
    strip = [r for r in rows if r["scenario"] == "strip" and r["factor"] == "gicp"]
    prior = [r for r in strip if r["strength"] == 1]
    absent = [r for r in strip if r["strength"] == 0]
    assert max(r["rotation_error_deg"] for r in prior) < 0.5
    # Mutation: removing the actual prior term must worsen known rotation.
    assert np.median([r["rotation_error_deg"] for r in absent]) > 1.0
    assert all(3.0 < r["depth_correction_mm"] < 5.0 for r in prior)
    assert all(r["iterations"] <= 20 for r in rows)
    moved = [r for r in rows if r["scenario"] == "moved" and r["factor"] == "gicp" and r["strength"] == 0.1]
    assert max(r["rotation_error_deg"] for r in moved) < 1.0


def test_same_native_solver_reproduces_fusion_failure_and_integrated_prior_recovers(native):
    fixture = load_fixture()
    with experiment(native, mode="post"):
        baseline = replay("ring", fixture=fixture)
    with experiment(native):
        integrated = replay("ring", fixture=fixture)
    old = next(r for r in baseline if r["frame"] == 4759)
    new = next(r for r in integrated if r["frame"] == 4759)
    assert old["status"] == "lost"
    assert old["details"]["reason"] == "fused_pose_contradicts_observation"
    assert new["status"] == "observed_candidate"
    assert new["accepted_frame"] == 4759
    assert new["map_version"] > integrated[0]["map_version"]
    assert_visible_recovery(integrated, fixture, 1)
    assert all(r["details"]["fit"]["numerical_solves"] == 1 for r in integrated[1:])
    assert all(r["details"]["fit"]["evaluations"] <= 50 for r in integrated[1:])


def test_integrated_prior_rejects_wrong_object_and_preserves_missing_state(native):
    meta, frames = load_fixture()
    changed = deepcopy(meta)
    changed["cases"]["wrong"] = [2735, 4759, 4765]
    with experiment(native):
        wrong = replay("wrong", fixture=(changed, frames))
        missing = replay("cube", fixture=(meta, frames), missing=True)
    assert all(r["status"] == "lost" and r["pose"] is None for r in wrong[1:])
    assert all(r["accepted_frame"] == wrong[0]["accepted_frame"] for r in wrong[1:])
    assert missing[1]["status"] != "observed_candidate"
    assert missing[1]["accepted_frame"] == missing[0]["accepted_frame"]
    np.testing.assert_array_equal(missing[1]["surface"], missing[0]["surface"])
    assert missing[2]["status"] == "observed_candidate"
    assert missing[2]["details"]["image_pairs"] == 0


def test_research_patch_restores_production_methods_even_after_exception(native):
    methods = (GrowingSurfaceSession.update, LibraryMatcher.refine, PoseKalmanFilter.correct)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        with experiment(native):
            raise RuntimeError("controlled interruption")
    assert methods == (GrowingSurfaceSession.update, LibraryMatcher.refine, PoseKalmanFilter.correct)


def test_native_prior_uses_fixed_object_point_independent_of_coordinate_origin(native):
    rng = np.random.default_rng(218)
    points = rng.uniform(-0.03, 0.03, (300, 3))
    target = points + [0.003, -0.001, 0.004]
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    cov = np.tile(np.diag([1.0, 1.0, 0.001]).reshape(1, 9), (len(points), 1))
    precision = np.diag([100.0] * 3 + [10000.0] * 3)
    predictions = []
    for shift in (np.zeros(3), np.array([0.3, -0.2, 0.5])):
        prepared = native.Prepared(target, normals, cov, points + shift, normals, cov)
        prior = np.eye(4)
        prior[:3, 3] = -shift
        result = prepared.align(prior, prior, precision, shift, 1.0, 0.03, 20, 2, 0.0, 0.0, "gicp")
        pose = np.asarray(result["pose"])
        predictions.append((points + shift) @ pose[:3, :3].T + pose[:3, 3])
    np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-6, rtol=0)
