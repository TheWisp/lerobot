"""Clear post-placement regression, with no expected-failure exemptions.

Run FLASH_RUN_PLACEMENT_TESTS=1 for optional-library fitting tests. These small
recorded checkpoints need neither UFM nor a GPU. Every backend uses the real
shared search and the original measured appearance, just like the session.
"""

import os
from functools import cache

import numpy as np
import pytest

from benchmarks.flash_gamepad_placement import METHODS, evaluate, feature_error, load, prepare, replay
from lerobot.policies.flash.surface_correspondence import rigid_fit


def test_feature_score_respects_button_symmetry_but_not_displacement():
    data, _ = load()
    points = data["features_xyz_540"]
    np.testing.assert_allclose(feature_error(points, points[[0, 4, 3, 2, 1]], np.eye(4)), 0)
    wrong = np.eye(4)
    wrong[0, 3] = 0.03
    assert feature_error(points, points, wrong).mean() > 0.02


def test_independent_visible_feature_annotations_are_self_consistent():
    data, metadata = load()
    for frame in metadata["frames"]:
        source, target = data["features_xyz_0"], data[f"features_xyz_{frame}"]
        # Only checks annotation quality. This fitted transform is NOT an input
        # to the tested tracker, its seeds, or its acceptance check.
        pose = rigid_fit(source, target)
        assert evaluate(data, 0, frame, pose)["mean_feature_error_mm"] < 1
        uv = np.rint(data[f"features_uv_{frame}"]).astype(int)
        assert data[f"mask_{frame}"][uv[:, 1], uv[:, 0]].all()


@pytest.mark.parametrize("method", ("small_gicp", "open3d_color"))
def test_recorded_protruding_gamepad_pose_is_rejected_without_optional_libraries(method):
    from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
    from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction

    data, metadata = load()
    current = prepare(data, metadata, 540)
    shape = ShapeConfig(scene_depth_constraints=False)
    gate = PoseUncertaintyConfig(max_visible_outlier_fraction=0.99)
    fit = WholeSurfaceFit(data[f"map_{method}_540"], current, data["intrinsics_540"], shape)
    assert not check_surface_contradiction(fit, data[f"seed_{method}_540"], gate)["accepted"]
    # A directly measured surface in this same frame must remain valid.
    observed = WholeSurfaceFit(current["obs"]["points"], current, data["intrinsics_540"], shape)
    assert check_surface_contradiction(observed, np.eye(4), gate)["accepted"]


@cache
def result(method, frame, clean):
    if method != "baseline":
        pytest.importorskip("small_gicp" if method == "small_gicp" else "open3d")
    return replay(method, frame, clean_start=clean)


recorded = pytest.mark.skipif(
    os.environ.get("FLASH_RUN_PLACEMENT_TESTS") != "1", reason="set FLASH_RUN_PLACEMENT_TESTS=1"
)
fit_methods = [
    "baseline",
    *(pytest.param(method, marks=recorded) for method in METHODS if method != "baseline"),
]


@pytest.mark.parametrize("method", fit_methods)
@pytest.mark.parametrize("frame", (540, 627))
def test_clear_face_stays_aligned_from_clean_start(method, frame):
    report = result(method, frame, True)
    # The pipeline's existing 5 mm geometric tolerance; no relaxed budget to
    # make a particular library pass. No 6D angle threshold on a flat toy.
    assert report["geometry"]["accepted"], report
    assert report["result"]["mean_feature_error_mm"] <= 5, report


@pytest.mark.parametrize("method", fit_methods)
@pytest.mark.parametrize("frame", (540, 627))
def test_returning_clear_face_recovers_after_recorded_history(method, frame):
    report = result(method, frame, False)
    assert report["geometry"]["accepted"], report
    assert report["result"]["mean_feature_error_mm"] <= 5, report


@recorded
@pytest.mark.parametrize("method", ("small_gicp", "open3d_color"))
def test_geometry_check_does_not_accept_known_wrong_clear_view_solution(method):
    report = result(method, 540, False)
    # Separate safety from recovery: declining a bad fit is preferable to
    # publishing it, even before a replacement can be found.
    assert not report["geometry"]["accepted"] or report["result"]["mean_feature_error_mm"] <= 5, report


@recorded
@pytest.mark.parametrize("method", METHODS[1:])
def test_library_recovery_does_not_secretly_call_scipy(method, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("Library selection must not silently fall back to SciPy")

    monkeypatch.setattr("lerobot.policies.flash.surface_grounding.least_squares", forbidden)
    report = replay(method, 627, clean_start=False)
    assert report["solver"]["library"] == method
    assert report["geometry"]["accepted"], report
    assert report["result"]["mean_feature_error_mm"] <= 5, report
