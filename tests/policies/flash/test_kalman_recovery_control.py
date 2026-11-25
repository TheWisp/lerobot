"""Rendered observations distinguish real recovery from a frozen prediction.

Historical filter comparisons and the production fallback share these controls.
The real observation builder, fitter and filter run; no fitted pose is mocked.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.flash_kalman_recovery_control import HIDDEN, MODES, WARMUP, changed_scene, run_scene, scene
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.vision_config import VisionPipelineConfig


def test_screen_hides_the_entire_moving_object_without_leaking_its_depth():
    hidden = scene()
    visible = scene(occluded=False)
    for covered, uncovered in zip(hidden, visible, strict=True):
        np.testing.assert_array_equal(covered["truth"], uncovered["truth"])
        if covered["hidden"]:
            assert not covered["mask"].any()
            assert np.all(covered["depth"][uncovered["mask"]] < uncovered["depth"][uncovered["mask"]])
        else:
            np.testing.assert_array_equal(covered["depth"], uncovered["depth"])
    assert sum(f["hidden"] for f in hidden) == HIDDEN
    assert hidden[-1]["truth"][0, 3] == pytest.approx(0.04)
    assert not np.array_equal(visible[0]["mask"], visible[-1]["mask"])


@pytest.fixture(scope="module")
def recovery_comparison():
    frames = scene()
    return {mode: run_scene(frames, mode) for mode in MODES}


def test_accurate_recovery_can_pass_kalman_without_bypassing_rejection(recovery_comparison):
    index = WARMUP + HIDDEN
    strict = recovery_comparison["strict"]["rows"][index]
    broad = recovery_comparison["strict_broader_motion"]["rows"][index]
    reset = recovery_comparison["current_reset"]["rows"][index]

    # The scene's known geometry evaluates independently fitted candidates.
    for row in (strict, broad, reset):
        assert row["candidate_error"]["median_point_error_mm"] < 5
        assert row["candidate_error"]["rotation_error_deg"] < 1

    assert not strict["available"]
    assert strict["kalman"]["filter_reason"] == "innovation_conflict"
    assert strict["kalman"]["innovation_squared"] > strict["kalman"]["innovation_limit"]
    assert reset["kalman"]["reason"] == "reinitialized_after_contradiction"
    assert reset["available"]

    assert broad["available"]
    assert broad["kalman"]["filter_reason"] == "corrected"
    assert broad["kalman"]["prediction_failed_geometry"]
    assert broad["kalman"]["innovation_squared"] < broad["kalman"]["innovation_limit"]
    assert broad["error"]["median_point_error_mm"] < 5
    assert broad["pose"][0][3] > 0.03  # A filter that never moves cannot pass.


def test_pose_recovery_does_not_depend_on_growing_the_surface(recovery_comparison):
    for result in recovery_comparison.values():
        assert len({r["map_points"] for r in result["rows"]}) == 1
        assert result["rows"][0]["map_points"] > 0


def fallback_config():
    settings = json.loads((Path(__file__).parent / "fixtures/fallback_settings_20261120.json").read_text())
    pipeline = VisionPipelineConfig.from_dict(settings["config"])
    pipeline = replace(
        pipeline, shape=replace(pipeline.shape, registration=replace(pipeline.shape.registration, threads=1))
    )
    return pipeline, GrowingSurfaceConfig(**settings["surface_experiment"])


def test_rendered_control_uses_supplied_filter_settings():
    pytest.importorskip("small_gicp")
    pipeline, surface = fallback_config()
    surface = replace(surface, uncertainty=replace(surface.uncertainty, position_process_std_m_sqrt_s=0.017))
    result = run_scene(changed_scene(20)[:1], "current_reset", config=surface, pipeline_config=pipeline)
    assert result["effective_filter_config"]["position_process_std_m_sqrt_s"] == 0.017
    assert (
        result["effective_filter_config"]["reset_on_geometry_conflict"]
        == surface.uncertainty.reset_on_geometry_conflict
    )
    assert result["effective_surface_config"]["refresh_observed_surface"] == surface.refresh_observed_surface


def test_accuracy_scores_published_pose_when_internal_pose_differs(monkeypatch):
    original_update = GrowingSurfaceSession.update

    def different_output(session, *args, **kwargs):
        result = original_update(session, *args, **kwargs)
        published = np.asarray(result["camera_from_reference"]).copy()
        published[0, 3] += 0.1
        result["camera_from_reference"] = published.tolist()
        return result

    monkeypatch.setattr(GrowingSurfaceSession, "update", different_output)
    row = run_scene(scene()[:1], "current_reset")["rows"][0]
    assert row["error_basis"] == "published"
    assert row["internal_pose"][0][3] == 0
    assert row["pose"] == row["published_pose"]
    assert row["error"]["median_point_error_mm"] == pytest.approx(100)


@pytest.mark.parametrize(
    "fallback",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="Known limitation: checked fit can be 8.009 mm wrong on a 24-pixel strip",
            ),
        ),
    ],
)
def test_checked_fit_does_not_publish_wrong_thin_strip_pose(fallback):
    pytest.importorskip("small_gicp")
    pipeline, surface = fallback_config()
    if not fallback:
        surface = replace(surface, allow_fitted_pose_fallback=False)
    result = run_scene(changed_scene(45, strip=24), "current_reset", config=surface, pipeline_config=pipeline)
    # Renderer truth only scores output: it never enters fitting or filtering.
    for row in result["rows"]:
        if row["available"] and not row["hidden"]:
            assert row["error"]["median_point_error_mm"] < 5, row
            assert row["error"]["rotation_error_deg"] < 1, row
