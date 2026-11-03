"""Rendered observations distinguish real recovery from a frozen prediction.

These characterize an experiment, not a recommendation to change defaults.
The real observation builder, fitter and filter run; no fitted pose is mocked.
"""

import numpy as np
import pytest

from benchmarks.flash_kalman_recovery_control import HIDDEN, MODES, WARMUP, run_scene, scene


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
