"""Known-pose controls for recovery from an overconfident wrong prediction.

Render a measured box surface, then move it without image/group motion hints.
The actual fitter must recover the movement even when ordinary Kalman fusion
stays near the wrong prediction. Renderer truth only scores published output;
it is never supplied to fitting or filtering. These are estimator controls,
not robot execution tests or evidence about segmentation accuracy.
"""

from dataclasses import replace

import pytest

from benchmarks.flash_kalman_recovery_control import WARMUP, changed_scene, run_scene
from tests.policies.flash.test_kalman_recovery_control import fallback_config


@pytest.mark.parametrize(
    "angle,translation",
    [
        pytest.param(20, -0.04, id="translate-left-40mm"),
        pytest.param(20, 0.04, id="translate-right-40mm"),
        pytest.param(45, 0.02, id="turn-25deg-and-translate-20mm"),
    ],
)
def test_checked_fit_recovers_actual_motion_despite_overconfident_prediction(angle, translation):
    pytest.importorskip("small_gicp")
    pipeline, surface = fallback_config()
    surface = replace(
        surface,
        grow=False,
        refresh_observed_surface=False,
        uncertainty=replace(
            surface.uncertainty,
            initial_position_std_m=1e-6,
            initial_rotation_std_deg=0.0001,
            position_process_std_m_sqrt_s=1e-6,
            rotation_process_std_deg_sqrt_s=0.0001,
            unmeasured_rotation_std_deg_sqrt_s=None,
        ),
    )
    frames = changed_scene(angle, translation=translation)[: WARMUP + 2]
    recovered = run_scene(frames, "current_reset", config=surface, pipeline_config=pipeline)["rows"]
    without_fallback = run_scene(
        frames,
        "current_reset",
        config=replace(surface, allow_fitted_pose_fallback=False),
        pipeline_config=pipeline,
    )["rows"]

    # Establish that the inputs actually exercise the recovery contract: a
    # good fitted measurement exists, but ordinary fusion contradicts depth.
    for rows in (recovered, without_fallback):
        first = rows[WARMUP]
        assert first["candidate_error"]["median_point_error_mm"] < 5
        assert first["candidate_error"]["rotation_error_deg"] < 1
        assert first["kalman"]["reason"] == "fused_pose_contradicts_observation"
        assert not first["kalman"]["accepted"]
    assert not without_fallback[WARMUP]["available"]
    assert recovered[WARMUP]["available"]
    assert recovered[WARMUP]["fallback"]["accepted"]

    # Score the public result against known motion, including the next frame.
    # Freezing the old pose cannot pass either the translation or turn cases.
    for row in recovered[WARMUP:]:
        assert row["available"]
        assert row["status"] == "observed_candidate"
        assert row["error_basis"] == "published"
        assert row["error"]["median_point_error_mm"] < 5
        assert row["error"]["rotation_error_deg"] < 1

    # Recovery must not require map growth. An ordinary correction resumes
    # on the following frame using the recovered pose as its prediction.
    assert len({row["map_version"] for row in recovered}) == 1
    assert recovered[WARMUP + 1]["kalman"]["accepted"]
    assert recovered[WARMUP + 1]["fallback"] is None
