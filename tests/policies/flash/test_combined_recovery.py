"""The parked filter fix must survive combination with surface recovery.

These checks forbid the recorded wrong jump. Recovery is tested separately in
test_recorded_recovery_sequences: rejecting every candidate cannot pass that.
"""

from dataclasses import asdict, replace

import numpy as np
import pytest

from benchmarks.flash_full_pose_score import score_transforms
from benchmarks.flash_recorded_recovery import load, replay
from benchmarks.flash_recovery_combinations import FILTERS, configurations
from benchmarks.flash_recovery_matrix_report import classify


def test_recovery_comparison_separates_wrong_missing_and_correct_output():
    truth = np.repeat(np.eye(4)[None], 3, axis=0)
    estimates = truth.copy()
    estimates[1, 0, 3] = 0.1
    # Missing output gets an identity placeholder only for array shape; it must
    # not earn the perfect score that this placeholder would otherwise receive.
    available = np.array([True, True, False])
    points = np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]])
    distance, angle, _ = score_transforms(estimates, available, truth, points)
    assert classify(available, distance, angle).tolist() == ["within", "outside", "missing"]


@pytest.mark.parametrize("previous_reset", [False, True])
def test_reset_comparison_changes_filter_policy_not_uncertainty_or_geometry(previous_reset):
    _, meta = load("gamepad_271")
    provenance = {"config": meta["vision"], "surface_experiment": meta["surface"]}
    provenance["surface_experiment"]["uncertainty"]["reset_on_geometry_conflict"] = previous_reset
    reset = configurations(provenance, "recovery", "reset_broader")
    strict = configurations(provenance, "recovery", "strict_broader")
    assert reset[0] == strict[0]
    assert reset[1].uncertainty.reset_on_geometry_conflict
    assert not strict[1].uncertainty.reset_on_geometry_conflict
    assert (
        replace(strict[1], uncertainty=replace(strict[1].uncertainty, reset_on_geometry_conflict=True))
        == reset[1]
    )
    assert reset[1].uncertainty.position_process_std_m_sqrt_s == 0.03
    assert FILTERS["reset_broader"] is FILTERS["reset"]
    assert FILTERS["reset_broader"] is FILTERS["strict_broader"]


def test_saved_no_reset_setting_rejects_the_recorded_wrong_gamepad_without_factory_override():
    _, meta = load("gamepad_271")
    vision, surface = configurations(
        {"config": meta["vision"], "surface_experiment": meta["surface"]}, "recovery", "strict_broader"
    )
    changes = asdict(surface)
    assert not changes["uncertainty"]["reset_on_geometry_conflict"]
    result = replay("gamepad_271", surface_changes=changes, observation_changes=asdict(vision.observation))
    detail = result["result"]["diagnostics"]["kalman"]
    assert detail["prior_rejected_by_depth_mask"]
    assert detail["innovation_squared"] > detail["innovation_limit"]
    assert not detail["accepted"]
    assert result["result"]["camera_from_reference"] is None
    assert not result["surface_changed"]


@pytest.mark.parametrize("fitting", ["baseline", "recovery"])
@pytest.mark.parametrize("filter_name", ["strict", "strict_broader"])
def test_failed_previous_geometry_does_not_authorize_recorded_wrong_gamepad_pose(fitting, filter_name):
    _, meta = load("gamepad_271")
    vision, surface = configurations(
        {"config": meta["vision"], "surface_experiment": meta["surface"]}, fitting, filter_name
    )
    report = replay(
        "gamepad_271",
        surface_changes=asdict(surface),
        observation_changes=asdict(vision.observation),
        filter_factory=FILTERS[filter_name],
    )
    details = report["result"]["diagnostics"]["kalman"]
    # This actual depth/mask/fitter output disproves the old pose, but the new
    # proposal still fails the unchanged Kalman consistency test.
    assert details["prior_rejected_by_depth_mask"]
    assert details["innovation_squared"] > details["innovation_limit"]
    assert not details["accepted"]
    assert details["reason"] == "innovation_conflict"
    assert report["result"]["camera_from_reference"] is None
    assert not report["surface_changed"]
