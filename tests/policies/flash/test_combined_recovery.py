"""Recovery comparison bookkeeping, and refusal of the recorded wrong gamepad pose."""

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


@pytest.mark.parametrize(
    ("fitting", "filter_name", "filter_override"),
    [
        ("recovery", "strict_broader", False),
        ("baseline", "strict", True),
        ("baseline", "strict_broader", True),
        ("recovery", "strict", True),
        ("recovery", "strict_broader", True),
    ],
)
def test_recorded_wrong_gamepad_alignment_is_not_published(fitting, filter_name, filter_override):
    # Outcome only: nothing is shown and the surface is not grown, whichever
    # check refuses the proposal.
    _, meta = load("gamepad_271")
    vision, surface = configurations(
        {"config": meta["vision"], "surface_experiment": meta["surface"]}, fitting, filter_name
    )
    report = replay(
        "gamepad_271",
        surface_changes=asdict(surface),
        observation_changes=asdict(vision.observation),
        **({"filter_factory": FILTERS[filter_name]} if filter_override else {}),
    )
    assert report["result"]["camera_from_reference"] is None, report["result"]["diagnostics"].get("reason")
    assert not report["surface_changed"]
