"""Known rendered truth still guards fusion with the innovation veto disabled."""

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.flash_kalman_recovery_control import HIDDEN, WARMUP, run_scene
from benchmarks.flash_recovery_fixture import restore
from benchmarks.flash_recovery_matrix import scenarios
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter


def ungated(pose, center, time_s, config):
    return PoseKalmanFilter(
        pose,
        center,
        time_s,
        replace(config, reset_on_geometry_conflict=False),
    )


@pytest.mark.parametrize("name", list(scenarios()))
def test_ungated_rendered_visibility_and_corruption_controls(name):
    result = run_scene(
        scenarios()[name],
        "strict_broader_motion",
        grow=name == "corrupted_then_clear",
        filter_factory=ungated,
    )
    rows = result["rows"]
    middle = rows[WARMUP : WARMUP + HIDDEN]
    if name == "corrupted_then_clear":
        assert all(not row["available"] for row in middle)
        assert all(row["map_version"] == rows[WARMUP - 1]["map_version"] for row in middle)
        evaluated = rows[:WARMUP] + rows[WARMUP + HIDDEN :]
    elif name == "hidden_motion":
        # No observation of hidden motion: evaluate recovery once visible.
        evaluated = rows[:WARMUP] + rows[WARMUP + HIDDEN :]
    else:
        evaluated = rows
    for row in evaluated:
        assert row["available"], (name, row["index"], row["reason"])
        assert row["error"]["median_point_error_mm"] < 5
        assert row["error"]["rotation_error_deg"] < 1
    if name == "partial_stationary":
        assert all(row["measurement_count"] for row in middle)
    if name == "hidden_stationary":
        before = np.asarray(rows[WARMUP - 1]["covariance"])
        hidden = np.asarray(rows[WARMUP + HIDDEN - 1]["covariance"])
        assert np.linalg.eigvalsh(hidden - before).min() > 0


def test_recorded_wrong_mug_fit_stays_unpublished_without_innovation_gate():
    session, prepared, meta, _ = restore("strict", filter_factory=ungated)
    before = session.store.points.copy()
    result = session.update(prepared, meta["time_s"])
    assert result["camera_from_reference"] is None
    assert result["diagnostics"]["reason"] == "no_observation_compatible_candidate"
    np.testing.assert_array_equal(session.store.points, before)
