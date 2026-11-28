"""Recorded recovery checkpoints: a rejected alignment must leave the surface
unchanged, and the captured frame's process noise must be applied once."""

import os
from functools import cache

import numpy as np
import pytest

from benchmarks.flash_recorded_recovery import replay


@cache
def result(name):
    mode = os.environ.get("FLASH_RECOVERY_EXPERIMENT", "baseline")
    if mode == "partial_depth_search":
        return replay(
            name,
            surface_changes={"partial_surface_matching": True},
            observation_changes={"component_gap_m": None, "depth_outlier_sigma": 3.0},
        )
    if mode in {"partial_filtered_depth_search", "recent_view"}:
        return replay(
            name,
            surface_changes={
                "partial_surface_matching": True,
                "fit_view_limit": 1 if mode == "recent_view" else None,
            },
            observation_changes={
                "component_gap_m": None,
                "depth_outlier_sigma": 3.0,
                "filtered_depth_for_fitting": True,
            },
        )
    if mode != "baseline":
        raise ValueError(f"Unknown recovery experiment: {mode}")
    return replay(name)


@pytest.mark.parametrize(
    "name", ["gamepad_271", "mustard_180", "yellow_68", "clock_39", "clock_100", "drill_46"]
)
def test_rejected_recorded_alignment_does_not_modify_the_surface(name):
    report = result(name)
    if report["result"]["camera_from_reference"] is None:
        assert not report["surface_changed"]


def test_snapshot_process_uncertainty_is_not_added_twice():
    report = result("gamepad_271")
    actual = report["result"]["diagnostics"]["kalman"]["prior_covariance"]
    np.testing.assert_allclose(actual, report["data"]["prior_covariance"], rtol=1e-9, atol=1e-12)
