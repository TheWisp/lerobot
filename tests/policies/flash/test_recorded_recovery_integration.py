"""Real failure checkpoints, before changing the fitter.

Unlabelled rig clips have explicitly limited checks: visible-surface agreement
is NOT a full 6D accuracy label. Their final results still require video review.
No xfail: run these to see remaining failures during the recovery experiment.
"""

import os
from functools import cache

import numpy as np
import pytest

from benchmarks.flash_recorded_recovery import replay
from lerobot.policies.flash.surface_observation import move


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


def returned_pose(report):
    pose = report["result"]["camera_from_reference"]
    assert pose is not None, report["result"]["diagnostics"].get("reason")
    return np.asarray(pose)


@pytest.mark.parametrize("name", ["mustard_180", "clock_39", "clock_100", "drill_46"])
def test_visible_labelled_object_recovers_accurate_pose(name):
    report = result(name)
    returned_pose(report)
    # Existing benchmark's loose budget. Do not tune it to this implementation.
    assert report["error"]["cloud_mm"] <= 20, report["error"]
    assert report["error"]["angle_deg"] <= 20, report["error"]


def test_drill_preparation_does_not_replace_near_object_with_far_depth_patch():
    report = result("drill_46")
    data = report["data"]
    expected = move(data["reference"], data["label_pose"])
    center = expected.mean(0)
    # Evaluation-only object extent from the source observation and dataset
    # pose, with the same 20 mm allowance. Never fed into observation cleanup.
    radius = np.linalg.norm(expected - center, axis=1).max() + 0.02
    measured = report["prepared"]["current"]["obs"]["points"]
    assert np.mean(np.linalg.norm(measured - center, axis=1) <= radius) >= 0.75


def test_visible_yellow_block_recovers_in_the_observed_region():
    report = result("yellow_68")
    pose = returned_pose(report)
    source = move(report["data"]["reference"], pose)
    observed = report["prepared"]["current"]["obs"]["points"]
    # This clip shows the same broad face before/after a push. A position-only
    # check against its measured centre catches a stale/lost track. It cannot
    # establish which equivalent orientation of the block is correct.
    assert np.linalg.norm(np.median(source, axis=0) - np.median(observed, axis=0)) <= 0.02


def test_gamepad_recovered_pose_agrees_with_visible_face_direction():
    report = result("gamepad_271")
    pose = returned_pose(report)
    source = report["data"]["reference"]
    observed = report["data"]["evaluation_visible_patch"]
    _, _, reference_axes = np.linalg.svd(source - source.mean(0), full_matrices=False)
    _, _, observed_axes = np.linalg.svd(observed - observed.mean(0), full_matrices=False)
    cosine = abs((pose[:3, :3] @ reference_axes[-1]) @ observed_axes[-1])
    angle = np.rad2deg(np.arccos(np.clip(cosine, 0, 1)))
    # A limited geometric consistency diagnostic for this flat toy, not an
    # independent pose label or permission to accept an arbitrary 180° flip.
    assert angle <= 20, {"visible_face_normal_disagreement_deg": angle}


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
