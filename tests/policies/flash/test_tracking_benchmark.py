"""The benchmark reads the same on-disk journal consumed by FLASH replay."""

from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.flash_tracking_benchmark import compare_results, markdown, score_suite

BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks"


def _case(tmp_path, *, complete=True, contradictory=False):
    root = tmp_path / "episode"
    evidence = root / "flash/v1/observation-evidence/episode_000000"
    masks = root / "flash/v1/imported-masks/object"
    evidence.mkdir(parents=True)
    masks.mkdir(parents=True)
    (root / "flash/v1/source.json").write_text("{}")
    source_mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    for i in range(2):
        cv2.imwrite(str(masks / f"{i:06d}.png"), source_mask)
    (evidence / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": complete,
                "frames_completed": 2,
                "total_frames": 2,
                "roles": ["object"],
            }
        )
    )
    (evidence / "observation.json").write_text(
        json.dumps(
            {
                "evidence_identity": "fixture",
                "config_bundle": "fixture",
                "mask_source": "recorded",
                "frames": 2,
            }
        )
    )
    (root / "fixture").write_text("{}")
    pose = np.eye(4).tolist()
    for i in range(2):
        row = {
            "frame_index": i,
            "image_size": [2, 2],
            "masks_rle": {"object": [[0, 1]]},
            "estimates": {
                "object": {
                    "evidence": "observed" if i == 0 else "lost",
                    "pose": pose if i == 0 or contradictory else None,
                    "estimator": "rgbd" if i == 0 else "none",
                }
            },
            "proposals": {
                "object": {
                    "failure": None if i == 0 else "no pose",
                    "visible_support": {"depth_residual_p90_m": 0.002},
                }
            },
        }
        with gzip.open(evidence / f"{i:06d}.json.gz", "wt") as stream:
            json.dump(row, stream)
    case = {
        "id": "fixture",
        "kind": "lerobot_rgbd",
        "domain": "fixture",
        "artifact_root": str(root),
        "expected_frames": 2,
    }
    return {"schema": "flash-tracking-suite-v1", "name": "fixture", "cases": [case]}, evidence


def test_availability_mask_agreement_and_accuracy_are_distinct(tmp_path):
    suite, _ = _case(tmp_path)
    result = score_suite(suite)
    role = result["cases"][0]["roles"]["object"]
    assert role["states"] == {"observed": 1, "lost": 1}
    assert role["direct_observation_fraction"] == 0.5
    assert role["pose_available_fraction"] == 0.5
    assert role["mask_iou_median_vs_supplied"] == 0.5
    assert role["observed_depth_residual_p90_mm_median"] == 2.0
    assert role["pose_accuracy"]["status"] == "unavailable_no_pose_labels"
    assert role["longest_lost_streak_visible_frames"] == 1
    assert "no labeled pose score" in markdown(result)


def test_prepared_demo_reports_coverage_without_inventing_visibility_or_accuracy(tmp_path):
    suite, evidence = _case(tmp_path)
    (evidence / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "frames_completed": 2,
                "total_frames": 2,
                "roles": ["object"],
                "source_sha256": "source",
            }
        )
    )
    (evidence.parent / "bundle.json").write_text("{}")
    suite["cases"][0].update(kind="flash_prepared_demo", evidence_dir=str(evidence))
    result = score_suite(suite)
    role = result["cases"][0]["roles"]["object"]
    assert role["states"] == {"observed": 1, "lost": 1}
    assert role["reference_visible_frames"] is None
    assert role["frames_scored"] == 2
    assert role["visibility_basis"] == "all_frames_no_visibility_labels"
    assert role["mask_present_frames"] == 2
    assert role["mask_iou_median_vs_supplied"] is None
    assert role["pose_accuracy"]["status"] == "unavailable_no_independent_pose_labels"
    assert "visibility unlabeled" in markdown(result)


@pytest.mark.parametrize(
    "complete,contradictory,match",
    [(False, False, "incomplete evidence"), (True, True, "pose/state contradiction")],
)
def test_incomplete_or_contradictory_trace_cannot_score(tmp_path, complete, contradictory, match):
    suite, _ = _case(tmp_path, complete=complete, contradictory=contradictory)
    with pytest.raises(ValueError, match=match):
        score_suite(suite)


def test_missing_local_data_is_explicit_and_strict_mode_fails(tmp_path):
    suite = {
        "schema": "flash-tracking-suite-v1",
        "name": "fixture",
        "cases": [
            {
                "id": "missing",
                "kind": "lerobot_rgbd",
                "domain": "fixture",
                "artifact_root": str(tmp_path / "absent"),
            }
        ],
    }
    assert score_suite(suite)["cases"][0]["status"] == "missing_local_artifact"
    with pytest.raises(FileNotFoundError):
        score_suite(suite, require_all=True)


def test_comparison_keeps_availability_separate_from_conditional_error(tmp_path):
    suite, _ = _case(tmp_path)
    baseline = score_suite(suite)
    candidate = deepcopy(baseline)
    candidate["cases"][0]["roles"]["object"]["frame_states"][1] = "predicted"
    comparison = compare_results(baseline, candidate)[0]
    assert comparison["available_frames_before"] == 1
    assert comparison["available_frames_after"] == 2
    assert "paired_labeled_pose_error" not in comparison
    candidate["cases"][0]["capture"]["input_sha256"] = "changed"
    with pytest.raises(ValueError, match="source changed"):
        compare_results(baseline, candidate)


def test_saved_baseline_matches_frozen_case_manifest():
    suite_path = BENCHMARKS / "flash_tracking_suite.json"
    suite = json.loads(suite_path.read_text())
    baseline = json.loads((BENCHMARKS / "flash_tracking_baseline.json").read_text())
    assert baseline["suite_sha256"] == hashlib.sha256(suite_path.read_bytes()).hexdigest()
    assert {case["id"] for case in suite["cases"]} == {case["id"] for case in baseline["cases"]}
    assert all(case["status"] == "scored" for case in baseline["cases"])
