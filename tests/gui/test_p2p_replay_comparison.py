# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""A better mean must not hide accepted failures or incomplete evaluation coverage."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "compare_p2p_replays", Path(__file__).parents[2] / "benchmarks/compare_p2p_replays.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def fixture(tmp_path, before, after, *, lost=None):
    (tmp_path / "suite.json").write_text(json.dumps({"sequences": {"moving_object": len(before)}}))
    for label, values in (("baseline", before), ("fixed", after)):
        root = tmp_path / "moving_object" / label
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps(dict.fromkeys(comparison.INPUT_KEYS, "same")))
        flags = np.zeros(len(values), bool) if lost is None else np.asarray(lost, bool)
        np.savez(root / "errors.npz", add=values, adds=values, lost=flags)
    return tmp_path


def compare(root):
    return comparison.compare_suite(root, bad_add_mm=50, auc_drop_pp=0, p95_rise_mm=0, extra_lost_frames=0)


def test_improved_average_cannot_hide_one_new_large_accepted_error(tmp_path):
    root = fixture(tmp_path, np.full(100, 0.02), np.r_[np.full(99, 0.001), 0.06])
    result = compare(root)
    row = result["sequences"][0]
    assert row["fixed"]["add_auc_percent"] > row["baseline"]["add_auc_percent"]
    assert row["fixed"]["add_p95_mm"] < row["baseline"]["add_p95_mm"]
    assert not result["passed"]
    assert row["regressions"] == ["more high-error frames reported as not lost"]


def test_lost_frames_still_contribute_their_pose_error(tmp_path):
    root = fixture(tmp_path, [0.01, 0.02], [0.01, 0.5], lost=[False, True])
    result = compare(root)
    assert not result["passed"]
    assert result["sequences"][0]["fixed"]["add_max_mm"] == 500


@pytest.mark.parametrize("after", ([0.01], [0.01, np.nan], [0.01, np.inf]))
def test_missing_or_invalid_estimates_cannot_disappear_from_the_score(tmp_path, after):
    with pytest.raises(ValueError):
        compare(fixture(tmp_path, [0.01, 0.02], after))


def test_mismatched_inputs_cannot_be_called_a_paired_comparison(tmp_path):
    root = fixture(tmp_path, [0.01, 0.02], [0.01, 0.02])
    path = root / "moving_object/fixed/manifest.json"
    value = json.loads(path.read_text())
    value["depth"] = "different depth"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different inputs"):
        compare(root)


def test_unrun_sequence_blocks_the_suite(tmp_path):
    root = fixture(tmp_path, [0.01, 0.02], [0.01, 0.02])
    (root / "suite.json").write_text(json.dumps({"sequences": {"moving_object": 2, "occluded": 5}}))
    with pytest.raises(FileNotFoundError):
        compare(root)


def test_identical_runs_pass_and_auc_uses_all_frames(tmp_path):
    assert compare(fixture(tmp_path, [0.01, 0.02], [0.01, 0.02]))["passed"]
    assert comparison.auc(np.array([0.0, 0.2])) == 50.0


def test_marking_estimates_lost_cannot_hide_an_availability_regression(tmp_path):
    root = fixture(tmp_path, [0.01, 0.02], [0.01, 0.02])
    np.savez(root / "moving_object/fixed/errors.npz", add=[0.01, 0.02], adds=[0.01, 0.02], lost=[True, True])
    assert compare(root)["sequences"][0]["regressions"] == ["more lost frames than allowed"]
