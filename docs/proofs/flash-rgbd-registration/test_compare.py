"""Checks for benchmark geometry and two-sided measured-depth gating."""

import compare
import cv2
import numpy as np
from compare import estimate_pair, measured_pairs, move, robust_fit
from scipy.spatial.transform import Rotation


def test_robust_fit_recovers_known_rigid_motion_with_outliers():
    cv2.setRNGSeed(0)
    generator = np.random.default_rng(12)
    source = generator.normal(0, 0.015, (100, 3)) + [0, 0, 0.35]
    rotation = Rotation.from_euler("xyz", [0.11, -0.06, 0.09]).as_matrix()
    target = source @ rotation.T + [0.008, -0.011, 0.003]
    target[:20] += generator.normal(0, 0.04, (20, 3))
    estimated, inliers, _ = robust_fit(source, target)
    assert estimated is not None
    assert inliers >= 80
    np.testing.assert_allclose(move(source[20:], estimated), target[20:], atol=1e-6)


def test_correspondence_requires_valid_source_and_target_depth():
    depth = np.full((4, 4), 0.5)
    frame_a = {"depth": depth, "usable": np.zeros((4, 4), bool)}
    frame_b = {"depth": depth, "usable": np.zeros((4, 4), bool)}
    frame_a["usable"][1, 1] = frame_a["usable"][2, 1] = True
    frame_b["usable"][1, 2] = True
    k = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]])
    source, target = measured_pairs(
        frame_a, frame_b, np.array([[1, 1], [1, 2]]), np.array([[2, 1], [2, 2]]), k
    )
    assert source.shape == target.shape == (1, 3)
    np.testing.assert_allclose(source[0], [0.005, 0.005, 0.5])
    np.testing.assert_allclose(target[0], [0.01, 0.005, 0.5])


def test_missing_object_yields_no_delta_in_either_frame():
    blank = {"usable": np.zeros((4, 4), bool), "mask": np.zeros((4, 4), bool)}
    visible = {"usable": np.ones((4, 4), bool), "mask": np.ones((4, 4), bool)}
    for previous, current in ((blank, blank), (visible, blank), (blank, visible)):
        pose, inliers, _, pairs = estimate_pair(None, "ufm", previous, current, np.eye(3))
        assert pose is None
        assert inliers == pairs == 0


def test_pose_only_cli_does_not_require_a_gripper_recording(tmp_path, monkeypatch):
    import csv
    import json
    import sys

    recording = tmp_path / "recording"
    recording.mkdir()
    np.savetxt(recording / "cam_K.txt", np.eye(3))
    output = tmp_path / "output"
    monkeypatch.setattr(compare, "RECORDING", recording)

    def no_gripper():
        raise AssertionError("This recording has no gripper evaluation data")

    monkeypatch.setattr(compare, "reference", no_gripper)
    monkeypatch.setattr(compare, "frame_data", lambda frame, k: {})
    step = np.eye(4)
    step[0, 3] = 0.01
    monkeypatch.setattr(compare, "estimate_pair", lambda *args: (step, 10, 0.0, 10))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--method",
            "colored-icp-968",
            "--start",
            "0",
            "--stop",
            "2",
            "--pose-only",
            "--output",
            str(output),
        ],
    )
    compare.main()
    saved = np.load(output / "trace.npz")
    np.testing.assert_array_equal(saved["frames"], [0, 1, 2])
    np.testing.assert_allclose(saved["poses"][:, 0, 3], [0, 0.01, 0.02])
    rows = list(csv.DictReader((output / "trace.csv").open()))
    assert len(rows) == 2
    assert all("gripper" not in key for key in rows[0])
    assert json.loads((output / "summary.json").read_text())["valid"] == 2


def test_gripper_metrics_preserve_every_saved_short_benchmark_value():
    import csv

    saved = np.load(compare.HERE / "results/ufm/trace.npz")
    rows = list(csv.DictReader((compare.HERE / "results/ufm/trace.csv").open()))
    ee, landmarks = compare.reference()
    for index, row in enumerate(rows, start=1):
        poses = saved["poses"]
        step = poses[index] @ np.linalg.inv(poses[index - 1]) if int(row["valid"]) else None
        actual = compare.gripper_metrics(
            ee, landmarks, poses[index], step, int(saved["frames"][0]), int(row["frame"])
        )
        for name, value in actual.items():
            np.testing.assert_allclose(value, float(row[name]), atol=1e-8, equal_nan=True)
