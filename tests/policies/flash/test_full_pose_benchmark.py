"""Check the full-pose benchmark's relative-transform scoring contract."""

import numpy as np

from benchmarks.flash_full_pose_score import score_transforms, summarize


def test_pose_score_uses_object_motion_and_counts_missing_frames() -> None:
    source = np.array([[0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [-0.1, 0.0, 1.0]])
    truth = np.repeat(np.eye(4)[None], 3, axis=0)
    angle = np.deg2rad(10)
    truth[1:, :3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    truth[1:, 0, 3] = 0.02
    predicted = truth.copy()
    predicted[1] = np.eye(4)
    available = np.array([True, True, False])

    cloud, rotation, center = score_transforms(predicted, available, truth, source)
    assert 15 < cloud[1] < 30
    assert np.isclose(rotation[1], 10)
    assert 14 < center[1] < 15
    assert np.isnan(cloud[2])
    summary = summarize(available, cloud, rotation, center)
    assert summary["available"] == 1
    assert summary["total"] == 2
    assert summary["within_10mm_10deg"] == 0
