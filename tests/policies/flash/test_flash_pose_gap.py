"""Brief RGB-D matcher losses must not erase still-fresh target poses."""

import numpy as np
import pytest

from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.pose_gap import PoseGapBackend
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def test_pose_gap_preserves_measurement_time_and_accepts_later_target_motion():
    def estimate(time_s, evidence, x=0.0):
        pose = None if evidence == Evidence.LOST else np.eye(4)
        if pose is not None:
            pose[0, 3] = x
        return PoseEstimate(
            "target", pose, time_s, time_s + 0.01, evidence, "target-v1", (True,) * 3, (True,) * 3
        )

    class Backend:
        def __init__(self):
            self.results = iter(
                (
                    estimate(0.0, Evidence.PREDICTED),  # Symmetry-reduced RGB-D observation.
                    estimate(0.2, Evidence.LOST),
                    estimate(0.5, Evidence.LOST),
                    estimate(0.6, Evidence.OBSERVED, 0.04),
                )
            )

        def update(self, _frame):
            return (next(self.results),)

    backend = PoseGapBackend(Backend(), 0.4)
    store = LatestPoseStore()
    results = []
    for time_s in (0.0, 0.2, 0.5, 0.6):
        frame = RGBDFrame(
            int(time_s * 10),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.ones((1, 1), dtype=np.uint16),
            np.eye(3),
            0.001,
            np.eye(4),
            time_s,
            time_s + 0.01,
            "camera-v1",
        )
        result = backend.update(frame)[0]
        store.publish((result,))
        assert store.snapshot()[1]["target"] is result
        results.append(result)

    assert results[1].evidence == Evidence.PREDICTED
    assert results[1].source_time_s == 0.0
    assert results[1].available_time_s == pytest.approx(0.21)
    assert results[2].evidence == Evidence.LOST
    assert results[3].pose[0, 3] == 0.04  # A real move is accepted after the gap.
    assert store.snapshot()[1]["target"] is results[3]
