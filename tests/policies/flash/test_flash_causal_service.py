"""A delayed pose anchor must replay against its own RGB-D frame."""

import time

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.causal_service import CausalPoseService, CausalTrackingConfig
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def frame(image, frame_id):
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    return RGBDFrame(
        frame_id,
        rgb,
        np.full(image.shape, 1000, dtype=np.uint16),
        np.array([[100.0, 0, 48.0], [0, 100.0, 48.0], [0, 0, 1]]),
        0.001,
        np.eye(4),
        frame_id / 30,
        frame_id / 30,
        "camera-v1",
    )


def test_causal_config_rejects_unbounded_or_invalid_tracking():
    valid = {
        "history_frames": 60,
        "queue_frames": 60,
        "min_inliers": 12,
        "max_frame_gap_s": 0.1,
        "fit_threshold_m": 0.005,
    }
    CausalTrackingConfig(**valid, image_scale=0.5)
    with pytest.raises(ValueError, match="invalid causal tracking configuration"):
        CausalTrackingConfig(**{**valid, "image_scale": 0.0})


def test_delayed_anchor_uses_retained_frame_and_all_intervening_motion():
    image = np.random.default_rng(4).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * i], [0, 1, 0]]), (96, 96)), i) for i in range(3)
    ]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    store = LatestPoseStore()
    service = CausalPoseService(store)
    service.start()
    try:
        for item in frames:
            service.submit_frame(item)
        anchor = PoseEstimate(
            "object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(0, (anchor,), {"object": mask})
        deadline = time.monotonic() + 2
        while service.published < 2 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.anchors == 1
        pose = store.snapshot()[1]["object"]
        assert pose.source_time_s == frames[2].source_time_s
        assert pose.evidence == Evidence.PREDICTED
        center = pose.pose @ np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(center[:3], [0.04, 0.0, 1.0], atol=0.005)
    finally:
        service.close()
