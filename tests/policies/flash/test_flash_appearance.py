"""Multiview appearance matches feed the common metric pose estimator."""

import numpy as np

from lerobot.policies.flash.appearance import (
    Sam3MaskSource,
)
from lerobot.policies.flash.observations import RGBDFrame

POINTS = np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, 0.02, 0], [0, 0, 0]])


def frame(frame_id, pixels, source_time):
    depth = np.zeros((24, 24), dtype=np.uint16)
    depth[np.rint(pixels[:, 1]).astype(int), np.rint(pixels[:, 0]).astype(int)] = 1000
    k = np.array([[100.0, 0, 10], [0, 100.0, 10], [0, 0, 1.0]])
    return RGBDFrame(
        frame_id,
        np.zeros((24, 24, 3), dtype=np.uint8),
        depth,
        k,
        0.001,
        np.eye(4),
        source_time,
        source_time,
        "camera-v1",
    )


def test_sam3_mask_source_shares_one_adapter_and_resets_on_discontinuity():
    class Adapter:
        def __init__(self, **kwargs):
            self.calls = []

        def set_camera(self, camera_id):
            self.calls.append(("camera", camera_id))

        def set_control(self, settings):
            self.calls.append(("control", settings))

        def reset(self):
            self.calls.append(("reset",))

        def segment(self, rgb):
            self.calls.append(("segment",))
            return {"tool": np.ones(rgb.shape[:2], dtype=bool), "ring": np.zeros(rgb.shape[:2], dtype=bool)}

    made = []

    def factory(**kwargs):
        adapter = Adapter(**kwargs)
        made.append(adapter)
        return adapter

    correction = {
        "enabled": True,
        "interval_frames": 16,
        "min_detector_score": 0.8,
        "min_tracker_score": 0.8,
        "min_mask_overlap": 0.5,
    }
    source = Sam3MaskSource(
        {"held": "tool", "target": "ring"},
        camera_id="overhead",
        detection_correction=correction,
        adapter_factory=factory,
    )
    xy = np.array([[10.0, 10.0]])
    assert set(source.masks(frame(1, xy, 1.0))) == {"held", "target"}
    source.masks(frame(2, xy, 2.0))
    source.masks(frame(1, xy, 3.0))
    source.reset_tracking()
    source.masks(frame(12, xy, 12.0))
    assert len(made) == 1
    assert made[0].calls.count(("segment",)) == 4
    assert made[0].calls.count(("reset",)) == 2
    assert (
        "control",
        {"prompt": "tool . ring", "text_detection": True, "detection_correction": correction},
    ) in made[0].calls
