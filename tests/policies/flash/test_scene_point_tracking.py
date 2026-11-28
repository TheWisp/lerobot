"""Scene point tracking invariants."""

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.observations import RGBDFrame
from lerobot.policies.flash.scene_groups import ScenePointTracker


def _camera_frame(rgb: np.ndarray, frame_id: int) -> RGBDFrame:
    return RGBDFrame(
        frame_id,
        rgb,
        np.full(rgb.shape[:2], 1000, dtype=np.uint16),
        np.array([[100.0, 0, 48.0], [0, 100.0, 48.0], [0, 0, 1.0]]),
        0.001,
        np.eye(4),
        frame_id / 30,
        frame_id / 30,
        "cal",
    )


def test_scene_tracks_keep_identities_on_every_frame():
    image = np.random.default_rng(4).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    tracker = ScenePointTracker(image_scale=1.0)
    first = tracker.update(_camera_frame(image, 0))
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 0]]), (96, 96))
    second = tracker.update(_camera_frame(shifted, 1))
    shared = np.intersect1d(first.ids, second.ids)
    assert len(shared) >= 12
    before = first.base_xyz[np.searchsorted(first.ids, shared)]
    after = second.base_xyz[np.searchsorted(second.ids, shared)]
    assert np.median(after[:, 0] - before[:, 0]) == pytest.approx(0.02, abs=0.004)

    missing = _camera_frame(shifted, 2)
    missing = RGBDFrame(
        missing.frame_id,
        missing.rgb,
        np.zeros_like(missing.depth),
        missing.intrinsics,
        missing.depth_m_per_unit,
        missing.base_from_camera,
        missing.source_time_s,
        missing.available_time_s,
        missing.calibration_version,
    )
    assert len(tracker.update(missing).ids) == 0
    restored = tracker.update(_camera_frame(shifted, 3))
    assert len(np.intersect1d(second.ids, restored.ids)) >= 12

    interrupted = _camera_frame(shifted, 4)
    interrupted = RGBDFrame(
        interrupted.frame_id,
        interrupted.rgb,
        interrupted.depth,
        interrupted.intrinsics,
        interrupted.depth_m_per_unit,
        interrupted.base_from_camera,
        2.0,
        2.0,
        interrupted.calibration_version,
    )
    after_gap = tracker.update(interrupted)
    assert len(np.intersect1d(second.ids, after_gap.ids)) >= 12
    assert tracker.gaps == 1
