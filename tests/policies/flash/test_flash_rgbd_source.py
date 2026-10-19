"""LeRobot's aligned camera frames become capture-timed FLASH observations."""

import numpy as np
import pytest

from lerobot.policies.flash.rgbd_source import LeRobotRGBDSource


class Camera:
    def __init__(self):
        self.calls = 0
        self.captured_s = 10.0

    def read_color_and_aligned_depth_with_timestamp(self, color_mode=None, timeout_ms=200):
        self.calls += 1
        bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        bgr[0, 0] = [1, 2, 3]
        return bgr, np.full((2, 3), 1000, dtype=np.uint16), self.captured_s


def test_rgbd_source_uses_one_frameset_and_capture_time_calibration():
    camera = Camera()
    calibrated_at = []

    def camera_pose(time_s):
        calibrated_at.append(time_s)
        return np.eye(4)

    source = LeRobotRGBDSource(
        camera,
        intrinsics=np.eye(3),
        depth_m_per_unit=0.001,
        base_from_camera_at=camera_pose,
        calibration_version="overhead-v1",
        color_order="bgr",
        max_capture_age_s=0.1,
        clock=lambda: 10.05,
    )
    first = source.read()
    assert camera.calls == 1
    assert calibrated_at == [10.0]
    assert first.frame_id == 0 and first.source_time_s == 10.0 and first.available_time_s == 10.05
    assert first.rgb[0, 0].tolist() == [3, 2, 1]
    np.testing.assert_allclose(first.metric_depth(), 1.0)
    assert source.read().frame_id == 1

    camera.captured_s = 9.8
    with pytest.raises(ValueError, match="stale"):
        source.read()
