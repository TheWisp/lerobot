"""The RGB-D observation and its host-clock time must be one atomic snapshot."""

from threading import Event, Lock

import numpy as np

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera


class RunningThread:
    def is_alive(self):
        return True


def test_aligned_rgbd_returns_capture_timestamp_without_changing_old_api():
    camera = object.__new__(RealSenseCamera)
    camera.rs_pipeline = object()
    camera.rs_profile = object()
    camera.use_depth = True
    camera.post_grab_processor = None
    camera.thread = RunningThread()
    camera.frame_lock = Lock()
    camera.new_frame_event = Event()
    camera.latest_color_frame = np.zeros((2, 3, 3), dtype=np.uint8)
    camera.latest_depth_frame = np.ones((2, 3), dtype=np.uint16)
    camera.latest_timestamp = 123.456
    camera.new_frame_event.set()

    color, depth, timestamp = camera.read_color_and_aligned_depth_with_timestamp()
    assert color is camera.latest_color_frame
    assert depth is camera.latest_depth_frame
    assert timestamp == 123.456

    camera.new_frame_event.set()
    legacy_color, legacy_depth = camera.read_color_and_aligned_depth()
    assert legacy_color is color and legacy_depth is depth
