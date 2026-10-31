"""The RGB-D observation and its host-clock time must be one atomic snapshot."""

from threading import Event, Lock, Thread

import numpy as np
import pytest

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.utils.errors import DeviceNotConnectedError


class RunningThread:
    def is_alive(self):
        return True


def test_aligned_rgbd_returns_capture_timestamp_without_changing_old_api():
    camera = object.__new__(RealSenseCamera)
    camera.rs_pipeline = object()
    camera.rs_profile = object()
    camera.serial_number = "test-camera"
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


def test_reader_cannot_consume_one_cached_frameset_twice_during_publication():
    class PausingEvent(Event):
        def __init__(self):
            super().__init__()
            self.set_count = 0
            self.second_set_started = Event()
            self.release_second_set = Event()

        def set(self):
            self.set_count += 1
            if self.set_count == 2:
                self.second_set_started.set()
                assert self.release_second_set.wait(2)
            super().set()

    class Frame:
        def __init__(self, value):
            self.data = np.full((2, 3), value, dtype=np.uint16)

        def get_data(self):
            return self.data

    class Frameset:
        def __init__(self, value):
            self.frame = Frame(value)

        def get_color_frame(self):
            return self.frame

        def get_depth_frame(self):
            return self.frame

    camera = object.__new__(RealSenseCamera)
    camera.rs_pipeline = object()
    camera.rs_profile = object()
    camera.serial_number = "test-camera"
    camera.use_rgb = True
    camera.use_depth = True
    camera.align_to_color = None
    camera.post_grab_processor = None
    camera.frame_lock = Lock()
    camera.new_frame_event = PausingEvent()
    camera.stop_event = Event()
    camera.latest_color_frame = None
    camera.latest_depth_frame = None
    camera.latest_timestamp = None
    camera._postprocess_image = lambda frame, depth_frame=False: frame
    frames = iter((Frameset(1), Frameset(2)))

    def read_hardware():
        try:
            return next(frames)
        except StopIteration:
            camera.stop_event.wait(2)
            raise DeviceNotConnectedError("stop test camera") from None

    camera._read_from_hardware = read_hardware
    camera.thread = Thread(target=camera._read_loop, daemon=True)
    camera.thread.start()
    try:
        assert camera.new_frame_event.second_set_started.wait(2)
        result = []
        reader = Thread(
            target=lambda: result.append(camera.read_color_and_aligned_depth_with_timestamp(timeout_ms=500)),
            daemon=True,
        )
        reader.start()
        camera.new_frame_event.release_second_set.set()
        reader.join(2)
        assert not reader.is_alive()
        assert len(result) == 1
        assert np.all(result[0][0] == 2)
        with pytest.raises(TimeoutError):
            camera.read_color_and_aligned_depth_with_timestamp(timeout_ms=30)
    finally:
        camera.new_frame_event.release_second_set.set()
        camera.stop_event.set()
        camera.thread.join(2)


def test_cached_rgbd_frames_do_not_hold_realsense_buffers():
    color_buffer = bytearray(2 * 3 * 3)
    depth_buffer = bytearray(2 * 3 * 2)
    color = np.frombuffer(color_buffer, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.frombuffer(depth_buffer, dtype=np.uint16).reshape(2, 3)
    color[:] = 17
    depth[:] = 42

    class Frame:
        def __init__(self, data):
            self.data = data

        def get_data(self):
            return self.data

    class Frameset:
        def get_color_frame(self):
            return Frame(color)

        def get_depth_frame(self):
            return Frame(depth)

    camera = object.__new__(RealSenseCamera)
    camera.rs_pipeline = object()
    camera.rs_profile = object()
    camera.serial_number = "test-camera"
    camera.use_rgb = True
    camera.use_depth = True
    camera.align_to_color = type("Align", (), {"process": lambda self, frame: frame})()
    camera._apply_depth_filters = lambda frame: frame
    camera._postprocess_image = lambda frame, depth_frame=False: frame
    camera.post_grab_processor = None
    camera.frame_lock = Lock()
    camera.new_frame_event = Event()
    camera.stop_event = Event()
    camera.latest_color_frame = None
    camera.latest_depth_frame = None
    camera.latest_timestamp = None
    calls = 0
    finish = Event()

    def read_hardware():
        nonlocal calls
        calls += 1
        if calls == 1:
            return Frameset()
        finish.wait(2)
        raise DeviceNotConnectedError("end fake stream")

    camera._read_from_hardware = read_hardware
    camera.thread = Thread(target=camera._read_loop, daemon=True)
    camera.thread.start()
    try:
        assert camera.new_frame_event.wait(2)
        cached_color, cached_depth, _ = camera.read_color_and_aligned_depth_with_timestamp()
        assert cached_color.flags.owndata and cached_depth.flags.owndata
        color[:] = 99
        depth[:] = 77
        assert np.all(cached_color == 17)
        assert np.all(cached_depth == 42)
    finally:
        finish.set()
        camera.stop_event.set()
        camera.thread.join(2)
