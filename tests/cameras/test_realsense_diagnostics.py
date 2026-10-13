"""Diagnostic recording must preserve sensor pairing and the live filter history."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyrealsense2")

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig, camera_realsense as module


class Frame:
    def __init__(self, values, timestamp=100.0, number=7):
        self.values, self.timestamp, self.number = values, timestamp, number

    def get_data(self):
        return self.values

    def get_timestamp(self):
        return self.timestamp

    def get_frame_timestamp_domain(self):
        return "hardware_clock"

    def get_frame_number(self):
        return self.number

    def supports_frame_metadata(self, key):
        return key == module.rs.frame_metadata_value.sensor_timestamp

    def get_frame_metadata(self, key):
        assert self.supports_frame_metadata(key)
        return int(self.timestamp * 1000)

    def get_units(self):
        return 0.001


def camera():
    return RealSenseCamera(RealSenseCameraConfig(serial_number_or_name="042", use_depth=True))


def test_observer_preserves_filtered_output_and_captures_pre_temporal_depth(monkeypatch):
    cam = camera()
    cam.enable_decimation = cam.enable_hole_filling = False
    calls = []

    def transform(name, delta):
        def process(frame):
            calls.append(name)
            return Frame(frame.values + delta)

        return SimpleNamespace(process=process)

    cam.depth_to_disparity = transform("to_disparity", 0)
    cam.spatial_filter = transform("spatial", 1)
    cam.temporal_filter = transform("temporal", 2)
    cam.disparity_to_depth = transform("to_depth", 0)
    monkeypatch.setattr(module.rs, "disparity_transform", lambda reverse: transform("observer_to_depth", 0))
    source = Frame(np.full((2, 2), 400, dtype=np.uint16))
    baseline = cam._apply_depth_filters(source)
    calls.clear()
    stages = {}
    observed = cam._apply_depth_filters(source, diagnostic_stages=stages)
    np.testing.assert_array_equal(observed.values, baseline.values)
    np.testing.assert_array_equal(stages["aligned_raw_depth"], 400)
    np.testing.assert_array_equal(stages["before_temporal_depth"], 401)
    assert calls.count("temporal") == 1
    assert calls.index("observer_to_depth") < calls.index("temporal")
    source.values[:] = 0
    assert (stages["aligned_raw_depth"] == 400).all(), "saved raw data must own its memory"


def test_reader_records_the_same_pair_it_publishes_with_separate_sensor_times(monkeypatch):
    cam = camera()
    cam.enable_decimation = cam.enable_spatial = cam.enable_temporal = cam.enable_hole_filling = False
    rgb = Frame(np.full((2, 2, 3), 40, np.uint8), 101.5, 10)
    depth = Frame(np.full((2, 2), 410, np.uint16), 100.0, 12)
    frameset = SimpleNamespace(get_color_frame=lambda: rgb, get_depth_frame=lambda: depth)
    cam.align_to_color = SimpleNamespace(process=lambda value: value)
    cam.stop_event = threading.Event()
    monkeypatch.setattr(cam, "_read_from_hardware", lambda: frameset)
    monkeypatch.setattr(cam, "_postprocess_image", lambda value, **kwargs: value)
    request = {"count": 1, "frames": [], "done": threading.Event()}
    cam._diagnostic_capture = request
    original_publish = cam._publish

    def publish(*args):
        original_publish(*args)
        cam.stop_event.set()

    monkeypatch.setattr(cam, "_publish", publish)
    cam._read_loop()
    assert request["done"].is_set()
    sample = request["frames"][0]
    np.testing.assert_array_equal(sample["rgb"], cam.latest_color_frame)
    np.testing.assert_array_equal(sample["filtered_depth"], cam.latest_depth_frame)
    assert sample["host_published_monotonic_s"] == cam.latest_timestamp
    assert sample["color_metadata"]["timestamp_ms"] == 101.5
    assert sample["depth_metadata"]["timestamp_ms"] == 100
    assert sample["color_metadata"]["frame_number"] == 10
    assert sample["depth_metadata"]["frame_number"] == 12
    assert sample["color_metadata"]["actual_exposure"] is None
    assert sample["color_metadata"]["sensor_timestamp"] == 101500


def test_timeout_cleans_up_request_and_late_frame_cannot_enter_next_capture(monkeypatch):
    cam = camera()
    cam.rs_profile = object()
    cam.rs_pipeline = object()
    cam.thread = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(TimeoutError):
        cam.capture_diagnostics(frames=1, timeout_s=0.001)
    assert cam._diagnostic_capture is None
    old = {"count": 1, "frames": [], "done": threading.Event()}
    new = {"count": 1, "frames": [], "done": threading.Event()}
    cam._diagnostic_capture = new
    cam._record_diagnostic_frame(old, {"old": True})
    assert not new["frames"]
    with pytest.raises(RuntimeError, match="already"):
        cam.capture_diagnostics(frames=1)
    assert cam._diagnostic_capture is new


def test_recording_stops_at_requested_count():
    cam = camera()
    request = {"count": 1, "frames": [], "done": threading.Event()}
    cam._diagnostic_capture = request
    cam._record_diagnostic_frame(request, {"index": 1})
    cam._record_diagnostic_frame(request, {"index": 2})
    assert request["frames"] == [{"index": 1}]
