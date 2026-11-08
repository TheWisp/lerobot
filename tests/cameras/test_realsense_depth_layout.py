# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The grab thread's depth layout, driven without hardware.

A robot declares its depth feature as ``(H, W, 1)`` and the dataset writer
rejects anything else, so the cached depth map must carry the channel axis on
every branch of the grab loop. The aligned branch (RGB + depth, the default
when depth is enabled) used to cache ``(H, W)``. The post-grab processor
protocol, on the other hand, takes the plain ``(H, W)`` map.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyrealsense2")

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

H, W = 720, 1280


class _Frame:
    def __init__(self, arr):
        self._arr = arr

    def get_data(self):
        return self._arr


class _Frameset:
    def __init__(self, color, depth, infrared=None):
        self._color, self._depth = color, depth
        self._infrared = infrared

    def get_color_frame(self):
        return _Frame(self._color)

    def get_depth_frame(self):
        return _Frame(self._depth)

    def get_infrared_frame(self, index):
        return _Frame(self._infrared[index - 1]) if self._infrared is not None else None


class _Align:
    def process(self, frameset):
        return frameset


def _camera_with_one_aligned_grab(depth_2d: np.ndarray) -> RealSenseCamera:
    """A camera whose grab loop sees exactly one aligned frameset and stops."""
    cam = RealSenseCamera(
        RealSenseCameraConfig(serial_number_or_name="0", fps=30, width=W, height=H, use_depth=True)
    )
    cam.align_to_color = _Align()
    cam.stop_event = threading.Event()
    color = np.zeros((H, W, 3), np.uint8)

    def one_grab():
        cam.stop_event.set()
        return _Frameset(color, depth_2d)

    cam._read_from_hardware = one_grab
    return cam


def test_depth_processing_names_the_filters_that_run():
    """The recording's camera block carries this, so it must say what the grab
    loop does, not what was configured: the filters run on the aligned path only."""
    cam = RealSenseCamera(
        RealSenseCameraConfig(serial_number_or_name="0", use_depth=True, enable_temporal=False)
    )
    assert cam.depth_processing == {
        "decimation": True,
        "spatial": True,
        "temporal": False,
        "hole_filling": True,
    }
    depth_only = RealSenseCamera(
        RealSenseCameraConfig(serial_number_or_name="0", use_rgb=False, use_depth=True)
    )
    assert not any(depth_only.depth_processing.values())


def test_depth_filters_switch_live_and_trace_the_applied_chain(monkeypatch):
    from lerobot.cameras.realsense import camera_realsense

    calls = []
    instances = []

    class Filter:
        def __init__(self, name):
            self.name = name
            instances.append(self)

        def set_option(self, *_):
            pass

        def process(self, value):
            calls.append(self.name)
            return value

    fake_rs = SimpleNamespace(
        option=SimpleNamespace(filter_magnitude=1, filter_smooth_alpha=2, filter_smooth_delta=3),
        spatial_filter=lambda: Filter("spatial"),
        temporal_filter=lambda: Filter("temporal"),
        hole_filling_filter=lambda: Filter("holes"),
        disparity_transform=lambda forward: Filter("to_disparity" if forward else "to_depth"),
    )
    monkeypatch.setattr(camera_realsense, "rs", fake_rs)
    cam = RealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name="0",
            use_depth=True,
            enable_decimation=False,
            enable_spatial=False,
            enable_temporal=False,
            enable_hole_filling=False,
        )
    )
    frame = object()
    assert cam._apply_depth_filters(frame) is frame
    assert calls == []
    assert cam.last_applied_depth_processing == (
        dict.fromkeys(("decimation", "spatial", "temporal", "hole_filling"), False),
        0,
    )

    cam.set_depth_processing(spatial=True, temporal=True, hole_filling=True)
    assert cam._apply_depth_filters(frame) is frame
    assert calls == ["to_disparity", "spatial", "temporal", "to_depth", "holes"]
    assert cam.last_applied_depth_processing[0] == {
        "decimation": False,
        "spatial": True,
        "temporal": True,
        "hole_filling": True,
    }
    assert cam.last_applied_depth_processing[1] == 1
    original_temporal = cam.temporal_filter
    cam.set_depth_processing(spatial=False, temporal=False, hole_filling=False)
    calls.clear()
    cam._apply_depth_filters(frame)
    assert calls == []
    assert cam.last_applied_depth_processing[1] == 2
    cam.set_depth_processing(spatial=False, temporal=True, hole_filling=False)
    calls.clear()
    cam._apply_depth_filters(frame)
    assert calls == ["to_disparity", "temporal", "to_depth"]
    assert cam.temporal_filter is not original_temporal  # no stale temporal history
    assert cam.last_applied_depth_processing[1] == 3


def test_aligned_depth_is_cached_channel_last():
    cam = _camera_with_one_aligned_grab(np.full((H, W), 443, np.uint16))

    cam._read_loop()

    assert cam.latest_depth_frame.shape == (H, W, 1)
    assert cam.latest_depth_frame.dtype == np.uint16
    assert cam.latest_color_frame.shape == (H, W, 3)


def test_before_and_after_depth_are_same_frame_and_owned():
    measured = np.full((H, W), 443, np.uint16)
    measured[8, 9] = 0
    cam = _camera_with_one_aligned_grab(measured)
    filled = measured.copy()
    filled[8, 9] = 442
    cam._apply_depth_filters = lambda _: _Frame(filled)
    pairs = []
    cam.aligned_frame_with_raw_listener = lambda rgb, after, before, stamp, seq: pairs.append(
        (seq, after[8, 9, 0], before[8, 9])
    )

    cam._read_loop()

    assert pairs == [(0, 442, 0)]
    assert cam.latest_depth_frame[8, 9, 0] == 442
    assert cam.latest_depth_before_filters[8, 9] == 0
    measured[8, 9] = 999
    assert cam.latest_depth_before_filters[8, 9] == 0


def test_stereo_pair_is_synchronized_and_owned():
    depth = np.full((H, W), 443, np.uint16)
    cam = _camera_with_one_aligned_grab(depth)
    cam.config.enable_infrared = True
    left = np.full((480, 848), 17, np.uint8)
    right = np.full((480, 848), 29, np.uint8)
    cam._read_from_hardware = lambda: _Frameset(np.zeros((H, W, 3), np.uint8), depth, (left, right))
    events = []
    cam.aligned_frame_with_stereo_listener = lambda *args: events.append(args)
    original_read = cam._read_from_hardware

    def one_grab():
        cam.stop_event.set()
        return original_read()

    cam._read_from_hardware = one_grab
    cam._read_loop()
    assert len(events) == 1
    np.testing.assert_array_equal(events[0][3], left)
    np.testing.assert_array_equal(events[0][4], right)
    assert events[0][-1] == 0
    left[:] = 0
    assert cam.latest_infrared_pair[0][0, 0] == 17


def test_post_grab_processor_receives_the_plain_map_and_consumes_it():
    cam = _camera_with_one_aligned_grab(np.full((H, W), 443, np.uint16))
    seen = {}

    class _Processor:
        def process_frame_with_depth(self, color, depth):
            seen["shape"] = depth.shape
            return color

    cam.post_grab_processor = _Processor()

    cam._read_loop()

    assert seen["shape"] == (H, W)
    assert cam.latest_depth_frame is None
