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
    def __init__(self, color, depth):
        self._color, self._depth = color, depth

    def get_color_frame(self):
        return _Frame(self._color)

    def get_depth_frame(self):
        return _Frame(self._depth)


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


def test_aligned_depth_is_cached_channel_last():
    cam = _camera_with_one_aligned_grab(np.full((H, W), 443, np.uint16))

    cam._read_loop()

    assert cam.latest_depth_frame.shape == (H, W, 1)
    assert cam.latest_depth_frame.dtype == np.uint16
    assert cam.latest_color_frame.shape == (H, W, 3)


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
