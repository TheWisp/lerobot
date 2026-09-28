# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The dataset viewer's JPEG funnel must colour a depth map, not cast it.

The reader returns depth as float millimetres ``(1, H, W)``; the funnel used
to multiply by 255 and clamp, which turned every depth tile into a white
sheet.
"""

import torch

from lerobot.gui.frame_cache import depth_to_rgb_preview, encode_frame_to_jpeg

H, W = 720, 1280


def _bench_depth() -> torch.Tensor:
    depth = torch.zeros((1, H, W), dtype=torch.float32)
    depth[0, :, : W // 2] = 300.0  # near, mm
    depth[0, :, W // 2 :] = 900.0  # far
    depth[0, :10, :10] = 0.0  # no measurement
    return depth


def test_preview_is_warm_near_cool_far_and_black_where_invalid():
    rgb = depth_to_rgb_preview(_bench_depth())

    assert rgb.shape == (3, H, W)
    assert rgb.dtype == torch.uint8
    assert torch.all(rgb[:, :10, :10] == 0)
    near, far = rgb[:, H // 2, 100], rgb[:, H // 2, W - 100]
    assert not torch.equal(near, far)
    assert near[0] > far[0]  # more red at the near end
    assert far[2] > near[2]  # more blue at the far end


def test_preview_accepts_every_single_channel_layout():
    chw = _bench_depth()
    hw = chw[0]
    hwc = chw.permute(1, 2, 0)
    ref = depth_to_rgb_preview(chw)
    assert torch.equal(depth_to_rgb_preview(hw), ref)
    assert torch.equal(depth_to_rgb_preview(hwc), ref)


def test_preview_of_a_frame_without_measurements_is_black():
    assert torch.all(depth_to_rgb_preview(torch.zeros((1, H, W))) == 0)


def test_jpeg_funnel_routes_depth_through_the_colormap():
    from torchvision.io import decode_jpeg

    jpeg = encode_frame_to_jpeg(_bench_depth())

    img = decode_jpeg(torch.frombuffer(bytearray(jpeg), dtype=torch.uint8))
    assert img.shape == (3, H, W)
    # Not the white sheet a cast produces: the two halves differ.
    assert not torch.equal(img[:, H // 2, 100], img[:, H // 2, W - 100])
    assert int(img[:, H // 2, 100].float().mean()) < 250
