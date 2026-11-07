# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Intrinsics describe the published frames, so a rotation setting must turn
them with the picture: the same physical ray, read through the rotated pixel
with the rotated numbers, is the original ray turned by the same quarter."""

import cv2
import pytest

from lerobot.cameras.intrinsics import CameraIntrinsics

# The D435 colour stream at 1280x720, as its SDK reported it.
D435 = CameraIntrinsics(width=1280, height=720, fx=906.30, fy=906.29, cx=632.84, cy=375.97)


def _ray(i: CameraIntrinsics, u: float, v: float) -> tuple[float, float]:
    return (u - i.cx) / i.fx, (v - i.cy) / i.fy


@pytest.mark.parametrize("u,v", [(100, 50), (1279, 719), (632.84, 375.97)])
def test_quarter_turns_keep_the_ray(u, v):
    x, y = _ray(D435, u, v)
    cw = D435.rotated(cv2.ROTATE_90_CLOCKWISE)
    assert (cw.width, cw.height) == (720, 1280)
    assert _ray(cw, (D435.height - 1) - v, u) == pytest.approx((-y, x), abs=1e-9)
    ccw = D435.rotated(cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert _ray(ccw, v, (D435.width - 1) - u) == pytest.approx((y, -x), abs=1e-9)
    flipped = D435.rotated(cv2.ROTATE_180)
    assert _ray(flipped, (D435.width - 1) - u, (D435.height - 1) - v) == pytest.approx((-x, -y), abs=1e-9)


def test_no_rotation_is_identity_and_unknown_codes_are_refused():
    assert D435.rotated(None) is D435
    with pytest.raises(ValueError):
        D435.rotated(99)


def test_round_trips_through_a_dict():
    i = CameraIntrinsics(
        640, 480, 600.0, 601.0, 320.5, 240.5, model="brown_conrady", coeffs=(0.1, 0.0, 0.0, 0.0, 0.0)
    )
    assert CameraIntrinsics.from_dict(i.to_dict()) == i
    assert i.to_dict()["coeffs"] == [0.1, 0.0, 0.0, 0.0, 0.0]
