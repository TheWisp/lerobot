#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation

#: The SDK's depth modes, by the name used in a profile. ``none`` opens the
#: camera as a plain rectified stereo pair with no depth computed.
DEPTH_MODES = ("none", "performance", "quality", "ultra", "neural_light", "neural", "neural_plus")

#: Eye resolutions the SDK names, keyed by (width, height) of one eye.
RESOLUTIONS = {
    (2208, 1242): "HD2K",
    (1920, 1080): "HD1080",
    (1920, 1200): "HD1200",
    (1280, 720): "HD720",
    (960, 600): "SVGA",
    (672, 376): "VGA",
}


@CameraConfig.register_subclass("zed")
@dataclass
class ZEDCameraConfig(CameraConfig):
    """A Stereolabs ZED camera through the ZED SDK.

    The SDK rectifies both eyes and, when a depth mode is set, computes depth on
    the GPU aligned to the left eye. ``pyzed`` is not on PyPI: it is the wheel
    that ships with the SDK installed on the host (``/usr/local/zed``), so a
    host needs the SDK before this camera can open. See ``docs/`` for the
    install step.

    Attributes:
        serial_number: The unit to open. ``None`` opens the first ZED found.
        fps, width, height: The stream to request. ``width`` x ``height`` is one
            eye as published, after ``rotation``; the sensor's frame, which must
            be one of the SDK's resolutions (``RESOLUTIONS``), is the transpose
            of that under a quarter turn.
        depth_mode: One of ``DEPTH_MODES``. The neural modes need the SDK's AI
            model optimized once per host, which the SDK does on first use.
        use_depth: Publish depth as ``<camera>_depth``, (H, W, 1) uint16
            millimetres aligned to the left eye, 0 where there is no measurement.
            Needs a ``depth_mode`` other than ``none``.
        stereo_split: Publish both eyes as ``<camera>_l`` and ``<camera>_r``,
            the names the side-by-side UVC path uses, so a dataset recorded
            either way reads the same. Off, only the left eye is published under
            ``<camera>``.
        depth_min_m, depth_max_m: The SDK's working range for depth.
        fill_holes: Let the SDK fill unmeasured pixels. Off by default so a
            stored 0 means "no measurement" rather than a guess.
        color_mode, rotation, warmup_s: As for the other cameras.
    """

    serial_number: int | None = None
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    depth_mode: str = "neural_light"
    use_depth: bool = False
    stereo_split: bool = True
    depth_min_m: float = 0.2
    depth_max_m: float = 10.0
    fill_holes: bool = False

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        if self.depth_mode not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {DEPTH_MODES}, got {self.depth_mode!r}")
        if self.use_depth and self.depth_mode == "none":
            raise ValueError("use_depth needs a depth_mode other than 'none'")
        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
        if self.width is not None:
            quarter = self.rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270)
            sensor = (self.height, self.width) if quarter else (self.width, self.height)
            if sensor not in RESOLUTIONS:
                raise ValueError(
                    f"{sensor[0]}x{sensor[1]} (the sensor's frame behind a published {self.width}x{self.height}) "
                    f"is not a ZED eye resolution; one of {sorted(RESOLUTIONS)}"
                )
        if not 0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("need 0 < depth_min_m < depth_max_m")
