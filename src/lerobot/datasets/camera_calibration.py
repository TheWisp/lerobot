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
"""What a dataset records about the cameras that produced its frames.

A camera feature's ``info`` dict, the same place the depth unit and the
quantizer live, gets a ``camera`` block with the unit's intrinsics for the
frames as stored. A 3D view, or anything that measures in the images, can
then read a dataset without the camera attached. The block is written after
the robot connects, because that is when a driver knows its calibration, and
only for cameras that do; a UVC camera without a calibration writes nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lerobot.datasets.io_utils import write_info
from lerobot.utils.constants import OBS_IMAGES

CAMERA_INFO_KEY = "camera"
#: How the intrinsics' axes relate to the pictures: the pinhole convention
#: every vendor SDK deprojects into.
OPTICAL_FRAME = "optical: x right, y down, z forward"


def camera_info_block(cam: Any, intrinsics: Any = None) -> dict[str, Any] | None:
    """The block for one camera object, or ``None`` when it has no intrinsics.
    ``intrinsics`` selects another of the camera's calibrations, the right
    eye's of a stereo camera, in place of its primary one."""
    intrinsics = intrinsics if intrinsics is not None else getattr(cam, "intrinsics", None)
    if intrinsics is None:
        return None
    block: dict[str, Any] = {"source": "device", "frame": OPTICAL_FRAME, "intrinsics": intrinsics.to_dict()}
    for field, attr in (("serial", "serial_number"), ("model", "device_name")):
        value = getattr(cam, attr, None)
        if value:
            block[field] = str(value)
    return block


def _feature_blocks(cam_key: str, cam: Any) -> list[tuple[str, dict[str, Any], str | None]]:
    """``(feature key, block, aligned_to)`` for every feature this camera
    publishes: its colour feature, or both eyes of a stereo camera, and the
    depth feature, which shares the left (or only) colour feature's geometry
    and, when the driver says what it did to the pixels (``depth_processing``),
    carries that too, so a reader knows whether a hole is the sensor's own."""
    block = camera_info_block(cam)
    if block is None:
        return []
    if getattr(cam, "stereo_split", False):
        from lerobot.cameras.stereo import stereo_channel_keys

        left_key, right_key = (f"{OBS_IMAGES}.{k}" for k in stereo_channel_keys(cam_key))
        out = [(left_key, block, None)]
        right_block = camera_info_block(cam, getattr(cam, "intrinsics_right", None))
        if right_block is not None:
            out.append((right_key, right_block, None))
        colour_key = left_key
    else:
        colour_key = f"{OBS_IMAGES}.{cam_key}"
        out = [(colour_key, block, None)]
    depth_block = block
    processing = getattr(cam, "depth_processing", None)
    if processing:
        depth_block = dict(block, depth_processing=dict(processing))
    out.append((f"{OBS_IMAGES}.{cam_key}_depth", depth_block, colour_key))
    return out


def attach_camera_intrinsics(
    meta: Any,
    cameras: Mapping[str, Any],
    *,
    overwrite: bool = False,
    feature_names: Mapping[str, str] | None = None,
) -> list[str]:
    """Write each camera's block into its colour feature (both eyes, for a
    stereo camera) and, when the depth feature has the same frame size, into
    that one too with ``aligned_to`` naming the colour feature it shares a
    geometry with. Persists ``info.json`` when anything was written. Returns
    the feature keys written.

    ``feature_names`` maps a camera's key to the name its features carry when
    the two differ: a bimanual robot keeps per-arm cameras under their bare
    names but publishes them as ``left_<name>``. Without the map the key is
    the name.

    A depth feature of another size is left alone: the colour intrinsics do
    not describe it, and the driver's depth stream intrinsics are not carried.
    """
    written: list[str] = []
    for cam_key, cam in cameras.items():
        name = (feature_names or {}).get(cam_key, cam_key)
        for key, block, aligned_to in _feature_blocks(name, cam):
            feature = meta.features.get(key)
            if feature is None:
                continue
            size = (block["intrinsics"]["height"], block["intrinsics"]["width"])
            shape = [int(x) for x in (feature.get("shape") or [])]
            if len(shape) >= 2 and (shape[0], shape[1]) != size:
                continue
            info = feature.setdefault("info", {})
            if CAMERA_INFO_KEY in info and not overwrite:
                continue
            info[CAMERA_INFO_KEY] = dict(block, **({"aligned_to": aligned_to} if aligned_to else {}))
            written.append(key)
    if written:
        write_info(meta.info, meta.root)
    return written
