"""Frame-bound selection and snapshot transport for camera inspection."""

import base64
import json
import struct
from pathlib import Path

import cv2
import numpy as np

from .appearance import Sam3MaskSource
from .perception import RGBDFrame


def tracker_depth_layout(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Adapt the camera's H×W×1 recording layout to FLASH's H×W depth map."""
    if depth.shape == (*rgb.shape[:2], 1):
        return depth[..., 0]
    if depth.shape == rgb.shape[:2]:
        return depth
    raise ValueError("camera depth is not aligned with RGB")


def write_snapshot(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def image_snapshot(frame: RGBDFrame) -> dict:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode the camera image")
    return {
        "frame_id": frame.frame_id,
        "captured_s": frame.source_time_s,
        "width": frame.rgb.shape[1],
        "height": frame.rgb.shape[0],
        "intrinsics": frame.intrinsics.tolist(),
        "depth_m_per_unit": frame.depth_m_per_unit,
        "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
    }


def preview_depth_packet(frame: RGBDFrame) -> bytes:
    """Frame ID plus aligned uint16 depth for the optional live Depth preview.

    The preview replaces one file at camera-preview cadence; it is not a
    second tracking input or a retained recording.
    """
    return struct.pack("<I", frame.frame_id) + frame.depth.astype("<u2", copy=False).tobytes()


def seed_frame(command: dict, displayed: RGBDFrame | None, latest: RGBDFrame | None) -> RGBDFrame:
    """Bind a click to the RGB-D frame that produced the displayed image."""
    if command.get("point") is not None:
        if displayed is None or command.get("frame_id") != displayed.frame_id:
            raise ValueError("That image is no longer available; click the current image again")
        x, y = command["point"]
        if not (0 <= x < displayed.rgb.shape[1] and 0 <= y < displayed.rgb.shape[0]):
            raise ValueError("The selected point is outside the camera image")
        return displayed
    frame = latest
    if frame is None:
        raise ValueError("Wait for the camera image")
    return frame


def new_selection(adapter, label: str, point, sequence: int):
    """Reuse weights, explicitly forget the old instance, and advance click IDs.

    SAM's video reset preserves clicked masks for scrub/recovery; selecting a
    different object must clear those masks through its public control API.
    """
    sequence += 1
    adapter.set_control({"click_seq": sequence, "clicks": {"debug": []}})
    adapter.reset()
    sequence += 1
    source = Sam3MaskSource(
        {"object": label},
        camera_id="debug",
        seed_points_by_role={"object": tuple(point)} if point else None,
        click_sequence_start=sequence,
        adapter_factory=lambda **_: adapter,
    )
    return source, sequence
