"""Frame-aligned display packet from the live shared session; no estimation."""

import json
import struct

import cv2
import numpy as np

from .flash_pose_overlay import reference_axes


def encode_rgbd_scene(frame, header):
    depth = frame.metric_depth()
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype("<f4")
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Cannot encode RGB frame")
    header = {
        **header,
        "camera": {
            "width": depth.shape[1],
            "height": depth.shape[0],
            "intrinsics": frame.intrinsics.tolist(),
        },
        "depth_bytes": depth.nbytes,
    }
    metadata = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    return struct.pack("<I", len(metadata)) + metadata + depth.tobytes() + encoded.tobytes()


def live_scene_packet(frame, live, row, masks, overlays):
    objects = {}
    for role, reference in live.engine.references.items():
        track = live.engine.candidates[live.method][role]
        result = row["objects"][role]
        contours, _ = cv2.findContours(
            masks[role].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        group = result[live.method + "_details"].get("group", {})
        objects[role] = {
            "name": live.mask_source.prompts[role],
            "surface": track.store.points.tolist(),
            "pose": result[live.method],
            "axis_points": reference_axes(reference).tolist(),
            "status": result[live.method + "_status"],
            "contours": [c[:, 0].tolist() for c in contours],
            "group_tracks": {**group, "links": overlays[role].update(group)["links"]},
        }
    # A missing mask/initialization still needs a useful image and status.
    for role in live.engine.sessions:
        objects.setdefault(
            role,
            {
                "name": live.mask_source.prompts[role],
                "surface": [],
                "pose": None,
                "status": row["objects"][role][live.method + "_status"],
                "contours": [],
            },
        )
    return encode_rgbd_scene(
        frame,
        {"frame": frame.frame_id, "time_s": frame.source_time_s, "method": live.method, "objects": objects},
    )
