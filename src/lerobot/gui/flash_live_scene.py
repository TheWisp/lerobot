"""Frame-aligned display packet from the live shared session; no estimation."""

import json
import struct

import cv2
import numpy as np

from .flash_pose_overlay import reference_axes


def loss_reason(status, details):
    """Explain an unavailable pose using the checks made on this frame."""
    if status != "lost":
        return None
    reason = details.get("reason")
    if reason == "no_mask":
        return "Object mask missing"
    if reason == "insufficient_depth":
        return "Too few usable depth points"
    if reason == "not_initialized":
        return "No starting surface yet"
    if reason == "surface_removed":
        return "Saved surface unavailable"
    if reason == "surface_group_disagreement":
        return "Surface and tracked point group disagree"
    prediction = details.get("prediction_geometry")
    check = prediction or (details.get("fit") or {}).get("candidate_check") or {}
    subject = "Prediction" if prediction else "Fit"
    if check.get("scene_depth_coherent_protrusion"):
        cause = f"{subject} protrudes into measured empty space"
    elif check.get("has_visible_overlap") is False:
        cause = f"{subject} misses the visible surface"
    elif check.get("geometry_accepted") is False:
        cause = f"{subject} disagrees with measured depth"
    elif reason == "insufficient_surface_overlap" or check.get("accepted") is False:
        cause = "Too little overlap with the saved surface"
    else:
        cause = "No pose passed the current image checks"
    return cause


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
        status = result[live.method + "_status"]
        details = result[live.method + "_details"]
        objects[role] = {
            "name": live.mask_source.prompts[role],
            "surface": track.store.points.tolist(),
            "pose": result[live.method],
            "axis_points": reference_axes(reference).tolist(),
            "status": status,
            "reason": loss_reason(status, details),
            "reason_code": details.get("reason") if status == "lost" else None,
            "contours": [c[:, 0].tolist() for c in contours],
            "group_tracks": {**group, "links": overlays[role].update(group)["links"]},
        }
    # A missing mask/initialization still needs a useful image and status.
    for role in live.engine.sessions:
        result = row["objects"][role]
        status = result[live.method + "_status"]
        details = result[live.method + "_details"]
        objects.setdefault(
            role,
            {
                "name": live.mask_source.prompts[role],
                "surface": [],
                "pose": None,
                "status": status,
                "reason": loss_reason(status, details),
                "reason_code": details.get("reason") if status == "lost" else None,
                "contours": [],
            },
        )
    return encode_rgbd_scene(
        frame,
        {"frame": frame.frame_id, "time_s": frame.source_time_s, "method": live.method, "objects": objects},
    )
