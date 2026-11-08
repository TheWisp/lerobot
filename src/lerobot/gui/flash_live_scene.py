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
    before = frame.depth_before_filters
    before_bytes = before.astype("<u2", copy=False).tobytes() if before is not None else b""
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
        "depth_before_filter_bytes": len(before_bytes),
        "depth_m_per_unit": 1.0,  # The primary depth payload is already metric float32.
        "depth_before_filter_m_per_unit": frame.depth_m_per_unit,
        "source_frame_id": frame.frame_id,
        "available_time_s": frame.available_time_s,
        "base_from_camera": frame.base_from_camera.tolist(),
        "calibration_version": frame.calibration_version,
    }
    metadata = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    return struct.pack("<I", len(metadata)) + metadata + depth.tobytes() + before_bytes + encoded.tobytes()


def object_evidence(result, method):
    """The same saved decision fields in video inspection and live history."""
    status = result.get(method + "_status")
    details = result.get(method + "_details", {})
    return {
        "pose": result.get(method),
        "status": status,
        "reason": loss_reason(status, details),
        "reason_code": details.get("reason") if status == "lost" else None,
        "diagnostics": {key: value for key, value in details.items() if key != "group"},
    }


def frame_evidence(row, masks):
    # Preserve the exact masks, including holes. Display contours alone cannot
    # reproduce the observation supplied to the pipeline.
    runs = {}
    for role, mask in masks.items():
        edges = np.flatnonzero(np.diff(np.pad(mask.ravel().astype(np.int8), (1, 1))))
        runs[role] = np.column_stack((edges[::2], edges[1::2] - edges[::2])).tolist()
    return {"evidence_schema": 1, "result": row, "masks_rle": runs}


def live_scene_packet(
    frame, live, row, masks, matches=None, *, group_overlay=None, depth_processing=None, depth_source="sensor"
):
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
            **object_evidence(result, live.method),
            "axis_points": reference_axes(reference).tolist(),
            "surface_used": track._prepared_reference.points.tolist()
            if track._prepared_reference is not None
            else [],
            "contours": [c[:, 0].tolist() for c in contours],
            "group_tracks": group,
        }
    # A missing mask/initialization still needs a useful image and status.
    for role in live.engine.sessions:
        result = row["objects"][role]
        objects.setdefault(
            role,
            {
                "name": live.mask_source.prompts[role],
                "surface": [],
                **object_evidence(result, live.method),
                "contours": [],
            },
        )
    return encode_rgbd_scene(
        frame,
        {
            "frame": frame.frame_id,
            "time_s": frame.source_time_s,
            "method": live.method,
            "objects": objects,
            "scene_group_tracks": (
                group_overlay.update(
                    row.get("scene_group_tracks", []), frame_size=(frame.rgb.shape[1], frame.rgb.shape[0])
                )
                if group_overlay is not None
                else row.get("scene_group_tracks", [])
            ),
            **frame_evidence(row, masks),
            "image_matches": matches,
            "depth_processing": depth_processing,
            "depth_source": depth_source,
            "depth_filter_revision": frame.depth_filter_revision,
        },
    )
