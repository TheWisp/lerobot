"""Join offline RGB-D scene evidence to recorded motor frames by capture ID."""

from __future__ import annotations

from copy import deepcopy

import numpy as np


def attach_scene_evidence(motor_replay: dict, ticks: list[dict], scene_replay: dict) -> dict:
    """Add reprocessed vision to the action timeline without replacing recorded decisions.

    One camera capture may occur in several motor rows. Its mask and pose are
    repeated only for display on those rows, never re-run through the tracker.
    The accepted pose from the original motor trace remains the original run's
    decision; offline scene poses are separately named comparison layers.
    """
    if not scene_replay.get("evidence_available"):
        return motor_replay
    keyed = {}
    for scene_frame in scene_replay["frames"]:
        key = scene_frame.get("source_camera_frame_id")
        if key is None:
            key = scene_frame["frame_index"]
        if key in keyed:
            raise ValueError("scene replay contains duplicate camera captures")
        keyed[key] = scene_frame
    ticks_by_index = {tick["frame_index"]: tick for tick in ticks}
    replay = deepcopy(motor_replay)
    aligned = 0
    for motor_frame in replay["frames"]:
        index = motor_frame["frame_index"]
        tick = ticks_by_index.get(index)
        if tick is None:
            continue
        key = tick.get("camera_frame_id")
        if key is None:
            key = index
        scene_frame = keyed.get(key)
        if scene_frame is None:
            continue
        scene_snapshot = scene_frame["snapshot"]
        scene_camera = scene_snapshot["camera"]
        if tick.get("camera_source_time_s") is not None and (
            tick["camera_source_time_s"] != scene_camera["source_time_s"]
        ):
            raise ValueError("scene capture time differs from the motor trace")
        snapshot = motor_frame["snapshot"]
        recorded_camera = snapshot.get("camera")
        if recorded_camera and recorded_camera.get("frame_id") != key:
            raise ValueError("recorded camera and scene evidence have different capture IDs")
        if recorded_camera:
            for field in ("base_from_camera", "intrinsics"):
                if field in recorded_camera and not np.allclose(
                    recorded_camera[field], scene_camera[field], rtol=0, atol=1e-6
                ):
                    raise ValueError(f"recorded camera and scene evidence disagree on {field}")
        camera = {**scene_camera, "frame_id": key, "source_camera_frame_id": key}
        snapshot["camera"] = {**(recorded_camera or {}), **camera}
        snapshot.setdefault("recorded_at_s", tick.get("observed_at_s") or tick.get("scheduled_s"))
        snapshot["reprocessed_observed_masks"] = {
            role: {**mask, "frame_id": key} for role, mask in scene_snapshot.get("observed_masks", {}).items()
        }
        snapshot["reprocessed_scene_support"] = {**scene_snapshot["scene_support"], "frame_id": key}
        snapshot["reprocessed_raw_proposals"] = {
            role: {**proposal, "frame_id": key}
            for role, proposal in scene_snapshot.get("raw_proposals", {}).items()
        }
        for role, estimate in scene_snapshot.get("accepted", {}).items():
            snapshot.setdefault("layers", {}).setdefault(role, {})["reprocessed_scene"] = estimate
        snapshot["reprocessed_scene_status"] = scene_snapshot.get("pose_status", {})
        if scene_frame.get("evidence"):
            motor_frame["evidence"] = {**scene_frame["evidence"], "frame_id": key}
        motor_frame["scene_evidence_frame_index"] = scene_frame["frame_index"]
        aligned += 1
    # These points are defined in the reprocessed estimator's object frame.
    # A prepared skill may have a different object-frame origin and reference
    # patch, so never reuse its geometry to draw this estimate.
    replay["reprocessed_scene_geometry"] = scene_replay.get("geometry", {})
    replay["scene_evidence"] = {
        "available": True,
        "camera_frames": len(scene_replay["frames"]),
        "motor_frames_aligned": aligned,
        "source": "offline_rgbd_reprocessing",
    }
    return replay
