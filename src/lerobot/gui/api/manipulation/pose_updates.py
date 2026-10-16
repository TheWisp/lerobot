"""Common accepted pose observations, independent of an object's interaction role.

These are measurements in a named demo reference frame, never group/gripper
predictions. Reference selection and cross-instance registration can expand
behind this interface without changing attachment consumers.
"""

from dataclasses import dataclass

import numpy as np

from . import planning, runtime


@dataclass(frozen=True)
class PoseObservation:
    object: str
    reference_frame: int
    capture_mono: float
    reference_to_camera: np.ndarray
    source: str = "verified tracking"
    reference_point_cam: np.ndarray | None = None


def latest_verified(demo, obj: str) -> PoseObservation | None:
    """The latest accepted measurement, with the SAME camera timestamp as its pose."""
    with runtime._state.lock:
        state = runtime._state
        model = state.track.model
        if state.demo is not demo or not state.track.on or model is None:
            return None
        stamp = model.get("capture_mono")
        if stamp is None or obj not in demo.objects:
            return None
        if obj == planning._session_object(demo):
            pose = model.get("delta_cam")
            ref = (state.teach.keypoints.get("ref") or {}) if state.teach is not None else {}
            anchor = ref.get("delta")
            if pose is None or anchor is None:
                return None
            point = (model.get("surface_constraint") or {}).get("reference_point")
            if point is not None:
                point = (np.linalg.inv(anchor) @ np.r_[point, 1.0])[:3]
            pose = np.asarray(pose) @ np.asarray(anchor)
        else:
            target = state.targets.get(obj)
            if target is None or not target.hold_views:
                return None
            point = (target.last.get("surface_constraint") or {}).get("reference_point")
            at, pose = target.hold_views[-1]
            if at != model["t"]:
                return None  # a retained old pose is not a new observation
        return PoseObservation(
            obj,
            int(demo.objects[obj]["frame"]),
            float(stamp),
            np.array(pose, copy=True),
            reference_point_cam=None if point is None else np.array(point, copy=True),
        )


def attachment_sample(observation, tip, t_bc, demo_attachment, reference_point_base):
    """Adapt a common pose into gripper coordinates, downstream of estimation.

    demo_attachment maps the designation's BASE-frame points into the demo
    gripper. It must itself be verified; an uncertain mapping cannot be repaired
    merely by observing the live object more accurately.
    """
    live = np.linalg.inv(tip) @ t_bc @ observation.reference_to_camera @ np.linalg.inv(t_bc)
    point = np.r_[reference_point_base, 1.0]
    return {
        "stamp": observation.capture_mono,
        "frame": observation.reference_frame,
        "desired": (demo_attachment @ point)[:3],
        "observed": (live @ point)[:3],
        "rotation": demo_attachment[:3, :3] @ live[:3, :3].T,
    }
