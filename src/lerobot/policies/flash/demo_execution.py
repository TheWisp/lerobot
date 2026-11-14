"""Small keyframe executor: shared tracked poses -> robot-owned FK/IK -> actions.

A consecutive range stores measured tool poses relative to its selected object
at range entry (or the robot base), and the demonstrated gripper commands.
The reference is acquired once before entering each range. There is no new matcher,
confidence estimator, attachment inference or recovery search here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .demo_document import DemoDocument, tracking_selection
from .geometry import interpolate, transform
from .live_surface_session import LIVE_METHOD, LiveSurfaceSession
from .perception import RGBDFrame

PLAN_VERSION = 2
POSE_AVAILABLE = {"initialized", "observed_candidate", "predicted"}


def load_reference(path, roles, *, frame_id=0, time_s=0.0, calibration_version="demo"):
    with np.load(path, allow_pickle=False) as saved:
        initial = RGBDFrame(
            frame_id,
            saved["rgb"],
            saved["depth"].astype(np.float32),
            saved["intrinsics"],
            1.0,
            np.eye(4),
            time_s,
            time_s,
            calibration_version,
        )
        masks = {role: saved[role].astype(bool) for role in roles}
    if any(mask.shape != initial.depth.shape for mask in masks.values()):
        raise ValueError("Demo reference masks differ from its depth grid")
    return initial, masks


def compile_demo(dataset, episode, document, tracking_directory: Path, robot, camera_name):
    if not document.keyframes:
        raise ValueError("Mark at least one keyframe before applying the demo")
    if dataset.meta.info.robot_geometry != robot.geometry.to_dict():
        raise ValueError("Demo and selected robot geometry differ; use the robot that recorded this demo")
    model = robot.cartesian_model()
    if dataset.meta.info.robot_cartesian != model.description():
        raise ValueError("Demo is missing matching robot joint units. Record a new demo with this setup.")
    if document.camera != f"observation.images.{camera_name}":
        raise ValueError("The selected camera differs from the demo camera")
    inputs = json.loads((tracking_directory / "input.json").read_text())
    status = json.loads((tracking_directory / "status.json").read_text())
    if status["state"] != "complete" or tracking_selection(
        DemoDocument.model_validate(inputs["document"])
    ) != tracking_selection(document):
        raise ValueError("Track the current object selection before applying this demo")
    # One visible reference frame makes the demo->run object coordinates
    # reproducible. Later initialization would require another saved reference.
    roles = {key.reference for key in document.keyframes} - {"world"}
    if any(status["reference_frames"].get(role) != document.reference_frame for role in roles):
        raise ValueError(
            "Choose an object reference frame where every required object is visible, then track again"
        )
    if not (tracking_directory / "reference.npz").exists():
        raise ValueError("Track this demo again to save its object reference")
    load_reference(tracking_directory / "reference.npz", [obj.id for obj in document.objects])
    base_from_camera = np.asarray(robot.geometry.to_dict()["cameras"][camera_name]["T_base_cam"])
    first, _ = dataset.episode_rows(episode)
    state_names = dataset.meta.features["observation.state"]["names"]
    action_names = dataset.meta.features["action"]["names"]
    keys = []
    previous = None
    reference = np.eye(4)
    reference_frame = None
    reference_status = None
    for key in document.keyframes:
        row = dataset[first + key.frame]
        state = dict(zip(state_names, row["observation.state"].tolist(), strict=True))
        action = dict(zip(action_names, row["action"].tolist(), strict=True))
        tool = model.forward(state)
        if previous is None or key.frame != previous.frame + 1 or key.reference != previous.reference:
            # Same ranges as the editor: contiguous frames with one reference.
            # Re-express the whole demonstrated motion in its entry frame.
            # A held object moving with the tool must not cancel out the lift.
            reference_frame = key.frame
            reference, reference_status = np.eye(4), "world"
            if key.reference != "world":
                packet = json.loads((tracking_directory / "frames" / f"{key.frame:06d}.json").read_text())
                evidence = packet["objects"][key.reference]
                if evidence["status"] not in POSE_AVAILABLE or evidence["pose"] is None:
                    raise ValueError(
                        f"Range starting at frame {key.frame}: {key.reference} has no usable pose"
                    )
                reference = base_from_camera @ transform(evidence["pose"])
                reference_status = evidence["status"]
        gripper = float(action[model.gripper])
        gripper_observed = float(state[model.gripper])
        if not np.isfinite([gripper, gripper_observed]).all():
            raise ValueError("Keyframe gripper action is nonfinite")
        keys.append(
            {
                "frame": key.frame,
                "reference": key.reference,
                "reference_frame": reference_frame,
                "reference_status": reference_status,
                "label": key.label,
                "reference_from_tool": (np.linalg.inv(reference) @ tool).tolist(),
                "demo_base_from_reference": reference.tolist(),
                "demo_tool": tool.tolist(),
                "state": state,
                "gripper": gripper,
                "gripper_observed": gripper_observed,
            }
        )
        previous = key
    return {
        "version": PLAN_VERSION,
        "document": document.model_dump(mode="json"),
        "robot_geometry": robot.geometry.to_dict(),
        "robot_cartesian": model.description(),
        "base_from_camera": base_from_camera.tolist(),
        "keyframes": keys,
        "settings": inputs["settings"],
        "depth_source": inputs["depth_source"],
    }


class DemoRunTracker:
    """Use the existing frame session, primed by the demo's exact first observation."""

    def __init__(self, plan, reference_path, mask_source, **session_kwargs):
        self.plan = plan
        self.session = LiveSurfaceSession(plan["settings"], mask_source, **session_kwargs)
        self.engine = self.session.engine
        self.mask_source = mask_source
        self.reference_path = reference_path
        self.started = False
        self.method = LIVE_METHOD

    def step(self, frame, *, measured_masks=None):
        measured_masks = measured_masks or self.measure_masks(frame)
        return self.session.step(frame, measured_masks=measured_masks)

    def measure_masks(self, frame):
        if not self.started:
            initial, masks = load_reference(
                self.reference_path,
                self.engine.sessions,
                frame_id=frame.frame_id - 1,
                time_s=frame.source_time_s - 1,
                calibration_version=frame.calibration_version,
            )
            # Initialize SAM's memory with the same demo image, not the first
            # live image or click coordinates from a different camera view.
            self.mask_source.masks(initial)
            self.session.step(initial, measured_masks=(initial.frame_id, masks, 0.0))
            self.started = True
        return self.session.measure_masks(frame)

    def close(self):
        self.session.close()


class KeyframeExecutor:
    """One bounded servo step. Each range acquires its reference before starting.

    The caller owns the motor clock and dispatch. A reference is fixed for its
    demonstrated range; another range must acquire new evidence. Every step
    uses current measured joints. Stop is terminal, including during occlusion.
    """

    def __init__(self, plan, model, *, max_age_s=0.5, joint_speed_deg_s=30.0):
        self.plan, self.model = plan, model
        self.max_age_s, self.joint_speed_deg_s = max_age_s, joint_speed_deg_s
        self.index = 0
        self.stopped = False
        self.status = "ready"
        self.target = None
        self.evidence = None
        self.reference_frame = None
        self.reference = None

    def stop(self):
        self.stopped, self.status = True, "stopped"

    def step(self, now, dt, state, observations):
        if self.stopped or self.index == len(self.plan["keyframes"]):
            if not self.stopped:
                self.status = "complete"
            return None
        if not 0 < dt <= 0.1:
            self.status = "waiting: motor clock interrupted"
            return None
        key = self.plan["keyframes"][self.index]
        if self.reference_frame != key["reference_frame"]:
            self.target = None
            reference = np.eye(4)
            self.evidence = None
            if key["reference"] != "world":
                evidence = observations.get(key["reference"])
                if (
                    not evidence
                    or evidence["status"] not in POSE_AVAILABLE
                    or evidence.get("pose") is None
                    or not 0 <= now - evidence["source_time_s"] <= self.max_age_s
                    or not evidence["source_time_s"] <= evidence["available_time_s"] <= now
                ):
                    self.status = f"waiting: {key['reference']} needs a current pose before this range"
                    return None
                reference = np.asarray(self.plan["base_from_camera"]) @ transform(evidence["pose"])
                self.evidence = {
                    "reference": key["reference"],
                    "reference_frame": key["reference_frame"],
                    "status": evidence["status"],
                    "source_time_s": evidence["source_time_s"],
                    "available_time_s": evidence["available_time_s"],
                    "acquired_time_s": now,
                    "base_from_reference": reference.tolist(),
                }
            self.reference, self.reference_frame = reference, key["reference_frame"]
        target = self.reference @ transform(key["reference_from_tool"])
        self.target = target
        current = self.model.forward(state)
        distance = np.linalg.norm(target[:3, 3] - current[:3, 3])
        angle = Rotation.from_matrix(current[:3, :3].T @ target[:3, :3]).magnitude()
        reached = distance < 0.003 and angle < np.deg2rad(3)
        if reached:
            # Keep the demonstrated gripper value in the robot's native units.
            # Do not close on the way to the approach pose.
            action = {name: state[name] for name in self.model.joint_names}
            action[self.model.gripper] = key["gripper"]
            # The recorded command may close beyond a held object's aperture.
            # Requiring the motor to reach that command would wait forever at
            # contact. Use the demonstrated measured aperture as the milestone.
            observed = key["gripper_observed"]
            actual = state[self.model.gripper]
            direction = np.sign(key["gripper"] - observed)
            gripper_ready = (
                abs(actual - observed) <= 2 if direction == 0 else direction * (actual - observed) >= -2
            )
            if gripper_ready:
                self.index += 1
            self.status = "complete" if self.index == len(self.plan["keyframes"]) else "gripper"
            return action
        # Small Cartesian increments, then a joint-speed bound. One IK solve,
        # never a search/retry loop or an unbounded stale command queue.
        fraction = min(1.0, 0.05 * dt / max(distance, 1e-12), np.deg2rad(30) * dt / max(angle, 1e-12))
        try:
            action = self.model.inverse(state, interpolate(current, target, fraction))
        except ValueError as exc:
            self.status = f"waiting: {exc}"
            return None
        for name, scale in zip(self.model.joint_names, self.model.units_per_degree, strict=True):
            limit = self.joint_speed_deg_s * dt * scale
            action[name] = float(np.clip(action[name], state[name] - limit, state[name] + limit))
        # A constrained solver can return its seed when no local step is
        # possible. Do not keep sending an unchanged command labelled moving.
        if np.allclose(self.model.degrees(action), self.model.degrees(state)):
            self.status = f"waiting: IK made no progress ({distance * 1000:.1f} mm / {np.rad2deg(angle):.1f}° remaining)"
            return None
        action[self.model.gripper] = state[self.model.gripper]
        self.status = "moving"
        return action
