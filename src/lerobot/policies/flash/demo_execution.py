"""Small keyframe executor: shared tracked poses -> robot-owned FK/IK -> actions.

A consecutive range stores measured tool poses relative to its selected object
at range entry (or the robot base), and the demonstrated gripper commands.
The live reference can update the unissued command queue. There is no new
matcher, confidence estimator, attachment inference or recovery search here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .demo_document import DEMO_FPS, tracking_stale_reason
from .execution import TimedAction
from .experimental_surface_session import PoseHint
from .geometry import transform
from .live_surface_session import LIVE_METHOD, LiveSurfaceSession
from .observations import RGBDFrame
from .pipeline_version import PIPELINE_VERSION
from .surface_kalman import PoseKalmanFilter

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
    if dataset.fps != DEMO_FPS:
        raise ValueError(
            f"This demo was recorded at {dataset.fps} fps; FLASH replays demos at {DEMO_FPS} fps. "
            f"Record it again at {DEMO_FPS} fps."
        )
    if dataset.meta.info.robot_geometry != robot.geometry.to_dict():
        raise ValueError("Demo and selected robot geometry differ; use the robot that recorded this demo")
    model = robot.cartesian_model()
    if dataset.meta.info.robot_cartesian != model.description():
        raise ValueError("Demo is missing matching robot joint units. Record a new demo with this setup.")
    if document.camera != f"observation.images.{camera_name}":
        raise ValueError("The selected camera differs from the demo camera")
    inputs = json.loads((tracking_directory / "input.json").read_text())
    status = json.loads((tracking_directory / "status.json").read_text())
    if status["state"] != "complete":
        raise ValueError("Tracking must complete before applying this demo")
    reason = tracking_stale_reason(inputs, document)
    if reason:
        raise ValueError(f"{reason}; rebuild tracking before applying this demo")
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
        entry_demo_speed = None
        if previous is not None and key.frame > previous.frame + 1:
            # A marked rest frame followed by pre-grasp supplies the actual
            # recorded approach speed, even though the intervening images are
            # not themselves keyframes.
            positions = []
            for frame_index in range(previous.frame, key.frame + 1):
                recorded = dataset[first + frame_index]["observation.state"]
                recorded_state = dict(zip(state_names, recorded.tolist(), strict=True))
                positions.append(model.forward(recorded_state)[:3, 3])
            path_m = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            entry_demo_speed = path_m * dataset.fps / (key.frame - previous.frame)
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
                "entry_demo_speed_m_s": entry_demo_speed,
            }
        )
        previous = key
    return {
        "version": PLAN_VERSION,
        "pipeline_version": PIPELINE_VERSION,
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
        self.demo_reference_frame = None
        self.live_accepted_roles = set()
        self.method = LIVE_METHOD

    def step(self, frame, *, measured_masks=None):
        measured_masks = measured_masks or self.measure_masks(frame)
        hints = self._initial_pose_hints(frame, measured_masks[1])
        for role, hint in hints.items():
            track = self.engine.candidates[LIVE_METHOD][role]
            # The demo and live scene are separate observations. The demo's
            # motion prior must not pull this first live fit back to its old
            # camera position. Keep the surface; reset only pose prediction.
            track.pose = hint.camera_from_reference.copy()
            track.base_from_camera = frame.base_from_camera.copy()
            track.kalman = None
        result = self.session.step(
            frame,
            measured_masks=measured_masks,
            pose_hints=hints,
        )
        for role in hints:
            track = self.engine.candidates[LIVE_METHOD][role]
            if track.accepted_frame == frame.frame_id and track.kalman is None:
                track.kalman = PoseKalmanFilter(
                    track.pose, track.reference.mean(0), frame.source_time_s, track.config.uncertainty
                )
        for role, track in self.engine.candidates[LIVE_METHOD].items():
            if track.accepted_frame == frame.frame_id:
                self.live_accepted_roles.add(role)
        return result

    def _initial_pose_hints(self, frame, masks):
        """Use each measured live mask to start its first demo-to-scene fit.

        This is a starting pose only. The shared surface check must still
        accept the fitted pose before the executor can use it.
        """
        hints = {}
        depth = frame.depth.astype(np.float32) * frame.depth_m_per_unit
        for role, track in self.engine.candidates[LIVE_METHOD].items():
            if role in self.live_accepted_roles or track.reference is None:
                continue
            valid = masks[role] & np.isfinite(depth) & (depth > 0)
            y, x = np.nonzero(valid)
            if len(x) < 3:
                continue
            z = depth[y, x]
            intrinsics = frame.intrinsics
            measured = np.column_stack(
                (
                    (x - intrinsics[0, 2]) * z / intrinsics[0, 0],
                    (y - intrinsics[1, 2]) * z / intrinsics[1, 1],
                    z,
                )
            )
            pose = np.eye(4)
            pose[:3, 3] = np.median(measured, axis=0) - np.median(track.reference, axis=0)
            hints[role] = PoseHint(
                pose,
                "demo_live_mask_center",
                frame.source_time_s,
                frame.source_time_s,
                track.reference_frame,
            )
        return hints

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
            self.demo_reference_frame = initial.frame_id
            # Transfer the object's measured surface and appearance, not the
            # old scene's depth. Only a live accepted view may later veto a
            # live pose or become the last accepted image-match reference.
            for track in self.engine.candidates[LIVE_METHOD].values():
                track.accepted_frame = None
                track.accepted_pose = None
                track.accepted_observation = None
                track.accepted_intrinsics = None
            self.started = True
        return self.session.measure_masks(frame)

    def close(self):
        self.session.close()


@dataclass(frozen=True)
class PlannedDemoRange:
    """A demonstrated range expressed as robot API actions at 30 Hz."""

    actions: tuple[TimedAction, ...]
    next_index: int
    target: np.ndarray
    evidence: dict | None
    entry_speed_m_s: float
    seed_deg: np.ndarray
    demo_path_m: float
    demo_duration_s: float
    reference_pose: np.ndarray
    start_index: int


def _robot_action(model, joints_deg, gripper):
    """Convert the robot's IK degrees to its ordinary action units."""
    values = np.asarray(joints_deg, dtype=float)
    if values.shape != (len(model.joint_names),) or not np.isfinite(values).all():
        raise ValueError("Robot IK did not produce finite joints")
    if not np.isfinite(gripper):
        raise ValueError("Demo gripper action is nonfinite")
    return dict(zip(model.joint_names, values * model.units_per_degree, strict=True)) | {
        model.gripper: float(gripper)
    }


def _solve_robot_ik(model, seed_deg, target, gripper):
    seed = _robot_action(model, seed_deg, gripper)
    solved = model.inverse(seed, target)
    return np.asarray([solved[name] for name in model.joint_names]) / model.units_per_degree


def _reference_for_range(plan, first, observations):
    role = first["reference"]
    if role == "world":
        return np.eye(4), None
    observed = observations.get(role)
    if not observed or observed.get("pose") is None or observed.get("status") not in POSE_AVAILABLE:
        raise ValueError(f"{role} needs a pose before this range")
    reference = np.asarray(plan["base_from_camera"]) @ transform(observed["pose"])
    evidence = {
        "reference": role,
        "reference_frame": first["reference_frame"],
        "status": observed["status"],
        "source_time_s": observed.get("source_time_s"),
        "source_frame_id": observed.get("source_frame_id"),
        "available_time_s": observed.get("available_time_s"),
        "base_from_reference": reference.tolist(),
    }
    return reference, evidence


def plan_demo_range(plan, model, index, state, observations, now, *, fps=DEMO_FPS):
    """Convert a demo range to a queue; new targets rebuild the unissued suffix.

    The robot owns IK and joint units. The editor supplies only the recorded
    tool path, the latest reference pose, and the recorded gripper actions.
    No separate Cartesian gate, search, or per-tick IK sits in this path.
    """
    keys = plan["keyframes"]
    first = keys[index]
    end = index + 1
    while end < len(keys) and keys[end]["reference_frame"] == first["reference_frame"]:
        end += 1
    range_keys = keys[index:end]
    reference, evidence = _reference_for_range(plan, first, observations)
    if evidence is not None:
        evidence["planned_at_s"] = now
    targets = [reference @ transform(key["reference_from_tool"]) for key in range_keys]
    seed = model.degrees(state)
    joints = seed.copy()
    period = 1.0 / fps
    demo_positions = np.asarray(
        [
            transform(key.get("demo_tool", target))[:3, 3]
            for key, target in zip(range_keys, targets, strict=True)
        ]
    )
    path_m = float(np.linalg.norm(np.diff(demo_positions, axis=0), axis=1).sum())
    duration_s = (range_keys[-1]["frame"] - first["frame"]) * period
    speed = first.get("entry_demo_speed_m_s") or (path_m / duration_s if duration_s and path_m else 0.04)
    actions = []
    # Join a newly acquired range at the demo's measured approach speed.
    start_tool = model.forward(state)
    entry_distance = float(np.linalg.norm(targets[0][:3, 3] - start_tool[:3, 3]))
    entry_ticks = max(1, int(np.ceil(entry_distance / speed * fps)))
    first_joints = _solve_robot_ik(model, joints, targets[0], float(first["gripper"]))
    first_gripper = float(state[model.gripper])
    actions.append(TimedAction(0, _robot_action(model, seed, first_gripper), f"entry:{first['frame']}"))
    for step in range(1, entry_ticks + 1):
        ratio = step / entry_ticks
        action = _robot_action(
            model,
            seed + ratio * (first_joints - seed),
            first_gripper + ratio * (float(first["gripper"]) - first_gripper),
        )
        actions.append(TimedAction(len(actions), action, f"entry:{first['frame']}"))
    joints = first_joints
    for key, target in zip(range_keys, targets, strict=True):
        joints = _solve_robot_ik(model, joints, target, float(key["gripper"]))
        action = _robot_action(model, joints, float(key["gripper"]))
        actions.append(
            TimedAction(
                len(actions),
                action,
                f"demo:{key['frame']}",
                interval_complete=key is range_keys[-1],
                tool_target=tuple(tuple(float(value) for value in row) for row in target),
            )
        )
    return PlannedDemoRange(
        tuple(actions),
        end,
        targets[-1],
        evidence,
        speed,
        seed,
        path_m,
        duration_s,
        reference,
        index,
    )
