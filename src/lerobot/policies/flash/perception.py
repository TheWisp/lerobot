"""Backend-independent RGB-D, reference and attachment contracts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import numpy as np

from .execution import MotorTick
from .geometry import interpolate, inverse, transform
from .trajectory import AttachmentState, Evidence, PoseEstimate


@dataclass(frozen=True)
class RGBDFrame:
    """One aligned camera observation; source time is on the motor clock."""

    frame_id: int
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    depth_m_per_unit: float
    base_from_camera: np.ndarray
    source_time_s: float
    available_time_s: float
    calibration_version: str

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("RGB must be HxWx3 uint8")
        if self.depth.shape != self.rgb.shape[:2] or self.depth.dtype not in (np.uint16, np.float32):
            raise ValueError("depth must be aligned HxW uint16 or float32")
        if self.intrinsics.shape != (3, 3) or not np.isfinite(self.intrinsics).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        if not np.isfinite(self.depth_m_per_unit) or self.depth_m_per_unit <= 0:
            raise ValueError("invalid depth scale")
        if self.available_time_s < self.source_time_s:
            raise ValueError("availability cannot precede capture")
        object.__setattr__(self, "base_from_camera", transform(self.base_from_camera))

    def metric_depth(self) -> np.ndarray:
        return self.depth.astype(np.float32) * self.depth_m_per_unit


class LatestRGBDFrameStore:
    """Retain a capture-thread snapshot for asynchronous episode recording.

    The capture thread copies frame buffers before publication so a camera
    driver may reuse its next read buffer. Readers only take a short lock and
    receive immutable-by-convention arrays; they never wait for the sensor.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: RGBDFrame | None = None

    def publish(self, frame: RGBDFrame) -> None:
        rgb = frame.rgb.copy()
        depth = frame.depth.copy()
        rgb.setflags(write=False)
        depth.setflags(write=False)
        snapshot = RGBDFrame(
            frame.frame_id,
            rgb,
            depth,
            frame.intrinsics.copy(),
            frame.depth_m_per_unit,
            frame.base_from_camera.copy(),
            frame.source_time_s,
            frame.available_time_s,
            frame.calibration_version,
        )
        snapshot.intrinsics.setflags(write=False)
        snapshot.base_from_camera.setflags(write=False)
        with self._lock:
            if self._latest is not None and (
                snapshot.frame_id <= self._latest.frame_id
                or snapshot.source_time_s <= self._latest.source_time_s
            ):
                raise ValueError("camera frame IDs and capture times must increase")
            self._latest = snapshot

    def snapshot(self, *, no_later_than_s: float | None = None) -> RGBDFrame | None:
        with self._lock:
            frame = self._latest
        if frame is not None and no_later_than_s is not None and frame.source_time_s > no_later_than_s:
            return None
        return frame


@dataclass(frozen=True)
class ReferenceSnippet:
    """One annotated view registered into a stable object frame."""

    repo_id: str
    episode: int
    frame_index: int
    object_role: str
    object_from_camera: np.ndarray
    mask_artifact: str
    calibration_version: str

    def __post_init__(self) -> None:
        if not self.repo_id or not self.object_role or self.episode < 0 or self.frame_index < 0:
            raise ValueError("invalid reference snippet identity")
        object.__setattr__(self, "object_from_camera", transform(self.object_from_camera))


@dataclass(frozen=True)
class ObjectReference:
    role: str
    version: str
    snippets: tuple[ReferenceSnippet, ...]

    def __post_init__(self) -> None:
        if not self.role or not self.version or not self.snippets:
            raise ValueError("object reference needs role, version and snippets")
        if any(snippet.object_role != self.role for snippet in self.snippets):
            raise ValueError("reference snippets must share one object role")


class PoseBackend(Protocol):
    """Heavy implementations live outside the motor process.

    A backend may use Point2Pose, a different model, or a trained checkpoint;
    all return pose estimates with provenance and supported dimensions.
    """

    def initialize(self, references: tuple[ObjectReference, ...]) -> None: ...

    def update(self, frame: RGBDFrame) -> tuple[PoseEstimate, ...]: ...


class LatestPoseStore:
    """Short-lock handoff from an asynchronous estimator to the planner."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._version = 0
        self._poses: dict[str, PoseEstimate] = {}

    def publish(self, estimates: tuple[PoseEstimate, ...]) -> int:
        with self._lock:
            changed = False
            for estimate in estimates:
                previous = self._poses.get(estimate.role)
                if previous is not None and estimate.source_time_s <= previous.source_time_s:
                    continue
                self._poses[estimate.role] = estimate
                changed = True
            if changed:
                self._version += 1
            return self._version

    def snapshot(self) -> tuple[int, dict[str, PoseEstimate]]:
        with self._lock:
            return self._version, dict(self._poses)


class TimedEndEffectorHistory:
    """Interpolate measured EE poses onto camera capture times, without extrapolation.

    A pose published after a slow perception pass still belongs to its source
    image. Using the newest servo position to update the attachment would
    turn perception latency into a spurious gripper-to-object offset.
    """

    def __init__(self, *, capacity: int, max_sample_gap_s: float) -> None:
        if capacity < 2 or not np.isfinite(max_sample_gap_s) or max_sample_gap_s <= 0:
            raise ValueError("EE history needs capacity >= 2 and a positive sample-gap bound")
        self._samples: deque[tuple[float, np.ndarray]] = deque(maxlen=capacity)
        self._lock = Lock()
        self.max_sample_gap_s = max_sample_gap_s

    def append(self, source_time_s: float, base_from_ee: np.ndarray) -> None:
        if not np.isfinite(source_time_s):
            raise ValueError("EE sample time must be finite")
        pose = transform(base_from_ee)
        with self._lock:
            if self._samples and source_time_s <= self._samples[-1][0]:
                raise ValueError("EE history times must strictly increase")
            self._samples.append((source_time_s, pose.copy()))

    def at(self, source_time_s: float) -> np.ndarray | None:
        if not np.isfinite(source_time_s):
            return None
        with self._lock:
            samples = tuple(self._samples)
        for time_s, pose in samples:
            if source_time_s == time_s:
                return pose.copy()
        for (left_time, left), (right_time, right) in zip(samples, samples[1:], strict=False):
            if left_time < source_time_s < right_time:
                if right_time - left_time > self.max_sample_gap_s:
                    return None
                return interpolate(left, right, (source_time_s - left_time) / (right_time - left_time))
        return None


class AttachmentEstimate:
    """Gripper-to-object relation updated only by independent held evidence.

    During occlusion, the last verified attachment predicts object motion with
    the gripper; feeding that prediction back as a measurement would falsely
    tighten confidence and conceal slip.
    """

    def __init__(self) -> None:
        self._attachment: np.ndarray | None = None
        self._source_time_s: float | None = None
        self._reference_version: str | None = None

    def update(self, base_from_ee: np.ndarray, held: PoseEstimate) -> bool:
        """Accept independently observed held pose and a capture-time EE pose."""
        if held.evidence != Evidence.OBSERVED or held.pose is None or not held.supports():
            return False
        if self._source_time_s is not None and held.source_time_s <= self._source_time_s:
            return False
        self._attachment = inverse(base_from_ee) @ held.pose
        self._source_time_s = held.source_time_s
        self._reference_version = held.reference_version
        return True

    def update_from_history(self, history: TimedEndEffectorHistory, held: PoseEstimate) -> bool:
        """Fuse held evidence only when the motor history brackets its image time."""
        if held.evidence != Evidence.OBSERVED:
            return False
        ee_at_capture = history.at(held.source_time_s)
        return ee_at_capture is not None and self.update(ee_at_capture, held)

    def predict(
        self, base_from_ee: np.ndarray, now_s: float, max_age_s: float, role: str
    ) -> PoseEstimate | None:
        if (
            self._attachment is None
            or self._source_time_s is None
            or not 0 <= now_s - self._source_time_s <= max_age_s
        ):
            return None
        return PoseEstimate(
            role=role,
            pose=transform(base_from_ee) @ self._attachment,
            source_time_s=self._source_time_s,
            available_time_s=now_s,
            evidence=Evidence.PREDICTED,
            reference_version=self._reference_version or "",
            translation_supported=(True, True, True),
            rotation_supported=(True, True, True),
        )

    @property
    def gripper_from_object(self) -> np.ndarray | None:
        return None if self._attachment is None else self._attachment.copy()

    def state(self, now_s: float, max_age_s: float) -> AttachmentState | None:
        if (
            self._attachment is None
            or self._source_time_s is None
            or not 0 <= now_s - self._source_time_s <= max_age_s
        ):
            return None
        return AttachmentState(self._attachment, self._source_time_s, now_s, self._reference_version or "")


class EndEffectorKinematics(Protocol):
    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray: ...


class HeldAttachmentUpdater:
    """Update a rigid gripper-object relation from capture-time observations.

    A pregrasp observation may seed a fallback once grasp is activated; it is
    not treated as proof that the object followed the gripper. Later independent
    held observations correct the relation if they agree with configured jump
    bounds. A conflicting update withdraws the attachment until resolved,
    rather than snapping the commanded gripper goal to an ambiguous pose.
    """

    def __init__(
        self,
        role: str,
        reference_version: str,
        kinematics: EndEffectorKinematics,
        arm_joint_names: tuple[str, ...],
        *,
        history_capacity: int,
        max_motor_sample_gap_s: float,
        max_attachment_age_s: float,
        max_update_shift_m: float,
        max_update_turn_rad: float,
    ) -> None:
        if not role or not reference_version or not arm_joint_names:
            raise ValueError("held attachment needs a role, object frame and arm joints")
        values = (max_attachment_age_s, max_update_shift_m, max_update_turn_rad)
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("held attachment limits must be finite and positive")
        self.role = role
        self.reference_version = reference_version
        self.kinematics = kinematics
        self.arm_joint_names = arm_joint_names
        self.history = TimedEndEffectorHistory(
            capacity=history_capacity, max_sample_gap_s=max_motor_sample_gap_s
        )
        self.max_attachment_age_s = max_attachment_age_s
        self.max_update_shift_m = max_update_shift_m
        self.max_update_turn_rad = max_update_turn_rad
        self._lock = Lock()
        self._estimate = AttachmentEstimate()
        self._active = False
        self._activation_time_s: float | None = None
        self._conflicted = False
        self.has_postgrasp_observation = False
        self.last_rejection: str | None = None

    def on_motor_tick(self, tick: MotorTick) -> None:
        if tick.observed_at_s is None:
            raise ValueError("held attachment needs a timestamped motor observation")
        try:
            joints = np.array([float(tick.observed[f"{name}.pos"]) for name in self.arm_joint_names])
        except KeyError as exc:
            raise ValueError("motor observation lacks a held-attachment arm joint") from exc
        self.history.append(tick.observed_at_s, self.kinematics.forward_kinematics(joints))

    def activate(self, grasp_time_s: float, pregrasp: PoseEstimate | None = None) -> bool:
        if not np.isfinite(grasp_time_s):
            raise ValueError("grasp activation time must be finite")
        with self._lock:
            self._estimate = AttachmentEstimate()
            self._active = True
            self._activation_time_s = grasp_time_s
            self._conflicted = False
            self.has_postgrasp_observation = False
            self.last_rejection = None
            return bool(pregrasp is not None and self._accept_seed(pregrasp))

    def _accept_seed(self, estimate: PoseEstimate) -> bool:
        return (
            estimate.role == self.role
            and estimate.reference_version == self.reference_version
            and self._estimate.update_from_history(self.history, estimate)
        )

    def on_pose_update(self, _version: int, poses: dict[str, PoseEstimate]) -> bool:
        held = poses.get(self.role)
        if held is None or held.reference_version != self.reference_version:
            return False
        with self._lock:
            if (
                not self._active
                or held.source_time_s < self._activation_time_s
                or held.evidence != Evidence.OBSERVED
                or held.pose is None
                or not held.supports()
            ):
                return False
            ee = self.history.at(held.source_time_s)
            if ee is None:
                self.last_rejection = "held image is not bracketed by motor observations"
                return False
            candidate = inverse(ee) @ held.pose
            prior = self._estimate.gripper_from_object
            if prior is not None:
                relative = inverse(prior) @ candidate
                shift = float(np.linalg.norm(relative[:3, 3]))
                turn = float(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))
                if shift > self.max_update_shift_m or turn > self.max_update_turn_rad:
                    self._conflicted = True
                    self.last_rejection = f"held attachment disagrees by {shift:.4f} m, {turn:.4f} rad"
                    return False
            if not self._estimate.update(ee, held):
                return False
            self._conflicted = False
            self.has_postgrasp_observation = True
            self.last_rejection = None
            return True

    def state(self, now_s: float) -> AttachmentState | None:
        with self._lock:
            if not self._active or self._conflicted:
                return None
            return self._estimate.state(now_s, self.max_attachment_age_s)

    def deactivate(self) -> None:
        with self._lock:
            self._active = False
            self._activation_time_s = None
            self._estimate = AttachmentEstimate()
            self._conflicted = False
