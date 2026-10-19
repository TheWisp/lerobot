"""Backend-independent RGB-D, reference and attachment contracts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import numpy as np

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
