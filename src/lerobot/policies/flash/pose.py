"""Timestamped pose observations shared by tracking and action execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .geometry import transform


class Evidence(str, Enum):
    OBSERVED = "observed"
    PREDICTED = "predicted"
    LOST = "lost"


@dataclass(frozen=True)
class PoseEstimate:
    """Pose in the robot-base frame, with times on the motor monotonic clock.

    Supported translation axes use the robot-base frame; supported rotation
    axes describe rotations about the object's local X, Y and Z axes. They
    mark task-relevant evidence, not a claim that all six axes were observed.
    A partial pose remains available when the task is invariant to its
    unsupported axes.
    """

    role: str
    pose: np.ndarray | None
    source_time_s: float
    available_time_s: float
    evidence: Evidence
    reference_version: str
    translation_supported: tuple[bool, bool, bool]
    rotation_supported: tuple[bool, bool, bool]
    direct_translation_observed: bool = False
    estimator: str = "unspecified"
    last_direct_observation_s: float | None = None
    symmetry_canonicalized: bool = False
    source_frame_id: int | None = None

    def __post_init__(self) -> None:
        if self.pose is not None:
            object.__setattr__(self, "pose", transform(self.pose))
        if self.evidence != Evidence.LOST and self.pose is None:
            raise ValueError("observed or predicted evidence needs a pose")
        if self.available_time_s < self.source_time_s:
            raise ValueError("availability cannot precede source time")
        if len(self.translation_supported) != 3 or len(self.rotation_supported) != 3:
            raise ValueError("pose support must specify three translation and three rotation axes")
        if not self.estimator:
            raise ValueError("pose estimator provenance must be nonempty")
        if self.last_direct_observation_s is not None and self.last_direct_observation_s > self.source_time_s:
            raise ValueError("direct observation cannot postdate pose source")
        if self.source_frame_id is not None and (
            isinstance(self.source_frame_id, bool)
            or not isinstance(self.source_frame_id, int)
            or self.source_frame_id < 0
        ):
            raise ValueError("pose source frame ID must be a nonnegative integer")

    @property
    def has_direct_translation(self) -> bool:
        """A current image measured the centre, even if orientation was constrained."""
        return self.evidence == Evidence.OBSERVED or self.direct_translation_observed

    def supports(
        self,
        translation: tuple[bool, bool, bool] = (True, True, True),
        rotation: tuple[bool, bool, bool] = (True, True, True),
    ) -> bool:
        return all(
            not required or supported
            for required, supported in zip(translation, self.translation_supported, strict=True)
        ) and all(
            not required or supported
            for required, supported in zip(rotation, self.rotation_supported, strict=True)
        )

    def usable(
        self,
        now_s: float,
        max_age_s: float,
        allow_prediction: bool,
        translation: tuple[bool, bool, bool] = (True, True, True),
        rotation: tuple[bool, bool, bool] = (True, True, True),
    ) -> bool:
        return (
            self.pose is not None
            and self.evidence != Evidence.LOST
            and (allow_prediction or self.evidence == Evidence.OBSERVED)
            and 0 <= now_s - self.source_time_s <= max_age_s
            and now_s >= self.available_time_s
            and self.supports(translation, rotation)
        )


@dataclass(frozen=True)
class AttachmentState:
    """``T_E_H`` for the current rollout, distinct from a world pose."""

    gripper_from_object: np.ndarray
    source_time_s: float
    available_time_s: float
    reference_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "gripper_from_object", transform(self.gripper_from_object))
        if self.available_time_s < self.source_time_s:
            raise ValueError("attachment availability cannot precede source time")

    def usable(self, now_s: float, max_age_s: float) -> bool:
        return self.available_time_s <= now_s and 0 <= now_s - self.source_time_s <= max_age_s
