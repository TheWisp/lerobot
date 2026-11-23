"""FLASH: demonstration-driven spatial action execution.

The package intentionally has no robot or tracker singleton. A skill is an
ordinary LeRobot episode plus a versioned spatial annotation; runtime adapters
provide current poses and robot kinematics.
"""

from .geometry import held_object_goal, pregrasp_goal
from .skill import FlashSkill, Interval, MotionMode, SpatialSample

__all__ = ["FlashSkill", "Interval", "MotionMode", "SpatialSample", "held_object_goal", "pregrasp_goal"]
