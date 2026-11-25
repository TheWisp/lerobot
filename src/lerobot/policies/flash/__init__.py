"""FLASH: demonstration-driven spatial action execution.

The package intentionally has no robot or tracker singleton. A skill is an
ordinary LeRobot episode plus a versioned spatial annotation; runtime adapters
provide current poses and robot kinematics.
"""

from .geometry import held_object_goal, pregrasp_goal

__all__ = ["FlashSkill", "Interval", "MotionMode", "SpatialSample", "held_object_goal", "pregrasp_goal"]


def __getattr__(name):
    # Tracking primitives must remain usable without the action-execution layer.
    if name in {"FlashSkill", "Interval", "MotionMode", "SpatialSample"}:
        from . import skill

        value = getattr(skill, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
