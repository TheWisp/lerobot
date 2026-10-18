# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Which tracker output is a measurement (src/lerobot/showservo/docs/act_execution.md, "The loops, and who writes
what").

The trackers report each object's pose every camera frame, each with its own verdict; the world takes measurements.
This layer sits between them: for each object in a frame, a pose to take, or none and why. Neither the world nor the
engine sees a tracker. An estimator is anything with :meth:`PoseEstimator.estimate`; a better one — another model,
agreement between sources, a check against depth — replaces the one in use without touching the world or the engine.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One tracker's pose of one object in one camera frame: its motion since the demo (base frame), and the tracker's
    own verdict on it."""

    source: str
    motion: np.ndarray
    ok: bool
    why: str = ""


@dataclasses.dataclass(frozen=True)
class Estimate:
    """An object's pose in one frame, to take as a measurement; None when there is none to take. ``why`` says which."""

    motion: np.ndarray | None
    why: str


class PoseEstimator(Protocol):
    def estimate(self, t: float, candidates: dict[str, list[Candidate]]) -> dict[str, Estimate]:
        """Each object's estimate from the candidates of the camera frame captured at ``t``."""
        ...


POINT2POSE = "point2pose"  # the session tracker's fit, by its own verdict (found, the object not lost)
OLD_RUNNER = "old runner checks"  # the views the key-frame runner's checks certify, surface-corrected where they correct


class SourceVerdict:
    """One source's own verdict: an object's pose is that source's candidate when the source is confident in it."""

    def __init__(self, source: str):
        self.source = source

    def estimate(self, t: float, candidates: dict[str, list[Candidate]]) -> dict[str, Estimate]:
        out = {}
        for name, found in candidates.items():
            c = next((c for c in found if c.source == self.source), None)
            if c is None:
                out[name] = Estimate(None, f"no {self.source} candidate")
            elif c.ok:
                out[name] = Estimate(c.motion, f"{c.source}: confident")
            else:
                out[name] = Estimate(None, f"{c.source}: {c.why or 'not confident'}")
        return out


def tracker_verdict() -> SourceVerdict:
    """The rule the engine started with: Point2Pose's own verdict. It passes views every other check rejects
    (act_execution.md E30)."""
    return SourceVerdict(POINT2POSE)


def old_runner_checks() -> SourceVerdict:
    """The key-frame runner's checks: enough of the object's tracks seen with a depth that agrees, its fit pinning its
    middle, its surface agreeing (lerobot.gui.api.manipulation.tracking). Not the runner's carrying between views by
    the point groups, which needs to know what the gripper holds."""
    return SourceVerdict(OLD_RUNNER)
