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


class TrackerVerdict:
    """The trackers' own verdicts: an object's pose is its first candidate its tracker is confident in. It passes
    views every other check rejects (act_execution.md E30), and is the rule the engine started with."""

    def estimate(self, t: float, candidates: dict[str, list[Candidate]]) -> dict[str, Estimate]:
        out = {}
        for name, found in candidates.items():
            c = next((c for c in found if c.ok), None)
            if c is not None:
                out[name] = Estimate(c.motion, f"{c.source}: confident")
            else:
                out[name] = Estimate(
                    None, "; ".join(f"{c.source}: {c.why or 'not confident'}" for c in found)
                )
        return out
