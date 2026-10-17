# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Faults injected into an act on purpose, to see the engine find them and recover (tests, and the rig).

A perturbation changes what reaches the world, never what the engine believes it did: the gripper opened mid-carry
is a command the engine did not make, so it learns of the drop only from the camera, as it would of a real slip.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass
class OpenGripper:
    """Open the gripper to ``to`` for ``seconds`` once the act's commands reach ``s`` of the leg labelled ``leg`` (a
    key frame's label, as the demo editor shows it). Fires once."""

    leg: str
    s: float = 0.5
    seconds: float = 0.5
    to: float | None = None  # None: the demo's most open command
    fired_at: float | None = None

    def command(self, q: np.ndarray, cmd: Any, engine: Any, now: float) -> np.ndarray:
        if self.fired_at is None:
            leg = engine.tl.legs[cmd.leg]
            if cmd.kind != "leg" or leg.label != self.leg or cmd.s < self.s:
                return q
            self.fired_at = now
            engine._record("perturbation", now, kind="open gripper", leg=self.leg, s=cmd.s)
        if now - self.fired_at > self.seconds:
            return q
        g = engine.arm.gripper
        out = q.copy()
        grip = np.asarray(engine.demo.grip, dtype=float)
        out[g] = (
            self.to if self.to is not None else float(grip.min() if engine.arm.closing > 0 else grip.max())
        )
        return out


class Perturbations:
    """Every perturbation of one act, applied in order to each command on its way out."""

    def __init__(self, items: list[Any]):
        self.items = list(items)

    def command(self, q: np.ndarray, cmd: Any, engine: Any, now: float) -> np.ndarray:
        for item in self.items:
            q = item.command(q, cmd, engine, now)
        return q

    @classmethod
    def from_spec(cls, spec: list[dict[str, Any]] | None) -> Perturbations | None:
        """From the act request: ``[{"kind": "open_gripper", "leg": "pre-place 1", "s": 0.5, "seconds": 0.6}]``."""
        if not spec:
            return None
        kinds = {"open_gripper": OpenGripper}
        items = []
        for item in spec:
            item = dict(item)
            kind = item.pop("kind")
            if kind not in kinds:
                raise ValueError(f"a perturbation is one of {sorted(kinds)}")
            items.append(kinds[kind](**item))
        return cls(items)
