# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The card: the ONLY task-specific artifact in the system.

A card is a compiled demonstration — boundary conditions, not a trajectory. The
runtime reads cards and never names a task; if teaching a new task requires editing
runtime code, that is a bug in the runtime, and ``tests/showservo/test_card.py``
holds a structural test that says so.

The goal is stored the way invariant 3 demands — *object-frame*, never scene-frame.
A stage's ``goal_relation`` records where the held end sat **in the taught image**
alongside the taught target constellation; at runtime the target team's own
taught->live fit transports those positions into the live frame. Move the target and
the goal moves with it, for free, with no frame bookkeeping and nothing about the
surrounding scene baked in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Termination types the monitor knows how to wait for. A card may only ask for one of
# these; a typo must fail at load time, not at second 40 of an unattended attempt.
TERMINATIONS = ("pose_hold", "contact", "fission", "defission", "push_test")

# The two ends of the difference the servo drives to zero (invariant 1). "target" is
# what the stage acts upon; "held" is the end that moves. An EMPTY held team is
# meaningful and normal — see `Stage.held_end`.
TEAMS = ("target", "held")


@dataclass
class Keypoint:
    """One tracked feature: where it sat in the taught frame, and what it looks like.

    Pre: ``uv`` is in taught-image pixels; ``descriptor``, when present, is the
    binder's descriptor for this point (SIFT 128-d, DINO patch feature, ...) and is
    L2-normalised. ``xyz`` stays None for v0 — RGB is sufficient and depth is an
    optional bonus channel, so nothing downstream may require it.
    """

    uv: np.ndarray  # (2,) float64
    descriptor: np.ndarray | None = None  # (D,) float32, L2-normalised
    xyz: np.ndarray | None = None  # (3,) float64 or None

    def __post_init__(self):
        self.uv = np.asarray(self.uv, dtype=np.float64).reshape(2)
        if self.descriptor is not None:
            self.descriptor = np.asarray(self.descriptor, dtype=np.float32).ravel()
            assert self.descriptor.size > 0, "an empty descriptor cannot match anything"
            norm = float(np.linalg.norm(self.descriptor))
            assert abs(norm - 1.0) < 1e-3, f"descriptor must be L2-normalised, |d|={norm:.4f}"
        if self.xyz is not None:
            self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(3)


@dataclass
class GoalRelation:
    """The taught configuration: held-end positions in the TAUGHT image frame.

    Transport is the target team's job — ``servo.desired_held_uv`` applies the
    target's taught->live fit to ``held_uv``. Storing raw taught pixels (rather than
    a pre-baked transform) is what keeps the card readable, editable in the review
    screen, and free of any frame convention the runtime would have to agree on.

    ``spread_uv`` is the per-point, per-axis disagreement across demos (§4). At fewer
    than 5 demos it is ORDINAL ONLY — a ranking of which points the demos agreed on,
    never a calibrated tolerance — and ``n_demos`` is carried so no consumer can
    forget that.
    """

    held_uv: np.ndarray  # (N, 2) float64
    spread_uv: np.ndarray | None = None  # (N, 2) float64
    n_demos: int = 1

    def __post_init__(self):
        self.held_uv = np.asarray(self.held_uv, dtype=np.float64).reshape(-1, 2)
        assert len(self.held_uv) >= 1, "a goal with no held points specifies nothing to servo"
        assert self.n_demos >= 1
        if self.spread_uv is not None:
            self.spread_uv = np.asarray(self.spread_uv, dtype=np.float64).reshape(-1, 2)
            assert self.spread_uv.shape == self.held_uv.shape
            assert (self.spread_uv >= 0).all(), "spread is a magnitude"

    @property
    def tolerance_is_calibrated(self) -> bool:
        """False below 5 demos: the spread is a ranking, not a number to threshold on."""
        return self.n_demos >= 5


@dataclass
class Termination:
    """How the stage ends. ``params`` is passed verbatim to the monitor's detector."""

    type: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.type in TERMINATIONS, f"unknown termination {self.type!r}, want one of {TERMINATIONS}"


@dataclass
class Budget:
    """Abort conditions. Exhausting either is a *reported* failure, never a silent one."""

    seconds: float = 30.0
    retries: int = 3

    def __post_init__(self):
        assert self.seconds > 0 and self.retries >= 0


@dataclass
class Stage:
    """One bind-track-servo-terminate segment.

    Pre: ``teams`` has a non-empty ``target``; ``travel_dir`` is a unit 3-vector in
    the target/context frame giving the approach corridor and contact axis.
    """

    camera: str
    teams: dict[str, list[Keypoint]]
    goal_relation: GoalRelation
    travel_dir: np.ndarray  # (3,) unit
    termination: Termination
    budget: Budget = field(default_factory=Budget)
    grasp_aperture_expected: float | None = None
    name: str = ""

    def __post_init__(self):
        # Camera is a rig-supplied name, deliberately NOT an enum: which cameras exist
        # is a property of the rig, and a card that hardcoded "top"/"wrist" would stop
        # being portable the moment a rig names its cameras differently.
        assert isinstance(self.camera, str) and self.camera, "stage must name its camera"
        assert set(self.teams) <= set(TEAMS), f"unknown team(s) {set(self.teams) - set(TEAMS)}"
        assert self.teams.get("target"), "a stage with no target team has nothing to servo to"

        self.travel_dir = np.asarray(self.travel_dir, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(self.travel_dir))
        assert norm > 1e-9, "travel_dir must be a direction, not a zero vector"
        self.travel_dir = self.travel_dir / norm

        n_held = len(self.teams.get("held", []))
        if n_held:
            assert len(self.goal_relation.held_uv) == n_held, (
                f"goal has {len(self.goal_relation.held_uv)} held points but the held team has {n_held}"
            )

    @property
    def held_end(self) -> str:
        """``"held"`` when the stage tracks a grasped object, else ``"gripper"``.

        An empty held team is how a card says "the moving end is the robot itself"
        (every D1 stage before the grasp). The gripper's appearance belongs to the
        RIG, not the task, so it is supplied by the runtime and never stored in a
        card — putting it here would bake the robot into a task description.
        """
        return "held" if self.teams.get("held") else "gripper"

    def team_uv(self, team: str) -> np.ndarray:
        """Taught pixel coordinates of a team. Post: (N, 2); (0, 2) for an absent team."""
        pts = self.teams.get(team, [])
        if not pts:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([kp.uv for kp in pts])

    def team_xyz(self, team: str) -> np.ndarray | None:
        """Taught camera-frame 3D for a team. Post: (N, 3), or None if any point lacks it.

        All-or-nothing for the same reason as descriptors: a team where only some
        points carry 3D cannot be fitted coherently, and a short stack would silently
        misalign points with their coordinates.
        """
        pts = self.teams.get(team, [])
        if not pts or any(kp.xyz is None for kp in pts):
            return None
        return np.stack([kp.xyz for kp in pts])

    def team_descriptors(self, team: str) -> np.ndarray | None:
        """Post: (N, D) stacked descriptors, or None if any point lacks one."""
        pts = self.teams.get(team, [])
        if not pts or any(kp.descriptor is None for kp in pts):
            return None
        return np.stack([kp.descriptor for kp in pts])


@dataclass
class Card:
    """A task, compiled. Pre: at least one stage."""

    name: str
    stages: list[Stage]
    descriptor_space: str = "sift"  # which binder produced the descriptors

    def __post_init__(self):
        assert self.stages, "a card with no stages teaches nothing"

    # --- persistence -------------------------------------------------------------
    # JSON, not pickle: cards are reviewed and hand-edited (§4's review screen), and a
    # format a human cannot diff is a format nobody audits.

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "descriptor_space": self.descriptor_space,
            "stages": [_stage_to_dict(ch) for ch in self.stages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Card:
        return cls(
            name=d["name"],
            descriptor_space=d.get("descriptor_space", "sift"),
            stages=[_stage_from_dict(c) for c in d["stages"]],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Card:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _kp_to_dict(kp: Keypoint) -> dict:
    return {
        "uv": kp.uv.tolist(),
        "descriptor": None if kp.descriptor is None else kp.descriptor.tolist(),
        "xyz": None if kp.xyz is None else kp.xyz.tolist(),
    }


def _stage_to_dict(ch: Stage) -> dict:
    return {
        "name": ch.name,
        "camera": ch.camera,
        "teams": {k: [_kp_to_dict(p) for p in v] for k, v in ch.teams.items()},
        "goal_relation": {
            "held_uv": ch.goal_relation.held_uv.tolist(),
            "spread_uv": None if ch.goal_relation.spread_uv is None else ch.goal_relation.spread_uv.tolist(),
            "n_demos": ch.goal_relation.n_demos,
        },
        "travel_dir": ch.travel_dir.tolist(),
        "termination": {"type": ch.termination.type, "params": ch.termination.params},
        "budget": {"seconds": ch.budget.seconds, "retries": ch.budget.retries},
        "grasp": {"aperture_expected": ch.grasp_aperture_expected},
    }


def _stage_from_dict(d: dict) -> Stage:
    g = d["goal_relation"]
    return Stage(
        name=d.get("name", ""),
        camera=d["camera"],
        teams={
            k: [Keypoint(uv=p["uv"], descriptor=p.get("descriptor"), xyz=p.get("xyz")) for p in v]
            for k, v in d["teams"].items()
        },
        goal_relation=GoalRelation(
            held_uv=g["held_uv"], spread_uv=g.get("spread_uv"), n_demos=g.get("n_demos", 1)
        ),
        travel_dir=d["travel_dir"],
        termination=Termination(d["termination"]["type"], d["termination"].get("params", {})),
        budget=Budget(**d.get("budget", {})),
        grasp_aperture_expected=(d.get("grasp") or {}).get("aperture_expected"),
    )
