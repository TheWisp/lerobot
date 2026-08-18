# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Synthetic card fixtures shaped like the two demos v0 must support.

Deliberately synthetic: these are structural fixtures, and pointing them at a real
recorded dataset would couple the test suite to data that can be re-recorded or
cache-cleared out from under it.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.showservo.card import Budget, Card, GoalRelation, Keypoint, Stage, Termination


def unit_descriptor(seed: int, dim: int = 128) -> np.ndarray:
    """An L2-normalised descriptor — the card schema requires normalisation."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=dim).astype(np.float32)
    return d / np.linalg.norm(d)


def make_team(uv: np.ndarray, seed0: int = 0) -> list[Keypoint]:
    return [Keypoint(uv=p, descriptor=unit_descriptor(seed0 + i)) for i, p in enumerate(uv)]


TARGET_UV = np.array([[100.0, 100.0], [140.0, 104.0], [120.0, 140.0], [96.0, 136.0]])
HELD_UV = np.array([[200.0, 90.0], [232.0, 96.0], [214.0, 128.0]])


@pytest.fixture
def d1_stage() -> Stage:
    """Pick-and-place shape: no held team, so the moving end is the gripper itself."""
    return Stage(
        name="grasp-the-block",
        camera="top",
        teams={"target": make_team(TARGET_UV, seed0=0), "held": []},
        goal_relation=GoalRelation(held_uv=HELD_UV, n_demos=1),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("fission", {"sustain": 3}),
        budget=Budget(seconds=20.0, retries=3),
        grasp_aperture_expected=0.35,
    )


@pytest.fixture
def d2_stage() -> Stage:
    """Peg-in-hole shape: a real held team plus a push-test termination."""
    return Stage(
        name="insert-the-peg",
        camera="wrist",
        teams={"target": make_team(TARGET_UV, seed0=10), "held": make_team(HELD_UV, seed0=20)},
        goal_relation=GoalRelation(held_uv=HELD_UV, spread_uv=np.full_like(HELD_UV, 1.5), n_demos=3),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("push_test", {"probe_mm": 1.0}),
        budget=Budget(seconds=40.0, retries=4),
    )


@pytest.fixture
def d1_card(d1_stage) -> Card:
    return Card(name="d1-pick-and-place", stages=[d1_stage])


@pytest.fixture
def d2_card(d2_stage) -> Card:
    return Card(name="d2-peg-in-hole", stages=[d2_stage])
