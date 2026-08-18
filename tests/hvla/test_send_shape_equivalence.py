# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Does `--send-action-shape` change what a robot actually executes?

S1 can hand the robot either payload shape for the same policy output:

* ``chunk`` (default) — ``chunk[idx:]`` packed as an ``ActionChunk``, so a
  robot with a lookahead controller can read the value at ``now + L`` directly
  from the frames instead of extrapolating from stair-stepped 30 Hz input.
* ``dict`` — only the single frame at ``idx``.

For a robot *without* a lookahead controller, the two are supposed to be
indistinguishable: it collapses the chunk to ``frames[0]``, which is the frame
meant for "now" — the same one the dict path sends. That claim is what makes
the chunk default safe on every robot, so it is worth pinning rather than
believing.

The comparison is enumerative on purpose: both payloads are built from the same
policy output, then reduced the way a chunk-unaware driver reduces them, and the
delivered dictionaries compared key by key.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.policies.hvla.s1_process import _remaining_chunk_as_actionchunk
from lerobot.types import action_first_frame

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "gripper.pos"]
FPS = 30.0


def _dict_payload(action_np: np.ndarray, joint_names: list[str]) -> dict[str, float]:
    """The `dict` branch of s1_process, reproduced so the two can be compared."""
    return {name: float(action_np[i]) for i, name in enumerate(joint_names) if i < len(action_np)}


def _chunk_payload_as_delivered(chunk: np.ndarray, idx: int, action_np: np.ndarray, joint_names):
    """The `chunk` branch, reduced the way a chunk-unaware robot reduces it."""
    packed = _remaining_chunk_as_actionchunk(chunk, idx, joint_names, FPS, current_frame_override=action_np)
    return action_first_frame(packed)


def _policy_chunk(n_frames: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-90.0, 90.0, size=(n_frames, len(JOINTS)))


class TestAChunkUnawareRobotExecutesTheSameAction:
    @pytest.mark.parametrize("idx", [0, 1, 4, 7])
    def test_both_shapes_deliver_an_identical_action(self, idx):
        """The claim that makes `chunk` safe as the default on every robot."""
        chunk = _policy_chunk()
        action_np = chunk[idx]

        assert _chunk_payload_as_delivered(chunk, idx, action_np, JOINTS) == _dict_payload(action_np, JOINTS)

    def test_a_clamped_frame_survives_the_chunk_path(self):
        """The jump clamp rewrites the frame for "now" without touching the rest
        of the chunk. If the clamp were dropped on the chunk path, a rewrite
        meant to protect the arm would apply in `dict` mode only.
        """
        chunk = _policy_chunk()
        idx = 3
        clamped = chunk[idx] * 0.1  # what the clamp would substitute

        delivered = _chunk_payload_as_delivered(chunk, idx, clamped, JOINTS)
        assert delivered == _dict_payload(clamped, JOINTS)
        assert delivered != _dict_payload(chunk[idx], JOINTS), "the clamp must not be ignored"

    def test_the_frames_beyond_now_are_what_a_predictive_robot_gains(self):
        """The difference is real, it is just not visible to a robot that drops
        the horizon: everything after frames[0] is the lookahead a predictive
        controller reads instead of extrapolating.
        """
        chunk = _policy_chunk(n_frames=8)
        idx = 2
        packed = _remaining_chunk_as_actionchunk(chunk, idx, JOINTS, FPS)

        assert len(packed.frames) == 6, "frames cover [idx, N)"
        assert packed.fps == FPS
        for offset, frame in enumerate(packed.frames):
            assert frame == {n: pytest.approx(chunk[idx + offset][j]) for j, n in enumerate(JOINTS)}


class TestTheOneCaseWhereTheyDiverge:
    """A short action vector is handled differently by the two branches.

    `dict` filters with `if i < len(action_np)` and silently emits fewer keys;
    the chunk path indexes every joint name and raises. Neither is obviously
    right — a truncated action is a bug either way — but they do not agree, so
    the divergence is recorded rather than left for someone to hit on hardware.
    """

    def test_dict_truncates_where_chunk_raises(self):
        chunk = _policy_chunk()
        short = chunk[0][:2]  # policy emitted fewer joints than the robot has

        assert set(_dict_payload(short, JOINTS)) == {"shoulder_pan.pos", "shoulder_lift.pos"}
        with pytest.raises(IndexError):
            _chunk_payload_as_delivered(chunk, 0, short, JOINTS)
