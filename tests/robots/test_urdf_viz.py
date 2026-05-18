#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the URDF viz ProcessorStep + attach helper.

The pinocchio/meshcat-dependent class :class:`BimanualUrdfViz` isn't
exercised here — those tests live in the manual smoke runs (a real
hardware replay against the GUI iframe). What this file covers is the
glue around it:

  * ``UrdfVizMirrorStep`` reads observation joints and forwards them
    correctly to a fake viz object (no MeshCat needed).
  * ``CommandedJointsLogStep`` throttles its INFO output and reads
    the right action keys.
  * ``maybe_attach_urdf_viz`` builds the right step kind based on the
    robot's observation_features (unimanual vs bimanual) and survives
    a missing viz dependency.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lerobot.processor import TransitionKey
from lerobot.robots.so107_description.urdf_viz import (
    CommandedJointsLogStep,
    UrdfVizMirrorStep,
)


class FakeViz:
    """Records the set_arm_joints_deg calls instead of pushing to MeshCat."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray]] = []

    def set_arm_joints_deg(self, arm: str, joint_deg_7: np.ndarray) -> None:
        self.calls.append((arm, np.asarray(joint_deg_7, dtype=float).copy()))


SO107_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _obs(prefix: str = "", angles: tuple[float, ...] = (0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 20.0)) -> dict:
    return {f"{prefix}{name}.pos": float(v) for name, v in zip(SO107_MOTOR_NAMES, angles, strict=False)}


def _run(step, observation):
    transition = {TransitionKey.OBSERVATION: observation, TransitionKey.ACTION: {}}
    return step(transition)


# ── UrdfVizMirrorStep ─────────────────────────────────────────────────────


def test_unimanual_step_forwards_seven_joints_to_right_arm():
    viz = FakeViz()
    step = UrdfVizMirrorStep(viz=viz, bimanual=False, unimanual_arm="right")
    _run(step, _obs())
    assert len(viz.calls) == 1
    arm, joints = viz.calls[0]
    assert arm == "right"
    np.testing.assert_allclose(joints, [0.0, -90.0, 60.0, 0.0, -40.0, 0.0, 20.0])


def test_bimanual_step_calls_both_arms_with_their_prefixed_joints():
    viz = FakeViz()
    step = UrdfVizMirrorStep(viz=viz, bimanual=True)
    obs = {
        **_obs("left_", (10.0, -80.0, 50.0, 5.0, -30.0, 0.0, 30.0)),
        **_obs("right_", (-10.0, -70.0, 70.0, -5.0, -40.0, 0.0, 40.0)),
    }
    _run(step, obs)
    arms_called = sorted(c[0] for c in viz.calls)
    assert arms_called == ["left", "right"]
    by_arm = dict(viz.calls)
    np.testing.assert_allclose(by_arm["left"], [10.0, -80.0, 50.0, 5.0, -30.0, 0.0, 30.0])
    np.testing.assert_allclose(by_arm["right"], [-10.0, -70.0, 70.0, -5.0, -40.0, 0.0, 40.0])


def test_observation_unchanged_after_step():
    """The viz step is side-effect only; the observation passes through verbatim."""
    viz = FakeViz()
    step = UrdfVizMirrorStep(viz=viz, bimanual=False)
    obs = _obs()
    out = _run(step, obs)[TransitionKey.OBSERVATION]
    assert out == obs


def test_step_returns_silently_when_observation_lacks_a_joint():
    """If a key is missing the step skips that arm (not raise)."""
    viz = FakeViz()
    step = UrdfVizMirrorStep(viz=viz, bimanual=False)
    obs = _obs()
    obs.pop("gripper.pos")
    _run(step, obs)
    assert viz.calls == []  # incomplete data -> skip rather than partial render


def test_step_warns_once_per_arm_on_render_failure(caplog):
    """First failure is a WARNING; subsequent are DEBUG to avoid log spam."""

    class BrokenViz:
        def set_arm_joints_deg(self, arm, joints):
            raise RuntimeError("synthetic render failure")

    step = UrdfVizMirrorStep(viz=BrokenViz(), bimanual=False)
    with caplog.at_level(logging.WARNING, logger="lerobot.robots.so107_description.urdf_viz"):
        _run(step, _obs())
        _run(step, _obs())
        _run(step, _obs())
    warning_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
    # Exactly one WARNING — the rest were dropped to DEBUG.
    assert len(warning_lines) == 1
    assert "synthetic render failure" in warning_lines[0].getMessage()


# ── CommandedJointsLogStep ────────────────────────────────────────────────


def test_commanded_joints_log_throttles_by_wall_clock(caplog):
    step = CommandedJointsLogStep(log_interval_s=0.1, bimanual=False)
    action = {f"{m}.pos": 12.3 for m in SO107_MOTOR_NAMES}

    def _tick():
        return step({TransitionKey.OBSERVATION: {}, TransitionKey.ACTION: dict(action)})

    with caplog.at_level(logging.INFO, logger="lerobot.robots.so107_description.urdf_viz"):
        # First call logs.
        _tick()
        # Same-tick calls don't.
        for _ in range(20):
            _tick()
    info_lines = [r for r in caplog.records if r.levelno == logging.INFO and "commanded" in r.getMessage()]
    assert len(info_lines) == 1, f"expected 1 INFO line, got {len(info_lines)}"


def test_commanded_joints_log_action_passes_through():
    step = CommandedJointsLogStep(log_interval_s=10, bimanual=False)
    action = {f"{m}.pos": 1.0 for m in SO107_MOTOR_NAMES}
    out = step({TransitionKey.OBSERVATION: {}, TransitionKey.ACTION: dict(action)})
    assert out[TransitionKey.ACTION] == action


def test_commanded_joints_log_bimanual_formats_per_arm(caplog):
    step = CommandedJointsLogStep(log_interval_s=0, bimanual=True)
    action = {}
    for prefix in ("left_", "right_"):
        for m in SO107_MOTOR_NAMES:
            action[f"{prefix}{m}.pos"] = 5.0
    with caplog.at_level(logging.INFO, logger="lerobot.robots.so107_description.urdf_viz"):
        step({TransitionKey.OBSERVATION: {}, TransitionKey.ACTION: dict(action)})
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    # Two lines — one per arm.
    arm_msgs = [m for m in msgs if m.startswith("commanded[")]
    assert any("commanded[left]" in m for m in arm_msgs)
    assert any("commanded[right]" in m for m in arm_msgs)


# ── maybe_attach_urdf_viz helper ──────────────────────────────────────────


def test_maybe_attach_urdf_viz_detects_bimanual_robot():
    """Robots with left_<motor>.pos observation_features get bimanual=True step."""
    from lerobot.robots.so107_description import urdf_viz as uv

    steps: list = []
    fake_robot = SimpleNamespace(
        observation_features={
            "left_shoulder_pan.pos": float,
            "right_shoulder_pan.pos": float,
        }
    )
    with patch.object(uv, "BimanualUrdfViz", return_value=SimpleNamespace(url="http://x", calls=[])):
        viz = uv.maybe_attach_urdf_viz(steps, fake_robot)
    assert viz is not None
    assert len(steps) == 1
    assert isinstance(steps[0], UrdfVizMirrorStep)
    assert steps[0].bimanual is True


def test_maybe_attach_urdf_viz_detects_unimanual_robot():
    from lerobot.robots.so107_description import urdf_viz as uv

    steps: list = []
    fake_robot = SimpleNamespace(observation_features={"shoulder_pan.pos": float})
    with patch.object(uv, "BimanualUrdfViz", return_value=SimpleNamespace(url="http://x", calls=[])):
        uv.maybe_attach_urdf_viz(steps, fake_robot)
    assert len(steps) == 1
    assert steps[0].bimanual is False


def test_maybe_attach_urdf_viz_swallows_viz_construction_failure(caplog):
    """If BimanualUrdfViz raises (e.g. port 7000 busy), helper logs and returns None."""
    from lerobot.robots.so107_description import urdf_viz as uv

    steps: list = []
    fake_robot = SimpleNamespace(observation_features={})

    def boom(*args, **kwargs):
        raise OSError("port already in use")

    with patch.object(uv, "BimanualUrdfViz", side_effect=boom):
        with caplog.at_level(logging.WARNING, logger="lerobot.robots.so107_description.urdf_viz"):
            viz = uv.maybe_attach_urdf_viz(steps, fake_robot)
    assert viz is None
    assert steps == []
    assert any("port already in use" in r.getMessage() for r in caplog.records)


def test_maybe_attach_urdf_viz_tolerates_robot_without_features():
    """Robot stub missing observation_features just gets unimanual mode, doesn't crash."""
    from lerobot.robots.so107_description import urdf_viz as uv

    steps: list = []
    fake_robot = SimpleNamespace()  # no observation_features attribute
    with patch.object(uv, "BimanualUrdfViz", return_value=SimpleNamespace(url="http://x", calls=[])):
        uv.maybe_attach_urdf_viz(steps, fake_robot)
    assert len(steps) == 1
    assert steps[0].bimanual is False
