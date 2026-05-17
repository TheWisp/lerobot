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

"""Tests for QuestVRTeleop.

We test the pure-Python action computation (engagement, delta mapping, axis
conversion) without spinning up an HTTPS server, by calling the internal
``_on_frame`` callback directly with synthetic Quest frames. The server
itself is a thin aiohttp wrapper around our callback; covering it would
require integration testing with a real Quest, which we can't do here.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.teleoperators.quest_vr import QuestVRTeleop, QuestVRTeleopConfig


@pytest.fixture
def teleop():
    return QuestVRTeleop(QuestVRTeleopConfig(id="test", port=18443))


def _frame(
    quest_pos=(0.0, 1.0, -0.3),
    quest_quat=(0.0, 0.0, 0.0, 1.0),
    clutch=0.0,
    trigger=0.0,
    hand="right",
) -> dict:
    """Build a synthetic WebXR frame matching the JS payload format."""
    buttons = [0.0] * 12
    buttons[0] = float(trigger)  # gripper trigger
    buttons[1] = float(clutch)  # clutch grip
    return {
        "type": "frame",
        "t_quest_send": 0.0,
        "poses": [{"hand": hand, "pos": list(quest_pos), "rot": list(quest_quat), "buttons": buttons}],
    }


def test_config_registered():
    """Verify quest_vr is registered with draccus via @register_subclass."""
    from lerobot.teleoperators.config import TeleoperatorConfig

    # ChoiceRegistry holds subclasses keyed by name; access via internal API.
    cfg = QuestVRTeleopConfig(id="t")
    assert cfg.type == "quest_vr"
    assert isinstance(cfg, TeleoperatorConfig)


def test_factory_dispatch_creates_teleop():
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    t = make_teleoperator_from_config(QuestVRTeleopConfig(id="t", port=18443))
    assert isinstance(t, QuestVRTeleop)
    assert t.is_calibrated  # Quest VR teleop is calibration-free.


def test_action_features_shape(teleop):
    af = teleop.action_features
    assert af["shape"] == (8,)
    assert set(af["names"].keys()) == {
        "enabled",
        "target_x",
        "target_y",
        "target_z",
        "target_wx",
        "target_wy",
        "target_wz",
        "gripper_vel",
    }


def test_idle_action_when_disengaged_after_frame(teleop):
    """Disengaged frames should produce an all-zeros action (apart from gripper_vel)."""
    teleop._on_frame(_frame(clutch=0.0))
    a = teleop._cached_action
    assert a is not None
    assert a["enabled"] == 0.0
    assert a["target_x"] == 0.0 and a["target_y"] == 0.0 and a["target_z"] == 0.0
    assert a["target_wx"] == 0.0 and a["target_wy"] == 0.0 and a["target_wz"] == 0.0


def test_engage_snapshots_reference_pose(teleop):
    """Rising-edge on clutch should snapshot the pose used as the reference."""
    teleop._on_frame(_frame(quest_pos=(0.1, 1.2, -0.4), clutch=1.0))
    assert teleop._engaged is True
    assert teleop._quest_pos_at_engage is not None
    np.testing.assert_allclose(teleop._quest_pos_at_engage, [0.1, 1.2, -0.4])
    # First engage frame's delta = 0 (against itself).
    a = teleop._cached_action
    assert a["enabled"] == 1.0
    assert abs(a["target_x"]) < 1e-9
    assert abs(a["target_y"]) < 1e-9
    assert abs(a["target_z"]) < 1e-9


def test_engaged_delta_maps_quest_to_robot_frame(teleop):
    """Push controller forward in Quest (-Z direction) -> robot moves forward (URDF -Y)."""
    # Engage at origin.
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, 0.0), clutch=1.0))
    # Now push hand 5cm in -Z (forward, away from user's face).
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, -0.05), clutch=1.0))
    a = teleop._cached_action
    # SO-107 default mapping: ROBOT_FORWARD_IN_URDF = (0, -1, 0).
    # Quest delta z = -0.05 (forward); QUEST_TO_ROBOT_M column 2 = (0, 1, 0).
    # So delta_robot = (0, 1*(-0.05), 0) = (0, -0.05, 0). Hmm let me re-check.
    # The actual quest_delta_to_robot for delta_quest=(0,0,-0.05) yields:
    #   QUEST_TO_ROBOT_M @ (0,0,-0.05) = col2 * -0.05 = -ROBOT_FORWARD * -0.05
    #     = -(0,-1,0) * -0.05 = (0,-1,0)*0.05 = (0,-0.05,0)
    # i.e. URDF y = -0.05 (forward in our convention). Correct.
    np.testing.assert_allclose([a["target_x"], a["target_y"], a["target_z"]], [0.0, -0.05, 0.0], atol=1e-9)


def test_release_clears_engage_state(teleop):
    """After release, the next clutch should produce a fresh snapshot."""
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, 0.0), clutch=1.0))
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, 0.0), clutch=0.0))  # release
    assert teleop._engaged is False
    # Re-engage at a different position.
    teleop._on_frame(_frame(quest_pos=(0.2, 1.0, 0.0), clutch=1.0))
    np.testing.assert_allclose(teleop._quest_pos_at_engage, [0.2, 1.0, 0.0])


def test_gripper_vel_derived_from_trigger_change(teleop):
    """Trigger value delta should appear in gripper_vel."""
    teleop._on_frame(_frame(trigger=0.0))  # baseline
    teleop._on_frame(_frame(trigger=0.3))  # trigger pulled
    # gripper_vel = previous - current = 0.0 - 0.3 = -0.3 (close).
    assert teleop._cached_action["gripper_vel"] == pytest.approx(-0.3)


def test_get_action_returns_idle_when_no_frame_yet(teleop):
    """Without connecting / without a frame, get_action would raise via decorator.
    Internally _idle_action is what we return; sanity-check shape.
    """
    a = teleop._idle_action()
    assert a["enabled"] == 0.0
    assert all(a[k] == 0.0 for k in a)


def test_unknown_hand_frame_is_ignored(teleop):
    """A frame containing only the left controller should be a no-op when configured for right."""
    teleop._cached_action = None
    teleop._on_frame(_frame(hand="left", clutch=1.0))
    assert teleop._cached_action is None  # no update
