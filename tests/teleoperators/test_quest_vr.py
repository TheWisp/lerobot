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
        "gripper_pos",
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
    assert teleop._arm._engaged is True
    assert teleop._arm._quest_pos_at_engage is not None
    np.testing.assert_allclose(teleop._arm._quest_pos_at_engage, [0.1, 1.2, -0.4])
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
    # Push hand 3cm in -Z (forward, away from user's face). Stays under
    # the default 4cm/frame glitch cap so the step isn't clipped.
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, -0.03), clutch=1.0))
    a = teleop._cached_action
    # SO-107 default mapping: ROBOT_FORWARD_IN_URDF = (0, -1, 0).
    # quest_delta_to_robot for delta_quest=(0,0,-0.03) yields (0,-0.03,0).
    np.testing.assert_allclose([a["target_x"], a["target_y"], a["target_z"]], [0.0, -0.03, 0.0], atol=1e-9)


def test_release_clears_engage_state(teleop):
    """After release, the next clutch should produce a fresh snapshot."""
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, 0.0), clutch=1.0))
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, 0.0), clutch=0.0))  # release
    assert teleop._arm._engaged is False
    # Re-engage at a different position. Step <= max_pos_step_m_per_tick
    # (0.04 m default) so the glitch suppression doesn't kick in here.
    teleop._on_frame(_frame(quest_pos=(0.03, 1.0, 0.0), clutch=1.0))
    np.testing.assert_allclose(teleop._arm._quest_pos_at_engage, [0.03, 1.0, 0.0])


def test_position_glitch_is_suppressed(teleop):
    """A bogus huge jump in Quest position between frames should be clamped."""
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, 0.0), clutch=1.0))
    # Now simulate a tracking glitch: controller teleports 380mm in one frame.
    teleop._on_frame(_frame(quest_pos=(0.0, 1.0, -0.38), clutch=1.0))
    # The arm_controller should have clamped the jump to its 0.04 m cap.
    # _quest_pos_prev now holds the clamped pose.
    np.testing.assert_allclose(teleop._arm._quest_pos_prev, [0.0, 1.0, -0.04], atol=1e-6)
    # target_xyz (Quest delta from engage, mapped to robot frame) should
    # reflect only the clamped step, not the full glitched delta.
    a = teleop._cached_action
    assert abs(a["target_z"]) < 0.001
    # Quest Z = robot Y per QUEST_TO_ROBOT_M; -0.04 quest_z -> robot Y = +0.04
    assert a["target_y"] == pytest.approx(-0.04, abs=1e-6)


def test_gripper_pos_mapped_absolutely_from_trigger(teleop):
    """When clutch is engaged, trigger value maps directly to motor-space gripper position.

    Defaults: gripper_open_motor=50, gripper_closed_motor=90 (rest half-open,
    pull-to-close on the soft-limit side). Gating on clutch is intentional:
    a resting controller with a stray trigger touch shouldn't move the
    gripper.
    """
    teleop._on_frame(_frame(clutch=1.0, trigger=0.0))
    assert teleop._cached_action["gripper_pos"] == pytest.approx(50.0)
    teleop._on_frame(_frame(clutch=1.0, trigger=0.5))
    assert teleop._cached_action["gripper_pos"] == pytest.approx(70.0)
    teleop._on_frame(_frame(clutch=1.0, trigger=1.0))
    assert teleop._cached_action["gripper_pos"] == pytest.approx(90.0)


def test_gripper_pos_holds_last_value_when_disengaged(teleop):
    """Releasing the clutch latches the gripper at its last commanded value.

    Trigger movement while disengaged is IGNORED — prevents a controller
    sitting on a table with an accidental trigger touch from squeezing the
    gripper.
    """
    teleop._on_frame(_frame(clutch=1.0, trigger=0.5))
    assert teleop._cached_action["gripper_pos"] == pytest.approx(70.0)
    # Release clutch; trigger goes wild but gripper holds.
    teleop._on_frame(_frame(clutch=0.0, trigger=1.0))
    assert teleop._cached_action["gripper_pos"] == pytest.approx(70.0)
    teleop._on_frame(_frame(clutch=0.0, trigger=0.0))
    assert teleop._cached_action["gripper_pos"] == pytest.approx(70.0)
    # Re-engage with a new trigger value — gripper resumes tracking.
    teleop._on_frame(_frame(clutch=1.0, trigger=0.25))
    assert teleop._cached_action["gripper_pos"] == pytest.approx(60.0)


def test_get_action_returns_idle_when_no_frame_yet(teleop):
    """Without connecting / without a frame, get_action would raise via decorator.
    Internally _idle_action is what we return; sanity-check shape.
    """
    a = teleop._idle_action()
    assert a["enabled"] == 0.0
    # All target_* zero on idle. gripper_pos defaults to the open-motor
    # value (50 by default = half-open) so a freshly-connected teleop sits
    # at a sensible neutral position rather than slamming open or closed.
    for k, v in a.items():
        if k == "gripper_pos":
            assert v == 50.0  # gripper_open_motor default
        else:
            assert v == 0.0


def test_unknown_hand_frame_is_ignored(teleop):
    """A frame containing only the left controller should be a no-op when configured for right."""
    teleop._cached_action = None
    teleop._on_frame(_frame(hand="left", clutch=1.0))
    assert teleop._cached_action is None  # no update


# ── Bimanual variant ──────────────────────────────────────────────────────


from lerobot.teleoperators.quest_vr import (  # noqa: E402
    BimanualQuestVRTeleop,
    BimanualQuestVRTeleopConfig,
)


@pytest.fixture
def bimanual_teleop():
    return BimanualQuestVRTeleop(BimanualQuestVRTeleopConfig(id="test_bi", port=18444))


def _bimanual_frame(
    left_pos=(0.0, 1.0, -0.3),
    right_pos=(0.0, 1.0, -0.3),
    left_clutch=0.0,
    right_clutch=0.0,
    left_trigger=0.0,
    right_trigger=0.0,
) -> dict:
    """Build a synthetic WebXR frame with both controllers' poses."""

    def _pose(hand, pos, clutch, trigger):
        buttons = [0.0] * 12
        buttons[0] = float(trigger)
        buttons[1] = float(clutch)
        return {
            "hand": hand,
            "pos": list(pos),
            "rot": [0.0, 0.0, 0.0, 1.0],
            "buttons": buttons,
        }

    return {
        "type": "frame",
        "t_quest_send": 0.0,
        "poses": [
            _pose("left", left_pos, left_clutch, left_trigger),
            _pose("right", right_pos, right_clutch, right_trigger),
        ],
    }


def test_bimanual_config_type():
    cfg = BimanualQuestVRTeleopConfig(id="t")
    assert cfg.type == "bimanual_quest_vr"


def test_bimanual_factory_dispatch():
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    t = make_teleoperator_from_config(BimanualQuestVRTeleopConfig(id="t", port=18444))
    assert isinstance(t, BimanualQuestVRTeleop)


def test_bimanual_action_features_has_both_arms(bimanual_teleop):
    """16 keys total — 8 per arm with left_/right_ prefixes."""
    af = bimanual_teleop.action_features
    assert af["shape"] == (16,)
    names = set(af["names"].keys())
    per_arm = {
        "enabled",
        "target_x",
        "target_y",
        "target_z",
        "target_wx",
        "target_wy",
        "target_wz",
        "gripper_pos",
    }
    assert names == {f"{p}{k}" for p in ("left_", "right_") for k in per_arm}


def test_bimanual_idle_action_is_at_neutral(bimanual_teleop):
    """All EE-delta keys are zero; gripper_pos sits at each arm's open value."""
    a = bimanual_teleop._idle_action()
    assert len(a) == 16
    for k, v in a.items():
        if k.endswith("gripper_pos"):
            # Defaults: left_gripper_open_motor=50, right_gripper_open_motor=50.
            assert v == 50.0
        else:
            assert v == 0.0


def test_bimanual_independent_engagement(bimanual_teleop):
    """Engage only the right controller; left stays disengaged."""
    bimanual_teleop._on_frame(_bimanual_frame(right_clutch=1.0))
    a = bimanual_teleop._cached_action
    assert a is not None
    assert a["right_enabled"] == 1.0
    assert a["left_enabled"] == 0.0


def test_bimanual_both_engaged(bimanual_teleop):
    """Engage both controllers simultaneously; both arms produce enabled=1."""
    bimanual_teleop._on_frame(_bimanual_frame(left_clutch=1.0, right_clutch=1.0))
    a = bimanual_teleop._cached_action
    assert a["left_enabled"] == 1.0
    assert a["right_enabled"] == 1.0


def test_bimanual_per_arm_delta_independent(bimanual_teleop):
    """Move only the right hand; left target deltas stay zero."""
    bimanual_teleop._on_frame(
        _bimanual_frame(
            left_pos=(0.0, 1.0, -0.3),
            right_pos=(0.0, 1.0, -0.3),
            left_clutch=1.0,
            right_clutch=1.0,
        )
    )  # engage both at origin
    # Push right hand 3cm forward (under the 0.04m/frame glitch cap default).
    bimanual_teleop._on_frame(
        _bimanual_frame(
            left_pos=(0.0, 1.0, -0.3),
            right_pos=(0.0, 1.0, -0.33),
            left_clutch=1.0,
            right_clutch=1.0,
        )
    )
    a = bimanual_teleop._cached_action
    assert abs(a["left_target_x"]) < 1e-9
    assert abs(a["left_target_y"]) < 1e-9
    assert abs(a["left_target_z"]) < 1e-9
    # Right hand should have a non-zero delta (axis mapping verified by single-arm tests).
    right_norm = (a["right_target_x"] ** 2 + a["right_target_y"] ** 2 + a["right_target_z"] ** 2) ** 0.5
    assert right_norm > 0.02
