#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Property tests for the Quest VR teleoperator's pure-logic surface.

No Quest hardware, no HTTPS server: drives ``QuestArmController`` directly
with synthetic WebXR frames and asserts the properties that justify Quest
VR being usable as an intent source for a Cartesian-IK robot. Properties
covered (matching the proof checklist sketched in the PR body):

1. **Engage-relative deltas.** ``target_xyz`` is the position delta from
   the engage snapshot, scaled by ``position_scale``. ``rotvec`` is the
   orientation delta from the engage snapshot.
2. **No drift on hold.** Clutch held + hand held still = ``target_xyz``
   stays at zero across many WebXR frames.
3. **Glitch suppression.** A jump larger than ``max_pos_step_m_per_tick``
   between consecutive Quest poses is clipped before it enters the delta.
4. **Bimanual independence.** A frame with only one hand updates that
   arm's action keys and routes the other hand to tracking-lost (its
   action keys stay disengaged).
5. **Gripper trigger mapping.** Engaged + trigger=t maps motor pos
   linearly from open to closed. Disengaged frames hold the last command.
6. **Tracking-dropout re-anchor.** After a frame with no pose for this
   controller, the next tracked frame re-anchors engage state — so a
   long no-tracking gap can never leak into ``target_xyz`` as a slew.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.teleoperators.quest_vr.arm_controller import QuestArmController
from lerobot.teleoperators.quest_vr.configuration_quest_vr import QuestVRTeleopConfig
from lerobot.teleoperators.quest_vr.server import (
    QUEST_TO_ROBOT_M,
    quest_delta_to_robot,
)

# ── Helpers ───────────────────────────────────────────────────────────────


_CLUTCH_IDX = 1  # default
_TRIGGER_IDX = 0  # default
_OPEN_MOTOR = 0.0
_CLOSED_MOTOR = 100.0
_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def _make_controller(**overrides) -> QuestArmController:
    """Default controller params — identity scale, generous caps for unit tests."""
    defaults = {
        "clutch_button_index": _CLUTCH_IDX,
        "gripper_button_index": _TRIGGER_IDX,
        "position_scale": 1.0,
        "max_rot_step_rad_per_tick": np.pi,
        "max_pos_step_m_per_tick": 1.0,  # off by default — tests opt in
        "gripper_open_motor": _OPEN_MOTOR,
        "gripper_closed_motor": _CLOSED_MOTOR,
        "key_prefix": "",
    }
    defaults.update(overrides)
    return QuestArmController(**defaults)


def _pose(pos, *, clutch: float = 1.0, trigger: float = 0.0, rot=_IDENTITY_QUAT):
    """Build a WebXR-shaped pose dict for one controller."""
    buttons = [0.0, 0.0]
    buttons[_TRIGGER_IDX] = float(trigger)
    buttons[_CLUTCH_IDX] = float(clutch)
    return {"pos": list(pos), "rot": list(rot), "buttons": buttons}


# ── 1. Engage-relative deltas ─────────────────────────────────────────────


def test_disengaged_action_is_idle_and_keeps_gripper_open():
    ctrl = _make_controller()
    action = ctrl.process_pose(_pose([0.1, 0.2, 0.3], clutch=0.0, trigger=0.5))

    assert action["enabled"] == 0.0
    for k in ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"):
        assert action[k] == 0.0, f"{k} should be zero while disengaged"
    # Trigger touch while disengaged should NOT slam the gripper.
    assert action["gripper_pos"] == _OPEN_MOTOR


def test_engaged_targets_are_position_delta_from_engage_snapshot():
    ctrl = _make_controller()
    # Engage at a non-trivial Quest position.
    engage = np.array([0.5, 1.2, -0.3])
    a0 = ctrl.process_pose(_pose(engage, clutch=1.0))
    assert a0["enabled"] == 1.0
    # First engaged frame: hand hasn't moved since engage, so zero delta.
    assert pytest.approx(a0["target_x"], abs=1e-12) == 0.0
    assert pytest.approx(a0["target_y"], abs=1e-12) == 0.0
    assert pytest.approx(a0["target_z"], abs=1e-12) == 0.0

    # Move the hand 7 cm along Quest +x. target_xyz is QUEST_TO_ROBOT_M @ delta.
    dquest = np.array([0.07, 0.0, 0.0])
    a1 = ctrl.process_pose(_pose(engage + dquest, clutch=1.0))
    drobot_expected = quest_delta_to_robot(dquest)
    np.testing.assert_allclose(
        [a1["target_x"], a1["target_y"], a1["target_z"]],
        drobot_expected,
        atol=1e-12,
    )


def test_position_scale_multiplies_target_delta():
    ctrl = _make_controller(position_scale=0.5)
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))  # engage
    a = ctrl.process_pose(_pose([0.10, 0.0, 0.0], clutch=1.0))

    drobot = quest_delta_to_robot(np.array([0.10, 0.0, 0.0])) * 0.5
    np.testing.assert_allclose([a["target_x"], a["target_y"], a["target_z"]], drobot, atol=1e-12)


def test_axis_mapping_is_consistent_with_quest_to_robot_matrix():
    """The robot-frame delta for each Quest unit vector equals that column
    of QUEST_TO_ROBOT_M (sanity-checks the frame conversion is wired
    through with no surprise sign flips)."""
    for col, quest_axis in enumerate(np.eye(3)):
        ctrl = _make_controller()
        ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))
        a = ctrl.process_pose(_pose(quest_axis, clutch=1.0))
        np.testing.assert_allclose(
            [a["target_x"], a["target_y"], a["target_z"]],
            QUEST_TO_ROBOT_M[:, col],
            atol=1e-12,
            err_msg=f"Quest axis {col} mapping",
        )


# ── 2. No drift on hold ───────────────────────────────────────────────────


def test_held_clutch_with_still_hand_does_not_drift():
    ctrl = _make_controller()
    ctrl.process_pose(_pose([0.1, 0.2, 0.3], clutch=1.0))  # engage

    # 100 identical WebXR frames at the same Quest pose. target_xyz must
    # stay exactly zero — any drift here would slew the IK target.
    for _ in range(100):
        a = ctrl.process_pose(_pose([0.1, 0.2, 0.3], clutch=1.0))
        assert a["enabled"] == 1.0
        for k in ("target_x", "target_y", "target_z"):
            assert a[k] == 0.0, f"{k} drifted while clutch held + hand still"


# ── 3. Glitch suppression ─────────────────────────────────────────────────


def test_glitch_jump_is_clipped_to_max_pos_step_per_tick():
    cap = 0.02  # 2 cm/frame
    ctrl = _make_controller(max_pos_step_m_per_tick=cap)
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))  # engage at origin

    # 50 cm jump in one frame — clearly a tracking-teleport glitch.
    a = ctrl.process_pose(_pose([0.5, 0.0, 0.0], clutch=1.0))

    # |target_xyz| must be ≤ cap (with position_scale=1 the robot-frame
    # delta has the same magnitude as the Quest-frame step after clipping).
    mag = float(np.linalg.norm([a["target_x"], a["target_y"], a["target_z"]]))
    assert mag <= cap + 1e-9, f"|target| {mag} exceeded cap {cap} — glitch leaked"


def test_glitch_cap_off_when_set_to_zero():
    ctrl = _make_controller(max_pos_step_m_per_tick=0.0)  # off
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))
    a = ctrl.process_pose(_pose([0.5, 0.0, 0.0], clutch=1.0))

    mag = float(np.linalg.norm([a["target_x"], a["target_y"], a["target_z"]]))
    assert mag > 0.4, "0.0 cap should disable glitch suppression entirely"


# ── 4. Per-arm independence (via the teleop's frame callback) ────────────


def test_only_left_hand_updates_only_left_keys():
    cfg = QuestVRTeleopConfig(
        left_gripper_open_motor=_OPEN_MOTOR,
        left_gripper_closed_motor=_CLOSED_MOTOR,
        right_gripper_open_motor=_OPEN_MOTOR,
        right_gripper_closed_motor=_CLOSED_MOTOR,
    )
    # Build the teleop instance without going through connect() — we only
    # need the _on_frame path, no HTTPS server.
    from lerobot.teleoperators.quest_vr.teleop_quest_vr import QuestVRTeleop

    teleop = QuestVRTeleop(cfg)
    # Engage left, leave right tracking-lost (no entry in poses[]).
    teleop._on_frame({"poses": [{"hand": "left", **_pose([0.0, 0.0, 0.0], clutch=1.0)}]})
    teleop._on_frame({"poses": [{"hand": "left", **_pose([0.1, 0.0, 0.0], clutch=1.0)}]})

    action = teleop.get_action_raw()
    # Left engaged + moved.
    assert action["left_enabled"] == 1.0
    assert any(action[k] != 0.0 for k in ("left_target_x", "left_target_y", "left_target_z"))
    # Right untouched — never tracked, never engaged.
    assert action["right_enabled"] == 0.0
    for k in ("right_target_x", "right_target_y", "right_target_z"):
        assert action[k] == 0.0


def test_action_features_has_all_sixteen_prefixed_keys():
    cfg = QuestVRTeleopConfig()
    from lerobot.teleoperators.quest_vr.teleop_quest_vr import ACTION_KEYS, QuestVRTeleop

    teleop = QuestVRTeleop(cfg)
    feats = teleop.action_features
    assert set(feats["names"]) == set(ACTION_KEYS)
    assert len(feats["names"]) == 16  # 8 per arm × 2


# ── 5. Gripper trigger mapping ────────────────────────────────────────────


def test_gripper_pos_linearly_maps_trigger_when_engaged():
    ctrl = _make_controller()
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))  # engage

    # trigger=0 → open, trigger=0.5 → midpoint, trigger=1 → closed.
    for trig, expected in [(0.0, 0.0), (0.5, 50.0), (1.0, 100.0)]:
        a = ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0, trigger=trig))
        assert pytest.approx(a["gripper_pos"], abs=1e-9) == expected, f"trigger={trig}"


def test_disengaged_holds_last_gripper_command():
    ctrl = _make_controller()
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0, trigger=0.0))  # engage open
    a_closed = ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0, trigger=1.0))
    assert a_closed["gripper_pos"] == _CLOSED_MOTOR

    # Release clutch, wiggle trigger — gripper holds the last engaged command.
    a_held = ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=0.0, trigger=0.0))
    assert a_held["enabled"] == 0.0
    assert a_held["gripper_pos"] == _CLOSED_MOTOR


# ── 6. Tracking-dropout re-anchor ─────────────────────────────────────────


def test_tracking_lost_then_reacquire_reanchors_no_slew():
    """The whole point of on_tracking_lost: a controller that drops out of
    tracking while clutch is held cannot, on re-acquire, leak the moved-while-
    untracked offset into ``target_xyz`` as a slew. The re-acquire frame is a
    fresh rising edge."""
    ctrl = _make_controller()
    # Engage at origin, move 1 cm — this should produce a small delta.
    ctrl.process_pose(_pose([0.0, 0.0, 0.0], clutch=1.0))
    a_moved = ctrl.process_pose(_pose([0.01, 0.0, 0.0], clutch=1.0))
    assert float(np.linalg.norm([a_moved["target_x"], a_moved["target_y"], a_moved["target_z"]])) > 0.005

    # Tracking lost for one frame (controller occluded / asleep).
    idle = ctrl.on_tracking_lost()
    assert idle["enabled"] == 0.0

    # User physically walks 50 cm with controller off-tracked. On the next
    # tracked frame the clutch is still held. Must NOT produce a 50 cm delta:
    # the engage snapshot was dropped, so this re-engages at the current pose.
    a_reanchor = ctrl.process_pose(_pose([0.5, 0.0, 0.0], clutch=1.0))
    mag = float(np.linalg.norm([a_reanchor["target_x"], a_reanchor["target_y"], a_reanchor["target_z"]]))
    assert mag < 1e-9, f"|target| {mag} after re-acquire — engage snapshot was not dropped on tracking-lost"


# ── Config sanity ─────────────────────────────────────────────────────────


def test_config_registers_under_quest_vr_type_string():
    assert QuestVRTeleopConfig().type == "quest_vr"


def test_config_clutch_and_gripper_button_defaults():
    cfg = QuestVRTeleopConfig()
    # Default Quest 3 mapping: grip (1) = clutch, trigger (0) = gripper.
    assert cfg.clutch_button_index == 1
    assert cfg.gripper_button_index == 0
