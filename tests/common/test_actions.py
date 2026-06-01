# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the shared action manifest.

The orchestrator and the GUI both call :func:`actions_for_workflow`;
its outputs are the contract that keeps them in sync without an
inter-process protocol. These tests pin the manifest's shape so
silent drift can't break the hotkeys UI or the channel registration.
"""

from __future__ import annotations

import pytest

from lerobot.common.actions import (
    HVLA_RLT_ACTIONS,
    RECORD_ACTIONS,
    ActionDecl,
    actions_for_workflow,
    all_known_actions,
    register_actions,
)
from lerobot.common.control_channel import ControlChannel


def test_record_actions_have_keyboard_defaults():
    """Record loop's three verbs ship with their legacy bindings —
    so a fresh checkout's keyboard works without a config file."""
    names = {a.name: a for a in RECORD_ACTIONS}
    assert set(names) == {"exit_early", "rerecord_episode", "stop_recording"}
    # Multi-key compound preserved from legacy: left + esc fire
    # exit_early in addition to their own verb.
    assert names["exit_early"].default_keyboard_keys == ("right", "left", "esc")
    assert names["rerecord_episode"].default_keyboard_keys == ("left",)
    assert names["stop_recording"].default_keyboard_keys == ("esc",)


def test_hvla_rlt_actions_full_set():
    """HVLA RLT registers four operator-intent verbs with the keys
    documented for the operator (r / left / down / e)."""
    names = {a.name: a for a in HVLA_RLT_ACTIONS}
    assert set(names) == {"rlt_success", "rlt_abort", "rlt_ignore", "rlt_toggle_engage"}
    assert names["rlt_success"].default_keyboard_keys == ("r",)
    assert names["rlt_abort"].default_keyboard_keys == ("left",)
    assert names["rlt_ignore"].default_keyboard_keys == ("down",)
    assert names["rlt_toggle_engage"].default_keyboard_keys == ("e",)


def test_actions_for_workflow_record_is_base():
    """Record workflow gets exactly RECORD_ACTIONS + any globals."""
    result = actions_for_workflow("record")
    assert [a.name for a in result if a.name not in {a2.name for a2 in RECORD_ACTIONS}] == []
    assert {a.name for a in RECORD_ACTIONS} <= {a.name for a in result}


def test_actions_for_workflow_hvla_excludes_rlt_unless_flagged():
    """HVLA without ``rlt_mode`` is the same vocabulary as record —
    same flow-control verbs, no RLT-specific signals."""
    hvla = {a.name for a in actions_for_workflow("hvla", rlt_mode=False)}
    assert "rlt_success" not in hvla
    assert "exit_early" in hvla  # base flow-control still present


def test_actions_for_workflow_hvla_includes_rlt_when_flagged():
    """``rlt_mode=True`` adds the four RLT-specific verbs."""
    hvla_rlt = {a.name for a in actions_for_workflow("hvla", rlt_mode=True)}
    assert {"rlt_success", "rlt_abort", "rlt_ignore", "rlt_toggle_engage"} <= hvla_rlt


def test_actions_for_workflow_unknown_returns_globals_only():
    """Unknown workflow string returns just the globals (empty today)
    rather than raising — orchestrators with experimental workflow
    names don't crash the channel."""
    result = actions_for_workflow("totally_made_up")
    # Today globals are empty; assert the shape rather than a hard
    # length so adding a global later doesn't break this test.
    assert all(not a.name.startswith(("record.", "hvla.")) for a in result)


def test_all_known_actions_is_union():
    """``all_known_actions`` should include everything across every
    workflow + rlt_mode permutation. Used by the GUI to render the
    full hotkeys page."""
    union = {a.name for a in all_known_actions()}
    assert {a.name for a in RECORD_ACTIONS} <= union
    assert {a.name for a in HVLA_RLT_ACTIONS} <= union


def test_all_known_actions_dedupes_by_name():
    """An action shared across workflows (e.g. exit_early appears in
    both record and hvla) should appear once in the union — the GUI
    should render one row per action, not one row per workflow."""
    union = all_known_actions()
    names = [a.name for a in union]
    assert len(names) == len(set(names))


def test_register_actions_round_trips_through_channel():
    """``register_actions(channel, actions)`` should leave the channel
    able to dispatch each action's default keys to the action."""
    ch = ControlChannel()
    register_actions(ch, list(RECORD_ACTIONS))

    # Each registered action's keyboard bindings dispatch correctly.
    assert "exit_early" in ch.events
    assert "rerecord_episode" in ch.events
    assert "stop_recording" in ch.events

    # Multi-key: pressing left fires both exit_early and rerecord_episode.
    fired = ch.emit_for_keyboard_key("left")
    assert set(fired) == {"exit_early", "rerecord_episode"}
    assert ch.events["exit_early"] is True
    assert ch.events["rerecord_episode"] is True


def test_register_actions_handles_no_default_keys():
    """An action with no ``default_keyboard_keys`` registers cleanly —
    used for actions that ship without a keyboard binding."""
    ch = ControlChannel()
    register_actions(ch, [ActionDecl(name="mark_success", description="…")])
    assert "mark_success" in ch.events
    assert ch._actions["mark_success"].keyboard_keys == ()


@pytest.mark.parametrize("workflow", ["record", "hvla"])
def test_actions_for_workflow_outputs_are_actiondecl(workflow):
    """Type contract: callers can rely on the return shape — every
    entry has ``name``, ``description``, and ``default_keyboard_keys``."""
    for action in actions_for_workflow(workflow):
        assert isinstance(action, ActionDecl)
        assert isinstance(action.name, str) and action.name
        assert isinstance(action.description, str)
        assert isinstance(action.default_keyboard_keys, tuple)
