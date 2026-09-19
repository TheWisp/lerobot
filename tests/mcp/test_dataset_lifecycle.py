# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Duplicate and delete a whole dataset over MCP.

Unlike the episode tools, neither of these stages anything for the operator to
review — the copy exists, or the files are gone, when the call returns. So the
assertions are mostly about refusals, and about `delete_dataset` being
unreachable without an explicit confirm.

Both tools call the same `_datasets_core` functions the FastAPI routes use, so
what is verified here is the MCP surface: argument shape, the confirm gate, and
that typed errors reach the agent as errors rather than silent no-ops.
"""

from __future__ import annotations

import asyncio

import pytest

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState
from lerobot.mcp.server import build_server

pytest_plugins = ["tests.fixtures.dataset_factories"]


@pytest.fixture
def mcp_and_dataset(tmp_path, lerobot_dataset_factory):
    """One dataset under an MCP discovery root, with a shared AppState."""
    root = tmp_path / "owner" / "pick_place"
    root.mkdir(parents=True)
    lerobot_dataset_factory(
        root=root,
        repo_id="owner/pick_place",
        total_episodes=2,
        total_frames=20,
        total_tasks=1,
        use_videos=False,
        camera_features={},
    )
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    orig = datasets_module._app_state
    datasets_module.set_app_state(state)
    try:
        yield build_server(app_state=state, dataset_root=tmp_path), state, root
    finally:
        datasets_module._app_state = orig


def _call(mcp, name, args):
    _, structured = asyncio.run(mcp.call_tool(name, args))
    return structured


def test_duplicate_creates_a_sibling_under_the_same_owner(mcp_and_dataset):
    mcp, _, root = mcp_and_dataset
    result = _call(mcp, "duplicate_dataset", {"repo_id": "owner/pick_place", "new_name": "pick_place_v2"})
    assert result["repo_id"] == "owner/pick_place_v2"
    copy = root.parent / "pick_place_v2"
    assert (copy / "meta" / "info.json").is_file()
    assert (root / "meta" / "info.json").is_file(), "the original must survive"


def test_duplicate_rejects_a_name_that_would_escape_the_owner_folder(mcp_and_dataset):
    mcp, _, root = mcp_and_dataset
    with pytest.raises(Exception, match="single folder name"):
        _call(mcp, "duplicate_dataset", {"repo_id": "owner/pick_place", "new_name": "../elsewhere"})
    assert not (root.parent.parent / "elsewhere").exists()


def test_duplicate_refuses_to_overwrite(mcp_and_dataset):
    mcp, _, _ = mcp_and_dataset
    _call(mcp, "duplicate_dataset", {"repo_id": "owner/pick_place", "new_name": "taken"})
    with pytest.raises(Exception, match="Already exists"):
        _call(mcp, "duplicate_dataset", {"repo_id": "owner/pick_place", "new_name": "taken"})


def test_delete_without_confirm_destroys_nothing_and_says_what_it_would_cost(mcp_and_dataset):
    """A mis-parsed instruction must not be able to wipe a recording session."""
    mcp, _, root = mcp_and_dataset
    with pytest.raises(Exception, match="2 episodes"):
        _call(mcp, "delete_dataset", {"repo_id": "owner/pick_place"})
    assert (root / "meta" / "info.json").is_file()


def test_delete_with_confirm_removes_the_directory(mcp_and_dataset):
    mcp, _, root = mcp_and_dataset
    result = _call(mcp, "delete_dataset", {"repo_id": "owner/pick_place", "confirm": True})
    assert result["status"] == "ok"
    assert not root.exists()


def test_unknown_dataset_is_an_error_on_both_tools(mcp_and_dataset):
    mcp, _, _ = mcp_and_dataset
    with pytest.raises(Exception, match="not found locally"):
        _call(mcp, "duplicate_dataset", {"repo_id": "nope/missing", "new_name": "x"})
    with pytest.raises(Exception, match="not found locally"):
        _call(mcp, "delete_dataset", {"repo_id": "nope/missing", "confirm": True})
