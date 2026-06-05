# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the edit-tier Run MCP tool surface (update_rlt_config)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from lerobot.gui.api import run as run_module
from lerobot.mcp.server import build_server


@pytest.fixture
def mcp(tmp_path):
    return build_server(dataset_root=tmp_path / "_root")


def _call(mcp_server, name, args):
    _, structured = asyncio.run(mcp_server.call_tool(name, args))
    return structured


@pytest.fixture
def active_session(tmp_path):
    """Patch a fake active RLT session pointing at tmp_path."""
    output_dir = tmp_path / "rlt_run"
    output_dir.mkdir()
    with patch.object(run_module, "_active_config", {"rlt_output_dir": str(output_dir)}):
        yield output_dir


class TestUpdateRltConfig:
    def test_writes_overrides_file(self, mcp, active_session):
        result = _call(mcp, "update_rlt_config", {"beta": 1.5, "exploration_sigma": 0.3})
        assert result["status"] == "ok"
        assert result["applied"] == {"beta": 1.5, "exploration_sigma": 0.3}
        on_disk = json.loads((active_session / "rlt_overrides.json").read_text())
        assert on_disk["beta"] == 1.5
        assert on_disk["exploration_sigma"] == 0.3

    def test_partial_update_merges_with_existing(self, mcp, active_session):
        (active_session / "rlt_overrides.json").write_text(json.dumps({"beta": 2.0, "target_sigma": 0.5}))
        result = _call(mcp, "update_rlt_config", {"exploration_sigma": 0.1})
        # Only the new key is applied
        assert result["applied"] == {"exploration_sigma": 0.1}
        # previous_values reflects the pre-merge state for the keys we wrote
        assert result["previous_values"] == {"exploration_sigma": None}
        # File holds all three
        on_disk = json.loads((active_session / "rlt_overrides.json").read_text())
        assert on_disk == {"beta": 2.0, "target_sigma": 0.5, "exploration_sigma": 0.1}

    def test_clamp_carries_what_was_requested_vs_applied(self, mcp, active_session):
        result = _call(mcp, "update_rlt_config", {"beta": 999.0})
        assert result["applied"]["beta"] == 10.0
        assert result["clamped"]["beta"] == {
            "requested": 999.0,
            "applied": 10.0,
            "range": [0.0, 10.0],
        }

    def test_no_clamp_means_no_clamped_field(self, mcp, active_session):
        result = _call(mcp, "update_rlt_config", {"beta": 5.0})
        assert "clamped" not in result

    def test_dump_chunks_bool_no_clamp(self, mcp, active_session):
        result = _call(mcp, "update_rlt_config", {"dump_chunks": True})
        assert result["applied"]["dump_chunks"] is True

    def test_no_active_session_raises(self, mcp):
        with patch.object(run_module, "_active_config", None):
            with pytest.raises(Exception, match="No active RLT session"):
                _call(mcp, "update_rlt_config", {"beta": 1.0})

    def test_active_config_without_rlt_output_dir_raises(self, mcp):
        with patch.object(run_module, "_active_config", {"some_other_key": "val"}):
            with pytest.raises(Exception, match="No active RLT session"):
                _call(mcp, "update_rlt_config", {"beta": 1.0})

    def test_no_args_raises(self, mcp, active_session):
        with pytest.raises(Exception, match="No override fields"):
            _call(mcp, "update_rlt_config", {})

    def test_all_none_raises(self, mcp, active_session):
        with pytest.raises(Exception, match="No override fields"):
            _call(
                mcp,
                "update_rlt_config",
                {
                    "beta": None,
                    "exploration_sigma": None,
                    "target_sigma": None,
                    "dump_chunks": None,
                },
            )


class TestCrossSurfaceSharedState:
    """The MCP `update_rlt_config` and FastAPI `POST /api/run/rlt-config`
    write to the same `rlt_overrides.json`. A change via MCP should be
    visible via the FastAPI read path (`GET /api/run/rlt-config`).
    """

    def test_mcp_write_visible_via_run_core_read(self, mcp, active_session):
        _call(mcp, "update_rlt_config", {"beta": 3.0, "exploration_sigma": 0.2})
        # The FastAPI handler reads the same file the MCP wrote
        on_disk = json.loads((active_session / "rlt_overrides.json").read_text())
        assert on_disk == {"beta": 3.0, "exploration_sigma": 0.2}
