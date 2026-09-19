"""Runs tests/gui/chunk_player.test.js in node: the chunk player's decisions
as pure functions, with no browser, server, video or link."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TEST = Path(__file__).with_name("chunk_player.test.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_chunk_player_decisions():
    out = subprocess.run(["node", str(TEST)], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "chunk_player: ok" in out.stdout
