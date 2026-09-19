"""Runs tests/gui/chunk_treatments.test.js in node: the recipe's treatments as
pure pixel functions, mirroring overlays/effects.py."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TEST = Path(__file__).with_name("chunk_treatments.test.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_treatments_in_the_page():
    out = subprocess.run(["node", str(TEST)], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "chunk_treatments: ok" in out.stdout
