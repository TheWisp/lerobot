"""Runs tests/gui/mask_chrome.test.js in node: the mask layer's one chrome
decision for both picture paths."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TEST = Path(__file__).with_name("mask_chrome.test.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_chrome_decision():
    out = subprocess.run(["node", str(TEST)], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "mask_chrome: ok" in out.stdout
