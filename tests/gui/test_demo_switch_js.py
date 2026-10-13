"""Switching demos must not leave an editor that could save the old demo's marks."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_demo_switch_js():
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("demo_switch.test.js"))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
