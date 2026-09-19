"""The camera picker's read-back rule is JS, so its unit test runs under node; this
pytest wrapper invokes it (skipped when node is absent). It locks the rule both trainers
depend on: every camera ticked submits nothing, because an absent value already means
"use every camera" — while an empty selection must stay distinguishable from an absent
one. See camera_picker.test.js for the assertions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_camera_picker_js():
    test_js = Path(__file__).parent / "camera_picker.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
