"""The flag picker's read-back rule is JS, so its unit test runs under node; this
pytest wrapper invokes it (skipped when node is absent). It locks the rule inverse to
the camera picker's: nothing ticked submits nothing, because an absent value already
means "exclude nothing" — and an empty list is not a second spelling of it, since
DatasetConfig refuses one. See flag_picker.test.js for the assertions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_flag_picker_js():
    test_js = Path(__file__).parent / "flag_picker.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
