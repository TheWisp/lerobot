"""Fast editor regressions at the DOM/player/HTTP boundaries, without hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_editor_controls():
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("flash_editor_controls.test.js"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
