"""Execute the Robots-tab ZED camera form test under Node when available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_robot_zed_camera_js() -> None:
    test_js = Path(__file__).parent / "robot_zed_camera.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
