"""The image-status banner is JS, so its unit test runs under node; this pytest wrapper
invokes it (skipped when node is absent). It locks the rule that an image event the
renderer does not name must not silently disappear — which is how ``image_refresh_failed``,
the warning that a run is training on an unverified local copy, was first added invisible.
See training_image_banner.test.js for the assertions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_training_image_banner_js():
    test_js = Path(__file__).parent / "training_image_banner.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
