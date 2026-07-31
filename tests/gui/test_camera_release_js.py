"""The leave-tab camera release decision is JS, so its unit test runs under node; this pytest
wrapper invokes it (skipped when node is absent).

It locks the fix for a bug that read as nonsense from the UI: cameras were visible in the Robot
tab but unavailable to teleop, because leaving the tab stopped the *drawing* without releasing the
*handles*. See camera_release.test.js for the assertions, and
tests/gui/test_robot_cameras.py::TestPreviewReleaseOnLaunch for the backend half.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_camera_release_decision_js():
    test_js = Path(__file__).parent / "camera_release.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_app_js_releases_rather_than_only_stopping_preview():
    """The switchTab hook must call stopAllCameras, not just stopCameraPreview.

    The JS unit test above covers the decision function; this pins that app.js
    actually uses it. Without this, the predicate could be correct and unwired —
    which is exactly the state the bug was in.
    """
    app_js = Path(__file__).resolve().parents[2] / "src/lerobot/gui/static/app.js"
    src = app_js.read_text(encoding="utf-8")
    assert "CameraRelease.shouldReleaseCameras" in src, (
        "switchTab no longer consults the release predicate — leaving the Robot tab "
        "would hold camera handles invisibly again"
    )
    assert "stopAllCameras()" in src, "switchTab must release handles, not only stop polling"


def test_camera_release_is_loaded_by_the_page():
    """A module the page never loads is a module that does nothing."""
    index = Path(__file__).resolve().parents[2] / "src/lerobot/gui/static/index.html"
    assert "camera_release.js" in index.read_text(encoding="utf-8"), (
        "camera_release.js is not referenced by index.html, so window.CameraRelease "
        "is undefined in the browser and switchTab silently falls back"
    )
