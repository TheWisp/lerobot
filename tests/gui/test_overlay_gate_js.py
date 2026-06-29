"""The live-overlay draw gate is JS, so its unit test runs under node; this pytest wrapper invokes
it (skipped when node is absent). It locks the started-race fix: the overlay draws iff the backend
OverlayStateMachine reports ACTIVE, as a pure function of that state — never a frontend flag that
can desync during the worker's spawn. See overlay_gate.test.js for the assertions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_overlay_draw_gate_js():
    test_js = Path(__file__).parent / "overlay_gate.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
