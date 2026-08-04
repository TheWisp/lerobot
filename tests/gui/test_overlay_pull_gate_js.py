"""The data-overlay pull gate is JS, so its unit test runs under node; this pytest
wrapper invokes it (skipped when node is absent). It locks the measured regression:
overlay pulls must follow the worker's overlay seq (one pull per produced overlay,
respawn-safe), with per-frame ticks pulling only as a rate-limited fallback while
SSE is down — unconditional per-tick pulls cost ~4 fps of worker throughput.
See overlay_pull_gate.test.js for the assertions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_overlay_pull_gate_js():
    test_js = Path(__file__).parent / "overlay_pull_gate.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
