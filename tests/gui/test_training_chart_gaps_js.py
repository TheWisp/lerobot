"""The training charts' present-vs-missing decision is JS, so its unit test runs under node;
this pytest wrapper invokes it (skipped when node is absent). It locks the rule the resource
telemetry in the training DESIGN.md depends on: a reading the sampler could not take must render
as absent, never as a genuine 0 — while a measured 0 must still chart. See
training_chart_gaps.test.js for the assertions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_training_chart_gaps_js():
    test_js = Path(__file__).parent / "training_chart_gaps.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
