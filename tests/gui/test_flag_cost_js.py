"""The cost annotation is JS, so its unit test runs under node; this pytest
wrapper invokes it (skipped when node is absent). It pins two things the figure
depends on: that the chunk length comes from the form rather than a default,
since every policy names that field differently and three offer none, and that
an annotation overtaken by a newer one does not write into the box it no longer
describes. See flag_cost.test.js for the assertions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_flag_cost_js():
    test_js = Path(__file__).parent / "flag_cost.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
