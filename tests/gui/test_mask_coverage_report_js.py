"""What a finished mask pass reports about its own coverage.

The toast fired "Saved, but nothing was found" whenever any single camera came
back empty. On a rig whose cameras do not all see the object, that is the normal
outcome of every run: the operator was told nothing was found while the timeline
filled with masks, and advised to re-run something a re-run cannot change.

Only a pass where no camera found anything now gets that wording. The logic is
in JS, so the assertions run under node; this wrapper invokes them.
See mask_coverage_report.test.js.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_mask_coverage_report_js():
    test_js = Path(__file__).parent / "mask_coverage_report.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)  # noqa: S603, S607
    assert result.returncode == 0, result.stdout + result.stderr
