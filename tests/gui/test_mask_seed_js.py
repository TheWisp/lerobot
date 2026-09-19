"""The prompt rows a segmentation step opens with are decided in JS, so the unit test runs
under node; this pytest wrapper invokes it (skipped when node is absent). It locks the
regression where picking SAM3 on a dataset with saved masks blanked every prompt box, and
the opposite fault it was introduced fixing — saved names left in the rows so typing
appended to them. See mask_seed.test.js for the assertions."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_mask_prompt_seeding_js():
    test_js = Path(__file__).parent / "mask_seed.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)  # noqa: S603, S607
    assert result.returncode == 0, result.stdout + result.stderr
