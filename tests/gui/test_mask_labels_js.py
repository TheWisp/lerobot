"""How a stored-mask region's name is sized and placed is decided in JS, so the unit test
runs under node and this pytest wrapper invokes it (skipped when node is absent). It locks
the reported "weird large font ... that stacks on top of existing ones": a font floored in
canvas space grew with the tile, and labels on regions at the top edge landed on the same
pixels. See mask_labels.test.js for the assertions, and test_overlay_label_placement.py for
the Python half and the cross-language size parity."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_mask_labels_js():
    test_js = Path(__file__).parent / "mask_labels.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)  # noqa: S603, S607
    assert result.returncode == 0, result.stdout + result.stderr
