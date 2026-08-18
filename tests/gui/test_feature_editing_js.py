"""``feature_editing.js`` owns the Inspector panel and the per-feature timeline
rows, and nothing loaded it — its render rules were reachable only by driving a
browser. That is how the uniform-value case shipped uncovered: reverting the
suppression left the whole suite green.

The pure decision and formatting functions are exposed on
``window.FeatureEditing._internals`` and asserted under node. See
``feature_editing.test.js``; this wrapper runs it (skipped when node is absent).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_feature_editing_render_rules_js():
    test_js = Path(__file__).parent / "feature_editing.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
