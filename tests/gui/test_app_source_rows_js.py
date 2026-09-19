"""``app.js`` builds the dataset tree inside DOM code, so its row rules were
reachable only by driving a browser — which is how a placeholder shipped that
was named unlike its neighbours, sorted after every dataset, and missing
entirely from a source that had not been scanned. The pure part is now
``sourceRowsFor``; this pytest wrapper runs its node unit test (skipped when
node is absent). See app_source_rows.test.js for the assertions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_source_rows_js():
    test_js = Path(__file__).parent / "app_source_rows.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
