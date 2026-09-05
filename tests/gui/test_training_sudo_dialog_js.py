"""How the sudo-password dialog submits, which is client-side.

See training_sudo_dialog.test.js for the assertions. The submit path is two
round trips before a run exists — and it is offered only on hosts where a round
trip is slow — so the dialog looks idle for seconds after the click. That is the
window in which a second click, or a held Enter, would start a second run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_training_sudo_dialog_js():
    test_js = Path(__file__).parent / "training_sudo_dialog.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
