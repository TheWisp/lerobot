"""Which run the detail pane may show while another one is loading.

The decision is client-side — the pane is filled by an async fetch — so its test
runs under node. See training_run_switch.test.js for the assertions.

The pane's fetch reads a run's files through its host, one round trip per file.
Against a rig 226 ms away that is 3.4 s, long enough for the operator to click
again, and long enough for a poll's response to land after they have. On a local
host the same code returns in ~30 ms and neither hazard is observable, which is
why both reached the GUI.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_training_run_switch_js():
    test_js = Path(__file__).parent / "training_run_switch.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_loading_state_has_a_style():
    """The placeholder is what the operator looks at for those seconds.

    Unstyled it inherits the pane's default and reads as a rendering glitch
    rather than as a wait, which is the confusion it was added to remove.
    """
    assert ".training-loading" in (STATIC / "style.css").read_text(), (
        "the loading placeholder has no style rule"
    )
