"""The Run tab's live-video client is JS, so its unit test runs under node;
this pytest wrapper invokes it (skipped when node is absent).

It pins what has to be right before any picture can appear, and which a
browser test would only report as "no video": the offer describes one
receive-only stream per camera with H.264 first and opens the cycle channel,
because an answer can add none of those; tracks are named by the answer, not
by arrival order; and the age at the eye is worked out from the capture times
the channel announces, or reported as unknown. See live_video_client.test.js.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_live_video_client_js():
    test_js = Path(__file__).parent / "live_video_client.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
