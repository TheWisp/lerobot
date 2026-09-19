"""What must hold while the live overlay owns the video tiles is decided in JS,
so the unit test runs under node; this pytest wrapper invokes it (skipped when
node is absent).

It locks two defects seen while previewing SAM3 over a dataset: the transport
button offered "Play" while the overlay's own stream was playing (togglePlay
delegates to the stream and returned before touching isPlaying, so the app
believed it was paused for the whole stream), and stored masks or stills were
painted at the app's playhead underneath tiles the stream was painting from its
own clock. Both are silent -- the picture just looks wrong -- which is why the
rule is a checked invariant rather than a comment. See
transport_invariants.test.js for the assertions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_overlay_transport_invariants_js():
    test_js = Path(__file__).parent / "transport_invariants.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)  # noqa: S603, S607
    assert result.returncode == 0, result.stdout + result.stderr
