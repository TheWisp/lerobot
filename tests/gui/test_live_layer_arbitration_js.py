"""Regression tests for the two defects that desynced the SAM3 preview from
playback. The logic is in JS, so the tests run under node; this pytest wrapper
invokes them (skipped when node is absent).

Both were verified to FAIL when their fix is reverted:

* Narrowing "the live layer owns the tiles" back to ``OverlayStream.streaming``
  misses the window where the worker is loading and the stream has not started.
  masks.js refreshes stills when it leaves composited mode, and loadAllFrames
  treats any call as a scrub and stops the stream -- so enabling SAM3 killed
  the stream it had just started.
* Having the stream write the transport button's label directly, instead of
  moving the app's own isPlaying, leaves the two disagreeing: togglePlay
  delegates to the stream and returns before touching that flag, so every
  re-render offered "Play" over moving video.

Neither failure raises anything at runtime -- a mask from a different frame
just looks like bad segmentation -- which is why they are pinned here.
See live_layer_arbitration.test.js for the assertions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_live_layer_arbitration_js():
    test_js = Path(__file__).parent / "live_layer_arbitration.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)  # noqa: S603, S607
    assert result.returncode == 0, result.stdout + result.stderr
