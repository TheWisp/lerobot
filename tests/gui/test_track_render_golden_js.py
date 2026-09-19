# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Every timeline row type is pinned to the pixels it draws; this wrapper runs
that check under node (skipped when node is absent).

``renderTrackSvg`` draws six kinds of row through one function, so a change
aimed at one of them can move another with nothing to say so -- the other row
types were asserted only in shape ("three runs produce three rects"), never in
position. See ``track_render_golden.test.js``; regenerate with
``node tests/gui/regen_track_render_golden.js`` and read the diff.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_timeline_row_rendering_is_unchanged():
    test_js = Path(__file__).parent / "track_render_golden.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
