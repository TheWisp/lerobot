# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The timeline's lane primitives are JS, so their unit test runs under node;
this pytest wrapper invokes it (skipped when node is absent).

``timeline_lanes.js`` is the single place a lane's band is defined, read by the
drawing, the legend, the pending overlay and the hit test alike. It was
extracted from ``feature_editing.js``, where nine sites recomputed it and the
hit test agreed with the drawing only by inspection -- a disagreement there is
not a visual defect, it edits the lane below the bar that was clicked. See
``timeline_lanes.test.js`` for the assertions.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_timeline_lane_primitives_js():
    test_js = Path(__file__).parent / "timeline_lanes.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
