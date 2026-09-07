# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The press gesture's rule, unit-tested under node; this wrapper runs it.

What a press MEANS -- first movement past the slop is a drag, a release before
it is a click, another button abandons it, lost focus drops it, and exactly one
outcome ever runs -- was previously reachable only by driving a browser. Every
one of those clauses was a defect at some point. See ``press_gesture.test.js``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_press_gesture_rules_js():
    test_js = Path(__file__).parent / "press_gesture.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
