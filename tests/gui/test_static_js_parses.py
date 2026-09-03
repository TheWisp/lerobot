# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Every shipped static script must parse.

A script with a syntax error fails silently in the field: the browser logs one
console line nobody reads, the file's whole IIFE never runs, and every feature
it provides simply is not there. That exact failure shipped once — a patch
tool interpolated a real newline into a quoted string in overlay_stream.js,
and for two days the save-masks button did not exist and playback silently
fell back from the H.264 stream to the per-frame pull path, which read as
"SAM got slower". Nothing in pre-commit parses JavaScript, so this test does.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"
SCRIPTS = sorted(STATIC.glob("*.js"))
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed; CI and dev machines have it")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script: Path):
    proc = subprocess.run([NODE, "--check", str(script)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"{script.name} does not parse:\n{proc.stderr}"


def test_the_suite_actually_covers_the_page():
    """If the glob ever comes back empty the parse test would pass vacuously."""
    assert len(SCRIPTS) >= 10, [p.name for p in SCRIPTS]
