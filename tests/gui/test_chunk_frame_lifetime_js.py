"""Exercise chunk decoding with a bounded decoder output pool."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_chunk_frame_lifetime():
    test = Path(__file__).with_name("chunk_frame_lifetime.test.js")
    out = subprocess.run(["node", str(test)], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "chunk_frame_lifetime: ok" in out.stdout
