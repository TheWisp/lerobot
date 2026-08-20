"""``masks.js`` decodes what ``mask_codec.py`` encodes — asserted across the two
languages, not within either.

A decoder that disagrees with the encoder does not fail loudly: it draws a
region slightly different from the stored one, which an operator reads as a poor
segmentation rather than a broken client. So the fixtures in
``mask_codec_fixture.json`` are produced BY the Python encoder and decoded under
node, covering the shapes that break run-length coders — uniform masks with no
value changes at all, a checkerboard where every pixel is its own run, and
row-vs-column cases that only differ if the column-major convention is wrong.

Regenerate the fixture with ``tests/gui/regen_mask_fixture.py`` if the encoding
ever changes; a stale fixture would let the two drift apart silently.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_mask_decoder_matches_the_python_encoder():
    test_js = Path(__file__).parent / "masks.test.js"
    result = subprocess.run(["node", str(test_js)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fixture_is_present_and_covers_the_hard_cases():
    """The JS test is skipped without node, so the fixture's coverage is pinned here."""
    import json

    fixture = json.loads((Path(__file__).parent / "mask_codec_fixture.json").read_text())
    names = {case["name"] for case in fixture}
    # Uniform masks emit no value changes and still must cover every pixel;
    # the checkerboard is the worst case for the delta coding.
    assert {"empty", "full", "checker"} <= names, names
    # first_row vs first_column differ only under the column-major convention.
    assert {"first_row", "first_column"} <= names, names
    for case in fixture:
        assert len(case["flat_rowmajor"]) == case["h"] * case["w"], case["name"]
