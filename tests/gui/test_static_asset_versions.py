# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Changing a static asset must bump its ``?v=`` in index.html.

The GUI cache-busts by hand: ``app.js?v=43``, ``style.css?v=75``. Edit the file
without touching the number and the URL is unchanged, so a browser serves its
cached copy and the change is invisible in the page while being plainly present
on the server. That failure is silent and looks exactly like the feature not
working — a plain reload does not fix it, because there is nothing new to fetch.

This is a ratchet, not a checksum of correctness: it records the digest each
version was released with. Change the asset and the test fails until you bump
the version and re-record, which is the moment to remember every open tab is
holding the old file.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "gui" / "static"
INDEX = STATIC / "index.html"
FINGERPRINTS = Path(__file__).parent / "static_asset_versions.json"

# Every asset index.html versions. Read from the fingerprint file so adding one
# there is all it takes; the completeness test below keeps the two in step.
VERSIONED = tuple(sorted(json.loads(FINGERPRINTS.read_text())))


def _declared_version(asset: str) -> str:
    m = re.search(rf"/static/{re.escape(asset)}\?v=(\d+)", INDEX.read_text())
    assert m, f"{asset} is not referenced with a ?v= query in index.html"
    return m.group(1)


def _digest(asset: str) -> str:
    return hashlib.sha256((STATIC / asset).read_bytes()).hexdigest()[:16]


@pytest.mark.parametrize("asset", VERSIONED)
def test_asset_change_bumps_its_version(asset: str):
    recorded = json.loads(FINGERPRINTS.read_text())
    version, digest = _declared_version(asset), _digest(asset)
    was = recorded.get(asset)
    assert was is not None, f"{asset} has no recorded fingerprint; add one to {FINGERPRINTS.name}"

    if digest == was["sha256_16"]:
        assert version == was["v"], (
            f"{asset} is unchanged but its version moved {was['v']} -> {version}; "
            "re-record the fingerprint if that was deliberate"
        )
        return

    assert version != was["v"], (
        f"{asset} changed but index.html still says ?v={version}. Browsers key their cache on "
        f"that URL, so every open tab keeps the old file and the change appears not to work. "
        f"Bump the version and update {FINGERPRINTS.name} to sha256_16={digest}."
    )
    pytest.fail(
        f"{asset} changed and the version was bumped to {version} — update "
        f"{FINGERPRINTS.name} to sha256_16={digest} to record the release."
    )


def test_every_versioned_asset_is_recorded():
    """A new versioned asset must be added here, or it is silently unguarded."""
    referenced = set(re.findall(r"/static/([\w.]+)\?v=\d+", INDEX.read_text()))
    recorded = set(json.loads(FINGERPRINTS.read_text()))
    assert referenced == recorded, (
        f"index.html versions {sorted(referenced)} but fingerprints cover {sorted(recorded)}"
    )


def test_no_asset_is_loaded_twice():
    """Two <script> tags for one asset run it twice, and hide a version conflict.

    The completeness test above compares sets, so a duplicate reference is
    invisible to it. A rebase between two branches that both touched the script
    block is how one arrives: each side's tag survives, at each side's version.
    The browser then fetches the same file under two URLs and executes it twice,
    which double-registers whatever it binds at load.
    """
    referenced = re.findall(r"/static/([\w.]+)\?v=\d+", INDEX.read_text())

    duplicated = sorted({name for name in referenced if referenced.count(name) > 1})

    assert not duplicated, (
        f"index.html loads {duplicated} more than once. Keep the highest version and delete the other tag."
    )
