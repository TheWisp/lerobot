# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Changing a static asset must bump the ``?v=`` its callers ask for it with.

The GUI cache-busts by hand: ``app.js?v=43``, ``style.css?v=75``. Edit the file
without touching the number and the URL is unchanged, so a browser serves its
cached copy and the change is invisible in the page while being plainly present
on the server. That failure is silent and looks exactly like the feature not
working — a plain reload does not fix it, because there is nothing new to fetch.

index.html declares most of these, but not all: ``urdf_viz.html`` is an iframe
the scripts build, and went unguarded here long enough to be shipped twice
without a bump. A version guards a file only if the ratchet looks where the
file is actually asked for, so this reads every caller rather than the index.

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

# Every versioned asset. Read from the fingerprint file so adding one there is
# all it takes; the completeness test below keeps the two in step.
VERSIONED = tuple(sorted(json.loads(FINGERPRINTS.read_text())))

#: A `?v=` in any query a caller builds. index.html declares most of them as
#: the whole query; a tile embedded by a script carries the version after the
#: parameters that say what to show, so the number is not the first thing there.
#: The class excludes the quotes so a match cannot run past its own string.
_VERSION_OF = r"/static/{}\?[^`\"']*?\bv=(\d+)"
_ANY_VERSIONED = r"/static/([\w.]+)\?[^`\"']*?\bv=\d+"


def _callers():
    """Everything that asks for a static asset by URL. urdf_viz.html is not in
    index.html at all -- it is an iframe the scripts build -- and a version only
    guards what actually references it."""
    return [INDEX, *sorted(STATIC.glob("*.js"))]


def _declared_version(asset: str) -> tuple[str, str]:
    """The version, and the caller that declares it -- which a failure has to
    name, because it is the file to go and edit."""
    for src in _callers():
        m = re.search(_VERSION_OF.format(re.escape(asset)), src.read_text())
        if m:
            return m.group(1), src.name
    raise AssertionError(f"{asset} is asked for with no ?v= by index.html or any script")


def _digest(asset: str) -> str:
    return hashlib.sha256((STATIC / asset).read_bytes()).hexdigest()[:16]


@pytest.mark.parametrize("asset", VERSIONED)
def test_asset_change_bumps_its_version(asset: str):
    recorded = json.loads(FINGERPRINTS.read_text())
    (version, declared_in), digest = _declared_version(asset), _digest(asset)
    was = recorded.get(asset)
    assert was is not None, f"{asset} has no recorded fingerprint; add one to {FINGERPRINTS.name}"

    if digest == was["sha256_16"]:
        assert version == was["v"], (
            f"{asset} is unchanged but its version in {declared_in} moved {was['v']} -> {version}; "
            "re-record the fingerprint if that was deliberate"
        )
        return

    assert version != was["v"], (
        f"{asset} changed but {declared_in} still asks for ?v={version}. Browsers key their cache on "
        f"that URL, so every open tab keeps the old file and the change appears not to work. "
        f"Bump the version and update {FINGERPRINTS.name} to sha256_16={digest}."
    )
    pytest.fail(
        f"{asset} changed and the version was bumped to {version} — update "
        f"{FINGERPRINTS.name} to sha256_16={digest} to record the release."
    )


def test_every_versioned_asset_is_recorded():
    """A new versioned asset must be added here, or it is silently unguarded."""
    referenced = set()
    for src in _callers():
        referenced |= set(re.findall(_ANY_VERSIONED, src.read_text()))
    recorded = set(json.loads(FINGERPRINTS.read_text()))
    assert referenced == recorded, (
        f"callers version {sorted(referenced)} but fingerprints cover {sorted(recorded)}"
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


def test_one_file_is_asked_for_under_one_version():
    """``urdf_viz.html`` is cache-busted by its callers rather than by
    index.html. The ratchet above reads the version from whichever caller it
    finds first, so it cannot see the two disagreeing: the browser would hold
    two copies of one file, and an edit to what they share would reach only
    whichever caller happened to be bumped."""
    versions = {}
    for js in sorted(STATIC.glob("*.js")):
        for match in re.finditer(r"urdf_viz\.html\?([^`\"']*)", js.read_text()):
            found = re.search(r"\bv=(\d+)", match.group(1))
            assert found, f"{js.name} loads urdf_viz.html with no version"
            versions.setdefault(found.group(1), []).append(js.name)
    assert len(versions) == 1, f"urdf_viz.html is asked for under {versions}"
