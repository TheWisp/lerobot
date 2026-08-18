# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fail any test that writes to the developer's real state.

This has now happened twice with the same shape. A fixture wrote a ``tmp_path``
dataset into the real ``opened_datasets.json``, so the GUI opened with a
"Failed to open dataset" toast pointing at a long-deleted pytest directory. And
the hub suite left 105 fabricated transfer records in
``~/.config/lerobot/gui/hub_transfers.jsonl`` — repos like ``user/repo`` and
``u/ds`` — which the tray then presented to the developer as their own upload
history.

Both were written down as a rule beforehand, in
``.agents/skills/verifying-changes/SKILL.md``, and both shipped anyway. Prose
does not fire. This does.

**Why a runtime guard rather than a lint rule.** A linter has to guess which
paths are real by pattern-matching source. Both instances arrived by routes
nobody would have thought to grep for — one through a fixture, one through a
worker subprocess that inherits the environment and cannot see a monkeypatched
module. The guard observes what actually changed on disk, so it catches writers
that did not exist when the rule was written.

**Opting out.** A test that legitimately exercises real-path resolution marks
itself ``@pytest.mark.touches_user_state``. That is deliberately a marker rather
than a config entry: it shows up in the diff, in the test, next to the reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Small, fully walked. These hold *decisions the user made* and would notice
# losing: which folders are configured as dataset sources, which datasets to
# reopen on next launch, how past transfers ended, filed bug reports.
FULLY_WALKED = (
    Path.home() / ".config" / "lerobot",
    Path.home() / ".cache" / "lerobot",
)

# NOT watched: ~/.cache/huggingface/lerobot. This is a partial answer, not a
# principled one, and the distinction matters if you are extending this.
#
# What is principled: a *new* entry appearing there is correct behaviour. Tests
# legitimately download public fixture datasets, and losing a cache costs
# bandwidth rather than work. The first version of this guard watched the cache
# and CI failed a broad set of unrelated tests — the policy factory suite,
# backward-compat checks, anything pulling a fixture. On a developer's machine
# those files already exist so nothing appears to change; on a cold CI cache
# every one is new. Firing on that would train people to ignore the guard.
#
# What is NOT principled, and is simply missing: a cached dataset being
# *removed*, or a bogus one being fabricated. Both destroy or invent something
# the user sees — an early CI run caught a test creating a `test_user` dataset
# directory here, which the GUI would list as one of their own recordings, and
# some of a user's datasets are not on the Hub to re-download. Detecting removal
# is cheap (a depth-2 scan is ~2.4ms) and was skipped because separating removal
# from creation was more care than the first pass took, not because it is wrong.
#
# What is genuinely too expensive: content changes *inside* a dataset. That
# needs a deep walk over tens of thousands of frame files per test. The rule for
# that is no_real_datasets_in_tests — don't point a test at a real dataset.
#
# Tracked as a follow-up rather than left as a comment nobody reads.

MARKER = "touches_user_state"


def _walk(root: Path) -> dict[str, tuple[int, int]]:
    """Map path → (size, mtime_ns) for everything under `root`.

    Directories are recorded as well as files. Walking only files misses a test
    that creates an empty directory — which is still the developer's config
    tree being written to, and is how a half-finished write leaves its mark.

    `mtime_ns` rather than content: it also catches a write that is later
    restored, which a content hash would call unchanged even though the file
    was briefly wrong for anything reading it concurrently.
    """
    out: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames) + list(filenames):
            p = os.path.join(dirpath, name)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out[p] = (st.st_size, st.st_mtime_ns)
    return out


def _snapshot() -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for root in FULLY_WALKED:
        snap.update(_walk(root))
    return snap


def _describe(before: dict, after: dict) -> list[str]:
    created = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    lines = []
    for label, paths in (("created", created), ("modified", changed), ("removed", removed)):
        for p in paths[:5]:
            lines.append(f"  {label}: {p}")
        if len(paths) > 5:
            lines.append(f"  {label}: … and {len(paths) - 5} more")
    return lines


@pytest.fixture(autouse=True)
def guard_real_user_state(request):
    """Snapshot the user's real state, and fail the test that changed it."""
    if request.node.get_closest_marker(MARKER):
        yield
        return
    before = _snapshot()
    yield
    after = _snapshot()
    if before == after:
        return
    detail = "\n".join(_describe(before, after))
    pytest.fail(
        f"This test wrote to the developer's real state:\n{detail}\n\n"
        "Tests write to tmp_path. Redirect the path the code under test uses — "
        "and redirect BOTH channels when there are two: the module constant for "
        "in-process writers, and the env var for subprocesses, which inherit the "
        "environment and cannot see a monkeypatched module.\n"
        f"If touching the real path IS what this test exercises, mark it "
        f"@pytest.mark.{MARKER} with a reason.",
        pytrace=False,
    )
