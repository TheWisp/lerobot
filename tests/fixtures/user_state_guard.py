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

# Deliberately NOT watched: ~/.cache/huggingface/lerobot.
#
# The first version of this guard did watch it, and CI immediately failed a
# broad set of unrelated tests — the policy factory suite, backward-compat
# checks, anything that pulls a public fixture dataset. On a developer's machine
# those files already exist so nothing appears to change; on a cold CI cache
# every one of them is a new entry.
#
# That is not the defect this guards. A cache is populated by design, shared
# between runs, and re-downloadable — losing it costs bandwidth, not work. The
# two incidents behind this fixture were both in ~/.config, where the loss is a
# decision the user made and cannot get back. Guarding the cache would fire on
# correct behaviour, and a guard that cries wolf gets ignored by the same people
# it is meant to protect.
#
# A test writing *into* an existing real dataset is a different rule — see
# no_real_datasets_in_tests — and is not detectable here anyway.

MARKER = "touches_user_state"


def _walk(root: Path) -> dict[str, tuple[int, int]]:
    """Map path → (size, mtime_ns) for everything under `root`."""
    out: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return out
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
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
