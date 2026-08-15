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

# Small, fully walked. These hold GUI state a user can see in the product:
# configured dataset sources, opened datasets, transfer history, bug reports.
FULLY_WALKED = (
    Path.home() / ".config" / "lerobot",
    Path.home() / ".cache" / "lerobot",
)

# The dataset cache is far too large to walk per test — it holds every frame of
# every recording. Watched at owner/name depth instead, which is enough to catch
# a test creating or deleting a dataset. Editing frames inside an existing
# dataset is not caught here; `no_real_datasets_in_tests` is the rule for that,
# and a test would have to point at a real dataset root to manage it.
SHALLOW_ROOTS = (Path.home() / ".cache" / "huggingface" / "lerobot",)
SHALLOW_DEPTH = 2

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


def _shallow(root: Path, depth: int) -> dict[str, tuple[int, int]]:
    """Entry names down to `depth`, without descending into dataset contents."""
    out: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return out

    def rec(d: Path, level: int) -> None:
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        for e in entries:
            out[e.path] = (0, 0)
            if e.is_dir(follow_symlinks=False) and level < depth:
                rec(Path(e.path), level + 1)

    rec(root, 1)
    return out


def _snapshot() -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for root in FULLY_WALKED:
        snap.update(_walk(root))
    for root in SHALLOW_ROOTS:
        snap.update(_shallow(root, SHALLOW_DEPTH))
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
