# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The training image's cache boundary, pinned as an executable invariant.

A code-only rebuild is cheap only because the dependency install sits in a
Docker layer whose inputs a source edit cannot touch. That is a property of
*instruction ordering*, not of any function, so nothing else in the suite can
catch its loss — and losing it is silent: the image still builds, still runs,
still passes every other test. It just costs ~24 minutes per edit again
(measured in issue #98).

These tests read the real Dockerfile rather than a fixture, because the claim
is about the file that actually gets built.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.training"

# The only two files that declare dependencies. Copying anything else above the
# dependency sync re-couples the expensive layer to a file that changes often —
# which is exactly the regression these tests exist to catch.
DEPENDENCY_MANIFESTS = {"pyproject.toml", "uv.lock"}

# Every instruction that can pull host files into a layer, and therefore every
# instruction that can key the dependency layer on source. Checking COPY alone
# would let the identical regression through written as ADD.
FILE_INGESTING = {"COPY", "ADD"}


def _instructions(text: str) -> list[tuple[str, str]]:
    """Parse a Dockerfile into ``(INSTRUCTION, arguments)`` pairs.

    Line continuations are joined and comments dropped, so the assertions below
    survive reformatting and only fail on a real change of meaning.
    """
    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1].strip() + " "
            continue
        joined.append((buffer + line).strip())
        buffer = ""
    if buffer:
        joined.append(buffer.strip())

    out: list[tuple[str, str]] = []
    for line in joined:
        head, _, rest = line.partition(" ")
        out.append((head.upper(), rest.strip()))
    return out


def _copy_sources(args: str) -> list[str]:
    """Source operands of a COPY/ADD, i.e. everything but flags and the target."""
    parts = [p for p in args.split() if not p.startswith("--")]
    return parts[:-1]  # last operand is the destination


@pytest.fixture(scope="module")
def instructions() -> list[tuple[str, str]]:
    assert _DOCKERFILE.is_file(), f"missing {_DOCKERFILE}"
    return _instructions(_DOCKERFILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dependency_sync_index(instructions) -> int:
    """Index of the sync that installs dependencies without the project."""
    hits = [
        i
        for i, (op, args) in enumerate(instructions)
        if op == "RUN" and "uv sync" in args and "--no-install-project" in args
    ]
    assert len(hits) == 1, (
        "expected exactly one `uv sync --no-install-project` — it is the seam "
        f"that separates dependencies from first-party code, found {len(hits)}"
    )
    return hits[0]


def test_nothing_but_dependency_manifests_is_copied_before_the_dependency_sync(
    instructions, dependency_sync_index
):
    """The cache boundary itself. This is the assertion that carries the perf.

    Ordering alone is not enough: a `COPY . .` placed above the sync keeps every
    ordering assertion true while destroying the property, because the layer
    would then be keyed on files that change every commit.
    """
    copied: list[str] = []
    for op, args in instructions[:dependency_sync_index]:
        if op in FILE_INGESTING:
            copied.extend(_copy_sources(args))

    unexpected = sorted(set(copied) - DEPENDENCY_MANIFESTS)
    assert not unexpected, (
        f"{unexpected} are copied before the dependency install, so editing any "
        "of them invalidates it. Only files that declare dependencies "
        f"({sorted(DEPENDENCY_MANIFESTS)}) belong above that layer; move these below it."
    )


def test_project_source_is_copied_after_the_dependency_sync(instructions, dependency_sync_index):
    """`src/` below the sync is what makes a code edit reuse the layer above."""
    src_copies = [
        i
        for i, (op, args) in enumerate(instructions)
        if op in FILE_INGESTING and any(s.rstrip("/") == "src" for s in _copy_sources(args))
    ]
    assert src_copies, "the image must copy src/ at some point"
    assert min(src_copies) > dependency_sync_index, (
        "src/ is copied before dependencies are installed, so every code edit "
        "invalidates the dependency layer — the ~24-minute rebuild of issue #98"
    )


def test_project_is_installed_after_its_source_arrives(instructions, dependency_sync_index):
    """The dependency sync deliberately skips the project, so a later sync must
    install it; otherwise the image would ship without lerobot at all."""
    later_syncs = [
        i
        for i, (op, args) in enumerate(instructions)
        if op == "RUN"
        and "uv sync" in args
        and "--no-install-project" not in args
        and i > dependency_sync_index
    ]
    assert later_syncs, (
        "no `uv sync` after the --no-install-project one: the project would never be installed into the venv"
    )


def test_dependency_sync_is_locked(instructions, dependency_sync_index):
    """`--locked` is what ties the cached layer to uv.lock's content.

    Without it the layer could be reused against a lockfile it no longer
    matches, which turns a cache hit into stale dependencies.
    """
    _, args = instructions[dependency_sync_index]
    assert "--locked" in args


def test_both_syncs_request_the_same_extra(instructions, dependency_sync_index):
    """A drift here installs one dependency set and resolves against another."""
    extras = [
        set(re.findall(r"--extra\s+(\S+)", args))
        for op, args in instructions
        if op == "RUN" and "uv sync" in args
    ]
    assert len(extras) >= 2
    assert all(e == extras[0] for e in extras[1:]), f"the uv sync invocations disagree on --extra: {extras}"


def test_dependency_sync_does_not_bind_mount_the_source(instructions, dependency_sync_index):
    """A bind mount is the third way to feed source into that layer.

    `RUN --mount=type=bind,source=src,...` makes the sync depend on code without
    a COPY anywhere, so neither ordering nor the manifest allowlist would notice.
    """
    _, args = instructions[dependency_sync_index]
    assert "--mount" not in args, (
        "the dependency sync mounts host paths; if any of them is source, the "
        f"layer is keyed on code again: {args}"
    )


def test_dockerfile_torchcodec_matches_the_lock() -> None:
    """The GPU-decode layer reinstalls torchcodec as the +cu128 variant of the
    LOCKED version. If a lock upgrade moves torchcodec and this line is
    forgotten, the image quietly ships a mismatched pair — this fails instead."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    docker = (root / "docker/Dockerfile.training").read_text()
    m = re.search(r"torchcodec==([0-9.]+)\+cu128", docker)
    assert m, "Dockerfile.training lost its CUDA torchcodec layer"
    lock = (root / "uv.lock").read_text()
    linux_pin = re.search(r'name = "torchcodec", version = "([0-9.]+)"[^\n]*sys_platform == \'linux\'', lock)
    assert linux_pin, "could not find the linux torchcodec pin in uv.lock"
    assert m.group(1) == linux_pin.group(1), (
        f"Dockerfile installs torchcodec {m.group(1)}+cu128 but uv.lock pins "
        f"{linux_pin.group(1)} on linux — update the Dockerfile layer with the lock"
    )
