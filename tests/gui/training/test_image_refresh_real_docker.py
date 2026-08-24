# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prove the moving-tag refresh against Docker and a registry, not against a fake.

``test_orchestrator.py`` scripts a fake transport client: it fixes what
``image_inspect`` and ``image_pull`` return and checks the orchestrator's
decisions. That is worth having and runs everywhere in milliseconds, but it
proves the branch, not the behaviour — the assumption it encodes is that
``docker pull`` on an already-present tag actually replaces the local copy when
the tag has moved. That assumption is the entire fix, so it is tested here
against the real thing.

Nothing is stubbed: a real registry, a real push that moves ``:latest`` to
different content, a real ``docker pull`` through ``SubprocessClient``. The
images are ``FROM scratch`` plus one small file, so the whole exchange is a few
hundred bytes and the registry is the only download.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

from lerobot.gui.training.orchestrator import Orchestrator
from lerobot.gui.training.runs import RunPaths
from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

REGISTRY_IMAGE = "registry:2"


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return probe.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_usable(), reason="needs a working docker daemon and a local registry"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(*argv: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if check:
        assert r.returncode == 0, f"{' '.join(argv)}\n{r.stdout}\n{r.stderr}"
    return r


@pytest.fixture
def registry():
    """A throwaway registry on localhost, which docker treats as insecure-allowed."""
    port = _free_port()
    name = f"lerobot-test-registry-{uuid.uuid4().hex[:8]}"
    _run("docker", "run", "-d", "--rm", "-p", f"127.0.0.1:{port}:5000", "--name", name, REGISTRY_IMAGE)
    try:
        yield f"127.0.0.1:{port}"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)


def _publish(tag: str, marker: str, tmp_path: Path) -> str:
    """Build a minimal image whose content depends on ``marker``, push it, return its id."""
    ctx = tmp_path / f"ctx-{marker}"
    ctx.mkdir(exist_ok=True)
    (ctx / "content").write_text(marker)
    (ctx / "Dockerfile").write_text("FROM scratch\nCOPY content /content\n")
    _run("docker", "build", "-q", "-t", tag, str(ctx))
    _run("docker", "push", tag)
    return _run("docker", "image", "inspect", "-f", "{{.Id}}", tag).stdout.strip()


def _local_id(tag: str) -> str | None:
    r = _run("docker", "image", "inspect", "-f", "{{.Id}}", tag, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


@requires_docker
def test_a_moved_tag_is_actually_refreshed_on_disk(registry, tmp_path: Path) -> None:
    """The fix, end to end: the bytes that run are the ones the tag now points at.

    Before the change the orchestrator saw the tag present locally, reported a
    cache hit, and trained on whatever was there. Under the old pinned tag that
    copy was merely old; under a tag that moves it would be stale, with the
    cache hit reported as success.
    """
    tag = f"{registry}/moving:latest"
    keep = f"{registry}/moving:keep-v1"

    first = _publish(tag, "v1", tmp_path)
    _run("docker", "tag", tag, keep)  # hold a reference so v1 survives the retag

    second = _publish(tag, "v2", tmp_path)
    assert second != first, "the tag must genuinely have moved for this to prove anything"

    # The state a host is in after pulling weeks ago: the tag resolves locally
    # to bytes the registry no longer serves under it.
    _run("docker", "tag", keep, tag)
    assert _local_id(tag) == first, "precondition: the host holds the stale image"

    paths = RunPaths.for_run("refresh", runs_dir=tmp_path / "runs")
    paths.ensure_exists()
    client = SubprocessClient(SubprocessTransport(workdir=paths.root))

    Orchestrator._ensure_image(Orchestrator.__new__(Orchestrator), client, tag, paths)

    assert _local_id(tag) == second, "the moving tag was not refreshed: the host would train on stale bytes"
    types = [line for line in paths.events_jsonl.read_text().splitlines() if line.strip()]
    assert any("image_pull_started" in t for t in types), "a moving tag must be re-pulled"
    assert not any("image_cache_hit" in t for t in types), "a moving tag must never report a cache hit"


@requires_docker
def test_a_digest_reference_is_trusted_from_the_local_cache(registry, tmp_path: Path) -> None:
    """The other half: an immutable reference keeps the shortcut it deserves.

    Re-pulling everything would be correct but wasteful, and would undo the
    optimisation the cache-hit path exists for. A digest names its own content,
    so a local copy cannot be the wrong bytes.
    """
    tag = f"{registry}/pinned:latest"
    _publish(tag, "pinned", tmp_path)
    digest = _run("docker", "image", "inspect", "-f", "{{index .RepoDigests 0}}", tag).stdout.strip()
    assert "@sha256:" in digest, digest
    _run("docker", "pull", digest)

    paths = RunPaths.for_run("pinned", runs_dir=tmp_path / "runs")
    paths.ensure_exists()
    client = SubprocessClient(SubprocessTransport(workdir=paths.root))

    Orchestrator._ensure_image(Orchestrator.__new__(Orchestrator), client, digest, paths)

    body = paths.events_jsonl.read_text()
    assert "image_cache_hit" in body, "a digest already on the host needs no pull"
    assert "image_pull_started" not in body
