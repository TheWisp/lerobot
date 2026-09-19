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

import gzip
import hashlib
import io
import json
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import urllib.request
import uuid
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Registry:
    addr: str  # host:port, the prefix of every reference pushed to it
    name: str  # its container, whose log is the record of what it served


@pytest.fixture
def registry():
    """A throwaway registry on localhost, which docker treats as insecure-allowed."""
    port = _free_port()
    name = f"lerobot-test-registry-{uuid.uuid4().hex[:8]}"
    _run("docker", "run", "-d", "--rm", "-p", f"127.0.0.1:{port}:5000", "--name", name, REGISTRY_IMAGE)
    try:
        yield _Registry(addr=f"127.0.0.1:{port}", name=name)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)


# The registry's access log, one line per request: what docker actually asked
# it for. Layer and config downloads are ``GET .../blobs/<digest>``; a manifest
# check is ``HEAD`` or ``GET .../manifests/<ref>``.
_BLOB_GET = re.compile(r'"GET /v2/([^ ]+?)/blobs/(sha256:[0-9a-f]+) HTTP/[^"]*" 200')
_MANIFEST_CHECK = re.compile(r'"(?:GET|HEAD) /v2/([^ ]+?)/manifests/[^ ]+ HTTP/[^"]*" 200')


def _served(registry: _Registry) -> tuple[list[str], int]:
    """``(blob digests served, manifest checks answered)`` so far, for every repo."""
    log = subprocess.run(["docker", "logs", registry.name], capture_output=True, text=True, timeout=30)
    text = log.stdout + log.stderr
    return [digest for _repo, digest in _BLOB_GET.findall(text)], len(_MANIFEST_CHECK.findall(text))


def _push_build_without_docker(registry: _Registry, repo: str, marker: str) -> tuple[str, str, str]:
    """Publish ``<repo>:latest`` straight into the registry, bypassing docker.

    A build made with ``docker build`` leaves its layers in the local store,
    and a later pull would find them there and download nothing — which is
    the same observation as "nothing changed". A build the host has never
    seen is what a moved tag looks like from a host that pulled weeks ago.
    Returns the manifest, config and layer digests.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = marker.encode()
        info = tarfile.TarInfo("content")
        info.size, info.mtime = len(data), 0
        tar.addfile(info, io.BytesIO(data))
    tar_bytes = buf.getvalue()
    layer = gzip.compress(tar_bytes, mtime=0)
    layer_digest = "sha256:" + hashlib.sha256(layer).hexdigest()
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    config = json.dumps(
        {
            "architecture": arch,
            "os": "linux",
            "config": {},
            "rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(tar_bytes).hexdigest()]},
        }
    ).encode()
    config_digest = "sha256:" + hashlib.sha256(config).hexdigest()

    def put_blob(blob: bytes, digest: str) -> None:
        start = urllib.request.Request(f"http://{registry.addr}/v2/{repo}/blobs/uploads/", method="POST")
        with urllib.request.urlopen(start, timeout=30) as r:  # noqa: S310 — loopback registry
            assert r.status == 202, r.status
            location = r.headers["Location"]
        if location.startswith("/"):
            location = f"http://{registry.addr}{location}"
        sep = "&" if "?" in location else "?"
        upload = urllib.request.Request(
            f"{location}{sep}digest={digest}",
            data=blob,
            method="PUT",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(upload, timeout=30) as r:  # noqa: S310
            assert r.status == 201, r.status

    put_blob(layer, layer_digest)
    put_blob(config, config_digest)
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": config_digest,
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": layer_digest,
                    "size": len(layer),
                }
            ],
        }
    ).encode()
    put = urllib.request.Request(
        f"http://{registry.addr}/v2/{repo}/manifests/latest",
        data=manifest,
        method="PUT",
        headers={"Content-Type": "application/vnd.oci.image.manifest.v1+json"},
    )
    with urllib.request.urlopen(put, timeout=30) as r:  # noqa: S310
        assert r.status == 201, r.status
        return r.headers["Docker-Content-Digest"], config_digest, layer_digest


def _events(paths: RunPaths) -> list[str]:
    return [json.loads(line)["type"] for line in paths.events_jsonl.read_text().splitlines() if line.strip()]


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
def test_the_same_build_downloads_nothing_and_a_new_build_downloads_its_layers(
    registry: _Registry, tmp_path: Path
) -> None:
    """Both halves of "always re-pull a tag", measured where the bytes come from.

    The orchestrator asks docker for the tag before every launch, and docker
    fetches only what differs from the registry's manifest. The registry's
    access log is the proof. A pull of the build the host already holds
    answers manifest checks and serves no layer blobs, the tag still resolves
    to the same image, and the event says it was current. A pull after the
    tag moved serves exactly the new build's config and layer, the tag then
    resolves to that build's manifest, and the event says it was pulled.

    The new build is pushed straight into the registry, so the local store has
    never seen its layers and cannot skip them for the wrong reason. Both
    builds carry content unique to this run for the same reason: a layer with
    the same bytes as one a previous run pulled is already in the store.
    """
    repo, tag = "moving", f"{registry.addr}/moving:latest"
    run_id = uuid.uuid4().hex
    first = _publish(
        tag, f"v1-{run_id}", tmp_path
    )  # built and pushed: the host holds what the registry serves
    client = SubprocessClient(SubprocessTransport(workdir=tmp_path))
    orch = Orchestrator.__new__(Orchestrator)

    # ── same build ────────────────────────────────────────────────────────
    paths = RunPaths.for_run("same", runs_dir=tmp_path / "runs")
    paths.ensure_exists()
    blobs_before, checks_before = _served(registry)

    orch._ensure_image(client, tag, paths)

    blobs, checks = _served(registry)
    assert _local_id(tag) == first, "the tag must still resolve to the build the host had"
    assert blobs[len(blobs_before) :] == [], (
        f"an up-to-date tag re-downloaded layers: {blobs[len(blobs_before) :]}"
    )
    assert checks > checks_before, "the registry was never asked: the orchestrator must not skip a tag itself"
    assert _events(paths) == ["image_pull_started", "image_up_to_date"]

    # ── a different build under the same tag ──────────────────────────────
    manifest, config, layer = _push_build_without_docker(registry, repo, f"v2-{run_id}")
    paths = RunPaths.for_run("moved", runs_dir=tmp_path / "runs")
    paths.ensure_exists()
    blobs_before, _ = _served(registry)

    orch._ensure_image(client, tag, paths)

    blobs, _ = _served(registry)
    assert _local_id(tag) != first, "the moving tag was not refreshed: the host would train on stale bytes"
    repo_digest = _run("docker", "image", "inspect", "-f", "{{index .RepoDigests 0}}", tag).stdout.strip()
    assert repo_digest == f"{registry.addr}/{repo}@{manifest}", (
        "the tag must resolve to the registry's new manifest"
    )
    assert set(blobs[len(blobs_before) :]) == {config, layer}, (
        f"expected the new build's config and layer to be served, got {blobs[len(blobs_before) :]}"
    )
    assert _events(paths) == ["image_pull_started", "image_pulled"]


@requires_docker
def test_a_digest_reference_is_trusted_from_the_local_cache(registry, tmp_path: Path) -> None:
    """The other half: an immutable reference keeps the shortcut it deserves.

    Re-pulling everything would be correct but wasteful, and would undo the
    optimisation the cache-hit path exists for. A digest names its own content,
    so a local copy cannot be the wrong bytes.
    """
    tag = f"{registry.addr}/pinned:latest"
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
