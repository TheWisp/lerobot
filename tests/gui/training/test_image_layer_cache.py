# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prove the cache boundary against Docker itself, not against our model of it.

``test_image_layers.py`` reads the Dockerfile and reasons about what Docker
*would* do. That is worth having — it runs everywhere in milliseconds — but it
encodes an assumption. These tests remove the assumption by building twice with
the real engine and comparing layer identity.

The build context is synthesized from the real Dockerfile: every ``COPY`` is
kept verbatim, so the ordering under test is the shipped one, while the base
image and the two ``uv sync`` invocations are swapped for cheap stand-ins so the
whole thing runs in seconds instead of pulling CUDA and Torch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.gui.training.test_image_layers import _instructions

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.training"

# Markers stand in for the two `uv sync` calls. Layer identity around these is
# the whole measurement, so they must be distinguishable in `docker history`.
DEP_MARKER = "cache-probe-dependency-sync"
PROJECT_MARKER = "cache-probe-project-sync"


def _docker_env() -> dict[str, str]:
    """Environment for every docker call this module makes.

    One function so the skip guard and the build cannot disagree. They did:
    the guard consulted the inherited environment while the build ran with a
    wiped one, so on a rootless, Colima or remote-``DOCKER_HOST`` daemon the
    module declined to skip and then failed hard. Wiping also discarded
    registry credentials, sending the base-image pull out anonymous into Docker
    Hub's rate limit.
    """
    return {**os.environ, "DOCKER_BUILDKIT": "0"}


def _docker_usable() -> bool:
    """Whether the probe can run, decided the same way the probe itself builds.

    The guard and the build must agree on environment, or the guard passes on a
    daemon the build cannot reach (rootless, Colima, a remote ``DOCKER_HOST``)
    and the skip turns into a hard failure.
    """
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=20, env=_docker_env())
    except (subprocess.TimeoutExpired, OSError):
        return False
    return probe.returncode == 0


# Applied per test rather than to the module: the parsing and environment
# tests below are pure and must run everywhere, including in CI without a daemon.
requires_docker = pytest.mark.skipif(
    not _docker_usable(), reason="needs a working docker daemon to exercise the layer cache"
)


def _synthesize_dockerfile() -> str:
    """Real COPY ordering, trivial everything else."""
    lines = ["FROM busybox:latest", "WORKDIR /app"]
    for op, args in _instructions(_DOCKERFILE.read_text(encoding="utf-8")):
        if op == "COPY":
            # --chown names a user busybox does not have; the flag is irrelevant
            # to cache keying, which is content + instruction.
            kept = " ".join(p for p in args.split() if not p.startswith("--chown"))
            lines.append(f"COPY {kept}")
        elif op == "RUN" and "uv sync" in args:
            marker = DEP_MARKER if "--no-install-project" in args else PROJECT_MARKER
            lines.append(f"RUN echo {marker} > /{marker}")
    return "\n".join(lines) + "\n"


def _make_context(root: Path) -> None:
    """Minimal stand-ins for every path the real Dockerfile copies."""
    (root / "pyproject.toml").write_text("[project]\nname = 'probe'\n")
    (root / "uv.lock").write_text("version = 1\n")
    (root / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    (root / "README.md").write_text("probe\n")
    (root / "MANIFEST.in").write_text("include README.md\n")
    pkg = root / "src" / "lerobot"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("VERSION = '0'\n")


def _build(context: Path, tag: str) -> None:
    """Build the probe with the caller's environment, plus the legacy builder.

    The environment is inherited, not replaced: wiping it drops ``DOCKER_HOST``
    (so a rootless or remote daemon becomes unreachable) and the registry
    credentials that keep the base-image pull off the anonymous rate limit.

    ``DOCKER_BUILDKIT=0`` is required, not preference. BuildKit reports
    ``<missing>`` for intermediate layers in ``docker history``, which would
    make the layer-identity comparison compare nothing at all — the assertions
    below defend against that too, but the legacy builder is what actually
    exposes per-step layer ids.
    """
    result = subprocess.run(
        ["docker", "build", "-f", str(context / "Dockerfile"), "-t", tag, str(context)],
        capture_output=True,
        text=True,
        env=_docker_env(),
        timeout=600,
    )
    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        # A rate-limited or offline base-image pull says nothing about this
        # repository; skipping beats a red build somebody has to triage.
        if any(s in combined for s in ("toomanyrequests", "rate limit", "no such host")):
            pytest.skip(f"cannot pull the probe base image: {combined.strip()[-200:]}")
        raise AssertionError(f"probe build failed:\n{combined}")


def _layer_ids_by_marker(tag: str) -> dict[str, str]:
    """Map each marker to the id of the layer its RUN produced."""
    out = subprocess.run(
        ["docker", "history", "--no-trunc", "--format", "{{.ID}}\t{{.CreatedBy}}", tag],
        capture_output=True,
        text=True,
        check=True,
        env=_docker_env(),
    ).stdout
    return _parse_history(out)


def _parse_history(out: str) -> dict[str, str]:
    """Marker -> layer id, rejecting ids that cannot be compared.

    Split from the subprocess call so the "<missing>" defence can be tested
    without a daemon — it is the difference between this probe measuring
    something and passing vacuously.
    """
    found: dict[str, str] = {}
    for line in out.splitlines():
        layer_id, _, created_by = line.partition("\t")
        for marker in (DEP_MARKER, PROJECT_MARKER):
            if f"echo {marker}" in created_by:
                # BuildKit reports "<missing>" for intermediate layers. Comparing
                # two of those is trivially equal, which would make the headline
                # assertion pass without measuring anything.
                assert layer_id and layer_id != "<missing>", (
                    f"{marker} has no usable layer id ({layer_id!r}); the builder is "
                    "not exposing per-step layers, so this probe proves nothing"
                )
                found[marker] = layer_id
    return found


@pytest.fixture
def probe_images(tmp_path):
    """Build the probe, edit a source file, rebuild. Yields both tag names."""
    context = tmp_path / "ctx"
    context.mkdir()
    _make_context(context)
    (context / "Dockerfile").write_text(_synthesize_dockerfile())

    suffix = uuid.uuid4().hex[:8]
    before, after = f"lerobot-cacheprobe-a-{suffix}", f"lerobot-cacheprobe-b-{suffix}"
    try:
        _build(context, before)
        # A content change, not an mtime touch: Docker keys COPY on content, so
        # touching alone would prove nothing.
        (context / "src" / "lerobot" / "__init__.py").write_text("VERSION = '0'  # edited\n")
        _build(context, after)
        yield before, after
    finally:
        subprocess.run(["docker", "rmi", "-f", before, after], capture_output=True)


@requires_docker
def test_a_source_edit_reuses_the_dependency_layer(probe_images):
    """The claim, measured: editing source must not rebuild the dependency layer.

    Layer identity is the evidence. If the dependency install re-ran, Docker
    would have produced a new layer id for it.
    """
    before, after = probe_images
    dep_before = _layer_ids_by_marker(before).get(DEP_MARKER)
    dep_after = _layer_ids_by_marker(after).get(DEP_MARKER)

    assert dep_before and dep_after, "probe did not produce a dependency layer"
    assert dep_before == dep_after, (
        "the dependency layer was rebuilt after a source-only edit — the layer "
        "cache is not holding, which is the ~24-minute regression of issue #98"
    )


@requires_docker
def test_a_source_edit_does_rebuild_the_project_layer(probe_images):
    """The complement, and the reason this pair is not vacuous.

    A Dockerfile that cached *everything* would pass the test above while
    shipping stale code. The project install must re-run so the edit lands.
    """
    before, after = probe_images
    project_before = _layer_ids_by_marker(before).get(PROJECT_MARKER)
    project_after = _layer_ids_by_marker(after).get(PROJECT_MARKER)

    assert project_before and project_after, "probe did not produce a project layer"
    assert project_before != project_after, (
        "the project install was cached across a source edit, so the image would "
        "ship the previous revision's code"
    )


# ── Regressions from the review of this PR ────────────────────────────────────
#
# These are pure: they run without a daemon, which is the point — the defects
# they cover only appear on machines this suite would otherwise skip on.


def test_docker_calls_preserve_the_ambient_environment(monkeypatch):
    """The guard and the build must see the same daemon.

    Regression: the skip guard consulted the inherited environment while the
    build ran with a wiped one, so on a rootless / Colima / remote-DOCKER_HOST
    setup the module declined to skip and then failed hard. The wipe also threw
    away registry credentials, sending the base-image pull out anonymous.
    """
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")
    monkeypatch.setenv("DOCKER_CONFIG", "/home/someone/.docker")

    env = _docker_env()

    assert env["DOCKER_HOST"] == "unix:///run/user/1000/docker.sock"
    assert env["DOCKER_CONFIG"] == "/home/someone/.docker"
    assert env["DOCKER_BUILDKIT"] == "0"


def test_history_parsing_rejects_uncomparable_layer_ids():
    """A "<missing>" id makes the headline comparison vacuous, not true.

    BuildKit reports "<missing>" for intermediate layers. Two of those compare
    equal, so the dependency-layer assertion would pass while measuring nothing.
    If the builder pin is ever dropped, this must fail loudly instead.
    """
    buildkit_style = (
        f"<missing>\techo {DEP_MARKER} > /{DEP_MARKER}\n"
        f"<missing>\techo {PROJECT_MARKER} > /{PROJECT_MARKER}\n"
    )
    with pytest.raises(AssertionError, match="no usable layer id"):
        _parse_history(buildkit_style)


def test_history_parsing_reads_real_layer_ids():
    """The complement: legitimate output still parses, so the guard above is
    rejecting the broken case rather than everything."""
    classic_style = (
        f"sha256:aaa\techo {DEP_MARKER} > /{DEP_MARKER}\n"
        f"sha256:bbb\techo {PROJECT_MARKER} > /{PROJECT_MARKER}\n"
        'sha256:ccc\t/bin/sh -c #(nop) CMD ["/bin/bash"]\n'
    )
    assert _parse_history(classic_style) == {
        DEP_MARKER: "sha256:aaa",
        PROJECT_MARKER: "sha256:bbb",
    }
