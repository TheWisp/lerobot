# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared fixtures for training-tests, primarily the SSH loopback shim.

The ``ssh_loopback`` fixture establishes a throwaway keypair, appends its
pubkey to the user's ``~/.ssh/authorized_keys`` (tagged so the teardown
can grep it out cleanly), and confirms ``ssh user@127.0.0.1 true`` works.
On any precondition failure it `pytest.skip()`s the test cleanly — CI
runners without sshd, hosts without tmux/ssh-keygen, etc. all degrade
gracefully.

Tests opt in via ``@pytest.mark.requires_ssh_loopback`` (registered in
pyproject.toml). The collection hook below pre-filters by checking the
fastest preconditions (binary availability) so dependent tests skip
quickly without spinning up the fixture only to skip there.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest


def _loopback_prereqs_present() -> tuple[bool, str]:
    """Quick precheck: are the binaries + sshd reachable?

    Returns (ok, reason). The reason is only used when ok is False.
    Avoids spinning up a keypair when we already know we'll skip.
    """
    for bin_name in ("ssh", "ssh-keygen", "scp", "tmux"):
        if not shutil.which(bin_name) and not (Path.home() / ".local" / "bin" / bin_name).exists():
            return False, f"`{bin_name}` not on PATH (or ~/.local/bin)"
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user:
        return False, "no $USER / $LOGNAME"
    try:
        with socket.create_connection(("127.0.0.1", 22), timeout=1.0):
            pass
    except OSError as e:
        return False, f"sshd not reachable on 127.0.0.1:22 ({e})"
    return True, ""


def _can_loopback_with_key(key_path: Path, user: str) -> tuple[bool, str]:
    """Final go/no-go check — actually attempt a no-op SSH command. The
    fixture has just added our pubkey, so this confirms the round trip."""
    try:
        r = subprocess.run(
            [
                "ssh",
                "-i",
                str(key_path),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=3",
                f"{user}@127.0.0.1",
                "true",
            ],
            capture_output=True,
            timeout=8.0,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"ssh exec failed: {e}"
    if r.returncode != 0:
        return False, f"ssh returned {r.returncode}: {r.stderr.decode('utf-8', 'replace')[:200]}"
    return True, ""


@pytest.fixture(scope="session")
def ssh_loopback(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """Session-scoped SSH loopback environment.

    Yields a dict ``{host, port, user, key_path}`` suitable for passing to
    :class:`SshTransport`. The pubkey is removed from ``~/.ssh/authorized_keys``
    on session end (best-effort; survives a Ctrl-C but leaks on SIGKILL).

    Skips the test with a precise reason on any setup failure.
    """
    ok, why = _loopback_prereqs_present()
    if not ok:
        pytest.skip(f"ssh loopback unavailable: {why}")

    user = os.environ["USER"] if "USER" in os.environ else os.environ["LOGNAME"]
    key_dir = tmp_path_factory.mktemp("ssh-loopback-key")
    key_path = key_dir / "id_ed25519"
    subprocess.check_call(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-q",
            "-N",
            "",
            "-f",
            str(key_path),
            "-C",
            f"lerobot-pytest-{os.getpid()}",
        ]
    )
    pubkey = (key_path.with_suffix(".pub")).read_text().strip()

    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    auth_keys = ssh_dir / "authorized_keys"
    # The marker tag lets us grep our entry out of the file on teardown
    # even if other tests / processes have appended their own entries in
    # the meantime. Includes pid for uniqueness across overlapping runs.
    marker = f"# lerobot-pytest-{os.getpid()}"
    line = f"{pubkey} {marker}\n"
    with auth_keys.open("a") as f:
        f.write(line)
    auth_keys.chmod(0o600)

    try:
        ok, why = _can_loopback_with_key(key_path, user)
        if not ok:
            pytest.skip(f"ssh loopback handshake failed: {why}")
        yield {
            "host": "127.0.0.1",
            "port": 22,
            "user": user,
            "key_path": str(key_path),
        }
    finally:
        # Best-effort cleanup: remove only OUR line, leave everything else.
        if auth_keys.exists():
            kept = [ln for ln in auth_keys.read_text().splitlines(keepends=True) if marker not in ln]
            auth_keys.write_text("".join(kept))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``requires_ssh_loopback`` items when the cheap prereqs aren't
    met. The fixture itself will also skip on a per-test basis if the
    handshake fails — this just avoids spinning up the fixture for tests
    we already know will skip."""
    ok, why = _loopback_prereqs_present()
    if ok:
        return
    skip = pytest.mark.skip(reason=f"ssh loopback unavailable: {why}")
    for item in items:
        if "requires_ssh_loopback" in item.keywords:
            item.add_marker(skip)
