# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared fixtures for training-tests, primarily the SSH loopback shim.

The ``ssh_loopback`` fixture mirrors the production auth model: it
appends a throwaway keypair to ``~/.ssh/authorized_keys`` (server side)
AND a ``Host`` block to ``~/.ssh/config`` (client side, pointing at the
key), then verifies ``ssh <alias>`` works **without ``-i``**. Tests
then construct :class:`SshTransport` with just the alias — no
``key_path`` field, matching the design.

Tests opt in via ``@pytest.mark.requires_ssh_loopback`` (registered in
pyproject.toml). The collection hook below currently quarantines every
such test because the tmux teardown path can terminate the host's
entire graphical user session. Keep the fixture for future isolated
VM/container coverage, but do not enable it on a developer workstation.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

# SAFETY: Keep the real SSH/tmux loopback suite disabled until the process-group
# teardown in SshClient.stop() has been redesigned and verified in an isolated
# VM/container.  On an Ubuntu GNOME workstation, running this suite has
# repeatedly created tmux-spawn scopes and then caused the user's systemd
# manager to activate exit.target, terminating the entire graphical session.
# Availability of sshd/tmux is therefore NOT sufficient permission to run it.
_SSH_LOOPBACK_DISABLED_REASON = (
    "disabled for workstation safety: real SSH/tmux teardown can terminate "
    "the entire systemd user session (GNOME exit.target)"
)


# SAFETY: Keep the real SSH/tmux loopback suite disabled until the process-group
# teardown in SshClient.stop() has been redesigned and verified in an isolated
# VM/container.  On an Ubuntu GNOME workstation, running this suite has
# repeatedly created tmux-spawn scopes and then caused the user's systemd
# manager to activate exit.target, terminating the entire graphical session.
# Availability of sshd/tmux is therefore NOT sufficient permission to run it.
_SSH_LOOPBACK_DISABLED_REASON = (
    "disabled for workstation safety: real SSH/tmux teardown can terminate "
    "the entire systemd user session (GNOME exit.target)"
)


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


@pytest.fixture(scope="session")
def ssh_loopback(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """Session-scoped SSH loopback environment.

    Mirrors the **production auth model**: the GUI server never sees
    the private key; ``ssh`` resolves the identity from the user's
    own setup. For tests that means appending a temporary Host block
    to ``~/.ssh/config`` pointing at our throwaway key, then yielding
    only the alias — no key_path field, matching the design.

    Yields ``{host: <alias>, port: 22, user: <user>}``. Both the
    pubkey line in ``~/.ssh/authorized_keys`` and the Host block in
    ``~/.ssh/config`` are tagged with the test pid so teardown can
    grep them out cleanly (best-effort; survives Ctrl-C but leaks
    on SIGKILL).

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

    # Server-side: pubkey in authorized_keys so the test key can log in.
    auth_keys = ssh_dir / "authorized_keys"
    marker = f"# lerobot-pytest-{os.getpid()}"
    auth_line = f"{pubkey} {marker}\n"
    with auth_keys.open("a") as f:
        f.write(auth_line)
    auth_keys.chmod(0o600)

    # Client-side: ssh-config Host block so a plain ``ssh <alias>``
    # resolves to the right key + user + port without anyone passing
    # ``-i``. This is exactly how a real user would set up a host
    # under the new design.
    alias = f"lerobot-pytest-loopback-{os.getpid()}"
    ssh_config = ssh_dir / "config"
    config_block = (
        f"\n# BEGIN {marker}\n"
        f"Host {alias}\n"
        f"    HostName 127.0.0.1\n"
        f"    User {user}\n"
        f"    Port 22\n"
        f"    IdentityFile {key_path}\n"
        f"    IdentitiesOnly yes\n"
        f"    StrictHostKeyChecking accept-new\n"
        f"# END {marker}\n"
    )
    with ssh_config.open("a") as f:
        f.write(config_block)
    ssh_config.chmod(0o600)

    try:
        # Verify ssh <alias> works without -i, proving the config block
        # resolves correctly. Same single round trip a real user does
        # before pasting the alias into the GUI.
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=3",
                    alias,
                    "true",
                ],
                capture_output=True,
                timeout=8.0,
            )
            if r.returncode != 0:
                pytest.skip(
                    f"ssh loopback handshake failed: rc={r.returncode} "
                    f"stderr={r.stderr.decode('utf-8', 'replace')[:200]}"
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            pytest.skip(f"ssh loopback handshake failed: {e}")
        yield {
            "host": alias,
            "port": 22,
            "user": user,
        }
    finally:
        # Strip our entries from both files. Tagged comments make this
        # robust to other lines appearing between setup and teardown.
        if auth_keys.exists():
            kept = [ln for ln in auth_keys.read_text().splitlines(keepends=True) if marker not in ln]
            auth_keys.write_text("".join(kept))
        if ssh_config.exists():
            text = ssh_config.read_text()
            start = f"# BEGIN {marker}"
            end = f"# END {marker}"
            if start in text and end in text:
                head, _, rest = text.partition(start)
                _, _, tail = rest.partition(end)
                # ``head`` ends with the \n before our BEGIN; ``tail``
                # begins with the \n after our END. Concatenating loses
                # exactly our block.
                ssh_config.write_text(head + tail.lstrip("\n"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Unconditionally quarantine real SSH/tmux loopback tests.

    Do not weaken this to an availability check: these tests have repeatedly
    terminated the host's graphical user session during tmux teardown.
    """
    skip = pytest.mark.skip(reason=_SSH_LOOPBACK_DISABLED_REASON)
    for item in items:
        if "requires_ssh_loopback" in item.keywords:
            item.add_marker(skip)
