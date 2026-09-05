# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""How a remote operation becomes root, and what it refuses to do.

Every privileged remote call goes through ``sudo_exec`` so the decision lives in
one place. These pin that decision, because the failures it guards against are
quiet ones: a password placed where other processes can read it, or a host
refused for want of a privilege it never needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lerobot.gui.training.ssh_transport import SshClient
from lerobot.gui.training.transport import SshConnectionError, SshTransport, SudoUnavailableError


class _Recorded(SshClient):
    """An SshClient whose remote calls are recorded rather than made."""

    def __init__(self, *, passwordless: bool, password_accepted: bool = True, reachable: bool = True) -> None:
        super().__init__(SshTransport(host="rig.invalid", port=22, user="operator"))
        self._passwordless = passwordless
        self._password_accepted = password_accepted
        self._reachable = reachable
        self.calls: list[tuple[str, bytes | None]] = []

    def _exec(self, remote_cmd, *, timeout=30.0, stdin=None):
        self.calls.append((remote_cmd, stdin))
        if not self._reachable:
            # 255 is ssh's own exit status; it belongs to no remote command.
            return subprocess.CompletedProcess(
                args=["ssh"], returncode=255, stdout=b"", stderr=b"Connection refused"
            )
        refused = {
            "sudo -n true": not self._passwordless,
            "sudo -S -p '' true": not self._password_accepted,
        }
        rc = 1 if refused.get(remote_cmd, False) else 0
        return subprocess.CompletedProcess(args=["ssh"], returncode=rc, stdout=b"", stderr=b"")

    def close(self) -> None:
        pass


def test_passwordless_sudo_is_used_when_the_host_offers_it() -> None:
    """The cloud-image case: no prompt, and no password anywhere."""
    c = _Recorded(passwordless=True)

    c.sudo_exec("id -u")

    assert c.calls[-1][0] == "sudo -n id -u"
    assert all(stdin is None for _, stdin in c.calls), "nothing should have been written to stdin"


def test_a_password_is_supplied_on_stdin_not_in_the_command() -> None:
    """argv is readable by every process on the host via ``ps``.

    The password must therefore never appear in the command, and ``-p ''``
    keeps sudo's prompt out of captured output that may be shown or logged.
    """
    c = _Recorded(passwordless=False)

    c.sudo_exec("id -u", password="hunter2")

    cmd, stdin = c.calls[-1]
    assert cmd == "sudo -S -p '' id -u"
    assert stdin == b"hunter2\n"
    assert "hunter2" not in cmd


def test_a_host_offering_neither_route_is_refused_before_anything_runs() -> None:
    """Naming which of the two is missing, rather than surfacing sudo's own
    "a terminal is required", which describes our plumbing."""
    c = _Recorded(passwordless=False)

    with pytest.raises(SudoUnavailableError, match="passwordless sudo"):
        c.sudo_exec("id -u")

    assert not any("id -u" in cmd for cmd, _ in c.calls), "the command must not have run"


def test_availability_is_probed_rather_than_inferred_from_stderr() -> None:
    """``sudo -n true`` separates sudo's exit status from the command's.

    Reading stderr instead would confuse a command that failed on its own with
    one that never ran, and the two want opposite responses.
    """
    c = _Recorded(passwordless=True)

    assert c.can_sudo_without_password() is True
    assert c.calls[0][0] == "sudo -n true"


def test_a_rejected_password_is_told_apart_from_a_payload_that_failed() -> None:
    """sudo exits 1 for both, and its stderr is localised.

    Authenticating on its own first is what makes the difference visible: a
    rejected password is worth asking the operator to retype, whereas a script
    that ran and failed is not. Parsing sudo's message would decide this
    differently depending on the host's locale — the rig answers in Chinese.
    """
    c = _Recorded(passwordless=False, password_accepted=False)

    with pytest.raises(SudoUnavailableError, match="rejected the sudo password"):
        c.sudo_exec("bash /tmp/prereqs.sh", password="wrong")

    assert not any("prereqs.sh" in cmd for cmd, _ in c.calls), "the payload must not have run"


def test_an_accepted_password_runs_the_payload_once() -> None:
    """The probe must not cost the payload an extra execution."""
    c = _Recorded(passwordless=False, password_accepted=True)

    c.sudo_exec("bash /tmp/prereqs.sh", password="right")

    payload = [cmd for cmd, _ in c.calls if "prereqs.sh" in cmd]
    assert payload == ["sudo -S -p '' bash /tmp/prereqs.sh"]


def test_an_unreachable_host_is_not_reported_as_a_host_without_sudo() -> None:
    """``sudo -n true`` exiting 255 means ssh never ran it.

    Folding that into "no passwordless sudo" sends the operator looking for a
    sudoers problem on a machine that never answered — the same misdirection
    ``SshConnectionError`` exists to prevent elsewhere.
    """
    c = _Recorded(passwordless=False, reachable=False)

    with pytest.raises(SshConnectionError):
        c.can_sudo_without_password()

    with pytest.raises(SshConnectionError):
        c.sudo_exec("id -u", password="hunter2")


def test_the_remote_home_is_asked_once_per_host_not_once_per_client() -> None:
    """Clients are built per operation; the home belongs to the host.

    The orchestrator constructs a fresh ``SshClient`` for every run it touches,
    so a cache living on the client is no cache at all. It was one: listing
    twenty-two runs asked a single machine where its home directory was twenty
    times, 12.2 s of the 18.7 s that took. The answer cannot differ between two
    clients pointed at the same destination.
    """
    from lerobot.gui.training import ssh_transport

    ssh_transport._REMOTE_HOME_CACHE.clear()
    transport = SshTransport(host="rig.invalid", port=22, user="operator")
    asked: list[str] = []

    def fake_exec(self, remote_cmd, *, timeout=30.0, stdin=None):
        asked.append(remote_cmd)
        return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout=b"/home/operator\n", stderr=b"")

    original = SshClient._exec
    SshClient._exec = fake_exec
    try:
        roots = [SshClient(transport).run_root(f"run{i}", Path("/gui/runs") / f"run{i}") for i in range(5)]
    finally:
        SshClient._exec = original
        ssh_transport._REMOTE_HOME_CACHE.clear()

    assert asked.count("echo $HOME") == 1, (
        f"five clients on one host asked {asked.count('echo $HOME')} times: {asked}"
    )
    assert roots[0] == Path("/home/operator/.lerobot/runs/run0")
    assert roots[4] == Path("/home/operator/.lerobot/runs/run4")
