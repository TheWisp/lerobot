# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""When ``ensure_prereqs`` is allowed to reach for sudo.

The installer needs root. Deciding *whether to run it* does not: its three
conditions — is Docker installed, is the user in the docker group, is the
NVIDIA toolkit present and unheld — are all readable by anyone. Asking them
under sudo meant a host that was already set up got refused for want of
passwordless sudo it needed only to find out it had nothing to do.

These tests fix which branch is taken, because the failure they guard against
is not an exception — it is a privileged installer running on somebody's
working machine when it had no reason to.
"""

from __future__ import annotations

import subprocess

import pytest

from lerobot.gui.training.ssh_transport import SshClient
from lerobot.gui.training.transport import SshTransport


class _RecordingClient(SshClient):
    """An SshClient whose remote calls are recorded instead of made."""

    def __init__(self, *, probe_out: bytes = b"", probe_rc: int = 0, install_rc: int = 0) -> None:
        super().__init__(SshTransport(host="nowhere.invalid", user="nobody"))
        self._probe_out, self._probe_rc, self._install_rc = probe_out, probe_rc, install_rc
        self.commands: list[str] = []

    def _exec(self, remote_cmd: str, *, timeout: float = 30.0, stdin: bytes | None = None):
        self.commands.append(remote_cmd)
        if "sudo" in remote_cmd:
            return subprocess.CompletedProcess(
                args=[], returncode=self._install_rc, stdout=b"", stderr=b"denied"
            )
        return subprocess.CompletedProcess(
            args=[], returncode=self._probe_rc, stdout=self._probe_out, stderr=b""
        )

    def close(self) -> None:  # no socket was ever opened
        pass

    @property
    def escalated(self) -> bool:
        return any("sudo" in c for c in self.commands)


def test_a_provisioned_host_is_never_escalated_to() -> None:
    """Nothing missing: the installer must not run, and sudo is never invoked."""
    client = _RecordingClient(probe_out=b"")

    client.ensure_prereqs()

    assert not client.escalated, "a host with nothing to do must not be asked for root"
    assert len(client.commands) == 1, "one read-only probe, and nothing else"


def test_a_host_missing_docker_is_installed_onto() -> None:
    """The fresh-VM case this exists for still works."""
    client = _RecordingClient(probe_out=b" docker docker-group")

    client.ensure_prereqs()

    assert client.escalated, "something is genuinely missing; the installer must run"


def test_an_unanswerable_probe_escalates_rather_than_assuming() -> None:
    """If the probe could not run, "already fine" is the unsafe guess."""
    client = _RecordingClient(probe_rc=255, probe_out=b"")

    client.ensure_prereqs()

    assert client.escalated


def test_a_failed_install_names_what_was_missing() -> None:
    """The error a user sees should say why root was needed at all."""
    client = _RecordingClient(probe_out=b" nvidia-container-toolkit", install_rc=1)

    with pytest.raises(RuntimeError, match="nvidia-container-toolkit"):
        client.ensure_prereqs()
