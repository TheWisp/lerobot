# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The ssh destination, and telling ssh's own failure from the remote command's.

Both were silent faults. A user substituted for an absent one overrides the
operator's ``~/.ssh/config`` without saying so, and ssh's exit 255 reported as
whatever ran first sends the reader to the wrong subsystem.
"""

from __future__ import annotations

import subprocess

import pytest

from lerobot.gui.training.ssh_transport import SshClient
from lerobot.gui.training.transport import SshConnectionError, SshTransport, ssh_destination


def _client(user: str) -> SshClient:
    return SshClient(SshTransport(host="rig.invalid", port=22, user=user))


def _failed(returncode: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(args=["ssh"], returncode=returncode, stdout=b"", stderr=b"")


class TestDestination:
    def test_a_named_user_is_used(self):
        assert ssh_destination("deploy", "rig") == "deploy@rig"

    def test_no_user_means_ssh_decides(self):
        """Not "root" — the operator's ssh config must keep its say."""
        assert ssh_destination("", "rig") == "rig"

    def test_the_argv_carries_the_bare_host_when_no_user_was_named(self):
        argv = _client("")._ssh_argv("true")
        assert "rig.invalid" in argv
        assert not any(a.endswith("@rig.invalid") for a in argv), argv

    def test_the_argv_carries_user_at_host_when_one_was_named(self):
        assert "deploy@rig.invalid" in _client("deploy")._ssh_argv("true")


class TestUnreachable:
    def test_ssh_exit_255_is_a_connection_error(self):
        with pytest.raises(SshConnectionError, match="cannot connect"):
            _client("deploy")._raise_if_unreachable(_failed(255), "Permission denied")

    def test_a_remote_command_failing_is_left_alone(self):
        """Only 255 is ssh's own. Anything else is the command's, and the
        caller's existing error is the right one."""
        _client("deploy")._raise_if_unreachable(_failed(1), "apt-get failed")

    def test_an_unnamed_user_gets_the_hint_that_explains_it(self):
        """The likeliest cause of a refused connection is the field itself."""
        with pytest.raises(SshConnectionError, match="user@host"):
            _client("")._raise_if_unreachable(_failed(255), "Permission denied")

    def test_a_named_user_gets_no_such_hint(self):
        with pytest.raises(SshConnectionError) as e:
            _client("deploy")._raise_if_unreachable(_failed(255), "Permission denied")
        assert "user@host" not in str(e.value)
