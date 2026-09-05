"""Unit tests for SshClient.wait_until_ready — the post-spawn boot-race guard.

A freshly-spawned cloud VM reports RUNNING before sshd/cloud-init are up, so the
first remote op used to race the boot and die on a single 30 s attempt (the
round-2 ephemeral smoke-test failure). wait_until_ready polls until SSH answers.
``_exec`` is stubbed so these never touch a real network; sleep/clock injected.
"""

from __future__ import annotations

import subprocess

import pytest

from lerobot.gui.training.ssh_transport import SshClient
from lerobot.gui.training.transport import SshTransport


def _client(tmp_path):
    return SshClient(SshTransport(host="1.2.3.4", port=22, user="bot"), control_path_dir=tmp_path)


def _ok(stdout: bytes = b"/tmp/lerobot-prereqs-abc12345.sh\n"):
    """A successful remote call. The default stdout is a plausible ``mktemp``
    reply, since staging the prereqs script reads the path back from it."""
    return subprocess.CompletedProcess(args=["ssh"], returncode=0, stdout=stdout, stderr=b"")


class _Clock:
    """Monotonic clock that only advances when the injected sleep is called."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_retries_through_boot_then_succeeds(tmp_path, monkeypatch):
    clock = _Clock()
    slept: list[float] = []
    calls = {"n": 0}

    def fake_exec(remote_cmd, *, timeout, stdin=None):
        calls["n"] += 1
        if calls["n"] < 3:  # sshd not up yet
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
        return _ok()

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fake_exec)

    def fake_sleep(s):
        slept.append(s)
        clock.t += s

    client.wait_until_ready(timeout_s=300, poll_interval_s=5, sleep=fake_sleep, clock=clock)
    assert calls["n"] == 3
    assert slept == [5, 5]  # slept after each of the two failed attempts


def test_succeeds_on_first_attempt_when_host_is_up(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_exec(remote_cmd, *, timeout, stdin=None):
        calls["n"] += 1
        return _ok()

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fake_exec)
    client.wait_until_ready(sleep=lambda _s: pytest.fail("should not sleep"), clock=lambda: 0.0)
    assert calls["n"] == 1


def test_raises_after_deadline_naming_boot_or_security_group(tmp_path, monkeypatch):
    clock = _Clock()

    def always_timeout(remote_cmd, *, timeout, stdin=None):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", always_timeout)

    def fake_sleep(s):
        clock.t += s

    with pytest.raises(RuntimeError, match="security group"):
        client.wait_until_ready(timeout_s=20, poll_interval_s=5, sleep=fake_sleep, clock=clock)


def test_ensure_prereqs_stages_the_script_then_runs_it_as_root(tmp_path, monkeypatch):
    """The script is put on the host and named, not piped to ``sudo bash -s``.

    Escalation goes through ``sudo_exec``, which owns stdin for the password.
    While the script arrived on stdin there was nowhere for a password to go, so
    a host without passwordless sudo could not be provisioned at all.
    """
    calls = []
    closed = {"n": 0}

    def fake_exec(remote_cmd, *, timeout=30.0, stdin=None):
        calls.append((remote_cmd, stdin))
        return _ok()

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fake_exec)
    monkeypatch.setattr(client, "close", lambda: closed.__setitem__("n", closed["n"] + 1))

    client.ensure_prereqs()

    staged = [c for c in calls if c[1] is not None]
    assert staged, "the script was never written to the host"
    assert b"install" in staged[0][1].lower(), "that write was not the script"
    assert any("mktemp" in c[0] for c in calls), (
        "the host must pick the name: a path we choose in world-writable /tmp can be "
        "pre-created as a symlink, and this file is executed as root"
    )

    ran = [c[0] for c in calls if "bash /tmp/" in c[0]]
    assert ran and ran[0].startswith("sudo "), f"the script was not run as root: {ran}"
    assert "LEROBOT_PREREQS_SKIP_CONTAINER_SMOKE=1" in ran[0]

    assert any(c[0].startswith("rm -f /tmp/") for c in calls), "the staged script was left behind"
    assert closed["n"] == 1  # control master reset so the new group applies


def test_ensure_prereqs_raises_on_setup_failure(tmp_path, monkeypatch):
    def fail_exec(remote_cmd, *, timeout=30.0, stdin=None):
        # Sudo is available and the script stages fine; the install itself is
        # what fails, which is the case this covers.
        if "bash /tmp/" not in remote_cmd:
            return _ok()
        return subprocess.CompletedProcess(
            args=["ssh"], returncode=1, stdout=b"", stderr=b"boom: held packages"
        )

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fail_exec)
    monkeypatch.setattr(client, "close", lambda: pytest.fail("must not reset control on failure"))
    with pytest.raises(RuntimeError, match="prereqs setup failed"):
        client.ensure_prereqs()


def test_local_ensure_prereqs_is_noop():
    """SubprocessClient must not try to apt-install Docker on the user's box."""
    from pathlib import Path

    from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

    client = SubprocessClient(SubprocessTransport(workdir=Path(".")))
    assert client.ensure_prereqs() is None


def test_ensure_prereqs_says_so_when_the_script_cannot_be_staged(tmp_path, monkeypatch):
    """A host that will not take the file is a different failure from one whose
    install fails, and the message has to say which."""

    def fail_write(remote_cmd, *, timeout=30.0, stdin=None):
        if stdin is not None:  # the staging write
            return subprocess.CompletedProcess(
                args=["ssh"], returncode=1, stdout=b"", stderr=b"No space left on device"
            )
        return _ok()

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", fail_write)
    with pytest.raises(RuntimeError, match="could not stage"):
        client.ensure_prereqs()


def test_ensure_prereqs_without_sudo_or_password_refuses_before_installing(tmp_path, monkeypatch):
    """The case this whole seam exists for: an already-provisioned workstation
    with no passwordless sudo. It must name what is missing, not report
    whichever command happened to run first."""
    from lerobot.gui.training.transport import SudoUnavailableError

    def no_sudo(remote_cmd, *, timeout=30.0, stdin=None):
        if remote_cmd == "sudo -n true":
            return subprocess.CompletedProcess(args=["ssh"], returncode=1, stdout=b"", stderr=b"")
        return _ok()

    client = _client(tmp_path)
    monkeypatch.setattr(client, "_exec", no_sudo)
    with pytest.raises(SudoUnavailableError, match="passwordless sudo"):
        client.ensure_prereqs()
