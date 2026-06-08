# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Transport layer for training runs: how the GUI server reaches the training process.

Two transports, matching ``scripts/training/DESIGN.md`` § Components:

- :class:`SubprocessTransport` — training runs as a subprocess of the GUI server
  (workstation case). "Host" is the GUI server's own filesystem; all operations
  are local file I/O.
- :class:`SshTransport` — training runs over SSH on a remote host
  (Persistent / Ephemeral cases). Lands in the SSH transport follow-up; stub here.

The :class:`TransportClient` protocol is what the run orchestrator drives. The
operations it needs are intentionally narrow:

- :meth:`launch` — start a detached training process; returns a session id
- :meth:`is_alive` / :meth:`stop` — process lifecycle
- :meth:`read_text` / :meth:`read_bytes_from_offset` — pull the structured
  state files (progress.json, events.jsonl, checkpoints.jsonl, stderr.log) from
  the host without holding an open pipe
- :meth:`fetch_file` — copy a checkpoint file from the host to local storage
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# ── Transport configurations ───────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class SubprocessTransport:
    """Run training as a subprocess of the GUI server, on the same machine.

    The "host" is the GUI server's own filesystem; no SSH dance needed.
    Files the training process writes are directly readable; no SCP for
    checkpoint pull either (the GUI server opens them in place).
    """

    workdir: Path


@dataclass(frozen=True, kw_only=True)
class SshTransport:
    """Run training over SSH on a remote host."""

    host: str
    port: int = 22
    user: str = "root"
    key_path: str


HostTransport = SubprocessTransport | SshTransport


# ── TransportClient protocol ──────────────────────────────────────────────────


class TransportClient(Protocol):
    """Operations the run orchestrator drives, regardless of transport."""

    def launch(
        self,
        command: list[str],
        env: dict[str, str],
        workdir: Path,
        log_path: Path,
    ) -> int:
        """Start a detached process running ``command``. Returns a session id.

        Pre: ``workdir`` exists or can be created. ``log_path``'s parent
        exists or can be created.
        Post: returned process is detached from the caller's process group
        (survives the GUI server restarting). stdout + stderr are merged
        and written to ``log_path``.
        """
        ...

    def is_alive(self, session_id: int) -> bool:
        """Is the launched process still running?"""
        ...

    def stop(self, session_id: int, *, force: bool = False) -> None:
        """Send SIGTERM (graceful) or SIGKILL (``force=True``). Idempotent —
        no error if the process is already dead."""
        ...

    def read_text(self, path: Path) -> str | None:
        """Read a file's full text content from the host. Returns None if
        the file doesn't exist."""
        ...

    def read_bytes_from_offset(self, path: Path, offset: int) -> tuple[bytes, int]:
        """Read from ``offset`` to EOF. Returns ``(data, new_offset)``.

        For a missing file, returns ``(b"", offset)``. Used for incremental
        tailing of stderr.log without redownloading the whole thing each poll.
        """
        ...

    def fetch_file(self, src: Path, dst: Path) -> None:
        """Copy a file from the host's filesystem to a local destination.

        For SubprocessClient, both paths are local; this is just ``shutil.copy``.
        For SshClient (later), this is SCP. ``dst``'s parent is created if missing.
        """
        ...


# ── SubprocessClient ──────────────────────────────────────────────────────────


class SubprocessClient:
    """TransportClient impl for :class:`SubprocessTransport`.

    All operations are local file I/O — the "host" is the GUI server itself.
    Processes are launched detached (``start_new_session=True``) so they
    survive the GUI server restarting.
    """

    def __init__(self, transport: SubprocessTransport) -> None:
        self._transport = transport

    @property
    def workdir(self) -> Path:
        return self._transport.workdir

    def launch(
        self,
        command: list[str],
        env: dict[str, str],
        workdir: Path,
        log_path: Path,
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)
        # Merge our env on top of the GUI server's environment (the worker
        # needs PATH, HOME, etc.).
        full_env = {**os.environ, **env}
        # Open the log file in write-binary mode; ownership transfers to the
        # subprocess (its stdout/stderr handles point to it). The parent's
        # handle leaks intentionally — closed when the subprocess exits.
        log_f = log_path.open("wb")  # noqa: SIM115 - intentional handover to subprocess
        proc = subprocess.Popen(
            command,
            cwd=str(workdir),
            env=full_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from GUI server's process group
        )
        return proc.pid

    def is_alive(self, session_id: int) -> bool:
        # Two cases:
        # (1) The process is still our direct child (we launched it, haven't
        #     restarted). waitpid(WNOHANG) reaps zombies and returns (pid, _);
        #     if still running it returns (0, 0). Without this reap, kill(pid,0)
        #     returns True for zombies — would deadlock the alive check.
        # (2) After GUI server restart, the process is no longer our child
        #     (reparented to init). waitpid raises ChildProcessError; we fall
        #     back to kill(pid, 0).
        try:
            reaped, _ = os.waitpid(session_id, os.WNOHANG)
            # waitpid returns (pid, status) if reaped (process is dead),
            # (0, 0) if our child is still running.
            return reaped != session_id
        except ChildProcessError:
            pass  # not our child; fall through to signal probe
        try:
            os.kill(session_id, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def stop(self, session_id: int, *, force: bool = False) -> None:
        sig = signal.SIGKILL if force else signal.SIGTERM
        # Signal the whole process group so detached children are caught,
        # not just the leader. Idempotent — already-dead PID is a no-op.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(session_id), sig)

    def read_text(self, path: Path) -> str | None:
        try:
            return path.read_text()
        except FileNotFoundError:
            return None

    def read_bytes_from_offset(self, path: Path, offset: int) -> tuple[bytes, int]:
        assert offset >= 0, f"offset must be non-negative, got {offset}"
        try:
            with path.open("rb") as f:
                f.seek(offset)
                data = f.read()
                return data, offset + len(data)
        except FileNotFoundError:
            return b"", offset

    def fetch_file(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


# ── Factory ────────────────────────────────────────────────────────────────────


def make_client(transport: HostTransport) -> TransportClient:
    """Construct the appropriate :class:`TransportClient` for a given transport.

    Pre: ``transport`` is one of the known transport types.
    Post: returns a client ready to drive a training run on that transport.
    """
    if isinstance(transport, SubprocessTransport):
        return SubprocessClient(transport)
    if isinstance(transport, SshTransport):
        raise NotImplementedError(
            "SshClient lands in the SSH transport follow-up. "
            "The subprocess prototype is complete; SSH provider/transport "
            "are the next phase."
        )
    raise TypeError(f"Unknown transport: {type(transport).__name__}")
