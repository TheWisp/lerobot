# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training host registry — listing of where the user can run training.

A :class:`TrainingHost` is "anywhere training can run" — the GUI server's
own box (workstation case, auto-registered), a user-added remote box, or an
auto-spawned cloud VM (future). For the prototype, only the workstation host
is materialized; user-added and Ephemeral lands with SSH transport.

DESIGN.md § Modes table → this file is the registry surface those entries map to.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.gui.training.jobs import HOSTS_DIR, HostProfile
from lerobot.gui.training.transport import HostTransport, SshTransport, SubprocessTransport

logger = logging.getLogger(__name__)

# Stable id for the auto-registered workstation host. The "this-server" string
# is the host_id the UI shows in dropdowns and the orchestrator looks up.
WORKSTATION_HOST_ID = "this-server"


@dataclass(frozen=True)
class TrainingHost:
    """A host where training can run.

    ``transport`` carries the operational shape (subprocess vs SSH). The
    orchestrator never branches on host *type*; it dispatches via the
    transport (same code path for all hosts).
    """

    id: str
    display_name: str
    transport: HostTransport
    capabilities: dict[str, Any] = field(default_factory=dict)


# ── GPU detection ─────────────────────────────────────────────────────────────


def _detect_nvidia_gpu() -> dict[str, Any] | None:
    """Probe ``nvidia-smi`` for GPU presence + name. Returns None if no NVIDIA
    GPU is present or the tool isn't installed.

    Cheap (single subprocess at startup), no Python NVIDIA bindings needed.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return None
    # First GPU — v1 is single-GPU per host (DESIGN.md "What we don't try to do").
    name, mem_str = (s.strip() for s in lines[0].split(","))
    try:
        vram_mb = int(mem_str)
    except ValueError:
        vram_mb = 0
    return {"gpu_name": name, "vram_mb": vram_mb, "gpu_count_detected": len(lines)}


def workstation_host(workdir: Path | None = None) -> TrainingHost | None:
    """Return the auto-registered workstation host if a local GPU is detected.

    DESIGN.md § Modes (Workstation row): "GUI detects on first start and
    registers the host as 'This server'."

    Returns None if no NVIDIA GPU is present — workstation mode just doesn't
    show up in the dropdown.
    """
    caps = _detect_nvidia_gpu()
    if caps is None:
        return None
    return TrainingHost(
        id=WORKSTATION_HOST_ID,
        display_name="This server",
        transport=SubprocessTransport(workdir=workdir or Path.home() / ".cache" / "lerobot" / "runs"),
        capabilities=caps,
    )


# ── Profile → TrainingHost adapter ────────────────────────────────────────────


def profile_to_training_host(profile: HostProfile) -> TrainingHost:
    """Materialize a saved :class:`HostProfile` as a :class:`TrainingHost`.

    Used at two seams: (1) ``HostRegistry.auto()`` loading saved hosts at
    GUI startup, and (2) the POST /hosts handler registering a freshly
    saved profile without a server restart. Both paths converge here so
    SSH-transport construction is captured in one place.
    """
    return TrainingHost(
        id=profile.name,
        display_name=profile.display_name or profile.name,
        transport=SshTransport(host=profile.ssh_host, port=profile.ssh_port, user=profile.ssh_user),
        capabilities=dict(profile.capabilities),
    )


# ── Registry ──────────────────────────────────────────────────────────────────


class HostRegistry:
    """In-memory registry of training hosts the GUI knows about.

    The workstation host is auto-detected at construction time; user-added
    SSH hosts are loaded from ``~/.config/lerobot/training_hosts/`` on
    startup and mutated in place via :meth:`add` / :meth:`remove` /
    :meth:`replace` as the user touches the "Add SSH host" dialog.
    """

    def __init__(self, hosts: list[TrainingHost] | None = None) -> None:
        self._hosts: dict[str, TrainingHost] = {h.id: h for h in (hosts or [])}

    @classmethod
    def auto(cls, workdir: Path | None = None, hosts_dir: Path | None = None) -> HostRegistry:
        """Build a registry from auto-detected workstation + on-disk profiles.

        ``hosts_dir`` defaults to :data:`HOSTS_DIR`. Tests can point this at
        a tmp dir to avoid touching the user's real config.
        """
        hosts: list[TrainingHost] = []
        ws = workstation_host(workdir=workdir)
        if ws is not None:
            hosts.append(ws)
        for profile in HostProfile.load_all(hosts_dir or HOSTS_DIR):
            try:
                hosts.append(profile_to_training_host(profile))
            except Exception as e:
                logger.warning("skipping host profile %s: %s", profile.name, e)
        return cls(hosts)

    def list_hosts(self) -> list[TrainingHost]:
        return list(self._hosts.values())

    def get(self, host_id: str) -> TrainingHost | None:
        return self._hosts.get(host_id)

    def add(self, host: TrainingHost) -> None:
        assert host.id not in self._hosts, f"host id collision: {host.id!r}"
        self._hosts[host.id] = host

    def remove(self, host_id: str) -> bool:
        """Evict a host. Returns True if it was present, False otherwise.
        The caller is responsible for deleting the on-disk profile."""
        return self._hosts.pop(host_id, None) is not None

    def replace(self, host_id: str, host: TrainingHost) -> None:
        """Swap the entry under ``host_id`` for ``host``. The new host's
        ``id`` need not equal ``host_id`` (e.g., rename via delete + add)."""
        self._hosts.pop(host_id, None)
        self._hosts[host.id] = host
