# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Nebius Compute provider — Ephemeral VM lifecycle via the ``nebius`` CLI.

Implements :class:`HostProvider` end-to-end through an INJECTABLE command
runner (``run_cli``), mirroring how :class:`SshClient` was built: the
control flow, SKU resolution, cloud-init document, JSON→HostHandle
parsing, readiness polling, and destroy/verify logic are all unit-tested
against a fake runner. The default runner shells out to the real
``nebius`` CLI.

VERIFICATION STATUS (2026-06-15): the exact ``nebius`` subcommand argv
(marked ``# VERIFY`` below) is NOT yet confirmed against a live CLI —
this environment has no ``nebius`` binary, terraform, or SDK. The argv
builders are isolated so they can be corrected from one place once the
tooling is available; tests assert their STRUCTURE (platform/preset/disk
present, names match) rather than byte-exact strings. Grounded data: the
L40S SKU + cloud-init shape + boot-disk fields come from the known-good
Terraform spec that provisioned the GPU-smoke VM (2026-06-12), so they
are real, not guessed. See PR description + TODO.md for the live-smoke
plan.

No cost estimation by design — vendor prices drift and there is no
price-catalog API to validate against. The dialog links to Nebius's
pricing page; spend protection is mechanical (hard TTL, disk-size
warning, auto-destroy). See DESIGN.md § What we don't try to do.

Auth: inherits from ``~/.nebius/`` (the user's existing ``nebius`` CLI
profile) — same pattern as kubectl / gh / aws CLI inheritance. The GUI
never prompts for vendor credentials.

Defensive defaults (lessons from the 2026-06-05 disk-billing bill):

  - Inline-managed boot disk only (cascade-delete with the VM). Never a
    standalone disk; those survive VM-delete and accrue silent charges.
  - Warn on ``disk_gib > 256`` — most LeRobot training fits in <100 GB.
  - Hard TTL: schedule a server-side delete at ``spawn + ttl_seconds`` so
    the VM dies even if our GUI server does. REQUIRED by the protocol.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from lerobot.gui.training.providers.protocol import (
    GpuKind,
    HostHandle,
    SpawnSpec,
)

logger = logging.getLogger(__name__)

# A command runner: takes the nebius subcommand argv (WITHOUT the leading
# "nebius") + a timeout, returns the completed process (text mode). Injected
# so tests drive the provider with canned CLI output and no real network.
CliRunner = Callable[[list[str], float], "subprocess.CompletedProcess[str]"]

# ── SKU mapping ─────────────────────────────────────────────────────────────


# Canonical GpuKind → Nebius platform string. Lookup is explicit — a GPU
# Nebius doesn't offer raises at lookup time, before spawn.
#
# PROVENANCE: only L40S is VERIFIED — its platform/preset come from the
# Terraform that actually provisioned the GPU-smoke VM (gpu-l40s-d,
# 1gpu-16vcpu-96gb). The others are best-effort and flagged UNVERIFIED;
# confirm against `nebius compute platform list` before relying on them.
NEBIUS_GPU_PLATFORMS: dict[GpuKind, str] = {
    "L40S": "gpu-l40s-d",  # VERIFIED (Terraform, 2026-06-12)
    "L4": "gpu-l4-a",  # UNVERIFIED
    "H100": "gpu-h100-sxm5",  # UNVERIFIED
    "H200": "gpu-h200-sxm5",  # UNVERIFIED
}


# Preset must match the GPU platform. v1 is single-GPU per host.
NEBIUS_DEFAULT_PRESET: dict[GpuKind, str] = {
    "L40S": "1gpu-16vcpu-96gb",  # VERIFIED (Terraform, 2026-06-12)
    "L4": "1gpu-8vcpu-32gb",  # UNVERIFIED
    "H100": "1gpu-16vcpu-200gb",  # UNVERIFIED
    "H200": "1gpu-16vcpu-200gb",  # UNVERIFIED
}


# Boot image family. VERIFIED from the Terraform spec.
NEBIUS_IMAGE_FAMILY = "ubuntu24.04-cuda13.0-serverless"

# Soft cap to surface a warning. The 1.28 TB Terraform web-form default is
# what bit us; warn well below that.
NEBIUS_DISK_WARN_GIB = 256

GIB = 1024**3


# ── Provider ────────────────────────────────────────────────────────────────


class NebiusProvider:
    """Nebius Compute provider — Ephemeral VM lifecycle.

    Pre: a working ``nebius`` CLI on PATH with an authenticated
    ``~/.nebius/`` profile (only when the DEFAULT runner is used; tests
    inject their own).
    """

    id = "nebius"
    display_name = "Nebius (auto-managed)"

    def __init__(
        self,
        *,
        run_cli: CliRunner | None = None,
        ssh_public_key: str | None = None,
        ssh_user: str = "lerobot",
        poll_interval_s: float = 10.0,
        ready_timeout_s: float = 600.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._run_cli = run_cli or _default_run_cli
        self._ssh_user = ssh_user
        # Resolved lazily so construction never touches the filesystem in
        # tests that inject a key; a missing key only matters at spawn time.
        self._ssh_public_key = ssh_public_key
        self._poll_interval_s = poll_interval_s
        self._ready_timeout_s = ready_timeout_s
        self._sleep = sleep
        self._clock = clock

    # ── Lifecycle ───────────────────────────────────────────────────────

    def spawn(self, spec: SpawnSpec) -> HostHandle:
        """Provision a VM, attach an inline boot disk + static public IP,
        schedule the hard-TTL delete, and poll until SSH is reachable.

        Pre: ``spec.ttl_seconds > 0``; the GpuKind is in the SKU tables.
        Post: returned handle's ``ssh_host:ssh_port`` accepts SSH via the
        user's local key, and a server-side delete is scheduled at
        ``spawn + ttl_seconds`` regardless of what happens to this process.
        """
        if spec.ttl_seconds <= 0:
            raise ValueError(f"spec.ttl_seconds must be > 0, got {spec.ttl_seconds}")
        platform = self.platform_for(spec.gpu)
        preset = self.preset_for(spec.gpu)
        warning = self.disk_warning(spec.disk_gib)
        if warning:
            logger.warning("nebius spawn: %s", warning)

        name = _instance_name()
        pubkey = self._resolve_pubkey()
        user_data = build_cloud_init(self._ssh_user, pubkey)
        spawned_at = self._clock()

        # VERIFY: exact `nebius compute instance create` argv.
        create_argv = _create_argv(
            name=name,
            platform=platform,
            preset=preset,
            disk_gib=spec.disk_gib,
            user_data=user_data,
            region_hint=spec.region_hint,
        )
        proc = self._run_cli(create_argv, 300.0)
        if proc.returncode != 0:
            raise NebiusCliError("create", proc)
        resource_id = _parse_instance_id(proc.stdout)

        # Hard TTL FIRST, before the (potentially long) readiness poll —
        # a VM that never becomes reachable must still self-destruct.
        # VERIFY: exact scheduled-delete argv.
        self._schedule_ttl_delete(resource_id, spec.ttl_seconds)

        deadline = spawned_at + self._ready_timeout_s
        handle = self._poll_until_ready(
            resource_id=resource_id,
            spec=spec,
            expires_at_unix=int(spawned_at + spec.ttl_seconds),
            deadline=deadline,
        )
        logger.info("nebius spawn: %s ready at %s:%d", resource_id, handle.ssh_host, handle.ssh_port)
        return handle

    def destroy(self, handle: HostHandle) -> None:
        """Idempotent delete of the VM + inline boot disk + public IP.

        Does not touch ``handle.persistent_volume_id`` (lifecycled
        separately). A delete of an already-gone instance is treated as
        success — destroy is the cleanup path and must never wedge.
        """
        # VERIFY: exact delete argv.
        argv = ["compute", "instance", "delete", "--id", handle.provider_resource_id]
        proc = self._run_cli(argv, 120.0)
        if proc.returncode != 0 and not _looks_like_not_found(proc):
            raise NebiusCliError("delete", proc)

    def verify_destroyed(self, handle: HostHandle) -> bool:
        """True iff no billable resource for this handle remains.

        Re-gets the instance; not-found (or a DELETED/DELETING status) is
        success. Any live status is a failed teardown.
        """
        # VERIFY: exact get argv.
        argv = ["compute", "instance", "get", "--id", handle.provider_resource_id, "--format", "json"]
        proc = self._run_cli(argv, 60.0)
        if _looks_like_not_found(proc):
            return True
        if proc.returncode != 0:
            # Can't confirm — conservatively report NOT destroyed so the
            # caller (and the user) keep looking rather than assume clean.
            logger.warning("nebius verify_destroyed: get failed: %s", proc.stderr[:200])
            return False
        try:
            status = (json.loads(proc.stdout).get("status") or {}).get("state", "")
        except (json.JSONDecodeError, AttributeError):
            return False
        return str(status).upper() in {"DELETED", "DELETING"}

    # ── Internals ───────────────────────────────────────────────────────

    def _schedule_ttl_delete(self, resource_id: str, ttl_seconds: int) -> None:
        # VERIFY: Nebius's server-side scheduled-delete mechanism. The
        # protocol REQUIRES the VM self-destruct even if the GUI server is
        # gone, so this must be a vendor-side schedule, not a local timer.
        argv = [
            "compute",
            "instance",
            "delete",
            "--id",
            resource_id,
            "--after-seconds",
            str(ttl_seconds),
        ]
        proc = self._run_cli(argv, 60.0)
        if proc.returncode != 0:
            raise NebiusCliError("schedule-delete", proc)

    def _poll_until_ready(
        self,
        *,
        resource_id: str,
        spec: SpawnSpec,
        expires_at_unix: int,
        deadline: float,
    ) -> HostHandle:
        """Poll ``instance get`` until the VM is RUNNING with a public IP,
        then return the handle. Raises on timeout."""
        last_state = "?"
        while True:
            # VERIFY: exact get argv.
            argv = ["compute", "instance", "get", "--id", resource_id, "--format", "json"]
            proc = self._run_cli(argv, 60.0)
            if proc.returncode == 0:
                info = _parse_instance_info(proc.stdout)
                last_state = info.state
                if info.state.upper() == "RUNNING" and info.public_ip:
                    return HostHandle(
                        provider="nebius",
                        provider_resource_id=resource_id,
                        ssh_host=info.public_ip,
                        ssh_port=22,
                        ssh_user=self._ssh_user,
                        region=info.region or (spec.region_hint or "unknown"),
                        expires_at_unix=expires_at_unix,
                    )
            if self._clock() >= deadline:
                raise NebiusSpawnTimeoutError(
                    f"instance {resource_id} not reachable within "
                    f"{self._ready_timeout_s:.0f}s (last state={last_state!r})"
                )
            self._sleep(self._poll_interval_s)

    def _resolve_pubkey(self) -> str:
        if self._ssh_public_key:
            return self._ssh_public_key
        for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
            p = Path.home() / ".ssh" / name
            if p.exists():
                return p.read_text().strip()
        raise FileNotFoundError(
            "no SSH public key found under ~/.ssh — generate one with "
            "`ssh-keygen` so the spawned VM can authorize your login"
        )

    # ── SKU + warning helpers (pure; exposed for UI + tests) ─────────────

    @staticmethod
    def disk_warning(disk_gib: int) -> str | None:
        """Warning string if ``disk_gib`` is unusually large, else None.

        Disks bill continuously create-to-delete regardless of VM state —
        the structural lesson from the bill investigation.
        """
        if disk_gib > NEBIUS_DISK_WARN_GIB:
            return (
                f"Boot disk is {disk_gib} GiB (warning at >{NEBIUS_DISK_WARN_GIB}). "
                f"Disks bill continuously until deleted, regardless of VM "
                f"state. Most training fits in 50–100 GiB."
            )
        return None

    @staticmethod
    def platform_for(gpu: GpuKind) -> str:
        """SKU lookup for the nebius CLI's --platform flag."""
        if gpu not in NEBIUS_GPU_PLATFORMS:
            raise ValueError(
                f"Nebius has no platform mapping for GPU kind {gpu!r}. "
                f"Supported: {sorted(NEBIUS_GPU_PLATFORMS)}"
            )
        return NEBIUS_GPU_PLATFORMS[gpu]

    @staticmethod
    def preset_for(gpu: GpuKind) -> str:
        """SKU lookup for the nebius CLI's --preset flag (matching the platform)."""
        if gpu not in NEBIUS_DEFAULT_PRESET:
            raise ValueError(
                f"Nebius has no default preset for GPU kind {gpu!r}. "
                f"Supported: {sorted(NEBIUS_DEFAULT_PRESET)}"
            )
        return NEBIUS_DEFAULT_PRESET[gpu]


# ── Errors ──────────────────────────────────────────────────────────────────


class NebiusCliError(RuntimeError):
    """A ``nebius`` CLI invocation returned non-zero."""

    def __init__(self, op: str, proc: subprocess.CompletedProcess[str]) -> None:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        super().__init__(f"nebius {op} failed (rc={proc.returncode}): {tail}")


class NebiusSpawnTimeoutError(RuntimeError):
    """The VM did not become SSH-reachable within the readiness window."""


# ── Pure helpers (fully testable, no CLI) ────────────────────────────────────


class _InstanceInfo:
    __slots__ = ("state", "public_ip", "region")

    def __init__(self, state: str, public_ip: str | None, region: str | None) -> None:
        self.state = state
        self.public_ip = public_ip
        self.region = region


def _instance_name() -> str:
    """Unique, identifiable instance name. ``lerobot-eph-`` prefix makes
    stray VMs obvious in the Nebius console (and greppable for cleanup)."""
    return f"lerobot-eph-{uuid.uuid4().hex[:10]}"


def build_cloud_init(ssh_user: str, ssh_public_key: str) -> str:
    """cloud-init user-data granting ``ssh_user`` key-only login + NOPASSWD
    sudo. Shape mirrors the known-good GPU-smoke Terraform user_data.

    Pre: ``ssh_public_key`` is a single OpenSSH public-key line.
    """
    assert ssh_public_key and not ssh_public_key.startswith("/"), (
        f"expected public-key contents, not a path: {ssh_public_key!r}"
    )
    return (
        "#cloud-config\n"
        "users:\n"
        f"  - name: {ssh_user}\n"
        "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
        "    shell: /bin/bash\n"
        "    ssh_authorized_keys:\n"
        f"      - {ssh_public_key}\n"
    )


def _create_argv(
    *,
    name: str,
    platform: str,
    preset: str,
    disk_gib: int,
    user_data: str,
    region_hint: str | None,
) -> list[str]:
    # VERIFY: byte-exact flags against the real `nebius compute instance
    # create`. Structure is grounded in the Terraform resource: platform +
    # preset, an INLINE NETWORK_SSD boot disk from the cuda image family,
    # a static public IP, and cloud-init user-data. Tests assert the
    # structural invariants, not the exact strings.
    argv = [
        "compute",
        "instance",
        "create",
        "--name",
        name,
        "--platform",
        platform,
        "--preset",
        preset,
        "--boot-disk-type",
        "NETWORK_SSD",
        "--boot-disk-size-bytes",
        str(disk_gib * GIB),
        "--boot-disk-image-family",
        NEBIUS_IMAGE_FAMILY,
        "--public-ip-static",
        "--user-data",
        user_data,
        "--format",
        "json",
    ]
    if region_hint:
        argv += ["--region", region_hint]
    return argv


def _parse_instance_id(stdout: str) -> str:
    """Pull the instance id out of `instance create --format json` output."""
    try:
        d = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise NebiusSpawnError(f"could not parse instance-create JSON: {e}") from e
    # Nebius wraps the resource under metadata.id (VERIFY against real output).
    rid = (d.get("metadata") or {}).get("id") or d.get("id")
    if not rid:
        raise NebiusSpawnError(f"instance-create JSON had no id: {stdout[:200]}")
    return str(rid)


def _parse_instance_info(stdout: str) -> _InstanceInfo:
    """Extract (state, public_ip, region) from `instance get --format json`."""
    d = json.loads(stdout)
    status = d.get("status") or {}
    state = str(status.get("state", "") or "")
    region = (d.get("metadata") or {}).get("region") or status.get("region")
    public_ip = None
    # VERIFY shape: status.network_interfaces[].public_ip_address.address
    for nic in status.get("network_interfaces") or []:
        addr = (nic.get("public_ip_address") or {}).get("address")
        if addr:
            public_ip = str(addr)
            break
    return _InstanceInfo(state=state, public_ip=public_ip, region=region)


def _looks_like_not_found(proc: subprocess.CompletedProcess[str]) -> bool:
    blob = ((proc.stderr or "") + (proc.stdout or "")).lower()
    return "not found" in blob or "notfound" in blob or "does not exist" in blob


class NebiusSpawnError(RuntimeError):
    """Spawn produced output we couldn't interpret."""


def _default_run_cli(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Default runner: shell out to the real ``nebius`` CLI."""
    return subprocess.run(
        ["nebius", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
