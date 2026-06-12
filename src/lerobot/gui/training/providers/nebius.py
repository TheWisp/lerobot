# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Nebius Compute provider — Ephemeral VM lifecycle via the ``nebius`` CLI.

Stub implementation in this commit (C1). Real ``nebius`` CLI subprocess
calls land in C2. The structure here pins:

  - Class shape conforms to :class:`HostProvider`.
  - SKU mapping table (GpuKind → Nebius platform string) lives in one place.

No cost estimation by design — vendor prices drift and there is no
price-catalog API to validate against. The dialog links to Nebius's
pricing page; spend protection is mechanical (hard TTL, disk-size
warning, auto-destroy). See DESIGN.md § What we don't try to do.

Auth: inherits from ``~/.nebius/`` (user's existing ``nebius`` CLI
profile). The GUI does NOT prompt for credentials in MVP — same pattern
as kubectl / gh / aws CLI inheritance.

Defensive defaults (lessons from the 2026-06-05 disk-billing bill):

  - Inline-managed boot disk only (cascade-delete with the VM). Never
    standalone disks unless the user explicitly opts in. Standalone
    disks survive VM-delete and accrue silent charges.
  - Warn on ``disk_gib > 256`` — most LeRobot training fits in <100 GB.
  - Idempotency: ``get-by-name`` before ``create`` (Nebius has no
    ``--idempotency-key``; the recommended pattern is check-then-create).
  - Hard TTL: schedule a delete async-op at ``spawn_time + ttl_seconds``
    via ``nebius compute instance delete --schedule`` so the VM dies
    even if our GUI server dies.
"""

from __future__ import annotations

from lerobot.gui.training.providers.protocol import (
    GpuKind,
    HostHandle,
    SpawnSpec,
)

# ── SKU mapping ─────────────────────────────────────────────────────────────


# Canonical GpuKind → Nebius platform string + matching preset. Lookup is
# explicit — vendors that don't offer a given GPU raise at lookup time,
# before spawn. Update when Nebius adds GPU SKUs.
NEBIUS_GPU_PLATFORMS: dict[GpuKind, str] = {
    "L4": "gpu-l4-a",
    "L40S": "gpu-l40s-a",
    "H100": "gpu-h100-sxm5",
    "H200": "gpu-h200-sxm5",
}


# Preset must match the GPU platform. For MVP, each GPU has one
# canonical single-GPU preset; multi-GPU presets land later.
NEBIUS_DEFAULT_PRESET: dict[GpuKind, str] = {
    "L4": "1gpu-8vcpu-32gb",
    "L40S": "1gpu-8vcpu-32gb",
    "H100": "1gpu-16vcpu-200gb",
    "H200": "1gpu-16vcpu-200gb",
}


# Soft cap to surface a warning. The 1.28 TB default that bit us was
# Nebius's web-form default; we warn on anything above ~2.5x what a
# typical training run actually needs.
NEBIUS_DISK_WARN_GIB = 256


# ── Provider ────────────────────────────────────────────────────────────────


class NebiusProvider:
    """Nebius Compute provider.

    C1 (this commit): stub — methods present, shapes correct, no CLI
    subprocess yet. Tests pin the SKU tables + defensive defaults.

    C2: real CLI integration. ``spawn`` shells out to ``nebius compute
    instance create`` and waits for the VM to become reachable. ``destroy``
    issues delete + verifies via ``verify_destroyed``.
    """

    id = "nebius"
    display_name = "Nebius (auto-managed)"

    def __init__(self) -> None:
        # In C2, this is where we'd resolve auth (default to inheriting
        # ~/.nebius/ profile, override via per-host config). For C1, no-op.
        pass

    # ── Lifecycle (stubbed in C1; real impl in C2) ──────────────────────

    def spawn(self, spec: SpawnSpec) -> HostHandle:
        raise NotImplementedError(
            "Nebius spawn lands in C2 — see src/lerobot/gui/training/DESIGN.md "
            "§ Phase 6.5 (Ephemeral mode + first managed provider). C1 "
            "lands the protocol + SKU tables; C2 wires the nebius CLI."
        )

    def destroy(self, handle: HostHandle) -> None:
        raise NotImplementedError("Nebius destroy lands in C2.")

    def verify_destroyed(self, handle: HostHandle) -> bool:
        raise NotImplementedError("Nebius verify_destroyed lands in C2.")

    # ── Helpers exposed for testing + UI surfacing ──────────────────────

    @staticmethod
    def disk_warning(disk_gib: int) -> str | None:
        """Return a warning string if ``disk_gib`` is unusually large.

        Surfaces the lesson from the bill investigation: the web-form
        default boot-disk size on Nebius is enormous (~1 TB) and most
        LeRobot training fits in <100 GB. Disk bills continuously from
        create-to-delete regardless of VM state.
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
