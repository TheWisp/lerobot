# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""HostProvider protocol — the abstraction over vendor-specific VM lifecycles.

The protocol is intentionally narrow:

  - estimate_cost(spec) → CostSnapshot          (pure, no side effects)
  - spawn(spec)         → HostHandle             (creates VM, waits for SSH)
  - destroy(handle)     → None                   (destroys VM + disk + IP)
  - verify_destroyed(handle) → bool              (idempotency check)
  - current_cost(handle) → CostSnapshot          (accrued $ since spawn)

What is deliberately NOT in the protocol:

  - stop() / start()  — Lambda has neither; including would force a
    fiction on vendors that don't support it. Ephemeral mode never
    stops; it destroys. Stopped-but-not-destroyed VMs cause the disk-
    billing-surprise the whole design is meant to prevent.
  - ssh_run() / ssh_tail() — vendor-agnostic. SshConnection + the polling
    loop in :mod:`lerobot.gui.training.worker` live above the protocol.
  - Auth flow — each provider reads its own credential source (e.g.
    `~/.nebius/`, RunPod env var, AWS profile). The GUI's API-key UI
    is a separate concern from the protocol.
  - Upload/download — checkpoints flow through HF Hub (see the design
    doc's "Recovery from preemption" section). Providers don't move files.
  - Preemption events — handled by :class:`PollScheduler` (3 fails →
    disconnected, 10 → lost contact). Providers don't emit events.

Vendor-agnostic Persistent mode (BYO SSH) implements this protocol as a
degenerate provider: spawn is "not supported" (the user provided the
SSH endpoint by hand); destroy is a no-op (the user owns the VM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# Narrow union — extend as we add providers. Adding a new vendor here
# AND registering it in training_providers/__init__.py is the full plug-in
# checklist.
ProviderId = Literal["persistent", "nebius"]


# Canonical GPU kinds the GUI exposes. Vendor-specific SKUs map onto these
# in each provider implementation (e.g., Nebius's "gpu-l40s-a" → "L40S",
# RunPod's "NVIDIA L40S" → "L40S"). Keeps the UI dropdown sane.
GpuKind = Literal[
    "RTX_4090",
    "L4",
    "L40S",
    "A100_40",
    "A100_80",
    "H100",
    "H200",
    "B200",
]


@dataclass(frozen=True, kw_only=True)
class SpawnSpec:
    """What the user asked for. Vendor-neutral.

    Pre: ``ttl_seconds`` > 0. ``estimated_cost_ceiling_usd`` > 0. The
    estimator must agree (within the spec's ceiling) before spawn is
    called — provider rejects spec otherwise.
    """

    gpu: GpuKind
    gpu_count: int = 1
    preemptible: bool = True  # Ephemeral defaults to spot/preemptible
    disk_gib: int = 100
    image: str  # docker ref the training container pulls
    region_hint: str | None = None  # provider may ignore; e.g. "us", "eu"
    # Hard kill — REQUIRED. The provider must enforce this server-side
    # (e.g., Nebius scheduled-delete async op; RunPod --terminate-after)
    # so a runaway VM dies even if our GUI server is gone.
    ttl_seconds: int
    # Refuse to spawn above this. Computed cost = compute_hourly *
    # (ttl_seconds / 3600) + storage_monthly * disk_gib * (ttl_seconds / 730h).
    estimated_cost_ceiling_usd: float


@dataclass(frozen=True, kw_only=True)
class HostHandle:
    """Live handle to a running host. The SSH path consumes this.

    All vendors converge on this shape — provider-specific identifiers
    are captured in ``provider_resource_id`` (opaque to the SSH layer).

    Pre: ``ssh_host:ssh_port`` reachable from the GUI server. ``ssh_user``
    can log in with ``ssh_key_path``.
    Post: ``destroy(handle)`` returns the system to a state where
    ``verify_destroyed(handle)`` returns True.
    """

    provider: ProviderId
    provider_resource_id: str  # vendor's VM/pod id; required for destroy()
    ssh_host: str
    ssh_port: int
    ssh_user: str = "root"
    ssh_key_path: str  # identity key for SSH; provider may have generated it
    # Persistent storage that survives this handle (None for fully-ephemeral).
    # Used so a follow-up Ephemeral spawn can reuse e.g. the HF cache.
    persistent_volume_id: str | None = None
    region: str
    # Provider's hard-TTL deadline as unix timestamp — never None. The GUI
    # uses this to display "auto-destroys at HH:MM" and as a sanity check.
    expires_at_unix: int


@dataclass(frozen=True, kw_only=True)
class CostSnapshot:
    """Cost numbers for UI display + spend-cap guard.

    Pre: rates are non-negative.
    """

    compute_hourly_usd: float
    storage_monthly_usd_per_gib: float
    # Accrued cost since spawn. For estimate_cost() (pre-spawn), this is
    # the projected total at TTL. For current_cost(handle), this is the
    # rate × elapsed wall-clock since spawn (best-effort; provider may
    # have a more accurate billing API).
    accrued_usd_estimate: float


@runtime_checkable
class HostProvider(Protocol):
    """Vendor adapter. Implementations live one-per-file in this package.

    Conformance is checked by ``isinstance(obj, HostProvider)`` thanks
    to ``runtime_checkable``.
    """

    id: ProviderId
    display_name: str

    def estimate_cost(self, spec: SpawnSpec) -> CostSnapshot:
        """Pure function. No API call. Caller uses this to reject expensive
        specs before any side effect."""
        ...

    def spawn(self, spec: SpawnSpec) -> HostHandle:
        """Provision the VM, attach disk, allocate public IP, wait until
        SSH is reachable. The provider MUST configure server-side TTL so
        the VM dies even if we lose control of it.

        Pre: ``estimate_cost(spec).accrued_usd_estimate <=
              spec.estimated_cost_ceiling_usd``.
        Post: returned handle's ssh_host:ssh_port accepts SSH with the
        returned ssh_key_path.
        """
        ...

    def destroy(self, handle: HostHandle) -> None:
        """Idempotent. Destroys VM + boot disk + public IP allocation.

        Does NOT touch ``handle.persistent_volume_id`` — that's lifecycled
        separately by the named-volume API on each vendor.
        """
        ...

    def verify_destroyed(self, handle: HostHandle) -> bool:
        """Re-list vendor resources matching this handle.

        Returns True iff nothing identifying this handle remains billable.
        Automates the "always check Compute → Disks after teardown"
        hygiene rule.
        """
        ...

    def current_cost(self, handle: HostHandle) -> CostSnapshot:
        """Best-effort cost-since-spawn for UI display.

        May estimate locally if the provider's billing API is async.
        """
        ...
