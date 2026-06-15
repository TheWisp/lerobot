# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training host providers — pluggable VM-lifecycle backends.

Each provider implements :class:`HostProvider` for one vendor. The
GUI's application code (training_worker, API endpoints, frontend)
talks only to the protocol; vendor specifics are contained per-file.

Adding a new vendor: drop a new file (e.g. ``runpod.py``), register it
in :func:`get_provider`. No changes to application code.

For the full design see ``src/lerobot/gui/training/DESIGN.md`` § "HostProvider
protocol (Phase 2.5)".
"""

from __future__ import annotations

from lerobot.gui.training.providers.nebius import NebiusProvider
from lerobot.gui.training.providers.persistent import PersistentSshProvider
from lerobot.gui.training.providers.protocol import (
    HostHandle,
    HostProvider,
    ProviderId,
    SpawnSpec,
)

__all__ = [
    "HostHandle",
    "HostProvider",
    "ProviderId",
    "SpawnSpec",
    "NebiusProvider",
    "PersistentSshProvider",
    "get_provider",
    "list_providers",
]


# Registry. Keep narrow — adding a vendor adds one line.
_PROVIDERS: dict[str, type] = {
    "persistent": PersistentSshProvider,
    "nebius": NebiusProvider,
}


def get_provider(provider_id: str, *, iam_token: str | None = None) -> HostProvider:
    """Resolve a vendor id to an instantiated HostProvider.

    ``iam_token`` is the caller's per-user vendor token (forwarded from the
    request header for LAN multi-user). Providers that take one use it to
    scope SDK calls to that user; those that don't (persistent) ignore it.
    None → the provider falls back to ambient credentials.

    Pre: ``provider_id`` is one of the registered providers.
    Post: a fresh provider instance for one spawn/destroy cycle.
    """
    if provider_id not in _PROVIDERS:
        raise ValueError(f"unknown provider id: {provider_id!r}. Registered providers: {sorted(_PROVIDERS)}")
    cls = _PROVIDERS[provider_id]
    if cls is NebiusProvider:
        return cls(iam_token=iam_token)
    return cls()


def list_providers() -> list[str]:
    """Return the registered provider ids, sorted for stable UI display."""
    return sorted(_PROVIDERS)
