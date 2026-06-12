# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for src/lerobot/gui/training/providers/.

Covers:
  - Protocol conformance via runtime_checkable
  - PersistentSshProvider degenerate behavior (no-op destroy, no-op cost,
    spawn raises, handle_from_profile constructs the right HostHandle)
  - NebiusProvider cost-math correctness (the table-driven part is the
    only real logic in C1; spawn/destroy stubbed for C2)
  - Registry: get_provider / list_providers
"""

from __future__ import annotations

import time

import pytest

from lerobot.gui.training.jobs import HostProfile
from lerobot.gui.training.providers import (
    NebiusProvider,
    PersistentSshProvider,
    get_provider,
    list_providers,
)
from lerobot.gui.training.providers.nebius import (
    NEBIUS_DISK_WARN_GIB,
    NEBIUS_GPU_HOURLY_USD,
    NEBIUS_SSD_MONTHLY_USD_PER_GIB,
)
from lerobot.gui.training.providers.protocol import (
    CostSnapshot,
    HostHandle,
    HostProvider,
    SpawnSpec,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _example_spec(**overrides) -> SpawnSpec:
    defaults: dict = {
        "gpu": "L40S",
        "gpu_count": 1,
        "preemptible": True,
        "disk_gib": 100,
        "image": "ghcr.io/thewisp/lerobot-training:latest",
        "ttl_seconds": 24 * 3600,
        "estimated_cost_ceiling_usd": 25.0,
    }
    defaults.update(overrides)
    return SpawnSpec(**defaults)


def _example_handle(**overrides) -> HostHandle:
    defaults: dict = {
        "provider": "persistent",
        "provider_resource_id": "test-id",
        "ssh_host": "10.0.0.5",
        "ssh_port": 22,
        "ssh_user": "feit",
        "region": "user-managed",
        "expires_at_unix": int(time.time()) + 3600,
    }
    defaults.update(overrides)
    return HostHandle(**defaults)


# ── Protocol conformance ────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_persistent_satisfies_protocol(self):
        assert isinstance(PersistentSshProvider(), HostProvider)

    def test_nebius_satisfies_protocol(self):
        assert isinstance(NebiusProvider(), HostProvider)

    def test_all_registered_providers_satisfy_protocol(self):
        for pid in list_providers():
            provider = get_provider(pid)
            assert isinstance(provider, HostProvider), (
                f"registered provider {pid!r} does not satisfy HostProvider protocol"
            )

    def test_provider_has_id_and_display_name(self):
        for pid in list_providers():
            provider = get_provider(pid)
            assert provider.id == pid
            assert provider.display_name  # non-empty


# ── Registry ────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown provider id"):
            get_provider("not-a-vendor")

    def test_get_provider_returns_correct_type(self):
        assert isinstance(get_provider("persistent"), PersistentSshProvider)
        assert isinstance(get_provider("nebius"), NebiusProvider)

    def test_list_providers_includes_v1_set(self):
        providers = list_providers()
        assert "persistent" in providers
        assert "nebius" in providers

    def test_list_providers_sorted(self):
        providers = list_providers()
        assert providers == sorted(providers)


# ── PersistentSshProvider ───────────────────────────────────────────────────


class TestPersistentProvider:
    def test_cost_is_zero(self):
        snap = PersistentSshProvider().estimate_cost(_example_spec())
        assert snap.compute_hourly_usd == 0.0
        assert snap.storage_monthly_usd_per_gib == 0.0
        assert snap.accrued_usd_estimate == 0.0

    def test_spawn_raises_not_implemented(self):
        # Persistent hosts are added by the user, not spawned by the GUI.
        with pytest.raises(NotImplementedError, match="Persistent hosts are added"):
            PersistentSshProvider().spawn(_example_spec())

    def test_destroy_raises(self):
        """User owns the VM; the GUI never destroys it. Raising (instead
        of a silent no-op) means a caller that reaches for destroy() on
        the wrong host type fails loudly rather than believing resources
        were freed."""
        with pytest.raises(NotImplementedError, match="user-owned"):
            PersistentSshProvider().destroy(_example_handle())

    def test_verify_destroyed_raises(self):
        """Same fail-fast rationale as destroy(): a vacuous True would
        let a buggy teardown sweep report all-clean on a host type it
        should never have been asked about."""
        with pytest.raises(NotImplementedError, match="user-owned"):
            PersistentSshProvider().verify_destroyed(_example_handle())

    def test_current_cost_is_zero(self):
        snap = PersistentSshProvider().current_cost(_example_handle())
        assert snap.accrued_usd_estimate == 0.0

    def test_handle_from_profile(self):
        """HostProfile (BYO SSH) → HostHandle, with no private-key fields
        on either end. Auth is resolved by the user's local SSH setup
        (~/.ssh/config / agent / default keys); see DESIGN.md §
        Authentication."""
        profile = HostProfile(
            name="lab-h100",
            ssh_user="feit",
            ssh_host="192.168.1.50",
            ssh_port=22,
            persistent_volume="/data",
        )
        handle = PersistentSshProvider.handle_from_profile(profile)
        assert handle.provider == "persistent"
        assert handle.ssh_host == "192.168.1.50"
        assert handle.ssh_user == "feit"
        assert handle.ssh_port == 22
        assert handle.persistent_volume_id == "/data"
        # Persistent hosts effectively never expire (user owns lifecycle).
        # Sanity check: at least 50 years in the future.
        assert handle.expires_at_unix > int(time.time()) + 50 * 365 * 24 * 3600


# ── NebiusProvider — cost math (the only real logic in C1) ──────────────────


class TestNebiusCostMath:
    def test_preemptible_l40s_24h(self):
        """1 × L40S preemptible × 24h + 100 GiB disk × 24h.

        L40S preemptible = $0.90/hr; 24h = $21.60.
        Disk = $0.10/GiB/month × 100 GiB × (24/730)h = $0.329.
        Total = $21.93.
        """
        spec = _example_spec(gpu="L40S", preemptible=True, ttl_seconds=24 * 3600)
        snap = NebiusProvider().estimate_cost(spec)
        assert snap.compute_hourly_usd == 0.90
        assert snap.storage_monthly_usd_per_gib == 0.10
        # Compute + storage, within rounding error
        expected = 0.90 * 24 + 0.10 * 100 * (24 / 730)
        assert abs(snap.accrued_usd_estimate - expected) < 0.01

    def test_on_demand_more_expensive_than_preemptible(self):
        spec_preempt = _example_spec(preemptible=True)
        spec_ondemand = _example_spec(preemptible=False)
        cost_p = NebiusProvider().estimate_cost(spec_preempt).accrued_usd_estimate
        cost_o = NebiusProvider().estimate_cost(spec_ondemand).accrued_usd_estimate
        assert cost_o > cost_p

    def test_h100_is_more_expensive_than_l40s(self):
        cost_l40s = NebiusProvider().estimate_cost(_example_spec(gpu="L40S")).accrued_usd_estimate
        cost_h100 = NebiusProvider().estimate_cost(_example_spec(gpu="H100")).accrued_usd_estimate
        assert cost_h100 > cost_l40s

    def test_multi_gpu_scales_compute(self):
        cost_1 = NebiusProvider().estimate_cost(_example_spec(gpu="L40S", gpu_count=1)).accrued_usd_estimate
        cost_4 = NebiusProvider().estimate_cost(_example_spec(gpu="L40S", gpu_count=4)).accrued_usd_estimate
        # Storage is the same → compute portion quadruples
        # cost = compute * 4 + storage = (cost_1 - storage) * 4 + storage
        # so cost_4 should be between 3x and 4x cost_1
        assert 3.0 < cost_4 / cost_1 < 4.5

    def test_longer_ttl_scales_cost_linearly(self):
        cost_1h = NebiusProvider().estimate_cost(_example_spec(ttl_seconds=3600)).accrued_usd_estimate
        cost_24h = NebiusProvider().estimate_cost(_example_spec(ttl_seconds=24 * 3600)).accrued_usd_estimate
        # Both compute AND storage scale linearly with time, so ratio ≈ 24.
        assert 23.0 < cost_24h / cost_1h < 25.0

    def test_unsupported_gpu_raises(self):
        with pytest.raises(ValueError, match="doesn't offer GPU kind"):
            NebiusProvider().estimate_cost(_example_spec(gpu="RTX_4090"))


# ── NebiusProvider — defensive helpers (the lessons from the bill) ──────────


class TestNebiusDefensiveDefaults:
    def test_disk_warning_fires_above_threshold(self):
        warning = NebiusProvider.disk_warning(640)  # the 640 GB that bit us
        assert warning is not None
        assert "640" in warning
        assert "regardless of VM state" in warning

    def test_disk_warning_silent_below_threshold(self):
        assert NebiusProvider.disk_warning(100) is None
        assert NebiusProvider.disk_warning(NEBIUS_DISK_WARN_GIB) is None

    def test_disk_warning_fires_at_default_terraform_disk(self):
        # The 1.28 TB Terraform default that caused yesterday's bill
        warning = NebiusProvider.disk_warning(1280)
        assert warning is not None

    def test_platform_lookup(self):
        assert NebiusProvider.platform_for("L40S") == "gpu-l40s-a"
        assert NebiusProvider.platform_for("H100") == "gpu-h100-sxm5"

    def test_platform_lookup_unsupported_raises(self):
        with pytest.raises(ValueError, match="no platform mapping"):
            NebiusProvider.platform_for("RTX_4090")

    def test_preset_lookup(self):
        assert NebiusProvider.preset_for("L40S") == "1gpu-8vcpu-32gb"
        assert NebiusProvider.preset_for("H100") == "1gpu-16vcpu-200gb"


# ── NebiusProvider — lifecycle methods stubbed for C2 ──────────────────────


class TestNebiusStubbedLifecycle:
    """C1 lands cost math; spawn/destroy land in C2. These tests pin that
    the stubs explicitly raise (vs. silently returning None) so a caller
    that wires this up before C2 lands fails loudly."""

    def test_spawn_raises_notimplemented(self):
        with pytest.raises(NotImplementedError, match="lands in C2"):
            NebiusProvider().spawn(_example_spec())

    def test_destroy_raises_notimplemented(self):
        with pytest.raises(NotImplementedError, match="lands in C2"):
            NebiusProvider().destroy(_example_handle(provider="nebius"))

    def test_verify_destroyed_raises_notimplemented(self):
        with pytest.raises(NotImplementedError, match="lands in C2"):
            NebiusProvider().verify_destroyed(_example_handle(provider="nebius"))

    def test_current_cost_raises_notimplemented(self):
        with pytest.raises(NotImplementedError, match="lands in C2"):
            NebiusProvider().current_cost(_example_handle(provider="nebius"))


# ── Pricing table invariants ────────────────────────────────────────────────


class TestPricingTableInvariants:
    """Catch typos when refreshing prices."""

    def test_preemptible_always_cheaper_than_on_demand(self):
        for gpu, (preempt, ondemand) in NEBIUS_GPU_HOURLY_USD.items():
            assert preempt < ondemand, f"{gpu}: preempt ${preempt} should be less than on-demand ${ondemand}"

    def test_all_prices_positive(self):
        for gpu, (preempt, ondemand) in NEBIUS_GPU_HOURLY_USD.items():
            assert preempt > 0, f"{gpu}: preemptible price not positive"
            assert ondemand > 0, f"{gpu}: on-demand price not positive"

    def test_ssd_price_set(self):
        assert NEBIUS_SSD_MONTHLY_USD_PER_GIB > 0


# ── CostSnapshot smoke ──────────────────────────────────────────────────────


class TestCostSnapshot:
    def test_is_frozen(self):
        import dataclasses

        snap = CostSnapshot(
            compute_hourly_usd=1.0,
            storage_monthly_usd_per_gib=0.10,
            accrued_usd_estimate=10.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.compute_hourly_usd = 2.0  # type: ignore[misc] — frozen invariant
