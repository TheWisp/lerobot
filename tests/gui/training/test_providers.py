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
  - PersistentSshProvider degenerate behavior (every lifecycle method
    raises; handle_from_profile constructs the right HostHandle)
  - NebiusProvider SKU tables + defensive defaults (spawn/destroy
    stubbed for C2). No cost math by design — see DESIGN.md § What we
    don't try to do.
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from lerobot.gui.training.jobs import HostProfile
from lerobot.gui.training.providers import (
    NebiusProvider,
    PersistentSshProvider,
    get_provider,
    list_providers,
)
from lerobot.gui.training.providers.nebius import NEBIUS_DISK_WARN_GIB
from lerobot.gui.training.providers.protocol import (
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
        # L40S is the VERIFIED value (from the GPU-smoke Terraform).
        assert NebiusProvider.platform_for("L40S") == "gpu-l40s-d"
        assert NebiusProvider.platform_for("H100") == "gpu-h100-sxm5"

    def test_platform_lookup_unsupported_raises(self):
        with pytest.raises(ValueError, match="no platform mapping"):
            NebiusProvider.platform_for("RTX_4090")

    def test_preset_lookup(self):
        assert NebiusProvider.preset_for("L40S") == "1gpu-16vcpu-96gb"
        assert NebiusProvider.preset_for("H100") == "1gpu-16vcpu-200gb"


# ── NebiusProvider — lifecycle via an injected fake CLI ─────────────────────
#
# No live `nebius` binary here, so the provider runs against a fake runner
# returning canned JSON — same pattern as SshClient's fake subprocess. These
# pin control flow + parsing + the structural argv invariants; the byte-exact
# CLI flags are marked VERIFY in the impl and confirmed at the live smoke.


class _FakeCli:
    """Scriptable nebius CLI: maps subcommand (argv[:3]) → CompletedProcess,
    records every call. Sequence-aware for `instance get` so a spawn can go
    PROVISIONING → RUNNING across polls."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self._get_sequence: list[subprocess.CompletedProcess] = []
        self._responses: dict[tuple[str, ...], subprocess.CompletedProcess] = {}

    def set(self, key: tuple[str, ...], stdout="", returncode=0, stderr=""):
        self._responses[key] = subprocess.CompletedProcess([], returncode, stdout, stderr)

    def set_get_sequence(self, procs):
        self._get_sequence = list(procs)

    def __call__(self, argv, timeout):
        self.calls.append(argv)
        verb = tuple(argv[:3])
        if verb == ("compute", "instance", "get") and self._get_sequence:
            return self._get_sequence.pop(0)
        if verb in self._responses:
            return self._responses[verb]
        # Distinguish create vs scheduled-delete (both are instance/delete or
        # instance/create); default OK for anything unscripted.
        return subprocess.CompletedProcess(argv, 0, "{}", "")


def _ok(stdout, returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def _running_json(ip="195.242.29.74", region="eu-north1"):
    return json.dumps(
        {
            "metadata": {"id": "computeinstance-abc", "region": region},
            "status": {
                "state": "RUNNING",
                "network_interfaces": [{"public_ip_address": {"address": ip}}],
            },
        }
    )


def _nebius(**kw):
    kw.setdefault("ssh_public_key", "ssh-ed25519 AAAAFAKEKEY test@host")
    kw.setdefault("poll_interval_s", 0)
    kw.setdefault("sleep", lambda _s: None)
    return NebiusProvider(**kw)


class TestNebiusSpawn:
    def test_spawn_happy_path_returns_handle(self):
        cli = _FakeCli()
        cli.set(
            ("compute", "instance", "create"), stdout=json.dumps({"metadata": {"id": "computeinstance-abc"}})
        )
        cli.set_get_sequence(
            [
                _ok(
                    json.dumps(
                        {
                            "metadata": {"id": "computeinstance-abc"},
                            "status": {"state": "PROVISIONING", "network_interfaces": []},
                        }
                    )
                ),
                _ok(_running_json()),
            ]
        )
        handle = _nebius(run_cli=cli).spawn(_example_spec(ttl_seconds=3600))
        assert handle.provider == "nebius"
        assert handle.provider_resource_id == "computeinstance-abc"
        assert handle.ssh_host == "195.242.29.74"
        assert handle.ssh_user == "lerobot"
        assert handle.region == "eu-north1"

    def test_spawn_schedules_ttl_before_polling(self):
        """The hard-TTL delete must be scheduled BEFORE the readiness poll —
        a VM that never comes up must still self-destruct."""
        cli = _FakeCli()
        cli.set(("compute", "instance", "create"), stdout=json.dumps({"metadata": {"id": "i-1"}}))
        cli.set_get_sequence([_ok(_running_json())])
        _nebius(run_cli=cli).spawn(_example_spec(ttl_seconds=3600))
        verbs = [tuple(c[:3]) for c in cli.calls]
        sched_idx = next(i for i, c in enumerate(cli.calls) if "--after-seconds" in c)
        first_get_idx = verbs.index(("compute", "instance", "get"))
        assert sched_idx < first_get_idx

    def test_spawn_ttl_value_propagates(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "create"), stdout=json.dumps({"metadata": {"id": "i-1"}}))
        cli.set_get_sequence([_ok(_running_json())])
        _nebius(run_cli=cli).spawn(_example_spec(ttl_seconds=7200))
        sched = next(c for c in cli.calls if "--after-seconds" in c)
        assert sched[sched.index("--after-seconds") + 1] == "7200"

    def test_spawn_create_argv_has_verified_sku(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "create"), stdout=json.dumps({"metadata": {"id": "i-1"}}))
        cli.set_get_sequence([_ok(_running_json())])
        _nebius(run_cli=cli).spawn(_example_spec(gpu="L40S", disk_gib=50))
        create = next(c for c in cli.calls if tuple(c[:3]) == ("compute", "instance", "create"))
        assert "gpu-l40s-d" in create
        assert "1gpu-16vcpu-96gb" in create
        assert str(50 * (1024**3)) in create  # disk bytes
        # cloud-init carries the pubkey
        ud = create[create.index("--user-data") + 1]
        assert "ssh-ed25519 AAAAFAKEKEY" in ud

    def test_spawn_rejects_nonpositive_ttl(self):
        with pytest.raises(ValueError, match="ttl_seconds"):
            _nebius().spawn(_example_spec(ttl_seconds=0))

    def test_spawn_create_failure_raises(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "create"), returncode=1, stderr="quota exceeded")
        with pytest.raises(Exception, match="create failed"):
            _nebius(run_cli=cli).spawn(_example_spec())

    def test_spawn_times_out_when_never_ready(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "create"), stdout=json.dumps({"metadata": {"id": "i-1"}}))
        # Always PROVISIONING, never RUNNING.
        provisioning = _ok(
            json.dumps(
                {"metadata": {"id": "i-1"}, "status": {"state": "PROVISIONING", "network_interfaces": []}}
            )
        )
        cli.set(("compute", "instance", "get"), stdout=provisioning.stdout)
        clock = iter([0.0, 0.0, 100.0, 700.0])  # last exceeds the 600s default
        prov = _nebius(run_cli=cli, clock=lambda: next(clock))
        with pytest.raises(Exception, match="not reachable"):
            prov.spawn(_example_spec())


class TestNebiusDestroyVerify:
    def test_destroy_idempotent_on_not_found(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "delete"), returncode=1, stderr="instance not found")
        _nebius(run_cli=cli).destroy(_example_handle(provider="nebius"))  # no raise

    def test_destroy_real_failure_raises(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "delete"), returncode=1, stderr="permission denied")
        with pytest.raises(Exception, match="delete failed"):
            _nebius(run_cli=cli).destroy(_example_handle(provider="nebius"))

    def test_verify_destroyed_true_on_not_found(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "get"), returncode=1, stderr="instance does not exist")
        assert _nebius(run_cli=cli).verify_destroyed(_example_handle(provider="nebius")) is True

    def test_verify_destroyed_false_when_still_running(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "get"), stdout=_running_json())
        assert _nebius(run_cli=cli).verify_destroyed(_example_handle(provider="nebius")) is False

    def test_verify_destroyed_true_when_deleting(self):
        cli = _FakeCli()
        cli.set(("compute", "instance", "get"), stdout=json.dumps({"status": {"state": "DELETING"}}))
        assert _nebius(run_cli=cli).verify_destroyed(_example_handle(provider="nebius")) is True


class TestNebiusCloudInit:
    def test_cloud_init_grants_key_and_sudo(self):
        from lerobot.gui.training.providers.nebius import build_cloud_init

        doc = build_cloud_init("lerobot", "ssh-ed25519 AAAAKEY u@h")
        assert doc.startswith("#cloud-config")
        assert "name: lerobot" in doc
        assert "NOPASSWD:ALL" in doc
        assert "ssh-ed25519 AAAAKEY u@h" in doc

    def test_cloud_init_rejects_key_path(self):
        from lerobot.gui.training.providers.nebius import build_cloud_init

        with pytest.raises(AssertionError, match="not a path"):
            build_cloud_init("lerobot", "/home/u/.ssh/id_ed25519.pub")
