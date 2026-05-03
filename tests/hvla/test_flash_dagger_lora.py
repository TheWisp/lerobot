"""Tests for flash-DAgger LoRA module: attach, hash, save/load, merge, peel."""

from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.policies.hvla.flash_dagger.lora import (
    apply_lora_to_decoder,
    compute_base_hash,
    extract_lora_state_dict,
    load_lora_state_dict,
    lora_layer_diagnostics,
    merge_lora_into_base,
    peel_lora,
)


class _MockDecoderLayer(nn.Module):
    """Minimal HVLA-decoder-layer-shaped module: self_attn, multihead_attn, linear1, linear2."""

    def __init__(self, d: int = 32, nhead: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.linear1 = nn.Linear(d, 4 * d)
        self.linear2 = nn.Linear(4 * d, d)


class _MockInner(nn.Module):
    def __init__(self, n_layers: int = 2, d: int = 32):
        super().__init__()
        self.decoder_layers = nn.ModuleList(_MockDecoderLayer(d) for _ in range(n_layers))


class _MockPolicy(nn.Module):
    def __init__(self, n_layers: int = 2, d: int = 32):
        super().__init__()
        self.model = _MockInner(n_layers, d)


def _make_policy() -> _MockPolicy:
    torch.manual_seed(0)
    return _MockPolicy()


def test_apply_lora_zero_init_preserves_forward():
    """At attach time, B=0 so output must equal the base output exactly."""
    policy = _make_policy()
    layer = policy.model.decoder_layers[0]

    x = torch.randn(2, 5, 32)
    base_self_out, _ = layer.self_attn(x, x, x)
    base_linear1 = layer.linear1(x)

    apply_lora_to_decoder(policy, rank=4, alpha=8.0, ffn=True)

    new_self_out, _ = layer.self_attn(x, x, x)
    new_linear1 = layer.linear1(x)

    torch.testing.assert_close(new_self_out, base_self_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new_linear1, base_linear1, rtol=1e-5, atol=1e-5)


def test_apply_lora_only_lora_params_trainable():
    policy = _make_policy()
    apply_lora_to_decoder(policy, rank=4, alpha=8.0)
    for name, p in policy.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        assert p.requires_grad == is_lora, f"{name}: requires_grad={p.requires_grad}, is_lora={is_lora}"


def test_apply_lora_returns_param_counts():
    policy = _make_policy()
    n_lora, n_total = apply_lora_to_decoder(policy, rank=4, alpha=8.0)
    assert n_lora > 0
    assert n_total > n_lora


def test_compute_base_hash_stable_for_identical_policy():
    p1 = _make_policy()
    p2 = _make_policy()
    apply_lora_to_decoder(p1, rank=4, alpha=8.0)
    apply_lora_to_decoder(p2, rank=4, alpha=8.0)
    # Same seed → same base weights; hash should match
    assert compute_base_hash(p1) == compute_base_hash(p2)


def test_compute_base_hash_changes_with_base_weights():
    p1 = _make_policy()
    apply_lora_to_decoder(p1, rank=4, alpha=8.0)
    h1 = compute_base_hash(p1)
    # Mutate a base param
    with torch.no_grad():
        p1.model.decoder_layers[0].linear1.base.weight.add_(1.0)
    h2 = compute_base_hash(p1)
    assert h1 != h2


def test_compute_base_hash_invariant_to_lora_state():
    p1 = _make_policy()
    apply_lora_to_decoder(p1, rank=4, alpha=8.0)
    h1 = compute_base_hash(p1)
    # Mutate LoRA params (they shouldn't be in the hash)
    for n, p in p1.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            with torch.no_grad():
                p.add_(0.5)
    h2 = compute_base_hash(p1)
    assert h1 == h2


def test_extract_then_load_roundtrip():
    p1 = _make_policy()
    apply_lora_to_decoder(p1, rank=4, alpha=8.0)
    # Mutate LoRA params (so it's not all zeros)
    for n, p in p1.named_parameters():
        if "lora_B" in n:
            with torch.no_grad():
                p.normal_(std=0.1)
    sd = extract_lora_state_dict(p1)

    p2 = _make_policy()
    apply_lora_to_decoder(p2, rank=4, alpha=8.0)
    load_lora_state_dict(p2, sd, metadata=None, strict_hash=False)

    sd2 = extract_lora_state_dict(p2)
    assert sd.keys() == sd2.keys()
    for k in sd:
        torch.testing.assert_close(sd[k], sd2[k])


def test_peel_makes_forward_match_base_again():
    policy = _make_policy()
    layer = policy.model.decoder_layers[0]
    x = torch.randn(2, 5, 32)

    apply_lora_to_decoder(policy, rank=4, alpha=8.0, ffn=True)
    base_self_out, _ = layer.self_attn(x, x, x)
    base_linear1 = layer.linear1(x)

    # Mutate LoRA so it's no longer identity
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.1)
    after_self_out, _ = layer.self_attn(x, x, x)
    assert not torch.allclose(after_self_out, base_self_out, rtol=1e-3, atol=1e-3)

    n_peeled = peel_lora(policy)
    assert n_peeled > 0

    peeled_self_out, _ = layer.self_attn(x, x, x)
    peeled_linear1 = layer.linear1(x)
    torch.testing.assert_close(peeled_self_out, base_self_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(peeled_linear1, base_linear1, rtol=1e-5, atol=1e-5)


def test_merge_into_base_preserves_forward():
    """After merge: forward(x) is unchanged, but BA reset to identity."""
    policy = _make_policy()
    apply_lora_to_decoder(policy, rank=4, alpha=8.0, ffn=True)
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.1)

    layer = policy.model.decoder_layers[0]
    x = torch.randn(2, 5, 32)
    pre_merge_out, _ = layer.self_attn(x, x, x)
    pre_merge_linear1 = layer.linear1(x)

    n_merged = merge_lora_into_base(policy)
    assert n_merged > 0

    post_merge_out, _ = layer.self_attn(x, x, x)
    post_merge_linear1 = layer.linear1(x)

    torch.testing.assert_close(pre_merge_out, post_merge_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(pre_merge_linear1, post_merge_linear1, rtol=1e-4, atol=1e-4)

    # After merge, lora_B should be zero (BA reset to identity-equivalent)
    for n, p in policy.named_parameters():
        if "lora_B" in n:
            assert torch.allclose(p, torch.zeros_like(p))


def test_lora_layer_diagnostics_counts_layers():
    policy = _make_policy(n_layers=2)
    apply_lora_to_decoder(policy, rank=4, alpha=8.0, ffn=True)
    rows = lora_layer_diagnostics(policy)
    # n_layers * (q,k,v,o for self_attn + q,k,v,o for cross_attn + linear1 + linear2)
    # = 2 * (4 + 4 + 1 + 1) = 20
    assert len(rows) == 2 * (4 + 4 + 2)
    for r in rows:
        assert r["rank"] == 4
        # At init, BA = 0 so frobenius == 0
        assert r["frobenius"] == 0.0
        assert r["effective_rank"] == 0


def test_save_load_with_hash_mismatch_raises(tmp_path):
    """Loading a LoRA against a different base must raise."""
    from lerobot.policies.hvla.flash_dagger.persistence import load_lora, save_lora

    p1 = _make_policy()
    apply_lora_to_decoder(p1, rank=4, alpha=8.0)
    save_lora(p1, tmp_path, cycle=0, rank=4, alpha=8.0, apply_to_ffn=True)

    p2 = _make_policy()
    # Different base by mutating a weight before LoRA attach
    with torch.no_grad():
        p2.model.decoder_layers[0].linear1.weight.add_(0.5)
    apply_lora_to_decoder(p2, rank=4, alpha=8.0)

    cycle_path = tmp_path / "lora" / "cycle_0000.pt"
    try:
        load_lora(p2, cycle_path, strict_hash=True)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected hash-mismatch assertion"
