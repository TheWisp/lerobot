"""LoRA adapter management for online flash-DAgger.

Wraps the offline LoRA module from hvla/scripts/flash_dagger_lora.py with
the operations the online runtime needs:

  - base-hash binding (refuse to load a LoRA against the wrong base)
  - extract / load just the LoRA params (no base weights in the state-dict)
  - merge LoRA into the base in-place (for consolidation)
  - peel LoRA back to identity (B := 0; forward() == base())
  - per-layer LoRA diagnostics (||BA||_F, effective rank) for Layer B metrics
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn

from lerobot.policies.hvla.scripts.flash_dagger_lora import (
    LoRALinear,
    LoRAMultiheadAttention,
    apply_lora_to_decoder,
)

# Re-export so callers don't need to know the offline location.
__all__ = [
    "LoRALinear",
    "LoRAMultiheadAttention",
    "apply_lora_to_decoder",
    "LoRAMetadata",
    "compute_base_hash",
    "extract_lora_state_dict",
    "load_lora_state_dict",
    "merge_lora_into_base",
    "peel_lora",
    "lora_layer_diagnostics",
]


@dataclass(frozen=True)
class LoRAMetadata:
    """Metadata bound to a saved LoRA state-dict.

    `base_hash` is computed by `compute_base_hash` over the non-LoRA params
    of the policy at the time the LoRA was attached. Loading a LoRA whose
    `base_hash` differs from the current policy's hash is refused to prevent
    silent drift (a LoRA fit against base W_a is meaningless against W_b).
    """

    base_hash: str
    rank: int
    alpha: float
    apply_to_ffn: bool
    n_lora_params: int


def compute_base_hash(policy: nn.Module) -> str:
    """SHA-256 of non-LoRA params (the frozen base).

    Stable as long as the same set of base params with the same shapes and
    values is present. Iteration is `policy.named_parameters()` in
    insertion order, which is deterministic for a given module structure.
    """
    h = hashlib.sha256()
    for name, p in policy.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue
        h.update(name.encode("utf-8"))
        h.update(str(tuple(p.shape)).encode("utf-8"))
        # Hash bytes of the tensor in a deterministic float32 view (small
        # cost vs. the size of the model — runs once per session).
        h.update(p.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def extract_lora_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    """Pull just the lora_A / lora_B params from the policy.

    Shape mirrors `policy.state_dict()` but filtered. Suitable for torch.save.
    """
    sd = {}
    for name, p in policy.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            sd[name] = p.detach().clone()
    return sd


def load_lora_state_dict(
    policy: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: LoRAMetadata | None = None,
    *,
    strict_hash: bool = True,
) -> None:
    """Load a LoRA state-dict into `policy`, verifying the base hash.

    Preconditions:
      - `policy` already has LoRA attached (apply_lora_to_decoder was called)
      - `state_dict` keys are a subset of policy.state_dict() keys
      - if `metadata` is provided and `strict_hash`, `metadata.base_hash`
        must match `compute_base_hash(policy)`

    Postcondition: every LoRA param in `state_dict` is loaded into `policy`.
    Raises if a base-hash mismatch is detected.
    """
    if metadata is not None and strict_hash:
        current = compute_base_hash(policy)
        assert current == metadata.base_hash, (
            f"LoRA base-hash mismatch: saved={metadata.base_hash[:12]}... "
            f"current={current[:12]}... — refusing to load. The base policy "
            f"weights have changed since this LoRA was trained; either reload "
            f"the original base or retrain the LoRA."
        )

    policy_sd = dict(policy.named_parameters())
    missing, unexpected = [], []
    for name, val in state_dict.items():
        if name not in policy_sd:
            unexpected.append(name)
            continue
        with torch.no_grad():
            policy_sd[name].copy_(val.to(policy_sd[name].device, dtype=policy_sd[name].dtype))
    for name in policy_sd:
        if ("lora_A" in name or "lora_B" in name) and name not in state_dict:
            missing.append(name)
    if missing or unexpected:
        # Soft assertion: warn but don't fail. Partial loads are sometimes
        # intentional (e.g. peeling subset of layers).
        import logging

        logging.getLogger(__name__).warning(
            "LoRA load: %d missing keys, %d unexpected. First few: missing=%s unexpected=%s",
            len(missing),
            len(unexpected),
            missing[:3],
            unexpected[:3],
        )


def merge_lora_into_base(policy: nn.Module) -> int:
    """Absorb LoRA delta into base weights in place; reset LoRA to identity.

    For each LoRALinear: base.weight += scaling * (B @ A); then A := kaiming,
    B := 0. After merge the policy behaves identically at inference, but the
    correction is now permanent (cannot be peeled).

    Returns the number of LoRA layers merged.
    """
    n = 0
    for module in policy.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = module.lora_B @ module.lora_A  # [out, in]
                module.base.weight.add_(module.scaling * delta)
                # Reset LoRA to identity-equivalent state (B=0 => no-op)
                nn.init.kaiming_uniform_(module.lora_A, a=5**0.5)
                module.lora_B.zero_()
            n += 1
    return n


def peel_lora(policy: nn.Module) -> int:
    """Reset LoRA params so forward() == base forward(). Does NOT remove modules.

    B := 0, A := kaiming. After this, the policy behaves as if no LoRA is
    attached, but the LoRA modules are still in place ready to be re-fit.

    Returns the number of LoRA layers peeled.
    """
    n = 0
    with torch.no_grad():
        for module in policy.modules():
            if isinstance(module, LoRALinear):
                nn.init.kaiming_uniform_(module.lora_A, a=5**0.5)
                module.lora_B.zero_()
                n += 1
    return n


def lora_layer_diagnostics(policy: nn.Module) -> list[dict]:
    """Per-LoRA-layer diagnostics for Layer B monitoring.

    Returns a list of dicts, one per LoRALinear:
      - layer: dotted name in the policy
      - rank: nominal rank
      - frobenius: ||scaling * B @ A||_F
      - effective_rank: count of singular values > 1% of the max
    """
    rows = []
    name_lookup = dict(policy.named_modules())
    name_by_module = {id(m): n for n, m in name_lookup.items()}

    for module in policy.modules():
        if not isinstance(module, LoRALinear):
            continue
        with torch.no_grad():
            ba = module.scaling * (module.lora_B @ module.lora_A)
            frob = float(ba.norm().item())
            # Effective rank via SVD on the rank-r product (cheap: rank-r matrix)
            try:
                s = torch.linalg.svdvals(ba.float())
                if s.numel() == 0:
                    eff = 0
                else:
                    smax = float(s.max().item())
                    eff = int((s > smax * 0.01).sum().item()) if smax > 0 else 0
            except Exception:
                eff = -1
        rows.append(
            {
                "layer": name_by_module.get(id(module), "<unknown>"),
                "rank": module.rank,
                "frobenius": frob,
                "effective_rank": eff,
            }
        )
    return rows
