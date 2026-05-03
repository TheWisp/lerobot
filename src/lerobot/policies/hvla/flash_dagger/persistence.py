"""LoRA save / load with base-hash binding.

A flash-DAgger session writes:
    <output_dir>/lora/cycle_<n>.pt      — per-cycle LoRA + metadata
    <output_dir>/lora/latest.pt         — pointer to most recent (overwritten)
    <output_dir>/lora/merged_base.safetensors  — optional, written on merge

Each .pt blob is:
    {"state_dict": {lora_A/B params}, "metadata": {base_hash, rank, ...}}
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import torch

from lerobot.policies.hvla.flash_dagger.lora import (
    LoRAMetadata,
    compute_base_hash,
    extract_lora_state_dict,
    load_lora_state_dict,
)

logger = logging.getLogger(__name__)


def save_lora(
    policy,
    output_dir: Path,
    cycle: int,
    *,
    rank: int,
    alpha: float,
    apply_to_ffn: bool,
    overwrite_latest: bool = True,
) -> Path:
    """Snapshot LoRA params + metadata to disk.

    Returns the path to the per-cycle file. Also overwrites `latest.pt` when
    `overwrite_latest`.
    """
    output_dir = Path(output_dir)
    lora_dir = output_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)

    sd = extract_lora_state_dict(policy)
    meta = LoRAMetadata(
        base_hash=compute_base_hash(policy),
        rank=rank,
        alpha=alpha,
        apply_to_ffn=apply_to_ffn,
        n_lora_params=sum(t.numel() for t in sd.values()),
    )

    blob = {"state_dict": sd, "metadata": dataclasses.asdict(meta)}
    cycle_path = lora_dir / f"cycle_{cycle:04d}.pt"
    # Atomic-ish: write to .tmp then rename.
    tmp_path = cycle_path.with_suffix(".pt.tmp")
    torch.save(blob, tmp_path)
    tmp_path.replace(cycle_path)
    if overwrite_latest:
        latest_path = lora_dir / "latest.pt"
        tmp_path = latest_path.with_suffix(".pt.tmp")
        torch.save(blob, tmp_path)
        tmp_path.replace(latest_path)
    logger.info(
        "[flash-DAgger] saved LoRA cycle=%d → %s (params=%d, base_hash=%s...)",
        cycle,
        cycle_path,
        meta.n_lora_params,
        meta.base_hash[:12],
    )
    return cycle_path


def load_lora(
    policy,
    path: Path,
    *,
    strict_hash: bool = True,
) -> LoRAMetadata:
    """Load LoRA params from disk into `policy`. Verifies base hash.

    Returns the loaded metadata. Raises if base-hash mismatch (when strict).
    """
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    meta_dict = blob["metadata"]
    meta = LoRAMetadata(**meta_dict)
    load_lora_state_dict(policy, sd, metadata=meta, strict_hash=strict_hash)
    logger.info(
        "[flash-DAgger] loaded LoRA from %s (rank=%d, params=%d, base_hash=%s...)",
        path,
        meta.rank,
        meta.n_lora_params,
        meta.base_hash[:12],
    )
    return meta
