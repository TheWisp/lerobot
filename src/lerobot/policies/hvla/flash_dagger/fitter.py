"""Synchronous fit loop for online flash-DAgger.

The training loop preserves the offline phase-D recipe exactly:
    AdamW (no weight decay) + CosineAnnealingLR (eta_min = lr * 0.05)
    + bf16 autocast + per-sample loss .mean() + grad clip 1.0.

Two functions:
  fit_step_loop(...)
    Runs N optimizer steps over a torch.utils.data.DataLoader. Generic
    over the loss function — caller supplies `loss_fn(policy, batch) -> Tensor`.
    Returns the per-step training-loss curve.

  evaluate_loss(...)
    Computes mean loss over a list of frame dicts. Used for pre/post
    held-out evaluation.

Higher-level cycle orchestration (pre/post eval, swap-or-revert, metrics)
lives in system.py.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def fit_step_loop(
    policy: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    cosine_eta_min: float,
    device: torch.device,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> list[float]:
    """One synchronous fit cycle. Mutates `policy`'s trainable params.

    Preconditions:
      - LoRA already attached to `policy` (only LoRA params have requires_grad=True)
      - `loader` yields batches as dicts; loss_fn returns per-sample loss tensor
      - `steps <= len(loader)` typically; loop breaks at `steps`

    Postcondition:
      - Optimizer is created fresh inside (no carryover state across cycles)
      - Returns list of per-step mean training losses (length == actual steps run)
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    assert trainable, "no trainable params — was apply_lora_to_decoder() called?"

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=steps,
        eta_min=cosine_eta_min,
    )

    policy.train()
    curve: list[float] = []
    use_autocast = device.type == "cuda"

    for step, batch in enumerate(loader, start=1):
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        if use_autocast:
            with torch.autocast("cuda", dtype=autocast_dtype):
                losses = loss_fn(policy, batch)
                loss = losses.mean()
        else:
            losses = loss_fn(policy, batch)
            loss = losses.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        opt.step()
        sched.step()
        curve.append(float(loss.item()))
        if step >= steps:
            break

    return curve


@torch.no_grad()
def evaluate_loss(
    policy: torch.nn.Module,
    frames: Sequence[dict],
    loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
    collate_fn: Callable[[list[dict]], dict],
    *,
    batch_size: int,
    device: torch.device,
    passes: int = 1,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> float:
    """Mean loss over `frames`. Always runs in eval() mode.

    `passes > 1` averages multiple stochastic eval passes (helpful with
    flow-matching / diffusion losses where each forward samples a noise
    draw). Restores the original train/eval mode of `policy` on exit.
    """
    if not frames:
        return float("nan")

    was_training = policy.training
    policy.eval()
    use_autocast = device.type == "cuda"

    total_loss = 0.0
    total_count = 0
    try:
        for _ in range(passes):
            for start in range(0, len(frames), batch_size):
                batch_list = list(frames[start : start + batch_size])
                batch = collate_fn(batch_list)
                batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if use_autocast:
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        losses = loss_fn(policy, batch)
                else:
                    losses = loss_fn(policy, batch)
                total_loss += float(losses.sum().item())
                total_count += int(losses.numel())
    finally:
        if was_training:
            policy.train()

    return total_loss / max(total_count, 1)


class FrameListDataset(torch.utils.data.Dataset):
    """Adapt a list of pre-built frame dicts to a torch Dataset.

    Used when frames already match the HF dataset's per-sample schema
    (e.g. cached intervention frames re-loaded from disk for an offline
    smoke run).
    """

    def __init__(self, frames: list[dict[str, Any]]):
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        return self._frames[idx]


class InterventionChunkDataset(torch.utils.data.Dataset):
    """Build training chunks from per-tick intervention captures.

    Accepts a list of segments — one segment per contiguous intervention.
    Sliding-window chunks are built within each segment; no chunk ever
    spans a segment boundary (such a chunk would pair an obs from one
    intervention with actions from a different time/scene).

    Input: list[list[{"obs": dict[str, Tensor], "action": Tensor[action_dim]}]]
        outer list = segments, inner list = per-tick frames within a segment.

    Output dict sample matches the HF dataset's per-chunk schema:
        observation.* keys from `obs` at start tick t (within a segment)
        "action": Tensor[chunk_size, action_dim] = actions at [t, ..., t+chunk_size-1]
        "action_is_pad": Tensor[chunk_size] bool, all False (no padding for
            valid windows; segments shorter than chunk_size contribute 0 chunks)

    Valid starts within each segment: 0..len(segment)-chunk_size. If a
    segment is shorter than chunk_size, it contributes 0 chunks. If the
    whole dataset has 0 chunks, the fit cycle is skipped upstream.

    A flat ``list[frame]`` is accepted for backward compat; it's treated
    as a single segment.
    """

    def __init__(
        self,
        segments: list[list[dict[str, Any]]] | list[dict[str, Any]],
        chunk_size: int,
    ):
        # Backward-compat: if caller passes a flat list of frame dicts
        # (legacy single-segment shape), wrap it.
        if segments and isinstance(segments[0], dict):
            segments = [segments]  # type: ignore[list-item]
        self._segments = segments
        self._chunk_size = chunk_size
        # Cumulative valid-start counts per segment for O(1) index lookup.
        # segment i contributes max(0, len(seg)-chunk_size+1) chunks.
        self._segment_offsets: list[int] = []
        cumulative = 0
        for seg in self._segments:
            self._segment_offsets.append(cumulative)
            cumulative += max(0, len(seg) - chunk_size + 1)
        self._total = cumulative

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        # Locate (segment, offset-within-segment) without bisect (small list).
        seg_idx = 0
        for i in range(len(self._segment_offsets) - 1, -1, -1):
            if idx >= self._segment_offsets[i]:
                seg_idx = i
                break
        offset = idx - self._segment_offsets[seg_idx]
        seg = self._segments[seg_idx]
        start = seg[offset]
        # Format-agnostic: take everything from the start frame except its
        # "action" (we'll replace with a chunk). Two common shapes are
        # supported: legacy nested {"obs": dict, "action": ...} and the
        # current flat {"context": tensor, "action": ...} produced after
        # _encode_live_obs_batch.
        if "obs" in start and isinstance(start["obs"], dict):
            sample = dict(start["obs"])
        else:
            sample = {k: v for k, v in start.items() if k != "action"}
        actions = torch.stack([seg[offset + j]["action"] for j in range(self._chunk_size)])
        sample["action"] = actions
        sample["action_is_pad"] = torch.zeros(self._chunk_size, dtype=torch.bool)
        return sample
