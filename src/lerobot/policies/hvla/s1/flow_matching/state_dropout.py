"""Optional, training-only dropout of HVLA's projected robot-state token.

The module knows nothing about datasets, masks, actions or the training loop.
It preserves retained tokens without rescaling and never edits its input.
Its private RNG does not consume the random stream used for flow noise or RTC.
RNG state belongs to the trainer's optimizer checkpoint, not model weights,
so existing safetensors checkpoints retain exactly the same parameter keys.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class StateTokenDropout(nn.Module):
    """Zero entire [1, D] state tokens independently across a [B, 1, D] batch."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("state_dropout_p must be a finite number in [0, 1]")
        self.p = float(p)
        self._generator: torch.Generator | None = None

    def _rng(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed((torch.initial_seed() + 104729) % (2**63))
        return self._generator

    def forward(self, token: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return token
        drop = (
            torch.rand((token.shape[0], 1, 1), device=token.device, generator=self._rng(token.device))
            < self.p
        )
        return token.masked_fill(drop, 0)

    def rng_state(self) -> Tensor | None:
        """Training-only state; absent when disabled or not yet used."""
        return None if self._generator is None else self._generator.get_state()

    def restore_rng_state(self, state: Tensor | None, device: torch.device) -> None:
        """Restore on the training device; legacy checkpoints may omit this state.

        Like PyTorch's device RNG, this state is for resumes on the same device
        type. This does not promise sample-exact resumption of the data loader.
        """
        if state is not None and self.p > 0:
            self._rng(device).set_state(state.cpu())
