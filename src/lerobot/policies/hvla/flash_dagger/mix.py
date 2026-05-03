"""Three-way batch sampler for online flash-DAgger.

Same recipe as the offline phase_d (10/25/65 default) but operating purely
on pre-built sample lists — every entry in every pool is already a fully-
formed training sample (after pre-encoding old/forget pools at startup, and
encoding live captures at cycle start). No HF-dataset handle is needed at
fit time.

The sampler is a torch.utils.data.Dataset: __getitem__ rolls a categorical
each call to choose old / flashed / new, then samples uniformly within
the chosen pool.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch.utils.data


class ThreeWayMixDataset(torch.utils.data.Dataset):
    """Old / flashed / new mixed sampler.

    Args:
      old_samples: pre-encoded training samples drawn from the original
        training dataset's replay slot (forgetting protection).
      flashed_pools: list of pre-encoded sample lists, one per past correction
        (rehearsal of prior flashed episodes). Sampler picks a correction
        uniformly, then a sample uniformly within it.
      new_pool: pre-encoded samples for the current correction (the
        intervention's training portion).
      old_pct, flashed_pct: probabilities; new_pct = 1 - old - flashed.
      length: nominal __len__ for the DataLoader.

    If `flashed_pools` is empty (first correction of the session), the
    flashed_pct mass is folded into new_pct.
    """

    def __init__(
        self,
        old_samples: Sequence[dict],
        flashed_pools: Sequence[Sequence[dict]],
        new_pool: Sequence[dict],
        old_pct: float,
        flashed_pct: float,
        length: int,
    ):
        assert old_pct >= 0.0 and flashed_pct >= 0.0
        assert old_pct + flashed_pct < 1.0
        assert len(new_pool) > 0, "new_pool must be non-empty"

        self.old_samples = list(old_samples)
        self.flashed_pools = [list(p) for p in flashed_pools if len(p) > 0]
        self.new_pool = list(new_pool)
        self.length = length

        if not self.flashed_pools:
            # First correction — collapse flashed mass into new
            self.old_pct = old_pct
            self.flashed_pct = 0.0
        else:
            self.old_pct = old_pct
            self.flashed_pct = flashed_pct

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _idx: int) -> dict:
        r = random.random()
        if r < self.old_pct and self.old_samples:
            return self.old_samples[random.randrange(len(self.old_samples))]
        if r < self.old_pct + self.flashed_pct and self.flashed_pools:
            pool = self.flashed_pools[random.randrange(len(self.flashed_pools))]
            return pool[random.randrange(len(pool))]
        return self.new_pool[random.randrange(len(self.new_pool))]
