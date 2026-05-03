"""Online flash-DAgger: LoRA-based correction adapter for HVLA S1.

This is an *additional* module to HVLA, sibling of `rlt/`. The offline
recipe is documented at:

    src/lerobot/policies/hvla/research/flash_dagger/SUMMARY.md

The online version captures operator interventions (SPACE-key) into a
per-episode buffer and runs a synchronous LoRA fit at episode-end (or
session-end) on a three-way batch mix:

    10% old (training-set replay)  +  25% flashed (prior corrections this
    session)  +  65% new (current episode's intervention frames).

The fitted LoRA is hot-swapped into the live policy. Frozen base means a
bad fit can be peeled off without losing the original policy.

For v0 this is HVLA-specific. Generalization to ACT/PI0 is a future PR.
"""

from lerobot.policies.hvla.flash_dagger.config import FlashDaggerConfig
from lerobot.policies.hvla.flash_dagger.system import FlashDaggerSystem

__all__ = ["FlashDaggerConfig", "FlashDaggerSystem"]
