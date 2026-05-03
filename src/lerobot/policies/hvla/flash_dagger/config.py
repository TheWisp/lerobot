"""Flash-DAgger configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FlashDaggerConfig:
    # LoRA
    rank: int = 16
    alpha: float = 32.0
    apply_to_ffn: bool = True

    # Fit loop
    steps: int = 100
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    cosine_eta_min_frac: float = 0.05  # eta_min = lr * this

    # Three-way mix
    old_pct: float = 0.10
    flashed_pct: float = 0.25
    # new_pct = 1 - old_pct - flashed_pct (computed)

    # Pools sampled from the training dataset
    replay_pool_size: int = 5000  # for the "old" slot
    forget_val_size: int = 500  # held-out broad-task tripwire

    # Per-flashed-episode val split (for retention metric)
    val_pct: float = 0.2

    # Safety / tripwires
    forget_drift_abort_pct: float = 50.0  # if loss_old_val drift > X%, refuse swap

    # Triggers
    min_intervention_frames: int = 50  # below this, skip the fit (too little signal)

    # Persistence
    output_dir: Path = field(default_factory=lambda: Path("outputs/flash_dagger"))
    save_per_episode: bool = False  # if True, also snapshot LoRA per-episode

    # Misc
    num_workers: int = 4
    seed: int = 0

    @property
    def new_pct(self) -> float:
        return 1.0 - self.old_pct - self.flashed_pct

    def __post_init__(self) -> None:
        assert 0.0 <= self.old_pct <= 1.0
        assert 0.0 <= self.flashed_pct <= 1.0
        assert self.old_pct + self.flashed_pct < 1.0, (
            f"old_pct + flashed_pct must leave room for new_pct: "
            f"{self.old_pct} + {self.flashed_pct} = {self.old_pct + self.flashed_pct}"
        )
        assert self.rank > 0
        assert self.steps > 0
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
