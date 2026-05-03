"""Flash-DAgger monitoring + persisted metrics.

Layer A — fit / forget / retain (4 losses, per fit cycle):
    loss_new_train  — running training loss on the current correction
    loss_new_val    — held-out val of the current intervention
    loss_old_val    — random-sample training-set val (forget tripwire)
    loss_flashed_val — average val loss over previously-flashed corrections

Layer B — LoRA internals (per-layer, captured post-fit):
    frobenius     — ||scaling * B @ A||_F
    effective_rank — count of singular values > 1% of max

Outputs:
    <output_dir>/summary.jsonl            one row per fit cycle (all metrics)
    <output_dir>/curves/cycle_<n>.csv     per-step train loss for cycle n
    <output_dir>/layer_diag/cycle_<n>.csv per-layer diagnostics for cycle n
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """One fit cycle's measurements. Serialized to summary.jsonl."""

    cycle: int
    episode: int  # external episode counter
    correction_id: int  # FlashedEpisodePool's id
    n_intervention_frames: int
    n_train_frames: int
    n_val_frames: int
    n_steps: int
    wall_seconds: float  # sum of timing components below

    # Layer A
    loss_new_train_final: float
    loss_new_val_pre: float
    loss_new_val_post: float
    loss_old_val_pre: float
    loss_old_val_post: float
    loss_flashed_val_pre: float | None  # None on first cycle (empty pool)
    loss_flashed_val_post: float | None

    # Decisions
    swap_accepted: bool
    swap_reject_reason: str = ""

    # Layer B summary (full per-layer goes to layer_diag/cycle_<n>.csv)
    n_lora_layers: int = 0
    frobenius_max: float = 0.0
    effective_rank_max: int = 0

    # Timing breakdown (helps diagnose where wall-time goes)
    encode_live_seconds: float = 0.0
    pre_eval_seconds: float = 0.0
    fit_seconds: float = 0.0
    post_eval_seconds: float = 0.0
    save_seconds: float = 0.0

    # Per-cycle structure
    n_segments: int = 0  # interventions in this episode


@dataclass
class MetricsLogger:
    output_dir: Path
    summary_path: Path = field(init=False)
    curves_dir: Path = field(init=False)
    layer_diag_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.summary_path = self.output_dir / "summary.jsonl"
        self.curves_dir = self.output_dir / "curves"
        self.layer_diag_dir = self.output_dir / "layer_diag"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        self.layer_diag_dir.mkdir(parents=True, exist_ok=True)

    def write_cycle(self, m: CycleMetrics) -> None:
        with self.summary_path.open("a") as f:
            f.write(json.dumps(dataclasses.asdict(m)) + "\n")
        logger.info(
            "[flash-DAgger] cycle %d ep=%d cid=%d segs=%d frames=%d steps=%d "
            "t=%.1fs (enc=%.1f pre=%.1f fit=%.1f post=%.1f save=%.1f) | "
            "new_val %.4f→%.4f | old_val %.4f→%.4f%s | "
            "lora ‖BA‖_max=%.3f eff_rank_max=%d | accepted=%s%s",
            m.cycle,
            m.episode,
            m.correction_id,
            m.n_segments,
            m.n_intervention_frames,
            m.n_steps,
            m.wall_seconds,
            m.encode_live_seconds,
            m.pre_eval_seconds,
            m.fit_seconds,
            m.post_eval_seconds,
            m.save_seconds,
            m.loss_new_val_pre,
            m.loss_new_val_post,
            m.loss_old_val_pre,
            m.loss_old_val_post,
            (
                f" | flashed_val {m.loss_flashed_val_pre:.4f}→{m.loss_flashed_val_post:.4f}"
                if m.loss_flashed_val_pre is not None
                else ""
            ),
            m.frobenius_max,
            m.effective_rank_max,
            m.swap_accepted,
            f" ({m.swap_reject_reason})" if m.swap_reject_reason else "",
        )

    def write_curve(self, cycle: int, per_step_losses: list[float]) -> None:
        path = self.curves_dir / f"cycle_{cycle:04d}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss"])
            for i, v in enumerate(per_step_losses, start=1):
                w.writerow([i, f"{v:.6f}"])

    def write_layer_diag(self, cycle: int, rows: list[dict]) -> None:
        if not rows:
            return
        path = self.layer_diag_dir / f"cycle_{cycle:04d}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["layer", "rank", "frobenius", "effective_rank"])
            w.writeheader()
            for r in rows:
                w.writerow(r)


def now_seconds() -> float:
    return time.time()
