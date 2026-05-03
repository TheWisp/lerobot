"""End-to-end smoke test for online flash-DAgger.

Exercises the full pipeline OFFLINE (no robot, no teleop) by pretending an
existing eval episode is an "intervention" and pushing its frames through
the FlashDaggerSystem hooks. Useful for:

  - validating the integration before running on hardware
  - sanity-checking a new HVLA checkpoint against the online code path
  - bisecting fit-quality regressions independent of robot behavior

Workflow this exercises (matches the on-robot flow):
  1. Construct FlashDaggerSystem (attaches LoRA, samples replay/forget pools)
  2. on_intervention_start()
  3. on_tick(obs, action) per frame from a chosen eval episode
  4. on_intervention_end()
  5. on_episode_end(episode=N, success=True) → triggers fit cycle
  6. Inspect <output_dir>/summary.jsonl, curves/, layer_diag/, lora/

Usage:
    python -m lerobot.policies.hvla.flash_dagger.smoke \\
        --checkpoint outputs/flow_s1_no_s2_merged_raw/checkpoints/checkpoint-50000 \\
        --eval-repo-id eval/eval_cylinder_ring_assembly_apr_24 \\
        --train-repo-id thewisp/cylinder_ring_assembly_merged_raw \\
        --episodes 247 76 \\
        --output-dir outputs/flash_dagger_smoke \\
        --steps 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.hvla.flash_dagger import FlashDaggerConfig, FlashDaggerSystem

logger = logging.getLogger(__name__)


def _replay_episode_into_system(
    system: FlashDaggerSystem,
    eval_dataset,
    eval_starts,
    eval_ends,
    episode_idx: int,
    label: str,
) -> None:
    """Push one eval episode's frames through the lifecycle hooks.

    Each dataset frame is already in the chunk-shape compute_per_sample_loss
    expects, so we deconstruct: store the obs keys + the FIRST action of
    each chunk as the per-tick action. This produces N - chunk_size + 1
    valid training chunks at fit time, matching what live capture would
    produce from the same trajectory.
    """
    start = int(eval_starts[episode_idx])
    end = int(eval_ends[episode_idx])
    n_frames = end - start
    logger.info("[smoke] %s: replaying ep %d (%d frames)", label, episode_idx, n_frames)

    system.on_intervention_start()
    for i in range(start, end):
        frame = eval_dataset[i]
        # obs keys: everything except action / action_is_pad
        obs = {k: v for k, v in frame.items() if k not in ("action", "action_is_pad")}
        # action is [chunk_size, action_dim]; the per-tick action is the first row
        # (index 0 = "what action to take at this observation").
        action = frame["action"][0]
        system.on_tick(obs, action)
    system.on_intervention_end()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-repo-id", required=True)
    parser.add_argument("--train-repo-id", required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[247, 76],
        help="One or more eval episode indices to flash sequentially",
    )
    parser.add_argument("--output-dir", default="outputs/flash_dagger_smoke")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--steps", type=int, default=30, help="kept low for smoke; 100 for real")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--old-pct", type=float, default=0.10)
    parser.add_argument("--flashed-pct", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)

    # Reuse the offline driver's dataset + checkpoint loading (single source
    # of truth for HVLA's data pipeline).
    import safetensors.torch as sft

    from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
    from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy
    from lerobot.policies.hvla.s1.protocol import S2_LATENT_KEY
    from lerobot.policies.hvla.scripts.flash_dagger_phase_b import build_dataset

    ckpt_root = Path(args.checkpoint)
    ckpt_dir = ckpt_root / "pretrained_model"
    if not ckpt_dir.is_dir():
        ckpt_dir = ckpt_root
    with open(ckpt_dir / "config.json") as f:
        ckpt_cfg = json.load(f)

    s1_config = FlowMatchingS1Config(
        chunk_size=ckpt_cfg["chunk_size"],
        hidden_dim=ckpt_cfg["hidden_dim"],
        num_decoder_layers=ckpt_cfg["num_decoder_layers"],
        num_inference_steps=ckpt_cfg["num_inference_steps"],
        rtc_max_delay=ckpt_cfg["rtc_max_delay"],
        rtc_drop_prob=ckpt_cfg["rtc_drop_prob"],
    )
    s1_config.image_features = ckpt_cfg["image_features"]
    norm_stats = torch.load(ckpt_dir / "norm_stats.pt", map_location="cpu", weights_only=True)

    logger.info("[smoke] loading eval dataset %s", args.eval_repo_id)
    _, eval_ds, _, eval_starts, eval_ends = build_dataset(args.eval_repo_id, s1_config, norm_stats)
    logger.info("[smoke] loading train dataset %s", args.train_repo_id)
    _, train_ds, _, _, _ = build_dataset(args.train_repo_id, s1_config, norm_stats)

    logger.info("[smoke] building policy")
    policy = FlowMatchingS1Policy(s1_config).to(device)
    state_dict = sft.load_file(str(ckpt_dir / "model.safetensors"))
    policy.load_state_dict(state_dict, strict=False)

    fd_config = FlashDaggerConfig(
        rank=args.rank,
        alpha=args.alpha,
        steps=args.steps,
        batch_size=args.batch_size,
        old_pct=args.old_pct,
        flashed_pct=args.flashed_pct,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    # Smoke runs without S2 / shared cache. on_tick captures will look like
    # dataset-format (image keys are "observation.images.{cam}"); the system
    # detects this and skips obs_to_s1_batch.
    system = FlashDaggerSystem(
        policy=policy,
        s1_config=s1_config,
        train_dataset=train_ds,
        config=fd_config,
        device=device,
        s1_image_keys=list(s1_config.image_features.keys()),
        resize_to=None,
        shared_cache=None,
        s2_latent_key=S2_LATENT_KEY,
    )

    for n, ep in enumerate(args.episodes, start=1):
        _replay_episode_into_system(
            system, eval_ds, eval_starts, eval_ends, ep, label=f"ep{n}/{len(args.episodes)}"
        )
        # Buffer is non-empty; on_episode_end triggers the fit cycle.
        system.on_episode_end(episode=n, success=True)

    system.shutdown()
    logger.info("[smoke] done — outputs at %s", fd_config.output_dir)
    logger.info("[smoke] summary: %s", fd_config.output_dir / "summary.jsonl")


if __name__ == "__main__":
    main()
