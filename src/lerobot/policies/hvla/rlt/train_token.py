"""Phase 1: Offline RL token encoder-decoder training on demo data.

Trains the RL token to compress S1's observation encoder output into a single
768-dim vector that retains enough information to reconstruct the original
context tokens. The frozen S1 provides the context; only the encoder-decoder
parameters (phi) are trained.

Usage:
    python -m lerobot.policies.hvla.rlt.train_token \
        --s1-checkpoint outputs/flow_s1_no_s2_v1/checkpoints/last/pretrained_model/model.safetensors \
        --dataset-repo-id thewisp/cylinder_ring_assembly \
        --output-dir outputs/rlt_token_v1 \
        --steps 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies.hvla.rlt.config import RLTConfig
from lerobot.policies.hvla.rlt.token import (
    RLTokenDecoder,
    RLTokenEncoder,
    rl_token_reconstruction_loss,
)
from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy

logger = logging.getLogger(__name__)


def train(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    logging.getLogger().handlers[0].stream = sys.stderr

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / "train_token.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Command: %s", " ".join(sys.argv))

    torch.set_float32_matmul_precision("high")

    # --- Load frozen S1 ---
    logger.info("Loading frozen S1 from %s", args.s1_checkpoint)
    s1_policy = FlowMatchingS1Policy.from_pretrained(args.s1_checkpoint)
    s1_policy.to(device)
    s1_policy.eval()
    for p in s1_policy.parameters():
        p.requires_grad = False
    s1_config = s1_policy.config
    logger.info("S1 hidden_dim=%d, chunk_size=%d", s1_config.hidden_dim, s1_config.chunk_size)

    # --- RLT config ---
    rlt_config = RLTConfig(rl_token_dim=s1_config.hidden_dim)
    if args.steps:
        rlt_config.token_train_steps = args.steps
    if args.lr:
        rlt_config.token_lr = args.lr

    # --- Build encoder-decoder ---
    encoder = RLTokenEncoder(rlt_config).to(device)
    decoder = RLTokenDecoder(rlt_config).to(device)
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    logger.info("Encoder params: %.1fM | Decoder params: %.1fM", enc_params / 1e6, dec_params / 1e6)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=rlt_config.token_lr,
    )

    # --- Load dataset (reuse S1's dataset pipeline) ---
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset

    logger.info("Loading dataset: %s", args.dataset_repo_id)
    lerobot_dataset = LeRobotDataset(args.dataset_repo_id)

    s2_latents = None
    if args.s2_latent_path:
        s2_latents = np.load(args.s2_latent_path)
        logger.info("S2 latents: %s", s2_latents.shape)

    resize_to = None
    if args.resize_images:
        h, w = [int(x) for x in args.resize_images.split("x")]
        resize_to = (h, w)

    dataset = FlowMatchingDataset(
        lerobot_dataset,
        s2_latents=s2_latents,
        chunk_size=s1_config.chunk_size,
        max_delay_seconds=0.0,  # no delay aug for token training
        resize_to=resize_to,
        image_keys=list(s1_config.image_features.keys()),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Training loop ---
    logger.info("Training RL token for %d steps (batch=%d)", rlt_config.token_train_steps, args.batch_size)
    step = 0
    t0 = time.time()
    running_loss = 0.0
    data_iter = iter(dataloader)

    while step < rlt_config.token_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Extract context tokens from frozen S1 (no grad)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if s1_config.image_features:
                batch["observation.images"] = [
                    batch[key] for key in s1_config.image_features
                ]
            context = s1_policy.model.encode_observations(batch)  # [B, N_ctx, D]
            context = context.float()  # cast back to fp32 for encoder-decoder

        # Train encoder-decoder
        loss = rl_token_reconstruction_loss(encoder, decoder, context)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg_loss = running_loss / 100
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            logger.info(
                "step %d/%d | loss=%.6f | %.1f steps/s",
                step, rlt_config.token_train_steps, avg_loss, steps_per_sec,
            )
            running_loss = 0.0

        if step % args.save_freq == 0 or step == rlt_config.token_train_steps:
            save_dir = output_dir / f"checkpoint-{step}"
            save_dir.mkdir(exist_ok=True)
            torch.save(encoder.state_dict(), save_dir / "encoder.pt")
            torch.save(decoder.state_dict(), save_dir / "decoder.pt")
            torch.save(
                {"optimizer": optimizer.state_dict(), "step": step},
                save_dir / "optimizer.pt",
            )
            logger.info("Saved checkpoint at step %d → %s", step, save_dir)

    elapsed = time.time() - t0
    logger.info("Training complete: %d steps in %.1fs (%.1f steps/s)", step, elapsed, step / elapsed)


def main():
    parser = argparse.ArgumentParser(description="Train RL token encoder-decoder (Phase 1)")
    parser.add_argument("--s1-checkpoint", required=True, help="Path to frozen S1 checkpoint")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo ID")
    parser.add_argument("--s2-latent-path", default=None, help="S2 latents .npy (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--resize-images", default="224x224")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = main()
    train(args)
