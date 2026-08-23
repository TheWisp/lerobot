"""Training script for Flow Matching S1 with Training-Time RTC.

Implements the training procedure from:
  "Training-Time Action Conditioning for Efficient Real-Time Chunking"
  (arXiv:2512.05964, Mees et al., 2025)

Key differences from standard flow matching training:
  - Simulated inference delay: randomly replace first D actions with GT (unnoised)
  - Per-position timestep: prefix positions get t=0, future positions get t~Beta
  - Prefix dropout: with probability p, no prefix (simulates first chunk)
  - S2 latent delay augmentation (independent from RTC delay)

Usage:
    python -m lerobot.policies.hvla.s1.flow_matching.train \\
        --dataset-repo-id thewisp/cylinder_ring_assembly \\
        --s2-latent-path ~/.cache/.../s2_latents_pt_11997.npy \\
        --output-dir outputs/flow_s1_hvla \\
        --steps 100000 --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.common.resource_telemetry import ResourceSampler
from lerobot.common.training_log import TrainingHealthTracker
from lerobot.policies.hvla.s1.flow_matching import vision_encoders
from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy
from lerobot.policies.hvla.s1.protocol import S2_AGE_KEY, S2_LATENT_KEY

logger = logging.getLogger(__name__)


def checkpoint_config_dict(config: FlowMatchingS1Config) -> dict:
    """Serialize a config into what a checkpoint's ``config.json`` must contain.

    Postcondition: the result satisfies
    :meth:`FlowMatchingS1Config.from_checkpoint_dict`, which is what makes a
    checkpoint loadable. Lives at module level rather than inside ``train()``
    so a test can hold the writer and the reader against each other; while it
    was a closure, dropping a contract field here would have shipped
    unloadable checkpoints with the whole suite still green.
    """
    return {
        "type": "hvla_flow_s1",
        "feature_contract_version": FlowMatchingS1Config.FEATURE_CONTRACT_VERSION,
        "action_dim": config.action_dim,
        "action_feature_names": config.action_feature_names,
        "robot_state_feature": config.robot_state_feature,
        "state_dim": config.state_dim,
        "state_feature_names": config.state_feature_names,
        "chunk_size": config.chunk_size,
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "num_encoder_layers": config.num_encoder_layers,
        "num_decoder_layers": config.num_decoder_layers,
        "dim_feedforward": config.dim_feedforward,
        "s2_latent_dim": config.s2_latent_dim,
        "s2_proj_hidden": config.s2_proj_hidden,
        "num_inference_steps": config.num_inference_steps,
        "rtc_max_delay": config.rtc_max_delay,
        "rtc_drop_prob": config.rtc_drop_prob,
        "use_dino_backbone": config.use_dino_backbone,
        "backbone_dim": config.backbone_dim,
        "freeze_backbone": config.freeze_backbone,
        "image_features": config.image_features,
        "image_resize_shape": config.image_resize_shape,
        "dino_model": config.dino_model,
    }


def configure_from_dataset_features(
    config: FlowMatchingS1Config,
    features: dict,
    *,
    resize_to: tuple[int, int] | None,
    cameras: list[str] | None = None,
) -> None:
    """Resolve the S1 input/output contract from LeRobot dataset metadata.

    The metadata is the only source of truth here: robot type, motor count,
    state layout, and camera names are deliberately not inferred from names.

    ``cameras`` restricts which visual features become model inputs, named
    either bare (``top_l``) or fully (``observation.images.top_l``). Default is
    every camera in the dataset. An unknown name is an error rather than a
    silent no-op, because a typo would otherwise train a model on more cameras
    than intended and only surface at deployment.
    """
    try:
        action_feature = features["action"]
    except KeyError as exc:
        raise ValueError("HVLA Flow S1 training requires an 'action' feature") from exc

    action_shape = tuple(action_feature.get("shape", ()))
    if len(action_shape) != 1 or action_shape[0] <= 0:
        raise ValueError(f"HVLA Flow S1 requires a 1-D action feature, got shape={action_shape}")
    config.action_dim = int(action_shape[0])
    config.action_feature_names = list(action_feature.get("names") or [])
    if len(config.action_feature_names) != config.action_dim:
        raise ValueError(
            "Action metadata must provide one ordered name per value: "
            f"{len(config.action_feature_names)} names for {config.action_dim} values"
        )

    state_feature = features.get("observation.state")
    if state_feature is None:
        config.robot_state_feature = False
        config.state_dim = 0
        config.state_feature_names = []
    else:
        state_shape = tuple(state_feature.get("shape", ()))
        if len(state_shape) != 1 or state_shape[0] <= 0:
            raise ValueError(
                f"HVLA Flow S1 requires a 1-D observation.state feature, got shape={state_shape}"
            )
        config.robot_state_feature = True
        config.state_dim = int(state_shape[0])
        config.state_feature_names = list(state_feature.get("names") or [])
        if len(config.state_feature_names) != config.state_dim:
            raise ValueError(
                "State metadata must provide one ordered name per value: "
                f"{len(config.state_feature_names)} names for {config.state_dim} values"
            )

    image_keys = [
        key
        for key, feature in features.items()
        if key.startswith("observation.images.")
        and len(tuple(feature.get("shape", ()))) == 3
        and feature.get("dtype") in {"image", "video"}
    ]
    if not image_keys:
        raise ValueError(
            "HVLA Flow S1 training requires at least one visual feature under observation.images.*"
        )

    if cameras is not None:
        wanted = {c if c.startswith("observation.images.") else f"observation.images.{c}" for c in cameras}
        unknown = sorted(wanted - set(image_keys))
        if unknown:
            raise ValueError(
                f"--cameras names features this dataset does not have: {unknown}; "
                f"available: {sorted(image_keys)}"
            )
        available = len(image_keys)
        image_keys = [k for k in image_keys if k in wanted]
        if not image_keys:
            raise ValueError("--cameras selected no cameras")
        logger.info(
            "Cameras: using %d of %d (%s)",
            len(image_keys),
            available,
            ", ".join(k.removeprefix("observation.images.") for k in image_keys),
        )

    image_size = resize_to[0] if resize_to is not None else None
    config.image_features = dict.fromkeys(image_keys, image_size)
    config.image_resize_shape = resize_to
    config.validate_feature_contract(require_names=True)


class FlowMatchingDataset(torch.utils.data.Dataset):
    """Dataset with S2 latent loading and delay augmentation.

    Training-time RTC is handled inside the model's forward pass (not here),
    because it operates on the noisy action sequence during flow matching.
    This dataset provides: observations, S2 latent + age, target actions.
    """

    def __init__(
        self,
        lerobot_dataset,
        s2_latents: np.ndarray,  # [N_frames, 2048]
        chunk_size: int = 50,
        max_delay_seconds: float = 0.15,
        fps: float = 30.0,
        resize_to: tuple[int, int] | None = None,
        image_keys: list[str] | None = None,
    ):
        self.dataset = lerobot_dataset
        self.s2_latents = s2_latents
        self.chunk_size = chunk_size
        self.max_delay_frames = int(max_delay_seconds * fps)
        self.fps = fps
        self.resize_to = resize_to
        self.image_keys = image_keys

        # Build episode boundaries for clipping
        self._episode_starts = {}
        self._episode_ends = {}
        if hasattr(lerobot_dataset, "episode_data_index"):
            for ep_idx in range(len(lerobot_dataset.episode_data_index["from"])):
                start = lerobot_dataset.episode_data_index["from"][ep_idx].item()
                end = lerobot_dataset.episode_data_index["to"][ep_idx].item()
                for i in range(start, end):
                    self._episode_starts[i] = start
                    self._episode_ends[i] = end

        # Preload all actions into memory.
        # Avoids calling dataset[i] 50 times per sample for chunk construction
        import logging as _log

        _log.getLogger(__name__).info("Preloading actions for chunk construction...")
        if hasattr(lerobot_dataset, "hf_dataset") and "action" in lerobot_dataset.hf_dataset.column_names:
            action_data = lerobot_dataset.hf_dataset["action"]
            if isinstance(action_data[0], torch.Tensor):
                self._all_actions = torch.stack(list(action_data)).float()
            else:
                import numpy as _np

                self._all_actions = torch.tensor(_np.array(action_data), dtype=torch.float32)
        else:
            # Fallback: load one by one
            self._all_actions = torch.stack(
                [lerobot_dataset[i]["action"] for i in range(len(lerobot_dataset))]
            )
        _log.getLogger(__name__).info("Actions preloaded: %s", self._all_actions.shape)

        # Compute normalization stats (z-score: (x - mean) / std)
        self.action_mean = self._all_actions.mean(dim=0)  # [action_dim]
        self.action_std = self._all_actions.std(dim=0).clamp(min=1e-6)  # [action_dim]
        _log.getLogger(__name__).info(
            "Action norm stats: mean=[%.1f..%.1f] std=[%.1f..%.1f]",
            self.action_mean.min(),
            self.action_mean.max(),
            self.action_std.min(),
            self.action_std.max(),
        )

        # Normalize all preloaded actions
        self._all_actions = (self._all_actions - self.action_mean) / self.action_std

        # Preload and normalize states too
        if (
            hasattr(lerobot_dataset, "hf_dataset")
            and "observation.state" in lerobot_dataset.hf_dataset.column_names
        ):
            state_data = lerobot_dataset.hf_dataset["observation.state"]
            if isinstance(state_data[0], torch.Tensor):
                self._all_states = torch.stack(list(state_data)).float()
            else:
                import numpy as _np

                self._all_states = torch.tensor(_np.array(state_data), dtype=torch.float32)
            self.state_mean = self._all_states.mean(dim=0)
            self.state_std = self._all_states.std(dim=0).clamp(min=1e-6)
            self._all_states = (self._all_states - self.state_mean) / self.state_std
            _log.getLogger(__name__).info("States preloaded and normalized: %s", self._all_states.shape)
        else:
            self._all_states = None
            self.state_mean = None
            self.state_std = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        ep_start = self._episode_starts.get(idx, 0)
        ep_end = self._episode_ends.get(idx, len(self.dataset))

        # --- Build action chunk: [chunk_size, action_dim] (already normalized) ---
        indices = torch.arange(idx, idx + self.chunk_size).clamp(max=ep_end - 1)
        sample["action"] = self._all_actions[indices]  # [chunk_size, action_dim]
        sample["action_is_pad"] = torch.arange(self.chunk_size) >= (ep_end - idx)

        # --- Use normalized state ---
        if self._all_states is not None:
            sample["observation.state"] = self._all_states[idx]

        # --- S2 latent with delay augmentation (skip if training without S2) ---
        if self.s2_latents is not None:
            k = np.random.randint(0, self.max_delay_frames + 1)
            delayed_idx = max(idx - k, ep_start)
            s2_latent = torch.from_numpy(self.s2_latents[delayed_idx]).float()
            age_seconds = k / self.fps

            sample[S2_LATENT_KEY] = s2_latent
            sample[S2_AGE_KEY] = torch.tensor([age_seconds], dtype=torch.float32)

        # The underlying dataset decodes every camera it has. Anything not
        # selected travels into the batch at SOURCE resolution and is collated
        # through shared memory for nothing -- with one of four cameras selected
        # that is three full-size frames per sample, which exhausts a
        # container's /dev/shm and kills the loader workers.
        #
        # Deliberately outside the resize guard below: a run that does not
        # resize still pays the full collation cost, which is the case this
        # exists to prevent.
        if self.image_keys:
            for key in [k for k in sample if k.startswith("observation.images.")]:
                if key not in self.image_keys:
                    del sample[key]

        # --- Resize images if needed ---
        if self.resize_to is not None and self.image_keys:
            import torchvision.transforms.functional as TF

            for key in self.image_keys:
                if key in sample and isinstance(sample[key], torch.Tensor):
                    sample[key] = TF.resize(
                        sample[key],
                        list(self.resize_to),
                        interpolation=TF.InterpolationMode.BILINEAR,
                        antialias=True,
                    )

        return sample


def train(args):
    """Main training loop."""
    import sys

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    logging.getLogger().handlers[0].stream = sys.stderr  # ensure unbuffered

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-create train.log in output dir (appends on resume)
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("Command: %s", " ".join(sys.argv))

    # Parse resize before resolving the feature contract.
    resize_to = None
    if args.resize_images:
        h, w = (int(x) for x in args.resize_images.split("x"))
        resize_to = (h, w)

    # Load config
    encoder = vision_encoders.resolve(args.vision_encoder)
    config = FlowMatchingS1Config(
        chunk_size=args.chunk_size,
        num_inference_steps=args.num_inference_steps,
        rtc_max_delay=args.rtc_max_delay,
        rtc_drop_prob=args.rtc_drop_prob,
        hidden_dim=args.hidden_dim,
        num_decoder_layers=args.num_decoder_layers,
        dino_model=args.vision_encoder,
        # Derived, not asked for: the encoder determines its own token width,
        # and a hand-set backbone_dim that disagrees is a shape error one batch
        # into training.
        backbone_dim=encoder.embed_dim,
    )
    logger.info(
        "Vision encoder: %s (%s, %d-d patch tokens, patch %d)",
        args.vision_encoder,
        encoder.label,
        encoder.embed_dim,
        encoder.patch_size,
    )
    # Load dataset
    logger.info("Loading dataset: %s", args.dataset_repo_id)
    lerobot_dataset = LeRobotDataset(args.dataset_repo_id)
    configure_from_dataset_features(
        config,
        lerobot_dataset.meta.features,
        resize_to=resize_to,
        cameras=args.cameras,
    )

    logger.info(
        "Config: action=%d, state=%d, cameras=%s, chunk=%d, hidden=%d, "
        "dec_layers=%d, rtc_max_delay=%d, rtc_drop=%.2f, denoise_steps=%d",
        config.action_dim,
        config.state_dim,
        list(config.image_features),
        config.chunk_size,
        config.hidden_dim,
        config.num_decoder_layers,
        config.rtc_max_delay,
        config.rtc_drop_prob,
        config.num_inference_steps,
    )

    # Load S2 latents (optional — train without S2 conditioning if omitted)
    s2_latents = None
    if args.s2_latent_path:
        logger.info("Loading S2 latents from %s", args.s2_latent_path)
        s2_latents = np.load(args.s2_latent_path)
        logger.info("S2 latents shape: %s", s2_latents.shape)
    else:
        logger.info("No S2 latent path provided — training without S2 conditioning")

    # Wrap dataset
    dataset = FlowMatchingDataset(
        lerobot_dataset,
        s2_latents=s2_latents,
        chunk_size=config.chunk_size,
        max_delay_seconds=args.max_delay,
        resize_to=resize_to,
        image_keys=list(config.image_features.keys()),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    logger.info("Building FlowMatchingS1 model...")

    # TF32 matmul precision — free ~2× speedup on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")

    policy = FlowMatchingS1Policy(config).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(
        "Total params: %d (%.1fM) | Trainable: %d (%.1fM)",
        total_params,
        total_params / 1e6,
        trainable_params,
        trainable_params / 1e6,
    )

    # Optimizer with cosine LR schedule (matching Pi0/SmolVLA)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Cosine decay: warmup → peak_lr → decay to lr_decay
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)  # linear warmup
        progress = (step - config.warmup_steps) / max(args.steps - config.warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = config.lr_decay / config.lr
        return min_ratio + (1 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        # Support both standard (pretrained_model/) and legacy (flat) formats
        pretrained_dir = resume_dir / "pretrained_model"
        training_state_dir = resume_dir / "training_state"
        if pretrained_dir.is_dir():
            model_path = pretrained_dir / "model.safetensors"
            opt_path = training_state_dir / "optimizer.pt"
        else:
            model_path = resume_dir / "model.safetensors"
            opt_path = resume_dir / "optimizer.pt"

        if model_path.exists():
            import safetensors.torch as sft

            state_dict = sft.load_file(str(model_path))
            policy.load_state_dict(state_dict, strict=False)
            logger.info("Resumed model from %s", model_path)
        if opt_path.exists():
            opt_state = torch.load(str(opt_path), weights_only=True, map_location=device)
            optimizer.load_state_dict(opt_state["optimizer"])
            scheduler.load_state_dict(opt_state["scheduler"])
            start_step = opt_state.get("step", 0)
            logger.info("Resumed optimizer from %s (step %d)", opt_path, start_step)
        else:
            # Try to infer step from directory name
            try:
                start_step = int(resume_dir.name.split("-")[-1])
                for _ in range(start_step):
                    scheduler.step()
                logger.info("Resumed from step %d (no optimizer state, LR schedule advanced)", start_step)
            except ValueError:
                pass

    # Mixed precision
    use_amp = device.type == "cuda"
    if use_amp:
        logger.info("Using bf16 mixed precision + TF32 matmul")
    logger.info(
        "LR schedule: warmup %d → peak %.1e → cosine decay → %.1e",
        config.warmup_steps,
        config.lr,
        config.lr_decay,
    )

    # Save norm stats for inference (must denormalize model output)
    norm_stats = {
        "action_mean": dataset.action_mean,
        "action_std": dataset.action_std,
    }
    if dataset.state_mean is not None:
        norm_stats["state_mean"] = dataset.state_mean
        norm_stats["state_std"] = dataset.state_std

    def save_checkpoint(step):
        import json

        import safetensors.torch as sft

        ckpts_dir = output_dir / "checkpoints"
        ckpts_dir.mkdir(exist_ok=True)
        ckpt_dir = ckpts_dir / f"checkpoint-{step}"

        # Save in standard LeRobot format: pretrained_model/ + training_state/
        pretrained_dir = ckpt_dir / "pretrained_model"
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        training_state_dir = ckpt_dir / "training_state"
        training_state_dir.mkdir(parents=True, exist_ok=True)

        # Model weights
        sft.save_file(
            dict(policy.state_dict().items()),
            str(pretrained_dir / "model.safetensors"),
        )

        # Norm stats (HVLA-specific, alongside model weights)
        torch.save(norm_stats, str(pretrained_dir / "norm_stats.pt"))

        # config.json — identifies this as an HVLA checkpoint
        (pretrained_dir / "config.json").write_text(json.dumps(checkpoint_config_dict(config), indent=2))

        # train_config.json — training args for reproducibility
        train_config = {
            "dataset": {"repo_id": args.dataset_repo_id},
            "s2_latent_path": args.s2_latent_path,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_delay": args.max_delay,
            "resize_images": args.resize_images,
        }
        (pretrained_dir / "train_config.json").write_text(json.dumps(train_config, indent=2))

        # Training state (optimizer, scheduler, step)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            str(training_state_dir / "optimizer.pt"),
        )
        (training_state_dir / "training_step.json").write_text(json.dumps({"step": step}))

        # Update 'last' symlink
        last_link = ckpts_dir / "last"
        if last_link.exists() or last_link.is_symlink():
            # safe-destruct: symlink update (not user data)
            last_link.unlink()
        last_link.symlink_to(ckpt_dir.name)

        logger.info("Saved checkpoint (step %d): %s", step, ckpt_dir)

    # Training loop
    policy.train()
    step = start_step
    data_iter = iter(dataloader)
    logger.info("Starting training from step %d to %d...", step, args.steps)
    # Resource telemetry rides this trainer's structured record rather than a
    # flat line, because that is the format it prints. The fields are flat
    # finite numbers, which is all the record accepts.
    resources = ResourceSampler()
    resources.start()
    health = TrainingHealthTracker(
        batch_size=args.batch_size,
        total_steps=args.steps,
        peak_memory_gb=(
            (lambda: torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else None
        ),
        reset_peak_memory=(
            (lambda: torch.cuda.reset_peak_memory_stats(device)) if device.type == "cuda" else None
        ),
    )

    while step < args.steps:
        with health.measure_data_loading():
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward with bf16 autocast
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, loss_dict = policy(batch)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        health.step()
        is_log_step = step <= 5 or step % 100 == 0

        if is_log_step:
            cur_lr = optimizer.param_groups[0]["lr"]
            # These scalar reads synchronize CUDA once per log window. Measure
            # the window after that synchronization so throughput and ETA
            # include real GPU execution without stalling every training step.
            loss_value = loss.item()
            flow_loss_value = float(loss_dict["flow_loss"])
            grad_norm_value = grad_norm.item()
            sample = health.sample(
                step=step,
                values={
                    # Telemetry first, so a future field that collides with a
                    # training metric loses to it rather than silently
                    # replacing it. Nothing about this run may depend on the
                    # sampler.
                    **resources.drain(),
                    "loss": loss_value,
                    "flow_loss": flow_loss_value,
                    "grdn": grad_norm_value,
                    "lr": cur_lr,
                },
            )
            if sample.omitted_fields:
                logger.warning(
                    "Non-finite training metrics at step %d (omitted from structured record): %s",
                    step,
                    ", ".join(sample.omitted_fields),
                )
            logger.info(
                "step %d/%d | loss: %.4f | flow_loss: %.4f | grdn: %.3f | lr: %.1e "
                "| updt_s: %.3f | data_s: %.3f | %.0fms | %s",
                step,
                args.steps,
                loss_value,
                flow_loss_value,
                grad_norm_value,
                cur_lr,
                sample.values["updt_s"],
                sample.values["data_s"],
                sample.values["step_time_ms"],
                sample.record,
            )

        if step % args.save_freq == 0:
            with health.exclude_time():
                save_checkpoint(step)

        if is_log_step:
            # Exclude logging and checkpoint I/O from the next training
            # window's throughput/ETA estimate.
            health.reset()

    # Final save
    save_checkpoint(step)
    resources.stop()
    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train Flow Matching S1 with Training-Time RTC")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument(
        "--s2-latent-path",
        default=None,
        help="Path to S2 latents .npy file. If omitted, S1 trains without S2 conditioning.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action horizon (50 at 30Hz = 1.67s)")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=15,
        help="Denoising steps at inference (15 → ~130ms, expected_d≈4 at 30fps)",
    )
    parser.add_argument(
        "--rtc-max-delay",
        type=int,
        default=6,
        help="Max simulated delay in frames for training-time RTC (15 denoise steps ≈ 5 frames delay)",
    )
    parser.add_argument(
        "--rtc-drop-prob", type=float, default=0.2, help="Probability of no prefix (simulates first chunk)"
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.0,
        help="Max S2 latent delay in seconds (0 = always use aligned latent)",
    )
    parser.add_argument("--resize-images", type=str, default="224x224")
    parser.add_argument(
        "--cameras",
        type=lambda v: [c.strip() for c in v.split(",") if c.strip()],
        default=None,
        help="Comma-separated cameras to train on (default: every camera in the "
        "dataset). Names may be bare (top_l) or full (observation.images.top_l). "
        "Recorded in the checkpoint, so inference requests exactly these.",
    )
    parser.add_argument(
        "--vision-encoder",
        type=str,
        default=vision_encoders.DEFAULT_ENCODER,
        choices=sorted(vision_encoders.VISION_ENCODERS),
        help="Patch-token backbone. DINOv3 weights are gated: accept the licence "
        "upstream and log in first; they are not redistributed here.",
    )
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint dir (e.g., outputs/flow_s1_hvla_v2/checkpoint-5000)",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
