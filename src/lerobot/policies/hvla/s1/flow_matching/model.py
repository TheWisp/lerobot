"""Flow Matching S1 Action Policy with Training-Time RTC.

Architecture:
  Encoder: DINOv2 image features + state + S2 latent + age → context tokens
  Decoder: Flow matching denoiser with cross-attention to context
  Action+time fusion: concat → MLP(SiLU) (matching Pi0/SmolVLA)

Training-Time RTC (arXiv:2512.05964):
  Instead of separate prefix tokens, we simulate inference delay by replacing
  the first D positions in the noisy action sequence with ground-truth (unnoised)
  actions, and setting their per-position timestep to t=0 (fully clean).
  The model learns to "inpaint" the remaining positions conditioned on the prefix.
  At inference, replace those positions with actually-executed actions.
  No architecture changes needed — just masking during the flow matching process.

Flow Matching (Lipman et al., ICLR 2023):
  Training: x_t = t * noise + (1-t) * actions, predict velocity v = noise - actions
  Inference: Euler integration from noise (t=1) to actions (t=0) over N steps

References:
  [1] Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
  [2] Black et al., "π₀: A Vision-Language-Action Flow Model", 2024
  [3] Mees et al., "Training-Time Action Conditioning for Efficient Real-Time
      Chunking", arXiv:2512.05964, 2025
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.protocol import S2_LATENT_KEY, S2_AGE_KEY, ACTION_PREFIX_KEY

OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
ACTION = "action"


def _sinusoidal_embedding(
    timesteps: Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0,
) -> Tensor:
    """Sinusoidal timestep embedding, matching Pi0's approach."""
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(min_period), math.log(max_period), half, device=timesteps.device)
    )
    args = timesteps[..., None] * freqs  # [..., half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [..., dim]


class FlowMatchingS1Model(nn.Module):
    """Core model: observation encoder + flow matching action decoder.

    The decoder follows Pi0/SmolVLA design:
    - Action tokens cross-attend to observation context
    - Action+timestep fused via concat → MLP(SiLU) (not additive)
    - Causal self-attention among action tokens
    """

    def __init__(self, config: FlowMatchingS1Config):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        # --- Image backbone (DINOv2) ---
        if config.use_dino_backbone:
            self.backbone = torch.hub.load("facebookresearch/dinov2", config.dino_model, pretrained=True)
            if config.freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            self._backbone_grad_ckpt = config.backbone_gradient_checkpointing and not config.freeze_backbone
            self.image_proj = nn.Linear(config.backbone_dim, d)
        else:
            self.backbone = None
            self.image_proj = None

        # --- State projection ---
        self.state_proj = nn.Linear(config.state_dim, d)

        # --- S2 latent projection + age embedding ---
        self.s2_proj = nn.Sequential(
            nn.Linear(config.s2_latent_dim, config.s2_proj_hidden),
            nn.GELU(),
            nn.Linear(config.s2_proj_hidden, d),
        )
        if config.use_s2_age_embedding:
            self.s2_age_embedding = nn.Sequential(
                nn.Linear(1, 64),
                nn.GELU(),
                nn.Linear(64, d),
            )
            # Zero-init output so age=0 → zeros (backward compatible)
            nn.init.zeros_(self.s2_age_embedding[2].weight)
            nn.init.zeros_(self.s2_age_embedding[2].bias)
        else:
            self.s2_age_embedding = None

        # --- Observation encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, batch_first=True,
        )
        self.obs_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        # --- Flow matching decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, batch_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        # --- Action projections (matching Pi0/SmolVLA) ---
        self.action_in_proj = nn.Linear(config.action_dim, d)
        self.action_out_proj = nn.Linear(d, config.action_dim)

        # --- Action+time fusion: concat → MLP(SiLU) (Pi0/SmolVLA style) ---
        # Input: concat(action_emb[d], time_emb[d]) = 2d → d
        self.action_time_mlp_in = nn.Linear(2 * d, d)
        self.action_time_mlp_out = nn.Linear(d, d)

        # --- Position embedding for action sequence ---
        self.action_pos_embed = nn.Embedding(config.chunk_size, d)

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode images + state + S2 latent → context tokens [B, N_ctx, D]."""
        tokens = []

        # Image features from DINOv2
        if self.backbone is not None:
            images = batch.get(OBS_IMAGES, [])
            if not images and self.config.image_features:
                images = [batch[k] for k in self.config.image_features]
            for img in images:
                if self.config.freeze_backbone:
                    with torch.no_grad():
                        features = self.backbone.forward_features(img)
                        patch_tokens = features["x_norm_patchtokens"]  # [B, 256, 768]
                elif self._backbone_grad_ckpt and self.training:
                    def _backbone_fwd(x):
                        return self.backbone.forward_features(x)["x_norm_patchtokens"]
                    patch_tokens = torch.utils.checkpoint.checkpoint(
                        _backbone_fwd, img, use_reentrant=False,
                    )
                else:
                    features = self.backbone.forward_features(img)
                    patch_tokens = features["x_norm_patchtokens"]
                tokens.append(self.image_proj(patch_tokens))

        # State token
        if OBS_STATE in batch:
            state_token = self.state_proj(batch[OBS_STATE]).unsqueeze(1)  # [B, 1, D]
            tokens.append(state_token)

        # S2 latent token + age
        if S2_LATENT_KEY in batch:
            s2_token = self.s2_proj(batch[S2_LATENT_KEY]).unsqueeze(1)  # [B, 1, D]
            if self.s2_age_embedding is not None and S2_AGE_KEY in batch:
                age = batch[S2_AGE_KEY]  # [B, 1]
                s2_token = s2_token + self.s2_age_embedding(age).unsqueeze(1)
            tokens.append(s2_token)

        context = torch.cat(tokens, dim=1)  # [B, N_ctx, D]
        context = self.obs_encoder(context)
        return context

    def denoise_step(
        self,
        x_t: Tensor,                # [B, chunk_size, action_dim]
        context: Tensor,             # [B, N_ctx, D]
        timestep: Tensor,            # [B, chunk_size] per-position timestep
    ) -> Tensor:
        """Single denoising step: predict velocity field v(x_t, t, context).

        Args:
            x_t: current (possibly partially clean) action sequence
            context: encoded observation tokens
            timestep: per-position timestep [B, chunk_size]. For training-time RTC,
                prefix positions have t=0 (clean), future positions have t=t_flow.

        Returns:
            velocity prediction [B, chunk_size, action_dim]
        """
        B, T, A = x_t.shape
        d = self.config.hidden_dim

        # Per-position sinusoidal time embedding [B, T, D]
        t_emb = _sinusoidal_embedding(timestep, d)  # [B, T, D]

        # Project actions
        action_emb = self.action_in_proj(x_t)  # [B, T, D]

        # Fuse action + time via concat → MLP(SiLU) (Pi0/SmolVLA style)
        action_time = torch.cat([action_emb, t_emb], dim=-1)  # [B, T, 2D]
        action_time = self.action_time_mlp_in(action_time)     # [B, T, D]
        action_time = F.silu(action_time)
        action_time = self.action_time_mlp_out(action_time)    # [B, T, D]

        # Add position embeddings
        pos_ids = torch.arange(T, device=x_t.device)
        action_time = action_time + self.action_pos_embed(pos_ids).unsqueeze(0)

        # Causal self-attention mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x_t.device, dtype=torch.bool), diagonal=1,
        )

        # Cross-attend to observation context
        decoded = self.action_decoder(
            tgt=action_time,
            memory=context,
            tgt_mask=causal_mask,
        )

        velocity = self.action_out_proj(decoded)  # [B, T, action_dim]
        return velocity

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass with training-time RTC.

        Implements arXiv:2512.05964: simulate inference delay by replacing
        the first D action positions with ground-truth (unnoised) actions,
        and setting their per-position timestep to 0 (fully clean).

        The model learns to predict velocity for the entire sequence, but
        the prefix positions are already at the target — their velocity
        target is zero (noise - actions = 0 when x_t = actions).
        """
        actions = batch[ACTION]  # [B, T, action_dim]
        B, T, A = actions.shape
        device = actions.device

        # Encode observations (shared across all positions)
        context = self.encode_observations(batch)

        # Sample flow matching time from Beta distribution (scalar per sample)
        t_beta = torch.distributions.Beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
        ).sample((B,)).to(device)
        t_flow = t_beta * (self.config.time_max - self.config.time_min) + self.config.time_min

        # Sample noise
        noise = torch.randn_like(actions)

        # Noisy actions: x_t = t * noise + (1-t) * actions
        t_expand = t_flow[:, None, None]  # [B, 1, 1]
        x_t = t_expand * noise + (1 - t_expand) * actions

        # Velocity target: u = noise - actions
        u_target = noise - actions

        # --- Training-time RTC (arXiv:2512.05964) ---
        # Sample delay D ~ Uniform(0, max_delay) per sample.
        # With probability rtc_drop_prob, set D=0 (no prefix, simulates first chunk).
        # Replace x_t[:, :D] with ground-truth actions (unnoised).
        # Set per-position timestep: t[:D] = 0 (clean), t[D:] = t_flow.
        per_pos_t = t_flow[:, None].expand(B, T).clone()  # [B, T]

        # --- Training-time RTC delay sampling ---
        # Per arXiv:2512.05964 and Ψ₀ (arXiv:2603.12263):
        # Sample delay d, replace x_t[:,:d] with clean actions, set per-pos t=0,
        # and EXCLUDE prefix positions from loss.
        #
        # Delay distribution: centered on realistic inference delay (2-3 frames),
        # with prefix dropout for episode-start simulation.
        loss_mask = torch.ones(B, T, 1, device=device)  # 1 = include in loss

        if self.config.rtc_max_delay > 0:
            max_d = min(self.config.rtc_max_delay, T - 1)

            # Sample delay: Uniform(1, max_d) for most samples, 0 for dropout
            delays = torch.randint(1, max_d + 1, (B,), device=device)

            # Prefix dropout: with probability rtc_drop_prob, set delay=0 (no prefix)
            drop_mask = torch.rand(B, device=device) < self.config.rtc_drop_prob
            delays = delays * (~drop_mask).long()

            # Apply prefix and build loss mask
            for b in range(B):
                d = delays[b].item()
                if d > 0:
                    x_t[b, :d] = actions[b, :d]    # clean actions (no noise)
                    per_pos_t[b, :d] = 0.0           # timestep = 0 for prefix
                    loss_mask[b, :d] = 0.0           # exclude prefix from loss

        # Predict velocity with per-position timesteps
        v_pred = self.denoise_step(x_t, context, per_pos_t)

        # MSE loss on velocity — prefix positions excluded (arXiv:2512.05964, Ψ₀)
        if "action_is_pad" in batch:
            loss_mask = loss_mask * (~batch["action_is_pad"].unsqueeze(-1)).float()
        mse = F.mse_loss(v_pred, u_target, reduction="none")  # [B, T, A]
        loss = (mse * loss_mask).sum() / loss_mask.sum().clamp(min=1.0) / A

        loss_dict = {"flow_loss": loss.item()}
        return loss, loss_dict

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        num_steps: int | None = None,
        action_prefix: Tensor | None = None,
        prefix_len: int = 0,
    ) -> Tensor:
        """Generate actions via Euler integration with optional RTC prefix.

        At inference, the prefix (actually-executed actions) replaces the first
        `prefix_len` positions at every denoising step, matching the training-time
        RTC conditioning. No gradient guidance needed.

        Args:
            batch: observation batch
            num_steps: override denoising steps
            action_prefix: [B, D, action_dim] actually-executed actions for RTC
            prefix_len: number of prefix positions (D). If 0, no RTC.

        Returns:
            [B, chunk_size, action_dim] — generated action chunk
        """
        num_steps = num_steps or self.config.num_inference_steps
        device = next(self.parameters()).device

        for v in batch.values():
            if isinstance(v, Tensor):
                B = v.shape[0]
                break

        # Encode observations once (reused across denoising steps)
        context = self.encode_observations(batch)

        # Build per-position timestep template
        # Prefix positions: t=0 (clean), future positions: t=t_denoise
        T = self.config.chunk_size

        # Start from noise
        x_t = torch.randn(B, T, self.config.action_dim, device=device)

        # If we have a prefix, inject it into the initial noise
        if action_prefix is not None and prefix_len > 0:
            D = min(prefix_len, T - 1)
            x_t[:, :D] = action_prefix[:, :D]

        # Euler integration: t goes from 1.0 to 0.0
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t_val = 1.0 + i * dt

            # Per-position timestep: prefix=0 (clean), rest=t_val
            per_pos_t = torch.full((B, T), t_val, device=device)
            if action_prefix is not None and prefix_len > 0:
                per_pos_t[:, :D] = 0.0

            v = self.denoise_step(x_t, context, per_pos_t)
            x_t = x_t + dt * v

            # Re-inject prefix after each step (inpainting)
            if action_prefix is not None and prefix_len > 0:
                x_t[:, :D] = action_prefix[:, :D]

        return x_t


class FlowMatchingS1Policy(nn.Module):
    """Policy wrapper matching the S1Policy protocol.

    Implements training-time RTC (arXiv:2512.05964) for smooth chunk transitions.
    At inference, previously-executed actions serve as the prefix — no gradient
    guidance or architecture changes needed vs training.
    """

    def __init__(self, config: FlowMatchingS1Config):
        super().__init__()
        self.config = config
        self.model = FlowMatchingS1Model(config)
        self._action_queue = deque()
        # Normalization stats (loaded from checkpoint dir)
        self._action_mean = None  # [action_dim]
        self._action_std = None   # [action_dim]
        self._state_mean = None   # [state_dim]
        self._state_std = None    # [state_dim]

    @property
    def supports_rtc(self) -> bool:
        return self.config.rtc_max_delay > 0

    @property
    def needs_temporal_ensemble(self) -> bool:
        return False

    @property
    def rtc_prefix_length(self) -> int:
        return self.config.rtc_max_delay

    def reset(self) -> None:
        self._action_queue.clear()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict action chunk via flow matching with RTC inpainting.

        RTC prefix (ACTION_PREFIX_KEY) should contain actions from the PREVIOUS
        chunk's predictions (the overlap portion), in raw (unnormalized) space.
        The prefix length d equals the actual measured inference delay in frames.

        Returns: [B, chunk_size, action_dim] in original (unnormalized) action space
        """
        self.eval()
        batch = dict(batch)
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # Normalize state input
        if self._state_mean is not None and "observation.state" in batch:
            device = batch["observation.state"].device
            batch["observation.state"] = (
                batch["observation.state"] - self._state_mean.to(device)
            ) / self._state_std.to(device)

        # RTC prefix: previous chunk's predictions (raw space → normalize)
        action_prefix = batch.pop(ACTION_PREFIX_KEY, None)
        if action_prefix is not None and self._action_mean is not None:
            device = action_prefix.device
            action_prefix = (action_prefix - self._action_mean.to(device)) / self._action_std.to(device)
        prefix_len = action_prefix.shape[1] if action_prefix is not None else 0

        # Model predicts in normalized space
        actions_norm = self.model.sample_actions(
            batch,
            action_prefix=action_prefix,
            prefix_len=prefix_len,
        )

        # Denormalize output
        if self._action_mean is not None:
            device = actions_norm.device
            actions_norm = actions_norm * self._action_std.to(device) + self._action_mean.to(device)

        return actions_norm

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward: compute flow matching loss with training-time RTC."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return self.model(batch)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: FlowMatchingS1Config | None = None):
        """Load from checkpoint. Also loads norm_stats.pt if present."""
        import safetensors.torch
        from pathlib import Path

        if config is None:
            config = FlowMatchingS1Config()

        policy = cls(config)

        state_dict = safetensors.torch.load_file(checkpoint_path)
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing:
            import logging
            logging.warning("Missing keys: %s", missing)
        if unexpected:
            import logging
            logging.warning("Unexpected keys: %s", unexpected)

        # Load normalization stats from same directory
        ckpt_dir = Path(checkpoint_path).parent
        norm_path = ckpt_dir / "norm_stats.pt"
        if norm_path.exists():
            import logging
            norm_stats = torch.load(norm_path, weights_only=True)
            policy._action_mean = norm_stats.get("action_mean")
            policy._action_std = norm_stats.get("action_std")
            policy._state_mean = norm_stats.get("state_mean")
            policy._state_std = norm_stats.get("state_std")
            logging.info("Loaded norm stats from %s", norm_path)
        else:
            import logging
            logging.warning("No norm_stats.pt found at %s — running without normalization", ckpt_dir)

        return policy
