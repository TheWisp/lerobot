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

import logging
import math
import pathlib
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.protocol import ACTION_PREFIX_KEY, S2_AGE_KEY, S2_LATENT_KEY

OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
ACTION = "action"


def _sinusoidal_embedding(
    timesteps: Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> Tensor:
    """Sinusoidal timestep embedding, matching Pi0's approach."""
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(min_period), math.log(max_period), half, device=timesteps.device)
    )
    args = timesteps[..., None] * freqs  # [..., half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [..., dim]


def rtc_prefix_weights(
    delays: Tensor,
    chunk_size: int,
    soft_len: int,
    soft_hmax: int,
) -> Tensor:
    """Per-token RTC conditioning weights w_j in [0, 1] (arXiv:2605.25537, Eq. 'omega').

    Three regions per sample, given its delay d and endpoint
    e(d) = min(d + soft_len, soft_hmax):

        j < d          w = 1     committed prefix, fully clamped, no loss
        d <= j < e(d)  w = g(.)  soft window, partly prior-informed, in the loss
        j >= e(d)      w = 0     free tail, ordinary flow matching

    ``g`` is the linear schedule ``linspace(1, 0, n + 2)[1:-1]`` over a window of
    n tokens, i.e. strictly inside (0, 1) at both ends. Excluding the endpoints
    matters: g(0) == 1 would leave the first executed action fully clamped and
    therefore weightless in the loss, which is the very thing Soft RTC exists to
    avoid. This is also the schedule upstream LeRobot uses for inference-time
    RTC (``policies/rtc/modeling_rtc.py::_linweights``).

    Preconditions: ``delays`` is a 1-D integer tensor of per-sample delays, each
    in [0, chunk_size]; ``soft_len >= 0``; ``soft_hmax >= 0``.
    Postconditions: returns [B, chunk_size] float in [0, 1], non-increasing along
    j within each row. ``soft_len == 0`` yields exactly the binary mask
    ``1[j < d]``, and ``d == 0`` yields an all-zero row (the paper sets
    e(0) = 0, so a dropped prefix means ordinary flow matching).
    """
    assert delays.ndim == 1, f"expected 1-D delays, got shape {tuple(delays.shape)}"
    assert soft_len >= 0 and soft_hmax >= 0, "soft window bounds must be non-negative"

    device = delays.device
    B = delays.shape[0]
    d = delays.long()[:, None]  # [B, 1]
    j = torch.arange(chunk_size, device=device)[None, :]  # [1, T]

    # e(d) = min(d + L, hmax), never before d, and never past the chunk.
    e = torch.clamp(d + soft_len, max=min(soft_hmax, chunk_size))
    e = torch.maximum(e, d)

    omega = (j < d).float()  # committed prefix

    if soft_len > 0:
        n = (e - d).clamp(min=0)  # [B, 1] soft-window length per sample
        # linspace(1, 0, n+2)[1:-1] evaluated positionally: the k-th of n tokens
        # (k = j - d, zero-based) takes value 1 - (k + 1) / (n + 1).
        k = (j - d).float()
        g = 1.0 - (k + 1.0) / (n.float() + 1.0)
        in_soft = (j >= d) & (j < e)
        omega = torch.where(in_soft, g.clamp(0.0, 1.0), omega)

    # A dropped prefix (d == 0) means no conditioning at all.
    omega = torch.where(d == 0, torch.zeros_like(omega), omega)

    assert omega.shape == (B, chunk_size)
    return omega


class FlowMatchingS1Model(nn.Module):
    """Core model: observation encoder + flow matching action decoder.

    The decoder follows Pi0/SmolVLA design:
    - Action tokens cross-attend to observation context
    - Action+timestep fused via concat → MLP(SiLU) (not additive)
    - Causal self-attention among action tokens
    """

    def __init__(self, config: FlowMatchingS1Config):
        super().__init__()
        config.validate_feature_contract()
        self.config = config
        d = config.hidden_dim

        # --- Image backbone ---
        if config.use_dino_backbone:
            from .vision_encoders import actual_embed_dim, load_backbone

            self.backbone = load_backbone(config.dino_model)
            # backbone_dim is a second config field that has to agree with the
            # chosen encoder; when it does not, the mismatch would otherwise
            # surface as a shape error inside image_proj on the first batch.
            true_dim = actual_embed_dim(self.backbone)
            if true_dim is not None and true_dim != config.backbone_dim:
                raise ValueError(
                    f"Vision encoder {config.dino_model!r} produces {true_dim}-d patch tokens but "
                    f"backbone_dim is {config.backbone_dim}. Set backbone_dim={true_dim} (training "
                    "resolves this automatically; a hand-written config must match)."
                )
            if config.freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            self._backbone_grad_ckpt = config.backbone_gradient_checkpointing and not config.freeze_backbone
            self.image_proj = nn.Linear(config.backbone_dim, d)
        else:
            self.backbone = None
            self.image_proj = None

        # --- State projection ---
        self.state_proj = nn.Linear(config.state_dim, d) if config.robot_state_feature else None

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
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.obs_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        # --- Flow matching decoder (with cross-attention KV caching) ---
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d,
                    nhead=config.num_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

        # --- Action projections (matching Pi0/SmolVLA) ---
        self.action_in_proj = nn.Linear(config.action_dim, d)
        self.action_out_proj = nn.Linear(d, config.action_dim)

        # --- Action+time fusion: concat → MLP(SiLU) (Pi0/SmolVLA style) ---
        # Input: concat(action_emb[d], time_emb[d]) = 2d → d
        self.action_time_mlp_in = nn.Linear(2 * d, d)
        self.action_time_mlp_out = nn.Linear(d, d)

        # --- Position embedding for action sequence ---
        self.action_pos_embed = nn.Embedding(config.chunk_size, d)

        # Context-token layout (patches-per-camera etc.), recorded by encode_observations —
        # the attention-rollout overlay (flow_matching/saliency.py) slices per camera with it.
        self._ctx_layout: dict | None = None

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode images + state + S2 latent → context tokens [B, N_ctx, D]."""
        tokens = []

        # Image features from DINOv2 (batched across cameras for efficiency)
        if self.backbone is not None:
            images = batch.get(OBS_IMAGES, [])
            if not images and self.config.image_features:
                images = [batch[k] for k in self.config.image_features]
            if images:
                B = images[0].shape[0]
                N_cams = len(images)
                # Stack all cameras into one batch: [B*N_cams, C, H, W]
                stacked = torch.cat(images, dim=0)
                if self.config.freeze_backbone:
                    with torch.no_grad():
                        features = self.backbone.forward_features(stacked)
                        all_patches = features["x_norm_patchtokens"]  # [B*N_cams, 256, 768]
                elif self._backbone_grad_ckpt and self.training:

                    def _backbone_fwd(x):
                        return self.backbone.forward_features(x)["x_norm_patchtokens"]

                    all_patches = torch.utils.checkpoint.checkpoint(
                        _backbone_fwd,
                        stacked,
                        use_reentrant=False,
                    )
                else:
                    features = self.backbone.forward_features(stacked)
                    all_patches = features["x_norm_patchtokens"]
                # Camera patch blocks occupy the FIRST N_cams*patches positions of the context
                # (state + S2 tokens follow); record the layout so capture can slice per camera.
                self._ctx_layout = {"n_cams": N_cams, "patches_per_cam": int(all_patches.shape[1])}
                # Split back per camera and project
                per_cam = all_patches.reshape(N_cams, B, all_patches.shape[1], all_patches.shape[2])
                for i in range(N_cams):
                    tokens.append(self.image_proj(per_cam[i]))

        # State token
        if self.state_proj is not None and OBS_STATE in batch:
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

    def precompute_cross_attn_kv(self, context: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Pre-compute cross-attention K,V from context for all decoder layers.

        Called once before the denoising loop. The cached K,V are reused
        across all denoise steps since context doesn't change.
        """
        cached_kv = []
        for layer in self.decoder_layers:
            # nn.TransformerDecoderLayer stores cross-attention as multihead_attn
            mha = layer.multihead_attn
            # Project context through K,V weights (in_proj contains Q,K,V stacked)
            # For nn.MultiheadAttention with batch_first=True:
            #   in_proj_weight is [3*d, d], split into Q, K, V
            d = mha.embed_dim
            w = mha.in_proj_weight  # [3*d, d]
            b = mha.in_proj_bias  # [3*d]
            # K = context @ W_K^T + b_K, V = context @ W_V^T + b_V
            k = F.linear(context, w[d : 2 * d], b[d : 2 * d] if b is not None else None)
            v = F.linear(context, w[2 * d : 3 * d], b[2 * d : 3 * d] if b is not None else None)
            # Reshape for multi-head: [B, N, D] → [B, N, nhead, head_dim] → [B*nhead, N, head_dim]
            # Actually, keep as [B, N, D] — we'll handle heads in the forward
            cached_kv.append((k, v))
        return cached_kv

    def denoise_step(
        self,
        x_t: Tensor,  # [B, chunk_size, action_dim]
        context: Tensor,  # [B, N_ctx, D]
        timestep: Tensor,  # [B, chunk_size] per-position timestep
        cached_kv: list[tuple[Tensor, Tensor]] | None = None,
    ) -> Tensor:
        """Single denoising step: predict velocity field v(x_t, t, context).

        Args:
            x_t: current (possibly partially clean) action sequence
            context: encoded observation tokens
            timestep: per-position timestep [B, chunk_size]. For training-time RTC,
                prefix positions have t=0 (clean), future positions have t=t_flow.
            cached_kv: pre-computed cross-attention K,V (from precompute_cross_attn_kv)

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
        action_time = self.action_time_mlp_in(action_time)  # [B, T, D]
        action_time = F.silu(action_time)
        action_time = self.action_time_mlp_out(action_time)  # [B, T, D]

        # Add position embeddings
        pos_ids = torch.arange(T, device=x_t.device)
        action_time = action_time + self.action_pos_embed(pos_ids).unsqueeze(0)

        # Decoder with optional KV cache for cross-attention
        x = action_time
        for i, layer in enumerate(self.decoder_layers):
            if cached_kv is not None:
                # Post-norm decoder (matching nn.TransformerDecoderLayer norm_first=False):
                #   x = norm1(x + self_attn(x))
                #   x = norm2(x + cross_attn(x, memory))
                #   x = norm3(x + ffn(x))
                # Self-attention (unchanged)
                x = layer.norm1(x + layer.dropout1(layer.self_attn(x, x, x, need_weights=False)[0]))
                # Cross-attention with pre-computed K,V
                ck, cv = cached_kv[i]
                mha = layer.multihead_attn
                q = F.linear(
                    x, mha.in_proj_weight[:d], mha.in_proj_bias[:d] if mha.in_proj_bias is not None else None
                )
                nhead = mha.num_heads
                head_dim = d // nhead
                q = q.reshape(B, T, nhead, head_dim).transpose(1, 2)
                k = ck.reshape(B, -1, nhead, head_dim).transpose(1, 2)
                v = cv.reshape(B, -1, nhead, head_dim).transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).reshape(B, T, d)
                attn_out = mha.out_proj(attn_out)
                x = layer.norm2(x + layer.dropout2(attn_out))
                # FFN
                x = layer.norm3(
                    x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))
                )
            else:
                x = layer(x, context, tgt_mask=None)

        velocity = self.action_out_proj(x)  # [B, T, action_dim]
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
        t_beta = (
            torch.distributions.Beta(
                self.config.time_sampling_beta_alpha,
                self.config.time_sampling_beta_beta,
            )
            .sample((B,))
            .to(device)
        )
        t_flow = t_beta * (self.config.time_max - self.config.time_min) + self.config.time_min

        # Sample noise
        noise = torch.randn_like(actions)

        # Velocity target: u = noise - actions
        u_target = noise - actions

        # --- Training-time RTC, in its Soft RTC form ---
        # Hard RTC (arXiv:2512.05964, Ψ₀ arXiv:2603.12263) clamps positions
        # [0, d) to clean ground truth, gives them per-position timestep 0, and
        # drops them from the loss. Soft RTC (arXiv:2605.25537) is the same
        # construction with the binary mask 1[j < d] replaced by continuous
        # weights w_j, which is what the two lines below express:
        #
        #     t_j = (1 - w_j) * t_flow          per-token flow time
        #     x_t = t_j * noise + (1 - t_j) * A per-token interpolation
        #
        # (Our time convention is flipped relative to both papers: here t=0 is
        # clean data and t=1 is noise, so w=1 maps to t=0 rather than t=1.)
        #
        # Loss weight is (1 - w_j): fully clamped tokens contribute nothing,
        # soft tokens contribute in proportion to how editable they are, free
        # tokens contribute fully. At rtc_soft_len == 0 every line here reduces
        # to the previous binary behaviour exactly.
        #
        # Delay distribution is unchanged: Uniform(1, max_d), with a
        # rtc_drop_prob chance of d=0 standing in for the first chunk of an
        # episode.
        if self.config.rtc_max_delay > 0:
            max_d = min(self.config.rtc_max_delay, T - 1)
            delays = torch.randint(1, max_d + 1, (B,), device=device)
            drop_mask = torch.rand(B, device=device) < self.config.rtc_drop_prob
            delays = delays * (~drop_mask).long()
            omega = rtc_prefix_weights(
                delays, T, self.config.rtc_soft_len, self.config.rtc_soft_hmax
            )  # [B, T]
        else:
            omega = torch.zeros(B, T, device=device)

        per_pos_t = (1.0 - omega) * t_flow[:, None]  # [B, T]
        t_expand = per_pos_t[..., None]  # [B, T, 1]
        x_t = t_expand * noise + (1 - t_expand) * actions
        loss_mask = (1.0 - omega).unsqueeze(-1)  # [B, T, 1]

        # Predict velocity with per-position timesteps
        v_pred = self.denoise_step(x_t, context, per_pos_t)

        # MSE on velocity, weighted by how editable each token is.
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
        context: Tensor | None = None,
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
            context: [B, N_ctx, D] pre-computed context tokens. When provided,
                the internal ``encode_observations`` call is skipped. Used by
                the RLT inference path, which already computes the same
                context for the RL token encoder — avoids doing one DINOv2
                forward twice per inference.

        Returns:
            [B, chunk_size, action_dim] — generated action chunk
        """
        num_steps = num_steps or self.config.num_inference_steps
        device = next(self.parameters()).device

        for v in batch.values():
            if isinstance(v, Tensor):
                B = v.shape[0]
                break

        # Encode observations once (reused across denoising steps) — unless
        # the caller pre-computed it. They must have normalized the batch the
        # same way prepare_batch_for_encode_observations does; otherwise the
        # supplied context is inconsistent with the prefix/state on the batch.
        if context is None:
            context = self.encode_observations(batch)

        # Pre-compute cross-attention K,V (reused across all denoise steps)
        cached_kv = self.precompute_cross_attn_kv(context)

        # Build per-position timestep template
        # Prefix positions: t=0 (clean), future positions: t=t_denoise
        T = self.config.chunk_size

        # Start from noise
        x_t = torch.randn(B, T, self.config.action_dim, device=device)

        # Soft RTC inference (arXiv:2605.25537, Algorithm 1). Per denoising step
        # the previous chunk Y is blended into the denoiser input rather than
        # written over it, using the same weights the loss was trained with:
        #
        #     x~[j] = w_j * Y[j] + (1 - w_j) * x[j]
        #     t~[j] = (1 - w_j) * t          (flipped convention: 0 is clean)
        #     x    <- x~ + dt * v(x~, t~)
        #
        # Note the update is applied to the BLENDED state, not the pre-blend
        # one, so at w=1 positions each step starts from exactly Y. That makes
        # this identical to the previous "inject before the loop, re-inject
        # after every step" formulation whenever w is binary — the injection
        # simply moved from the bottom of the iteration to the top, which is
        # also where both reference implementations put it.
        #
        # Y needs to cover the soft window, not just the committed prefix:
        # e(d) = min(d + soft_len, soft_hmax) positions. Callers that supply
        # only d rows still work — the window is clipped to what they gave us.
        omega = None
        if action_prefix is not None and prefix_len > 0:
            D = min(prefix_len, T - 1)
            avail = min(action_prefix.shape[1], T)
            soft_len = min(self.config.rtc_soft_len, max(avail - D, 0))
            omega = rtc_prefix_weights(
                torch.tensor([D], device=device),
                T,
                soft_len,
                self.config.rtc_soft_hmax,
            )[0]  # [T]
            prior = torch.zeros(B, T, self.config.action_dim, device=device)
            prior[:, :avail] = action_prefix[:, :avail]
            w = omega[None, :, None]  # [1, T, 1]

        # Euler integration: t goes from 1.0 to 0.0
        dt = -1.0 / num_steps
        prefix_drift = None
        for i in range(num_steps):
            t_val = 1.0 + i * dt

            if omega is None:
                per_pos_t = torch.full((B, T), t_val, device=device)
            else:
                x_t = w * prior + (1.0 - w) * x_t
                per_pos_t = (1.0 - omega)[None, :] * t_val

            v = self.denoise_step(x_t, context, per_pos_t, cached_kv=cached_kv)
            x_t = x_t + dt * v

            # Measure prefix drift BEFORE re-inject (how much the model
            # perturbed the prefix positions).
            #
            # This is NOT a health signal, though it was read as one for a
            # while. Under x_t = t*noise + (1-t)*A with target v = noise - A, a
            # model handed a clean action at t=0 must predict E[noise - A | A]
            # = -A, not 0. So the Euler step at the pinned positions is
            # dt*v = (-1/N)(-A) = +A/N, and a drift of |A|/N per step is what a
            # correctly trained model produces. Measured on checkpoint-50000:
            # predicted 0.396 deg, actual 0.462 deg, best-fit slope 0.955,
            # r=0.853 — the residual is the only part that carries information.
            # A drift near zero would mean the model was ignoring the timestep
            # conditioning, which is the opposite of healthy.
            #
            # TODO(review): raised during the RTC seam investigation; confirm
            # this reading in the PR before relying on prefix_drift anywhere.
            if omega is not None:
                prefix_drift = (x_t[:, :D] - action_prefix[:, :D]).abs().mean().item()
                # What the model itself put at the pinned positions, before the
                # stomp. The returned chunk[0:D] is always exactly the prefix,
                # so without this there is no way to tell agreement from a
                # forcibly-suppressed disagreement. Only the final step is kept
                # — earlier ones are overwritten anyway — so this is one small
                # async device copy per inference, not one per denoise step.
                if i == num_steps - 1:
                    self._last_prefix_pre_inject = x_t[:, :D].detach().clone()
                    # Every earlier iteration re-establishes the committed
                    # tokens via the blend at the top of the loop; only the last
                    # one has no successor, so restore them here to keep the
                    # long-standing invariant that chunk[0:D] is exactly the
                    # prefix. Soft-window tokens are deliberately left alone —
                    # they are executed, and their whole purpose is to be
                    # editable.
                    hard = (omega >= 1.0)[None, :, None]
                    x_t = torch.where(hard, prior, x_t)

        # Store drift on the instance for external access
        self._last_prefix_drift = prefix_drift
        if action_prefix is None or prefix_len <= 0:
            self._last_prefix_pre_inject = None

        return x_t


#: Bound on the normalized state fed to the network, applied identically here
#: and in ``FlowMatchingDataset``, so training and serving see one transform.
#:
#: It exists for the degenerate case, which is closer than it sounds. A joint
#: held still for a whole recording gets a std at the 1e-6 numerical floor;
#: dividing by that, a difference far below sensor resolution becomes enormous.
#: Measured on GPU/0803_20260803_174402: left_joint_3.pos has mean 0.9508 and
#: std 1.0e-06, and a rig reading of 0.732 -- 0.22 degrees away -- normalized to
#: 218,569 sigma. One channel that size dominates the first linear layer and
#: corrupts the output for every joint, including the ones doing the work.
#:
#: 10 is not free. Measured on GPU/0803_20260803_174402, 0.66% of training
#: frames have at least one feature beyond it. Most are dead channels, but real
#: motion reaches it too: right_joint_7.vel peaks at 20.5 sigma on a joint whose
#: native std is 17 deg/s. Those transients are truncated, on both sides of
#: training, in exchange for bounding the degenerate case.
NORMALIZED_STATE_CLAMP = 10.0

_clamp_log = logging.getLogger(__name__)


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
        self._action_std = None  # [action_dim]
        # Features already reported by the normalized-state clamp, so a
        # 30 Hz inference loop logs each one once rather than every frame.
        self._clamped_state_features: set[str] = set()
        self._state_mean = None  # [state_dim]
        self._state_std = None  # [state_dim]

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

    def _relative_action_reference(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Select current raw state positions in checkpoint action order.

        Arm joints are relative while grippers remain absolute.  Name-based
        selection is required because OpenArm state interleaves position,
        velocity, and torque; the first ``action_dim`` state values are not the
        action-position vector.
        """
        if not self.config.use_relative_actions:
            return None
        if OBS_STATE not in batch:
            raise ValueError("Relative-action Flow S1 inference requires observation.state")

        state_indices = torch.tensor(
            [self.config.state_feature_names.index(name) for name in self.config.action_feature_names],
            dtype=torch.long,
            device=batch[OBS_STATE].device,
        )
        relative_mask = torch.tensor(
            [
                name.endswith(".pos") and "gripper" not in name.lower()
                for name in self.config.action_feature_names
            ],
            dtype=batch[OBS_STATE].dtype,
            device=batch[OBS_STATE].device,
        )
        return batch[OBS_STATE].index_select(-1, state_indices) * relative_mask

    def prepare_batch_for_encode_observations(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Return a shallow copy of ``batch`` normalized exactly the way
        ``FlowMatchingDataset`` does at training time, ready to pass to
        ``self.model.encode_observations``.

        Contract — the resulting batch:
          * has ``observation.state`` z-scored with training-time mean/std
            (if the policy has norm_stats)
          * has ``OBS_IMAGES`` populated from ``self.config.image_features``
            (so ``encode_observations`` takes the fast path)
          * leaves ``ACTION_PREFIX_KEY`` untouched here (RTC prefix handling
            lives with ``predict_action_chunk``, since it's denoise-specific)

        Used by ``predict_action_chunk`` for its own denoise pass, and by
        external callers that feed ``encode_observations`` directly (RLT
        inference needs context tokens for the RL token encoder). Sharing
        this one helper is what eliminates the train/infer state mismatch:
        both paths normalize identically.
        """
        batch = dict(batch)
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        if self._state_mean is not None and "observation.state" in batch:
            device = batch["observation.state"].device
            normalized = (batch["observation.state"] - self._state_mean.to(device)) / self._state_std.to(
                device
            )
            batch["observation.state"] = self._clamp_normalized_state(normalized)
        return batch

    def _clamp_normalized_state(self, normalized: Tensor) -> Tensor:
        """Bound the normalized state to +/-NORMALIZED_STATE_CLAMP, logging once per feature.

        What it exists for is a joint the task leaves still, whose training std
                lands on the 1e-6 numerical floor and turns a sub-degree reading
                difference into tens of thousands of sigma. Clamping the *result*
                rather than the denominator is deliberate: it is unit-independent, so
                one bound covers pos, vel and torque, whereas the ``.pos`` std floor
                leaves torque channels reaching 41 sigma untouched.

                ``FlowMatchingDataset`` applies the same bound, so this is not a
                train/serve skew for models trained after it landed. For a checkpoint
                trained before it, this clamp is applied at inference only -- a skew on
                the ~0.7% of in-distribution frames that reach it, taken deliberately
                because the alternative is an unbounded input.

                Logged, not silent — otherwise a broken encoder and a slightly-off rest
                pose produce identical inputs and neither is visible.
        """
        exceeded = normalized.abs() > NORMALIZED_STATE_CLAMP
        if exceeded.any():
            names = list(self.config.state_feature_names or [])
            flat = exceeded.any(dim=tuple(range(normalized.ndim - 1))) if normalized.ndim > 1 else exceeded
            for i in flat.nonzero(as_tuple=False).flatten().tolist():
                name = names[i] if i < len(names) else f"state[{i}]"
                if name in self._clamped_state_features:
                    continue
                self._clamped_state_features.add(name)
                worst = normalized[..., i].abs().max().item()
                train_std = (
                    float(self._state_std.flatten()[i]) if self._state_std is not None else float("nan")
                )
                _clamp_log.warning(
                    "Normalized state for %s reached %.4g sigma; clamped to %.0f. Its training "
                    "std is %.3g in dataset units, so the joint barely moved during recording "
                    "and a difference this small from the recorded mean is amplified enormously. "
                    "Raise --state-position-std-floor, or exclude the channel, rather than "
                    "hunting for a pose discrepancy. Reported once per feature.",
                    name,
                    worst,
                    NORMALIZED_STATE_CLAMP,
                    train_std,
                )
        return normalized.clamp(-NORMALIZED_STATE_CLAMP, NORMALIZED_STATE_CLAMP)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        num_steps: int | None = None,
        context: Tensor | None = None,
        prefix_len: int | None = None,
    ) -> Tensor:
        """Predict action chunk via flow matching with RTC inpainting.

        RTC prefix (ACTION_PREFIX_KEY) should contain actions from the PREVIOUS
        chunk's predictions (the overlap portion), in raw (unnormalized) space.
        The prefix length d equals the actual measured inference delay in frames.

        ``context``: pre-computed context tokens from ``encode_observations``.
        When supplied, the internal encode_observations call inside
        ``sample_actions`` is skipped. Caller is responsible for having
        computed context on a batch that matches what
        ``prepare_batch_for_encode_observations`` would produce (same state
        normalization + image-keys mapping). Used by the RLT inference path
        to avoid redundant DINOv2 forwards.

        Returns: [B, chunk_size, action_dim] in original (unnormalized) action space
        """
        self.eval()
        relative_reference = self._relative_action_reference(batch)
        # Shared prep (images, state normalization) — exact same transform
        # that FlowMatchingDataset applies at training time. Also used by
        # external callers like RLT inference.
        batch = self.prepare_batch_for_encode_observations(batch)

        # RTC prefix: previous chunk's predictions (raw space → normalize).
        # Kept out of ``prepare_batch_for_encode_observations`` because only
        # the denoising pass needs the prefix; the encode_observations path
        # doesn't consume it.
        action_prefix = batch.pop(ACTION_PREFIX_KEY, None)
        if action_prefix is not None and relative_reference is not None:
            action_prefix = action_prefix - relative_reference.unsqueeze(1)
        if action_prefix is not None and self._action_mean is not None:
            device = action_prefix.device
            action_prefix = (action_prefix - self._action_mean.to(device)) / self._action_std.to(device)
        # Rows handed in vs positions committed. Hard RTC passes exactly d rows
        # and these coincide. Soft RTC passes e(d) = d + soft_len rows so the
        # blended window has a prior; only the first d are committed, and
        # inferring the delay from the row count instead would hard-pin the
        # whole window and silently disable Soft RTC.
        if action_prefix is None:
            prefix_len = 0
        elif prefix_len is None:
            prefix_len = action_prefix.shape[1]
        else:
            assert 0 < prefix_len <= action_prefix.shape[1], (
                f"prefix_len={prefix_len} must be within the {action_prefix.shape[1]} "
                "rows supplied"
            )

        # Model predicts in normalized space
        actions_norm = self.model.sample_actions(
            batch,
            num_steps=num_steps,
            action_prefix=action_prefix,
            prefix_len=prefix_len,
            context=context,
        )

        # Denormalize output
        inner = self.model if hasattr(self, "model") else self
        _pre = getattr(inner, "_last_prefix_pre_inject", None)
        if self._action_mean is not None:
            device = actions_norm.device
            actions_norm = actions_norm * self._action_std.to(device) + self._action_mean.to(device)
            if _pre is not None:
                _pre = _pre * self._action_std.to(_pre.device) + self._action_mean.to(_pre.device)
        # Same units as the returned chunk, so the two can be compared directly.
        self._last_prefix_pre_inject_denorm = _pre
        if relative_reference is not None:
            actions_norm = actions_norm + relative_reference.unsqueeze(1)

        return actions_norm

    def compute_input_saliency(self, batch: dict[str, Tensor], num_steps: int = 4, grid: int = 64) -> dict:
        """Per-camera input-gradient attention map — see ``flow_matching/saliency.py`` (kept out of
        the policy class; this delegator is the documented per-policy overlay contract)."""
        from lerobot.policies.hvla.s1.flow_matching.saliency import compute_input_saliency

        return compute_input_saliency(self, batch, num_steps=num_steps, grid=grid)

    def compute_attention_rollout(self, batch: dict[str, Tensor], num_steps: int = 4, grid: int = 64) -> dict:
        """Per-camera attention-rollout map — see ``flow_matching/saliency.py``."""
        from lerobot.policies.hvla.s1.flow_matching.saliency import compute_attention_rollout

        return compute_attention_rollout(self, batch, num_steps=num_steps, grid=grid)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward: compute flow matching loss with training-time RTC."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return self.model(batch)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: FlowMatchingS1Config | None = None):
        """Load from checkpoint. Also loads norm_stats.pt if present.

        Supports both standard LeRobot format (pretrained_model/) and legacy
        flat format (model.safetensors in checkpoint dir).
        """
        import json
        from pathlib import Path

        import safetensors.torch

        path = Path(checkpoint_path)

        # Resolve checkpoint path: accept dir or file
        if path.is_dir():
            # Standard format: checkpoint-N/pretrained_model/model.safetensors
            pretrained_dir = path / "pretrained_model"
            if pretrained_dir.is_dir():
                model_file = pretrained_dir / "model.safetensors"
                norm_dir = pretrained_dir
            else:
                # Legacy flat format: checkpoint-N/model.safetensors
                model_file = path / "model.safetensors"
                norm_dir = path
        else:
            model_file = path
            norm_dir = path.parent

        # Load config from config.json if present and no config provided
        if config is None:
            config_file = norm_dir / "config.json"
            if config_file.exists():
                config = FlowMatchingS1Config.from_checkpoint_dict(json.loads(config_file.read_text()))
            else:
                raise ValueError(
                    "HVLA checkpoint does not contain config.json. Pass an explicit "
                    "FlowMatchingS1Config after verifying its feature contract; inference "
                    "will not guess cameras, state usage, or tensor layouts."
                )

        policy = cls(config)

        state_dict = safetensors.torch.load_file(str(model_file))
        # Remap old checkpoint key format (action_decoder.layers → decoder_layers)
        remapped = {}
        for k, v in state_dict.items():
            new_k = k.replace("model.action_decoder.layers.", "model.decoder_layers.")
            remapped[new_k] = v
        missing, unexpected = policy.load_state_dict(remapped, strict=False)
        if missing:
            import logging

            logging.warning("Missing keys: %s", missing)
        if unexpected:
            import logging

            logging.warning("Unexpected keys: %s", unexpected)

        # Load normalization stats
        norm_path = norm_dir / "norm_stats.pt"
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

            logging.warning("No norm_stats.pt found at %s — running without normalization", norm_dir)

        return policy
