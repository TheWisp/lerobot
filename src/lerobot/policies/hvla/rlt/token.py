"""RL Token encoder and decoder.

The encoder compresses S1's observation context tokens into a single bottleneck
vector (z_rl). The decoder reconstructs the original tokens for training the
encoder. At runtime only the encoder is used (frozen).

Architecture follows Xu et al. 2026, Sec IV-A:
  encoder: append learned readout embedding to S1 context, process with small
           transformer, take output at readout position → z_rl
  decoder: autoregressive reconstruction of context tokens from z_rl
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.hvla.rlt.config import RLTConfig


class RLTokenEncoder(nn.Module):
    """Compress S1 context tokens [B, N, D] → z_rl [B, D] via learned readout."""

    def __init__(self, config: RLTConfig):
        super().__init__()
        d = config.rl_token_dim

        # Learned readout embedding appended to context sequence
        self.readout_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.token_num_heads,
            dim_feedforward=config.token_ffn_dim,
            dropout=config.token_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.token_encoder_layers,
        )

    def forward(self, context: Tensor) -> Tensor:
        """
        Args:
            context: [B, N_ctx, D] from S1.encode_observations (detached)
        Returns:
            z_rl: [B, D] compressed representation
        """
        B = context.shape[0]
        readout = self.readout_embed.expand(B, -1, -1)        # [B, 1, D]
        augmented = torch.cat([context, readout], dim=1)       # [B, N+1, D]
        out = self.transformer(augmented)                      # [B, N+1, D]
        z_rl = out[:, -1, :]                                   # [B, D]
        return z_rl


class RLTokenDecoder(nn.Module):
    """Autoregressive reconstruction of S1 context tokens from z_rl.

    Used only during Phase 1 training. Teacher-forced: at training time,
    ground-truth context tokens are shifted and fed as input, with causal
    masking so position i only sees positions < i.
    """

    def __init__(self, config: RLTConfig):
        super().__init__()
        d = config.rl_token_dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.token_num_heads,
            dim_feedforward=config.token_ffn_dim,
            dropout=config.token_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=config.token_decoder_layers,
        )
        self.output_proj = nn.Linear(d, d)

    def forward(self, z_rl: Tensor, target_context: Tensor) -> Tensor:
        """Teacher-forced autoregressive reconstruction.

        Args:
            z_rl: [B, D] RL token (used as memory for cross-attention)
            target_context: [B, N, D] ground-truth context tokens (stop-grad)
        Returns:
            reconstructed: [B, N, D] predicted context tokens
        """
        B, N, D = target_context.shape

        # Memory: z_rl as single token for cross-attention
        memory = z_rl.unsqueeze(1)  # [B, 1, D]

        # Decoder input: shift right — prepend z_rl, drop last token
        # Position 0 input = z_rl, position i input = target_context[:, i-1]
        decoder_input = torch.cat([
            z_rl.unsqueeze(1),            # [B, 1, D]
            target_context[:, :-1, :],    # [B, N-1, D]
        ], dim=1)                         # [B, N, D]

        # Causal mask: position i can only attend to positions <= i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            N, device=z_rl.device, dtype=z_rl.dtype,
        )

        out = self.transformer(decoder_input, memory, tgt_mask=causal_mask)
        reconstructed = self.output_proj(out)  # [B, N, D]
        return reconstructed


def rl_token_reconstruction_loss(
    encoder: RLTokenEncoder,
    decoder: RLTokenDecoder,
    context: Tensor,
) -> Tensor:
    """Compute L_rto (paper Eq. 2): reconstruction loss for RL token.

    Per-token L2 norm squared, summed over tokens, averaged over batch.
    Not F.mse_loss (which averages over all dims — too weak).

    Args:
        encoder: RL token encoder
        decoder: RL token decoder
        context: [B, N, D] from S1.encode_observations (already detached / stop-grad)
    Returns:
        scalar loss
    """
    target = context.detach()
    z_rl = encoder(target)
    reconstructed = decoder(z_rl, target)
    # Paper Eq. 2: ||h(d([z_rl, z̃_{1:i-1}]))_i - z̃_i||^2 summed over i
    loss = ((reconstructed - target) ** 2).sum(dim=-1).mean()
    return loss
