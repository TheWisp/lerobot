#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Residual VQ-VAE over action chunks.

Port of `models/vq_behavior_transformer/vqvae.py` from the Patch Policy reference implementation.
The residual vector quantizer is the fork's vendored copy in `lerobot.policies.vqbet.vqbet_utils`,
which shares the reference's lucidrains lineage. The two copies differ in one constructor default,
`threshold_ema_dead_code` (reference 2, ours 0), so it is passed explicitly.
"""

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.vqbet.vqbet_utils import ResidualVQ


def weights_init_encoder(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class EncoderMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 16, hidden_dim: int = 128, layer_num: int = 1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.encoder(x))


class VqVae(nn.Module):
    def __init__(
        self,
        input_dim_h: int,  # actions per chunk
        input_dim_w: int,  # action dimension
        n_latent_dims: int,
        vqvae_n_embed: int,
        vqvae_groups: int,
        encoder_loss_multiplier: float,
        act_scale: float,
    ):
        super().__init__()
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale
        self.vq_layer = ResidualVQ(
            dim=n_latent_dims,
            num_quantizers=vqvae_groups,
            codebook_size=vqvae_n_embed,
            threshold_ema_dead_code=2,
        )
        self.encoder = EncoderMLP(input_dim=input_dim_w * input_dim_h, output_dim=n_latent_dims)
        self.decoder = EncoderMLP(input_dim=n_latent_dims, output_dim=input_dim_w * input_dim_h)

    def draw_code_forward(self, encoding_indices: Tensor) -> Tensor:
        """Code indices `[..., G]` -> latent `[..., D]`, the sum of the selected codebook vectors."""
        with torch.no_grad():
            return self.vq_layer.get_codebook_vector_from_indices(encoding_indices).sum(dim=0)

    def get_action_from_latent(self, latent: Tensor) -> Tensor:
        output = self.decoder(latent) * self.act_scale
        return einops.rearrange(output, "... (T A) -> ... T A", A=self.input_dim_w)

    def preprocess(self, state: Tensor) -> Tensor:
        state = state / self.act_scale
        if self.input_dim_h == 1:
            return state.squeeze(-2)
        return einops.rearrange(state, "... T A -> ... (T A)")

    def get_code(self, state: Tensor) -> tuple[Tensor, Tensor]:
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)
            state_vq, vq_code, _ = self.vq_layer(state_rep)
            return state_vq, vq_code

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        state = self.preprocess(state)
        state_rep = self.encoder(state)
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        dec_out = self.decoder(state_vq)

        vq_loss_state = vq_loss_state.sum()
        encoder_loss = (state - dec_out).abs().mean()
        vqvae_recon_loss = F.mse_loss(state, dec_out)

        loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
        loss_dict = {
            "loss": loss.detach().clone(),
            "vq_loss_state": vq_loss_state.detach().clone(),
            "vqvae_recon_loss": vqvae_recon_loss.detach().clone(),
            "encoder_loss": encoder_loss.detach().clone(),
        }
        return loss, vq_code, loss_dict
