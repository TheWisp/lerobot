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

"""Patch Policy: a VQ-BeT head over a block-causal GPT that reads frozen ViT patch tokens.

`PatchPolicyModel` is a port of `BehaviorTransformer` (`models/vq_behavior_transformer/bet.py`) from
the reference implementation, unconditional variant (no goal tokens). Attribute names match the
reference so its state dict loads directly. `PatchPolicy` wraps it in the LeRobot policy contract:
the frozen encoder, image resizing, the observation window at inference and the reference
evaluation loop's chunk averaging (`train_policy.py`, the `action_window_size > 1` branch).
"""

import logging
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torchvision.transforms.functional import resize as tv_resize

from lerobot.utils.constants import ACTION

from ..pretrained import PreTrainedPolicy
from .configuration_patch_policy import PatchConfig
from .encoder import build_encoder
from .gpt import GPT, GPTConfig
from .vqvae import VqVae


def repeat_start_to_length(x: Tensor, length: int, dim: int) -> Tensor:
    """Pad `x` to `length` along `dim` by repeating its first slice at the front."""
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    first = x.narrow(dim, 0, 1)
    repeat_shape = [1] * x.ndim
    repeat_shape[dim] = pad_size
    return torch.cat([first.repeat(*repeat_shape), x], dim=dim)


class MLP(nn.Sequential):
    """torchvision-style MLP with the reference's layer layout (Linear, ReLU, Dropout per hidden layer)."""

    def __init__(self, in_channels: int, hidden_channels: list[int]):
        layers: list[nn.Module] = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.0)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, hidden_channels[-1]), nn.Dropout(0.0)]
        super().__init__(*layers)


def batch_idx(x: Tensor, idx: Tensor) -> Tensor:
    """Index `x[..., idx, ...]` where `idx` matches a prefix of `x.shape`; keeps the trailing dims."""
    if idx.shape != x.shape[: idx.ndim]:
        raise ValueError(f"index shape {tuple(idx.shape)} must match a prefix of {tuple(x.shape)}")
    remaining_shape = x.shape[idx.ndim + 1 :]
    x = x.flatten(start_dim=idx.ndim + 1)
    indices = []
    for i in range(idx.ndim):
        shape = [1] * idx.ndim
        shape[i] = idx.shape[i]
        indices.append(torch.arange(idx.shape[i], device=idx.device).reshape(shape))
    indices.append(idx)
    return x[tuple(indices)].reshape(idx.shape + remaining_shape)


class FocalLoss(nn.Module):
    """Per-row focal loss; callers reduce, so a per-sample loss is one `view` away."""

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        logpt = F.log_softmax(input, dim=-1).gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()
        return -1 * (1 - pt) ** self.gamma * logpt


class PatchPolicyModel(nn.Module):
    """`BehaviorTransformer`, unconditional: patch tokens in, VQ-BeT code logits and offsets out."""

    def __init__(self, config: PatchConfig, obs_dim: int, act_dim: int, n_patches: int):
        super().__init__()
        self.config = config
        self._obs_window_size = config.n_obs_steps
        self._act_window_size = config.chunk_size
        self._act_dim = act_dim

        self._gpt_model = GPT(
            GPTConfig(
                block_size=config.n_obs_steps + config.chunk_size,
                n_patches=n_patches,
                input_dim=obs_dim,
                output_dim=config.gpt_output_dim,
                n_layer=config.gpt_n_layer,
                n_head=config.gpt_n_head,
                n_embd=config.gpt_n_embd,
                dropout=config.dropout,
            )
        )
        self._vqvae_model = VqVae(
            input_dim_h=config.chunk_size,
            input_dim_w=act_dim,
            n_latent_dims=config.vqvae_latent_dim,
            vqvae_n_embed=config.vqvae_n_embed,
            vqvae_groups=config.vqvae_groups,
            encoder_loss_multiplier=config.vqvae_encoder_loss_multiplier,
            act_scale=config.act_scale,
        )
        self._G = config.vqvae_groups
        self._C = config.vqvae_n_embed
        self._map_to_cbet_preds_bin = MLP(config.gpt_output_dim, [1024, 1024, self._G * self._C])
        self._map_to_cbet_preds_offset = MLP(
            config.gpt_output_dim, [1024, 1024, self._G * self._C * act_dim * config.chunk_size]
        )
        self._criterion = FocalLoss(gamma=config.focal_gamma)

        # Persisted so a checkpoint knows whether its VQ-VAE has been fit. The collected actions are
        # not: a resume during the collection phase starts collecting again.
        self.register_buffer("vqvae_is_fit", torch.tensor(False))
        self._collected_actions: list[Tensor] = []

    def train(self, mode: bool = True):
        super().train(mode)
        if bool(self.vqvae_is_fit):
            self._vqvae_model.eval()
        return self

    def _unpack_actions(self, action_seq: Tensor) -> Tensor:
        """`[N, T + W - 1, A]` -> `[N, T, W, A]`: the chunk of W actions starting at each of T steps."""
        n, total_w, act_dim = action_seq.shape
        act_w = self._act_window_size
        obs_w = total_w + 1 - act_w
        if obs_w != self._obs_window_size:
            raise ValueError(
                f"expected {self._obs_window_size + act_w - 1} actions per sample "
                f"(n_obs_steps + chunk_size - 1), got {total_w}"
            )
        return action_seq.unfold(1, act_w, 1).permute(0, 1, 3, 2).contiguous()

    def _maybe_fit_vq(self) -> None:
        if bool(self.vqvae_is_fit) or len(self._collected_actions) < self.config.vqvae_fit_steps:
            return
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed and torch.distributed.get_world_size() > 1:
            # The reference gathers every rank's actions before fitting. Not ported.
            raise NotImplementedError("Patch Policy's VQ-VAE fit is single-process only")

        all_actions = torch.cat(self._collected_actions)
        all_actions = einops.rearrange(all_actions, "N T W A -> (N T) (W A)")
        all_actions = torch.unique(all_actions, dim=0)
        all_actions = einops.rearrange(all_actions, "... (W A) -> ... W A", W=self._act_window_size)

        optim = torch.optim.Adam(
            self._vqvae_model.parameters(),
            lr=self.config.vqvae_lr,
            weight_decay=self.config.vqvae_weight_decay,
        )
        self._vqvae_model.train()
        logging.info(
            "Fitting Patch Policy's VQ-VAE on %d unique action chunks for %d epochs",
            len(all_actions),
            self.config.vqvae_iters,
        )
        for _ in range(self.config.vqvae_iters):
            shuffle_idx = torch.randperm(len(all_actions), device=all_actions.device)
            for i in range(0, len(all_actions), self.config.vqvae_batch_size):
                batch = all_actions[shuffle_idx[i : i + self.config.vqvae_batch_size]]
                loss, vq_code, loss_dict = self._vqvae_model(batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
        self._vqvae_model.eval()
        for param in self._vqvae_model.parameters():
            param.requires_grad_(False)
        logging.info(
            "VQ-VAE fit: %d distinct codes, %d distinct combinations, losses %s",
            len(torch.unique(vq_code)),
            len(torch.unique(vq_code, dim=0)),
            {k: round(v.item(), 5) for k, v in loss_dict.items()},
        )
        self.vqvae_is_fit.fill_(True)
        self._collected_actions = []

    def forward(
        self, obs_seq: Tensor, action_seq: Tensor | None, reduction: str = "mean"
    ) -> tuple[Tensor, Tensor | None, dict[str, float]]:
        """`obs_seq` `[N, T', P, E]` with T' <= n_obs_steps; `action_seq` `[N, T + W - 1, A]` or None.

        Returns the predicted action chunks `[N, T, W, A]`, the loss (None without actions; a scalar,
        or `[N]` with `reduction="none"`) and metrics.
        """
        if action_seq is not None and not bool(self.vqvae_is_fit) and self.training:
            self._collected_actions.append(self._unpack_actions(action_seq).detach())
            self._maybe_fit_vq()

        if obs_seq.shape[1] < self._obs_window_size:
            obs_seq = repeat_start_to_length(obs_seq, self._obs_window_size, dim=1)

        gpt_output = self._gpt_model(obs_seq)
        cbet_logits, cbet_offsets = self._forward_heads(gpt_output)
        predicted_action, decoded_action, sampled_centers, _ = self._sample_action(cbet_logits, cbet_offsets)

        if action_seq is None:
            return predicted_action, None, {}
        loss, loss_dict = self._calc_loss(
            action_seq, predicted_action, decoded_action, sampled_centers, cbet_logits, reduction
        )
        return predicted_action, loss, loss_dict

    def _calc_loss(
        self,
        action_seq: Tensor,
        predicted_action: Tensor,
        decoded_action: Tensor,
        sampled_centers: Tensor,
        cbet_logits: Tensor,
        reduction: str,
    ) -> tuple[Tensor, dict[str, float]]:
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        action_seq = self._unpack_actions(action_seq)
        n, t = action_seq.shape[:2]
        _, action_bins = self._vqvae_model.get_code(action_seq)
        action_bins_flat = einops.rearrange(action_bins, "N T ... -> (N T) ...")
        cbet_logits_flat = einops.rearrange(cbet_logits, "N T ... -> (N T) ...")

        # Per-sample terms `[N]`; their means equal the reference's batch-wide means.
        offset_loss = (action_seq - predicted_action).abs().mean(dim=(1, 2, 3))

        # `[:, -1, 0]` is the action a rollout would execute: current step, first action of its chunk.
        action_diff = F.mse_loss(action_seq[:, -1, 0], predicted_action[:, -1, 0])
        action_diff_tot = F.mse_loss(action_seq[:, -1], predicted_action[:, -1])
        action_diff_mean_res1 = (action_seq - decoded_action)[:, -1, 0].abs().mean()
        action_diff_mean_res2 = (action_seq - predicted_action)[:, -1, 0].abs().mean()
        action_diff_max = (action_seq - predicted_action)[:, -1, 0].abs().max()
        cbet_loss1 = self._criterion(cbet_logits_flat[:, 0], action_bins_flat[:, 0]).view(n, t).mean(dim=1)
        cbet_loss2 = self._criterion(cbet_logits_flat[:, 1], action_bins_flat[:, 1]).view(n, t).mean(dim=1)
        cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self.config.secondary_code_multiplier

        eq_mask = action_bins == sampled_centers
        equal_total_code_rate = (eq_mask.sum(-1) == self._G).float().mean()
        equal_single_code_rate = eq_mask[..., 0].float().mean()
        equal_single_code_rate2 = eq_mask[..., 1].float().mean()

        loss = cbet_loss + self.config.offset_loss_multiplier * offset_loss
        loss_dict = {
            "classification_loss": cbet_loss.mean().item(),
            "offset_loss": offset_loss.mean().item(),
            "total_loss": loss.mean().item(),
            "equal_total_code_rate": equal_total_code_rate.item(),
            "equal_single_code_rate": equal_single_code_rate.item(),
            "equal_single_code_rate2": equal_single_code_rate2.item(),
            "action_diff": action_diff.item(),
            "action_diff_tot": action_diff_tot.item(),
            "action_diff_mean_res1": action_diff_mean_res1.item(),
            "action_diff_mean_res2": action_diff_mean_res2.item(),
            "action_diff_max": action_diff_max.item(),
            "vqvae_is_fit": float(self.vqvae_is_fit),
        }
        if not bool(self.vqvae_is_fit):
            loss = loss * 0.0  # nothing to learn from codes the VQ-VAE has not defined yet
        if reduction == "mean":
            loss = loss.mean()
        return loss, loss_dict

    def _forward_heads(self, gpt_output: Tensor) -> tuple[Tensor, Tensor]:
        cbet_logits = self._map_to_cbet_preds_bin(gpt_output)
        cbet_logits = einops.rearrange(cbet_logits, "N T (G C) -> N T G C", G=self._G)
        cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
        cbet_offsets = einops.rearrange(
            cbet_offsets,
            "N T (G C W A) -> N T G C W A",
            G=self._G,
            C=self._C,
            W=self._act_window_size,
            A=self._act_dim,
        )
        return cbet_logits, cbet_offsets

    def _sample_action(
        self, cbet_logits: Tensor, cbet_offsets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cbet_probs = torch.softmax(cbet_logits, dim=-1)
        sampled_centers = einops.rearrange(
            torch.multinomial(cbet_probs.view(-1, self._C), num_samples=1),
            "(N T G) 1 -> N T G",
            N=cbet_probs.shape[0],
            T=cbet_probs.shape[1],
            G=self._G,
        )
        centers = self._vqvae_model.draw_code_forward(sampled_centers).clone().detach()
        decoded_action = self._vqvae_model.get_action_from_latent(centers).clone().detach()
        sampled_offsets = batch_idx(cbet_offsets, sampled_centers).sum(dim=2)
        predicted_action = decoded_action + sampled_offsets
        return predicted_action, decoded_action, sampled_centers, sampled_offsets


class PatchPolicy(PreTrainedPolicy):
    """Patch Policy (Zhou et al., 2026, arXiv:2607.18236)."""

    config_class = PatchConfig
    name = "patch_policy"

    def __init__(self, config: PatchConfig | None = None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.encoder = build_encoder(config.encoder, config.encoder_pretrained)
        if config.image_size % self.encoder.patch_size != 0:
            raise ValueError(
                f"image_size={config.image_size} is not a multiple of the encoder's patch size "
                f"{self.encoder.patch_size}"
            )
        self.n_views = len(config.image_features)
        patches_per_view = (config.image_size // self.encoder.patch_size) ** 2
        self.model = PatchPolicyModel(
            config,
            obs_dim=self.encoder.embed_dim,
            act_dim=config.action_feature.shape[0],
            n_patches=patches_per_view * self.n_views,
        )
        self.reset()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # the trainer calls train() every step; the encoder stays frozen
        return self

    def get_optim_params(self) -> list[dict]:
        decay, no_decay = self.model._gpt_model.decay_and_no_decay_parameters()
        decay = decay + list(self.model._map_to_cbet_preds_offset.parameters())
        # The reference adds the bin head to its AdamW with `add_param_group` and no weight decay of
        # its own, so that group inherits torch's AdamW default of 1e-2, not the configured value.
        bin_head = list(self.model._map_to_cbet_preds_bin.parameters())
        return [
            {"params": decay},
            {"params": bin_head, "weight_decay": 1e-2},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def reset(self) -> None:
        self._obs_queue: deque[Tensor] = deque(maxlen=self.config.n_obs_steps)
        self._chunk_history: deque[Tensor] = deque(maxlen=self.config.chunk_size)

    def _encode_images(self, images: list[Tensor]) -> Tensor:
        """Per-camera `[B, T, C, H, W]` (or `[B, C, H, W]`) -> tokens `[B, T, V * P, E]`.

        Each camera is resized to the `image_size` square on its own, so cameras may differ in
        resolution. The aspect ratio is not kept; the reference only ever sees square renders.
        """
        size = self.config.image_size
        frames = []
        for img in images:
            if img.ndim == 4:
                img = img[:, None]
            b, t = img.shape[:2]
            if tuple(img.shape[-2:]) != (size, size):
                flat = tv_resize(img.reshape(b * t, *img.shape[2:]), [size, size], antialias=True)
                img = flat.reshape(b, t, *flat.shape[1:])
            frames.append(img)
        x = torch.stack(frames, dim=2)
        b, t, v = x.shape[:3]
        with torch.no_grad():
            tokens = self.encoder(x.reshape(b * t * v, *x.shape[3:]))
        return tokens.reshape(b, t, v * tokens.shape[1], tokens.shape[2])

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """`reduction="none"` returns one loss per sample, `[B]`, for the trainer's sample weighting."""
        obs_seq = self._encode_images([batch[key] for key in self.config.image_features])
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions[:, None]
        _, loss, loss_dict = self.model(obs_seq, actions, reduction=reduction)
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Append the observation to the window and return the chunk predicted for it, `[B, W, A]`."""
        obs = self._encode_images([batch[key] for key in self.config.image_features])
        self._obs_queue.append(obs[:, -1])
        obs_seq = torch.stack(list(self._obs_queue), dim=1)
        predicted_action, _, _ = self.model(obs_seq, None)
        return predicted_action[:, -1]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """One action `[B, A]` per control tick.

        The model is queried every tick. With `chunk_size > 1` the action is the mean of the aligned
        predictions from the last `chunk_size` chunks, as in the reference evaluation loop.
        """
        chunk = self.predict_action_chunk(batch)
        if self.config.chunk_size == 1:
            return chunk[:, 0]
        self._chunk_history.append(chunk)
        n = len(self._chunk_history)
        aligned = torch.stack([past[:, n - 1 - i] for i, past in enumerate(self._chunk_history)], dim=0)
        return aligned.mean(dim=0)
