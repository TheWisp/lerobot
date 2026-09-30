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

"""Configuration for Patch Policy.

Patch Policy: Efficient Embodied Control via Dense Visual Representations
(Zhou, Cui, Langford, Tan, LeCun, Pinto, 2026, arXiv:2607.18236).
Reference implementation: https://github.com/gaoyuezhou/patch_policy (commit ebf94cf).
"""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("patch_policy")
@dataclass
class PatchConfig(PreTrainedConfig):
    """A VQ-BeT head over a block-causal GPT that reads every patch token of a frozen pretrained ViT,
    for every camera and each of the last `n_obs_steps` frames.

    The reference policy is visual-only: no proprioception and no language. `observation.state` is
    therefore ignored even when the dataset provides it.

    Names follow the reference repository except where LeRobot already has a name for the same thing:
    `window_size` is `n_obs_steps` and `action_window_size` is `chunk_size`. Defaults are the LIBERO
    Goal single-GPU recipe (`configs/train_libero_goal_1gpu.yaml` over `train_libero_goal.yaml`).

    The class is named `PatchConfig` because the policy factory derives the policy class name by
    replacing the `Config` suffix with `Policy`.
    """

    n_obs_steps: int = 2
    chunk_size: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Frozen vision encoder. Images are resized to an `image_size` square before encoding, and the
    # encoder applies its own input normalization, which is why VISUAL is IDENTITY above.
    encoder: str = "dinov2_vits14"
    encoder_pretrained: bool = True
    image_size: int = 224

    # Policy transformer (`models/vq_behavior_transformer/gpt.py`).
    gpt_n_layer: int = 6
    gpt_n_head: int = 6
    gpt_n_embd: int = 120
    gpt_output_dim: int = 256
    dropout: float = 0.0

    # Residual VQ-VAE over action chunks (`vqvae.py`). Actions from the first `vqvae_fit_steps`
    # training batches are collected, the VQ-VAE is fit on them for `vqvae_iters` epochs and then
    # frozen. The policy loss is zero until the fit has happened.
    vqvae_latent_dim: int = 512
    vqvae_n_embed: int = 16
    vqvae_groups: int = 2
    vqvae_fit_steps: int = 1000
    vqvae_iters: int = 300
    vqvae_batch_size: int = 2048
    vqvae_lr: float = 1e-3
    vqvae_weight_decay: float = 1e-4
    vqvae_encoder_loss_multiplier: float = 1.0
    act_scale: float = 1.0

    # Loss (`bet.py::_calc_loss`).
    offset_loss_multiplier: float = 100.0
    secondary_code_multiplier: float = 0.5
    focal_gamma: float = 2.0

    # Optimizer (`train_libero_goal.yaml::optim`). The reference trainer has no LR schedule.
    optimizer_lr: float = 5.5e-5
    optimizer_weight_decay: float = 2e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)

    def __post_init__(self):
        super().__post_init__()
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be at least 1, got {self.n_obs_steps}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {self.chunk_size}")
        if self.image_size < 1:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.vqvae_fit_steps < 1:
            raise ValueError(f"vqvae_fit_steps must be at least 1, got {self.vqvae_fit_steps}")
        if self.vqvae_groups != 2:
            # The code loss weights exactly two residual groups: the primary code x5 and the
            # secondary code x`secondary_code_multiplier`.
            raise ValueError(f"vqvae_groups must be 2, got {self.vqvae_groups}")
        if self.gpt_n_embd % self.gpt_n_head != 0:
            raise ValueError(
                f"gpt_n_embd={self.gpt_n_embd} must be divisible by gpt_n_head={self.gpt_n_head}"
            )
        # Checked here, not only when the model is built, so the GUI form rejects it before a run.
        if self.encoder.startswith("dinov2_"):
            if self.image_size % 14 != 0:
                raise ValueError(
                    f"image_size={self.image_size} must be a multiple of DINOv2's patch size, 14"
                )
        elif self.encoder != "tiny_test":
            raise ValueError(
                f"encoder must be a dinov2_* torch.hub name (or 'tiny_test'), got {self.encoder!r}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            betas=self.optimizer_betas,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if len(self.image_features) < 1:
            raise ValueError("Patch Policy is visual-only and needs at least one observation.images.* input")
        if self.action_feature is None:
            raise ValueError("Patch Policy needs an `action` output feature")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        # A chunk of `chunk_size` actions starts at every frame of the window: T + W - 1 actions.
        return list(range(1 - self.n_obs_steps, self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
