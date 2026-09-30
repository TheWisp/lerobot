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

"""Frozen patch-token encoders for Patch Policy.

Every encoder maps images `[..., 3, H, W]` in [0, 1] to patch tokens `[..., P, E]` and exposes
`patch_size` and `embed_dim`. `DinoV2PatchEncoder` is the reference's `models/encoder/dino.py`
(`x_norm_patchtokens`, ImageNet normalization). `TinyPatchEncoder` is a fixed random patchify for
tests that must run without the hub.
"""

import torch
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoV2PatchEncoder(nn.Module):
    patch_size = 14

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
        self.embed_dim = int(self.model.num_features)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        prefix = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        x = (x - self.mean) / self.std
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        return tokens.reshape(*prefix, *tokens.shape[1:])


class TinyPatchEncoder(nn.Module):
    """A frozen random linear patchify, for tests. Not a trained representation."""

    patch_size = 14

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        prefix = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return tokens.reshape(*prefix, *tokens.shape[1:])


def build_encoder(name: str, pretrained: bool = True) -> nn.Module:
    if name.startswith("dinov2_"):
        encoder = DinoV2PatchEncoder(name, pretrained=pretrained)
    elif name == "tiny_test":
        encoder = TinyPatchEncoder()
    else:
        raise ValueError(
            f"unknown Patch Policy encoder {name!r}; expected a dinov2_* hub name or 'tiny_test'"
        )
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder.eval()
