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

Every encoder maps images `[..., 3, S, S]` in [0, 1] to patch tokens `[..., P, E]` and exposes
`patch_size` and `embed_dim`. The families mirror the reference's `models/encoder/`:

| family  | reference file | tokens                                                     |
| ------- | -------------- | ---------------------------------------------------------- |
| dinov2  | `dino.py`      | torch.hub, `x_norm_patchtokens`                            |
| hf_vit  | `dinov3.py`, `webssl.py` | transformers, `last_hidden_state` minus CLS + registers |
| vjepa2  | `vjepa2.py`    | transformers, one frame repeated to a 2-frame tubelet      |

All three normalize with ImageNet statistics, as the reference does. `tiny_test` is a fixed random
patchify for tests that must run without any hub.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderSpec:
    family: str
    repo: str  # torch.hub entry point or Hugging Face repo id
    patch_size: int
    gated: bool = False  # weights need an accepted licence and a Hub token


ENCODERS: dict[str, EncoderSpec] = {
    # The paper's real-robot encoder and its default.
    "dinov2_vits14": EncoderSpec("dinov2", "dinov2_vits14", 14),
    "dinov2_vitb14": EncoderSpec("dinov2", "dinov2_vitb14", 14),
    # The paper's DINOv3 is ViT-S/16+ (`configs/encoder/dinov3_patch.yaml`, plus: true).
    "dinov3_vits16": EncoderSpec("hf_vit", "facebook/dinov3-vits16-pretrain-lvd1689m", 16, gated=True),
    "dinov3_vits16plus": EncoderSpec(
        "hf_vit", "facebook/dinov3-vits16plus-pretrain-lvd1689m", 16, gated=True
    ),
    "dinov3_vitb16": EncoderSpec("hf_vit", "facebook/dinov3-vitb16-pretrain-lvd1689m", 16, gated=True),
    # The paper's WebSSL (`models/encoder/webssl.py`).
    "webssl_dino300m": EncoderSpec("hf_vit", "facebook/webssl-dino300m-full2b-224", 14),
    # The paper's V-JEPA 2 is ViT-L (`models/encoder/vjepa2.py`); its processor crops to 256 px.
    "vjepa2_vitl": EncoderSpec("vjepa2", "facebook/vjepa2-vitl-fpc64-256", 16),
    "vjepa2_vitg": EncoderSpec("vjepa2", "facebook/vjepa2-vitg-fpc64-256", 16),
    "tiny_test": EncoderSpec("tiny", "", 14),
}


def patch_size_for(name: str) -> int:
    """The patch size of a registered encoder, so a config can check `image_size` before any download."""
    try:
        return ENCODERS[name].patch_size
    except KeyError:
        raise ValueError(f"unknown Patch Policy encoder {name!r}; choose one of {sorted(ENCODERS)}") from None


class _ImageNetNormalized(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _flatten(self, x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        prefix = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        return (x - self.mean) / self.std, prefix


class DinoV2PatchEncoder(_ImageNetNormalized):
    """`models/encoder/dino.py`: DINOv2 from torch.hub, `x_norm_patchtokens`."""

    def __init__(self, spec: EncoderSpec, pretrained: bool = True):
        super().__init__()
        self.patch_size = spec.patch_size
        self.model = torch.hub.load("facebookresearch/dinov2", spec.repo, pretrained=pretrained)
        self.embed_dim = int(self.model.num_features)

    def forward(self, x: Tensor) -> Tensor:
        x, prefix = self._flatten(x)
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        return tokens.reshape(*prefix, *tokens.shape[1:])


class HFPatchEncoder(_ImageNetNormalized):
    """`models/encoder/dinov3.py` and `webssl.py`: a transformers ViT, patch tokens only.

    `last_hidden_state` starts with the CLS token and any register tokens (4 for DINOv3, none for
    WebSSL); the prefix length comes from the model config rather than a constant.
    """

    def __init__(self, spec: EncoderSpec, pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(spec.repo)
        if int(config.patch_size) != spec.patch_size:
            raise ValueError(
                f"{spec.repo} has patch size {config.patch_size}, registry says {spec.patch_size}"
            )
        self.model = AutoModel.from_pretrained(spec.repo) if pretrained else AutoModel.from_config(config)
        self.patch_size = spec.patch_size
        self.embed_dim = int(config.hidden_size)
        self._num_prefix = 1 + int(getattr(config, "num_register_tokens", 0))

    def forward(self, x: Tensor) -> Tensor:
        x, prefix = self._flatten(x)
        tokens = self.model(pixel_values=x).last_hidden_state[:, self._num_prefix :]
        return tokens.reshape(*prefix, *tokens.shape[1:])


class VJEPA2PatchEncoder(_ImageNetNormalized):
    """`models/encoder/vjepa2.py`: a still image repeated into one temporal tubelet of the video model."""

    def __init__(self, spec: EncoderSpec, pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(spec.repo)
        if int(config.patch_size) != spec.patch_size:
            raise ValueError(
                f"{spec.repo} has patch size {config.patch_size}, registry says {spec.patch_size}"
            )
        self.model = AutoModel.from_pretrained(spec.repo) if pretrained else AutoModel.from_config(config)
        self.patch_size = spec.patch_size
        self.embed_dim = int(config.hidden_size)
        self._tubelet_size = int(config.tubelet_size)

    def forward(self, x: Tensor) -> Tensor:
        x, prefix = self._flatten(x)
        video = x[:, None].expand(-1, self._tubelet_size, -1, -1, -1)  # [N, frames, C, H, W]
        tokens = self.model.get_vision_features(video)
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


_FAMILIES = {"dinov2": DinoV2PatchEncoder, "hf_vit": HFPatchEncoder, "vjepa2": VJEPA2PatchEncoder}


def build_encoder(name: str, pretrained: bool = True) -> nn.Module:
    patch_size_for(name)  # raises for an unknown name
    spec = ENCODERS[name]
    if spec.family == "tiny":
        encoder: nn.Module = TinyPatchEncoder()
    else:
        try:
            encoder = _FAMILIES[spec.family](spec, pretrained=pretrained)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"encoder {name!r} needs {exc.name!r}; install it with: uv sync --extra transformers-dep"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised with the licence hint
            if spec.gated:
                raise RuntimeError(
                    f"could not load gated encoder {name!r} from {spec.repo}: accept the licence on "
                    f"https://huggingface.co/{spec.repo} and log in with `hf auth login`, then retry "
                    f"({type(exc).__name__}: {exc})"
                ) from exc
            raise
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder.eval()
