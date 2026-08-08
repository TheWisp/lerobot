# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Vision encoders Flow S1 can be trained with, and how to load each one.

One table so that adding an encoder is a row rather than an edit scattered
across the model, the trainer and the training form. The model previously
hardcoded ``torch.hub.load("facebookresearch/dinov2", …)``, which meant any
other family was unreachable, and kept ``backbone_dim`` as a second config
field a human had to keep in sync with the chosen model — a silent shape
mismatch at ``image_proj`` when they disagreed.

Weights are never vendored. Each entry names an upstream source that is
fetched at run time under whatever licence the operator has accepted:
DINOv2 is Apache-2.0, DINOv3 is under Meta's own licence with gated access,
and re-hosting either from this repository is deliberately not a thing this
module can express.

S1 consumes *patch* tokens (``x_norm_patchtokens``), not a pooled embedding,
so patch size matters to cost but not to correctness: the model reads the
token count off the tensor at runtime. A 224px input gives 16x16 = 256 tokens
at patch 14 and 14x14 = 196 at patch 16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VisionEncoder:
    """One selectable backbone.

    Attributes:
        source: ``"torch.hub"`` or ``"hf"``. DINOv2 ships through torch.hub;
            DINOv3's hub entrypoint downloads from ``dl.fbaipublicfiles.com``,
            which a HuggingFace licence grant does *not* cover (verified: 403
            with access granted on the Hub), so it loads through transformers.
        hub_repo: torch.hub repository, or the HF model id.
        embed_dim: Patch-token width. Advisory only — the loader reads the
            true value off the model and this is what it is checked against,
            so a wrong number here fails loudly instead of mis-shaping
            ``image_proj``.
        patch_size: Informational; drives token count and therefore cost.
        label: Shown in the training form.
        gated: Weights need an accepted licence and a Hub token.
        requires_extra: Optional dependency group this family needs, if any.
    """

    source: str
    hub_repo: str
    embed_dim: int
    patch_size: int
    label: str
    gated: bool = False
    requires_extra: str | None = None


# Keys are the value persisted in the checkpoint as ``dino_model``; existing
# checkpoints carry "dinov2_vits14", so that key must not be renamed.
VISION_ENCODERS: dict[str, VisionEncoder] = {
    "dinov2_vits14": VisionEncoder(
        source="torch.hub",
        hub_repo="facebookresearch/dinov2",
        embed_dim=384,
        patch_size=14,
        label="DINOv2 ViT-S/14 (22M, 384-d) — default",
    ),
    "dinov2_vitb14": VisionEncoder(
        source="torch.hub",
        hub_repo="facebookresearch/dinov2",
        embed_dim=768,
        patch_size=14,
        label="DINOv2 ViT-B/14 (86M, 768-d)",
    ),
    "dinov3_vits16": VisionEncoder(
        source="hf",
        hub_repo="facebook/dinov3-vits16-pretrain-lvd1689m",
        embed_dim=384,
        patch_size=16,
        label="DINOv3 ViT-S/16 (21M, 384-d) — gated weights",
        gated=True,
        requires_extra="transformers",
    ),
    "dinov3_vitb16": VisionEncoder(
        source="hf",
        hub_repo="facebook/dinov3-vitb16-pretrain-lvd1689m",
        embed_dim=768,
        patch_size=16,
        label="DINOv3 ViT-B/16 (86M, 768-d) — gated weights",
        gated=True,
        requires_extra="transformers",
    ),
}

DEFAULT_ENCODER = "dinov2_vits14"


def resolve(name: str) -> VisionEncoder:
    """Look up an encoder, failing with the list of what is available.

    Precondition: ``name`` is a key of :data:`VISION_ENCODERS`.

    Raises:
        ValueError: unknown encoder. A typo would otherwise surface as a
            torch.hub 404 naming a URL rather than a config field.
    """
    try:
        return VISION_ENCODERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown vision encoder {name!r}. Available: {sorted(VISION_ENCODERS)}. "
            "Add a row to VISION_ENCODERS to support another one."
        ) from None


class _HFPatchTokens(torch.nn.Module):
    """Give a transformers ViT the DINOv2 interface S1 already consumes.

    S1 calls ``forward_features(x)["x_norm_patchtokens"]``. A transformers model
    returns ``last_hidden_state`` with the CLS and register tokens still on the
    front, so the prefix is sliced off here rather than teaching the model about
    a second output convention. The prefix length comes from the config
    (``1 + num_register_tokens``) instead of a constant, because DINOv3 sizes
    differ in register count.
    """

    def __init__(self, model, embed_dim: int, num_prefix: int):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self._num_prefix = num_prefix

    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return {"x_norm_patchtokens": out.last_hidden_state[:, self._num_prefix :]}

    def forward(self, x):
        return self.forward_features(x)["x_norm_patchtokens"]


def _load_hf(spec, name: str):
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(spec.hub_repo)
    model = AutoModel.from_pretrained(spec.hub_repo)
    num_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0))
    return _HFPatchTokens(model, int(cfg.hidden_size), num_prefix)


def load_backbone(name: str):
    """Fetch an encoder's weights and return the model.

    Postcondition: the returned module exposes ``forward_features`` yielding
    ``x_norm_patchtokens``, which is the only interface S1 uses. HF-sourced
    encoders are wrapped to provide it.

    Gated weights raise when the operator has not accepted the licence or has no
    token; that error is re-raised naming the reason, because the raw failure is
    an opaque HTTP error.
    """
    import torch

    spec = resolve(name)
    try:
        if spec.source == "hf":
            return _load_hf(spec, name)
        return torch.hub.load(spec.hub_repo, name, pretrained=True)
    except ModuleNotFoundError as exc:
        extra = f" Install it with: uv sync --extra {spec.requires_extra}-dep" if spec.requires_extra else ""
        raise ModuleNotFoundError(
            f"Vision encoder {name!r} needs {exc.name!r}, which is not installed.{extra}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — re-raised with context below
        if spec.gated:
            raise RuntimeError(
                f"Could not load gated encoder {name!r} from {spec.hub_repo}. Accept the licence on "
                f"https://huggingface.co/{spec.hub_repo} with the account you are logged in as, then "
                "retry. This repository does not redistribute the weights. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        raise


def actual_embed_dim(backbone) -> int | None:
    """The loaded model's true patch-token width, or None if it does not say.

    Both DINO families expose ``embed_dim``; anything that does not is treated
    as unknown rather than assumed, so the caller can decide whether to trust
    the table.
    """
    dim = getattr(backbone, "embed_dim", None)
    return int(dim) if isinstance(dim, int) else None
