# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Flow S1's vision encoder is a choice, and every layer must agree on it.

The backbone used to be ``torch.hub.load("facebookresearch/dinov2", …)`` with
the repository hardcoded, so no other family was reachable; ``backbone_dim``
was a second config field kept in sync by hand, and a disagreement surfaced as
a shape error inside ``image_proj`` one batch into training.

The registry is now the single source. These pin the parts that rot silently:
that the form cannot offer an encoder the trainer would reject, that the width
is derived rather than typed twice, and that the key persisted in existing
checkpoints keeps working.
"""

from __future__ import annotations

import pytest

from lerobot.policies.hvla.s1.flow_matching.vision_encoders import (
    DEFAULT_ENCODER,
    VISION_ENCODERS,
    resolve,
)


def test_the_default_is_what_existing_checkpoints_carry():
    """Renaming this key orphans every S1 checkpoint on disk."""
    assert DEFAULT_ENCODER == "dinov2_vits14"
    assert DEFAULT_ENCODER in VISION_ENCODERS


def test_dinov3_is_offered():
    assert "dinov3_vits16" in VISION_ENCODERS
    assert "dinov3_vitb16" in VISION_ENCODERS


@pytest.mark.parametrize("name", sorted(VISION_ENCODERS))
def test_every_entry_is_complete(name):
    spec = resolve(name)

    assert spec.hub_repo, "an entry with no source cannot be loaded"
    assert spec.embed_dim > 0
    assert spec.patch_size > 0
    assert spec.label


def test_unknown_encoder_names_the_alternatives():
    """A typo should not surface as a torch.hub 404 quoting a URL."""
    with pytest.raises(ValueError, match="Unknown vision encoder"):
        resolve("dinov2_vitgiant")


class TestWeightsAreNotRedistributed:
    """Licensing is kept out of the repo by never vendoring weights.

    DINOv2 is Apache-2.0; DINOv3 is under Meta's own licence with gated access.
    Every entry names an upstream source fetched at run time under whatever the
    operator has accepted, so this repository never re-hosts either.
    """

    @pytest.mark.parametrize("name", sorted(VISION_ENCODERS))
    def test_each_entry_points_at_an_upstream_repo(self, name):
        assert "/" in resolve(name).hub_repo, "must be an upstream repo, not a local path"

    def test_gated_families_are_marked(self):
        """The flag drives the warning an operator sees before hitting a 401."""
        assert resolve("dinov3_vits16").gated is True
        assert resolve("dinov2_vits14").gated is False


class TestTheFormCannotDriftFromTheTrainer:
    """Two hand-maintained lists would disagree the first time one is edited."""

    def _field(self) -> dict:
        from lerobot.gui.api.training import list_policies

        recipe = next(p for p in list_policies() if p["type_name"] == "hvla_flow_s1")
        return next(f for f in recipe["fields"] if f["name"] == "vision_encoder")

    def test_the_form_offers_exactly_the_registry(self):
        assert sorted(self._field()["choices"]) == sorted(VISION_ENCODERS)

    def test_the_form_default_matches_the_trainer_default(self):
        assert self._field()["default"] == DEFAULT_ENCODER

    def test_it_is_a_dropdown(self):
        assert self._field()["type"] == "select"

    def test_every_choice_has_a_label(self):
        field = self._field()

        assert set(field["choice_labels"]) == set(field["choices"])

    def test_the_choice_reaches_the_trainer_as_a_flag(self):
        """A form field with no flag mapping is silently dropped at launch."""
        from lerobot.gui.training.recipes import HVLA_FLOW_S1_FIELD_TO_FLAG

        assert HVLA_FLOW_S1_FIELD_TO_FLAG["vision_encoder"] == "--vision-encoder"


class TestWidthIsDerivedNotTyped:
    def test_the_registry_knows_each_encoder_width(self):
        assert resolve("dinov2_vits14").embed_dim == 384
        assert resolve("dinov2_vitb14").embed_dim == 768
        assert resolve("dinov3_vits16").embed_dim == 384
        assert resolve("dinov3_vitb16").embed_dim == 768

    def test_patch_size_differs_across_families(self):
        """224px gives 256 tokens at patch 14 and 196 at patch 16.

        S1 reads the count off the tensor, so this is a cost difference rather
        than a correctness one — but it is why the two families are not
        interchangeable in a single checkpoint.
        """
        assert resolve("dinov2_vits14").patch_size == 14
        assert resolve("dinov3_vits16").patch_size == 16


class TestFailuresNameTheirCause:
    """Both DINOv3 failure modes are opaque unless the loader translates them.

    The dependency one fires before any weight is fetched and mentions only
    ``torchmetrics``, giving no hint it came from choosing an encoder; the gated
    one is an HTTP error. Neither tells an operator what to do.
    """

    def test_missing_dependency_names_the_encoder_and_the_extra(self, monkeypatch):
        from lerobot.policies.hvla.s1.flow_matching import vision_encoders

        def _raise(*_args, **_kwargs):
            raise ModuleNotFoundError("No module named 'transformers'", name="transformers")

        monkeypatch.setattr(vision_encoders, "_load_hf", _raise)

        with pytest.raises(ModuleNotFoundError) as excinfo:
            vision_encoders.load_backbone("dinov3_vits16")

        message = str(excinfo.value)
        assert "dinov3_vits16" in message, "must say which encoder caused it"
        assert "transformers" in message
        assert "--extra transformers-dep" in message, "must say how to fix it"

    def test_gated_failure_points_at_the_model_page(self, monkeypatch):
        """The first version said `huggingface-cli login`, which does not help.

        Access is per-model and granted on the model page; being logged in is
        necessary and not sufficient, as a 403 with a valid token showed.
        """
        from lerobot.policies.hvla.s1.flow_matching import vision_encoders

        def _raise(*_args, **_kwargs):
            raise RuntimeError("403 Client Error: not in the authorized list")

        monkeypatch.setattr(vision_encoders, "_load_hf", _raise)

        with pytest.raises(RuntimeError, match="huggingface.co/facebook/dinov3"):
            vision_encoders.load_backbone("dinov3_vits16")

    def test_ungated_failures_are_not_reinterpreted(self, monkeypatch):
        """A DINOv2 network blip must not be reported as a licence problem."""
        import torch

        from lerobot.policies.hvla.s1.flow_matching import vision_encoders

        def _raise(*_args, **_kwargs):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(torch.hub, "load", _raise)

        with pytest.raises(RuntimeError, match="connection reset"):
            vision_encoders.load_backbone("dinov2_vits14")


def test_the_dinov3_extra_is_declared():
    """The registry promises an extra; pyproject has to actually define it."""
    import tomllib
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    needed = {s.requires_extra for s in VISION_ENCODERS.values() if s.requires_extra}
    for extra in needed:
        assert f"{extra}-dep" in extras or extra in extras, f"extra {extra!r} is not declared"


class TestTheHFAdapterGivesS1TheInterfaceItExpects:
    """DINOv3 comes from transformers, which speaks a different dialect.

    S1 calls ``forward_features(x)["x_norm_patchtokens"]``. transformers returns
    ``last_hidden_state`` with CLS and register tokens still attached, so the
    adapter must strip exactly the right prefix — getting it wrong feeds S1 a
    few non-patch tokens that look like patches.
    """

    def _wrapped(self, num_register_tokens: int, seq_len: int, dim: int = 8):
        import torch

        from lerobot.policies.hvla.s1.flow_matching.vision_encoders import _HFPatchTokens

        class _Out:
            def __init__(self, t):
                self.last_hidden_state = t

        class _Model(torch.nn.Module):
            def forward(self, pixel_values=None):
                # Distinct values per token so slicing is observable.
                return _Out(
                    torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1).expand(1, seq_len, dim)
                )

        return _HFPatchTokens(_Model(), dim, 1 + num_register_tokens)

    def test_it_exposes_forward_features_with_the_expected_key(self):
        import torch

        out = self._wrapped(num_register_tokens=4, seq_len=201).forward_features(torch.zeros(1, 3, 224, 224))

        assert "x_norm_patchtokens" in out

    def test_cls_and_register_tokens_are_stripped(self):
        """224px at patch 16 is 196 patches; 201 - (1 CLS + 4 registers) = 196."""
        import torch

        tokens = self._wrapped(num_register_tokens=4, seq_len=201).forward_features(
            torch.zeros(1, 3, 224, 224)
        )
        patches = tokens["x_norm_patchtokens"]

        assert patches.shape[1] == 196
        assert patches[0, 0, 0].item() == 5.0, "first patch must be token 5, not token 0"

    def test_the_prefix_follows_the_register_count(self):
        """DINOv3 sizes differ in register count, so it cannot be a constant."""
        import torch

        patches = self._wrapped(num_register_tokens=0, seq_len=197).forward_features(
            torch.zeros(1, 3, 224, 224)
        )["x_norm_patchtokens"]

        assert patches.shape[1] == 196
        assert patches[0, 0, 0].item() == 1.0, "only CLS should be stripped"
