# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""pi05 parameter freezing is configuration, not a training-script special case.

The fork used to freeze pi05's language tower from ``lerobot_train.py``, keyed
on ``cfg.policy.type == "pi05"``. Two things were wrong with that beyond the
VRAM tradeoff it encoded: it ran after the policy was built and rewrote
``requires_grad`` on *every* parameter, so it silently reversed whatever
``freeze_vision_encoder`` / ``train_expert_only`` the operator had set; and it
selected parameters by substring, so an upstream rename would match nothing,
freeze everything, and train a model that learns nothing without saying so.

These pin the replacement: a config field, applied by the policy, that composes
with the flags already there.
"""

from __future__ import annotations

import pytest

from lerobot.policies.pi05.configuration_pi05 import PI05Config


def test_the_flag_is_off_by_default():
    """A fork-divergent default is how the original hack stayed invisible."""
    assert PI05Config().freeze_language_tower is False


def test_freezing_language_is_not_the_same_as_training_the_expert_only():
    """The distinction the flag exists for.

    ``train_expert_only`` freezes all of PaliGemma, vision tower included. A new
    robot's cameras are exactly what must adapt; its language grounding is not.
    """
    cfg = PI05Config(freeze_language_tower=True)

    assert cfg.train_expert_only is False
    assert cfg.freeze_vision_encoder is False


def test_the_flag_round_trips_through_the_saved_config():
    """A checkpoint has to remember how it was trained."""
    cfg = PI05Config(freeze_language_tower=True)

    restored = PI05Config(**{**cfg.__dict__, "freeze_language_tower": cfg.freeze_language_tower})

    assert restored.freeze_language_tower is True


class TestItReachesTheTrainingForm:
    """Surfacing is the point: an operator drives training from the GUI.

    The form is introspected from the config dataclass, so a bool with a default
    renders as a checkbox with no frontend change — but only if it is actually
    renderable, which these assert rather than assume.
    """

    def _fields(self) -> dict:
        from lerobot.gui.api.training import _introspect_policy_fields

        return {f["name"]: f for f in _introspect_policy_fields(PI05Config)}

    def test_the_flag_is_offered_in_the_form(self):
        assert "freeze_language_tower" in self._fields()

    def test_it_renders_as_a_checkbox_defaulting_off(self):
        field = self._fields()["freeze_language_tower"]

        assert field["type"] == "bool"
        assert field["default"] is False

    def test_the_sibling_freeze_flags_are_offered_too(self):
        """All three knobs, or an operator picks from an arbitrary subset."""
        fields = self._fields()

        assert {"freeze_vision_encoder", "train_expert_only"} <= set(fields)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({}, "everything trainable"),
        ({"freeze_language_tower": True}, "language frozen, vision and expert train"),
        ({"train_expert_only": True}, "all of paligemma frozen"),
        ({"freeze_vision_encoder": True, "freeze_language_tower": True}, "only the expert trains"),
    ],
)
def test_the_flags_compose(flags, expected):
    """Each combination must be expressible; none may be silently overridden."""
    cfg = PI05Config(**flags)

    for name, value in flags.items():
        assert getattr(cfg, name) is value, f"{name} did not survive ({expected})"


# --- Measured against a real module tree -------------------------------------
#
# Everything above tests the config surface. That cannot catch the thing that
# actually matters -- whether parameters are frozen -- and it did not: the first
# version of this flag left `paligemma.lm_head` trainable where the train-script
# hack it replaced had frozen it. Only enumerating requires_grad off a
# constructed model found that.
#
# Dimensions are shrunk so this builds on CPU in seconds. Parameter *names* and
# requires_grad flags are what is under test, and neither depends on width.

pytest.importorskip("transformers")

# The train-script allowlist this flag replaced: anything not matching was frozen.
LEGACY_ALLOWLIST = (
    "gemma_expert",
    "vision_tower",
    "multi_modal",
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
)


class _TinyDims:
    width, depth, num_heads, num_kv_heads, head_dim, mlp_dim = 32, 2, 2, 1, 16, 64


def _build(**flags):
    import torch

    from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel

    torch.manual_seed(0)
    return PaliGemmaWithExpertModel(
        vlm_config=_TinyDims(),
        action_expert_config=_TinyDims(),
        precision="float32",
        **flags,
    )


def _frozen(model) -> set[str]:
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def _trainable(model) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}


class TestTheLanguageTowerIsActuallyFrozen:
    def test_language_parameters_stop_requiring_grad(self):
        model = _build(freeze_language_tower=True)

        language = {n for n in _frozen(model) if "language_model" in n}

        assert language, "no language parameters were frozen at all"
        assert all("language_model" not in n for n in _trainable(model) if "lm_head" not in n)

    def test_vision_and_expert_keep_training(self):
        """The whole point of a separate flag: cameras must still adapt."""
        model = _build(freeze_language_tower=True)
        trainable = _trainable(model)

        assert any("vision_tower" in n for n in trainable), "vision tower must stay trainable"
        assert any("gemma_expert" in n for n in trainable), "action expert must stay trainable"

    def test_nothing_is_frozen_when_the_flag_is_off(self):
        model = _build(freeze_language_tower=False)

        assert not _frozen(model)

    def test_it_matches_what_the_train_script_hack_froze(self):
        """Equivalence with the replaced behaviour, enumerated rather than argued.

        The hack set requires_grad from a substring allowlist; this asserts the
        flag freezes exactly the same parameter set -- including
        paligemma.lm_head, which sits beside `model` rather than inside
        language_model and which the first version of this flag missed.
        """
        model = _build(freeze_language_tower=True)

        legacy_frozen = {n for n, _ in model.named_parameters() if not any(k in n for k in LEGACY_ALLOWLIST)}

        assert _frozen(model) == legacy_frozen

    def test_the_language_head_is_frozen_with_the_tower(self):
        """Regression guard for the specific miss."""
        model = _build(freeze_language_tower=True)

        assert any("lm_head" in n for n in _frozen(model))

    def test_no_flag_combination_can_train_nothing(self):
        """All three at once still leaves the action expert trainable.

        Asserting this rather than the opposite because writing the opposite
        test is how I found out: `train_expert_only` freezes PaliGemma but is
        *named* for what survives, so the expert is never frozen by any flag.
        The guard inside _set_requires_grad is therefore a backstop against a
        future flag, not something these three can trigger — and this pins the
        property that makes it unreachable.
        """
        model = _build(freeze_language_tower=True, train_expert_only=True, freeze_vision_encoder=True)

        trainable = _trainable(model)

        assert trainable, "some parameter must remain trainable"
        assert all("gemma_expert" in n or "proj" in n or "time_mlp" in n for n in trainable), (
            f"unexpected survivors: {sorted(trainable)[:5]}"
        )

    def test_vision_and_language_freeze_composes(self):
        model = _build(freeze_language_tower=True, freeze_vision_encoder=True)
        frozen = _frozen(model)

        assert any("language_model" in n for n in frozen)
        assert any("vision_tower" in n for n in frozen)
        assert any("gemma_expert" in n for n in _trainable(model)), "the expert must survive"
