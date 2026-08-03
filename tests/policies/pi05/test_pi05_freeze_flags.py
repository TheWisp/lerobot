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
