# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What a padded chunk position does, and does not, do to training.

Padding at an episode end is only safe because those positions are excluded from
the loss. That exclusion is one multiply buried in the loss expression, it has no
observable effect on any metric, and the boundary bug this suite also covers was
undetectable precisely because the same mask was inert. So it is pinned directly
rather than assumed.

The second test is the honest complement: a masked position is removed from the
loss but *not* from the input, because this model has no attention mask. Stating
that as a passing test means a future change that adds one will fail here and be
made deliberately, rather than quietly altering what padding costs.
"""

from __future__ import annotations

import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import ACTION, OBS_STATE, FlowMatchingS1Policy

CHUNK, REAL, BATCH, DIM = 6, 4, 2, 2
SEED = 1234


def _policy() -> FlowMatchingS1Policy:
    """A tiny vision-free S1: this is about the loss reduction, not the encoder."""
    cfg = FlowMatchingS1Config(
        chunk_size=CHUNK,
        action_dim=DIM,
        state_dim=DIM,
        hidden_dim=32,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        use_dino_backbone=False,
        image_features={},
        robot_state_feature=True,
        action_feature_names=["a", "b"],
        state_feature_names=["a", "b"],
        rtc_max_delay=0,  # the RTC prefix has its own mask; keep this test to padding
    )
    return FlowMatchingS1Policy(cfg).eval()


def _batch(tail_value: float) -> dict[str, torch.Tensor]:
    """Positions REAL.. are padding. ``tail_value`` fills them."""
    torch.manual_seed(7)
    actions = torch.randn(BATCH, CHUNK, DIM)
    actions[:, REAL:, :] = tail_value
    is_pad = torch.zeros(BATCH, CHUNK, dtype=torch.bool)
    is_pad[:, REAL:] = True
    return {
        ACTION: actions,
        OBS_STATE: torch.zeros(BATCH, DIM),
        "action_is_pad": is_pad,
    }


def _loss(policy, batch) -> float:
    """Seeded, so the flow-matching noise and timestep are identical per call."""
    torch.manual_seed(SEED)
    loss, _ = policy(batch)
    return float(loss.detach())


def test_padded_positions_contribute_nothing_to_the_loss():
    """Change only the padded targets; the objective must not move.

    ``denoise_step`` is stubbed so the prediction cannot depend on the padded
    inputs — this isolates the loss reduction, which is the property under test,
    from the attention coupling covered by the next test. What remains varying is
    exactly the velocity target at padded positions.
    """
    policy = _policy()
    policy.model.denoise_step = lambda x_t, context, timestep, cached_kv=None: torch.zeros_like(x_t)

    quiet = _loss(policy, _batch(0.0))
    wild = _loss(policy, _batch(1000.0))

    assert quiet == wild, (
        "padded positions changed the loss, so they are being scored: "
        f"{quiet} vs {wild}. The mask in the loss expression is not doing its job."
    )


def test_padded_positions_are_still_part_of_the_input():
    """They are excluded from the loss, not from the forward pass.

    ``x_t`` is built from every position including the padded tail, and the
    decoder attends across the whole chunk with no key-padding mask. So a padded
    value still reaches the prediction at real positions. This is why a flagged
    *bad* frame cannot be handled by masking alone the way a repeated real pose
    can — recorded here because nothing else in the codebase says it.
    """
    policy = _policy()

    quiet = _loss(policy, _batch(0.0))
    wild = _loss(policy, _batch(1000.0))

    assert quiet != wild, (
        "the padded tail no longer reaches the prediction — if an attention mask "
        "was added, that is a real improvement and this test should be updated to "
        "assert equality instead of removed."
    )
