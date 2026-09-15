# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""How S1's robot state is normalized, in one place for training and serving.

A joint the task leaves still gets a per-channel std at the numerical floor.
Dividing by it turns a reading a fraction of a degree from the recorded mean
into tens of thousands of sigma, and one channel that size dominates the first
linear layer. Two bounds answer that, and neither covers the other: a floor on
the std of position channels, which is unit-specific, and a clamp on the
normalized result, which is not.

Both are here rather than beside their callers so training and inference cannot
drift apart, and so each is a function a test can drive on its own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

#: Bound on the normalized state fed to the network. Applied by the dataset and
#: by the policy, so a model trained after it sees exactly what it is served.
NORMALIZED_STATE_CLAMP = 10.0

#: The suffix that marks a position channel. The floor is in dataset-native
#: units, so it is only meaningful for channels that share one.
POSITION_SUFFIX = ".pos"


def floor_position_std(
    state_std: Tensor, state_feature_names: Sequence[str] | None, floor: float
) -> tuple[Tensor, int]:
    """Raise every position channel's std to at least ``floor``.

    Pre: ``floor`` is finite and positive; ``state_feature_names`` holds one
    name per value of ``state_std``, in order, and at least one ends in
    ``.pos``. Each is a refusal rather than a silent no-op, because applying a
    positional floor to an order you cannot confirm floors the wrong channels.

    Post: returns a new tensor, and how many channels it raised. ``state_std``
    is not modified.
    """
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError(f"A position std floor must be finite and positive; got {floor}")

    count = state_std.shape[0]
    if state_feature_names is None or len(state_feature_names) != count:
        given = 0 if state_feature_names is None else len(state_feature_names)
        raise ValueError(
            "A positive state position std floor requires one ordered state feature name per "
            f"state value; got {given} names for {count} values"
        )

    is_position = torch.tensor(
        [name.endswith(POSITION_SUFFIX) for name in state_feature_names], dtype=torch.bool
    )
    if not is_position.any():
        raise ValueError(
            f"A positive state position std floor was requested but no state feature name "
            f"ends in '{POSITION_SUFFIX}'"
        )

    floored = state_std.clone()
    floored[is_position] = floored[is_position].clamp(min=floor)
    raised = int((state_std[is_position] < floor).sum().item())
    return floored, raised


def position_channels(state_feature_names: Sequence[str] | None) -> int:
    """How many channels the floor applies to, for reporting."""
    return sum(1 for n in state_feature_names or () if n.endswith(POSITION_SUFFIX))
