# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""How a check answers (src/lerobot/showservo/docs/act_execution.md, "What a check answers"): an object's measured
pose against the two outcomes' predictions, by the noise measured on that object, never by a fixed number."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.showservo.engine import predicates as p
from lerobot.showservo.engine.predicates import Answer

# ── the decision: the nearer of two predictions, once they are further apart than the noise ──


@pytest.mark.parametrize(
    ("observed", "answer"),
    [
        ((10.0, 0.0), Answer.YES),  # on the new prediction
        ((7.0, 1.0), Answer.YES),  # nearer the new by more than the noise
        ((0.0, 0.0), Answer.NO),  # on the old
        ((5.0, 0.0), Answer.UNKNOWN),  # half way
        ((10.0, 30.0), Answer.NO),  # far from both: something else moved it
    ],
)
def test_decide_takes_the_nearer_prediction_by_more_than_the_noise(observed, answer):
    v = p.decide(np.array(observed), new=np.array([10.0, 0.0]), old=np.zeros(2), noise=1.0, unit="px")
    assert v.answer is answer, v.why


def test_decide_cannot_tell_while_the_predictions_are_within_the_noise_of_each_other():
    v = p.decide(np.array([4.0, 0.0]), new=np.array([4.0, 0.0]), old=np.zeros(2), noise=2.5, unit="px")
    assert v.answer is Answer.UNKNOWN
    assert "apart" in v.why


def test_decide_scales_with_the_noise_not_with_a_constant():
    """The same observation reads success with steady views and can't tell with noisy ones (R9)."""
    obs, new, old = np.array([18.0, 0.0]), np.array([20.0, 0.0]), np.zeros(2)
    assert p.decide(obs, new=new, old=old, noise=2.0, unit="mm").answer is Answer.YES
    assert p.decide(obs, new=new, old=old, noise=12.0, unit="mm").answer is Answer.UNKNOWN
