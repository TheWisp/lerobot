# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The layer that says which tracker output is a measurement (src/lerobot/showservo/estimation.py)."""

from __future__ import annotations

import numpy as np

from lerobot.showservo.estimation import Candidate, SourceVerdict


def test_a_sources_verdict_takes_that_sources_candidate_when_it_is_confident_and_says_why_not():
    a, b = np.eye(4), np.eye(4) * 2.0
    out = SourceVerdict("second").estimate(
        1.0,
        {
            "block": [Candidate("first", a, ok=True), Candidate("second", b, ok=True)],
            "ring": [Candidate("first", a, ok=True), Candidate("second", b, ok=False, why="lost")],
            "cup": [Candidate("first", a, ok=True)],
        },
    )
    assert out["block"].motion is b and out["block"].why == "second: confident"
    assert out["ring"].motion is None and out["ring"].why == "second: lost"
    assert out["cup"].motion is None and out["cup"].why == "no second candidate"
