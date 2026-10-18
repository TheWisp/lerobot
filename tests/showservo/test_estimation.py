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

from lerobot.showservo.estimation import Candidate, TrackerVerdict


def test_the_trackers_verdict_takes_the_first_confident_candidate_and_says_why_not():
    a, b = np.eye(4), np.eye(4) * 2.0
    out = TrackerVerdict().estimate(
        1.0,
        {
            "block": [Candidate("first", a, ok=False, why="no fit"), Candidate("second", b, ok=True)],
            "ring": [Candidate("first", a, ok=False, why="lost"), Candidate("second", b, ok=False)],
        },
    )
    assert out["block"].motion is b and out["block"].why == "second: confident"
    assert out["ring"].motion is None
    assert out["ring"].why == "first: lost; second: not confident"
