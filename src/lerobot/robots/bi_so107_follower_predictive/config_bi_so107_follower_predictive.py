#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from typing import Literal

from ..bi_so107_follower.config_bi_so107_follower import BiSO107FollowerConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("bi_so107_follower_predictive")
@dataclass
class BiSO107FollowerPredictiveConfig(BiSO107FollowerConfig):
    """Config for the bimanual SO-107 follower with predictive-lookahead.

    Inherits every field of ``BiSO107FollowerConfig`` (per-arm ports,
    cameras, ``camera_read_strategy``, ``dry_run``, etc.) — recorded
    ``bi_so107_follower`` profiles and ``.trajectory.json`` files
    load against this config unchanged.

    The controller knobs are shared by both arms. Splitting them per-arm
    is a small follow-up if asymmetric tuning is ever needed; until then
    keeping them shared keeps the YAML / CLI surface small and matches
    the prototype's symmetric tuning on bi_so107 + white profile.

    Why this is registered as a distinct ``robot_type``: dataset action /
    state alignment differs from ``bi_so107_follower`` (controller
    compensates motor τ → ``state(t) ≈ intent(t)`` instead of
    ``state(t) ≈ leader(t − τ)``). Treating it as a separate embodiment
    prevents accidental mixing of the two recording regimes in training.
    See ``SO107FollowerPredictiveRobotConfig`` for the long-form rationale.
    """

    # Mirror the defaults from SO107FollowerPredictiveRobotConfig — these
    # are the values validated on bi_so107 + cylinder_ring_assembly in
    # scripts/proto_decoupled_teleop.py and scripts/backtest_lookahead.py.
    lookahead_ms: float = 80.0
    max_lookahead_ms: float = 110.0
    corrector_alpha: float = 1.0
    velocity_window_ms: float = 70.0
    control_rate_hz: float = 200.0
    adaptive: bool = True
    max_step_deg: float = 3.0
    velocity_estimator: Literal["quad", "linear", "forward_diff"] = "quad"
