#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Literal

from ..config import RobotConfig
from ..so_follower.config_so_follower import SOFollowerConfig


@RobotConfig.register_subclass("so107_follower_predictive")
@dataclass
class SO107FollowerPredictiveRobotConfig(RobotConfig, SOFollowerConfig):
    """Config for the SO-107 follower with predictive-lookahead controller.

    Registered under its own ``robot_type`` because the trained-policy
    contract differs from the plain ``so107_follower``: datasets recorded
    against this robot have ``action(t) ≈ state(t)`` (the operator's raw
    intent aligns with the motor's actual position because the controller
    transparently compensates motor τ), whereas plain so107_follower
    datasets have ``state(t) ≈ leader(t − τ)``. Treating predictive vs.
    pass-forward as separate embodiments prevents accidental mixing of
    incompatible recording regimes in training.

    All fields inherit the same defaults that the prototype validated on
    bi_so107 + P=16 + corrector α=0.3 (see scripts/proto_decoupled_teleop.py
    and the cylinder_ring_assembly backtest in scripts/backtest_lookahead.py).
    """

    # Initial lookahead in ms. The controller refines this online via
    # amplitude-gated cross-correlation when ``adaptive=True``. For
    # bi_so107 white profile at P=16 the converged value is ~80-110 ms.
    lookahead_ms: float = 80.0

    # Cap on adaptive lookahead. Past this, extrapolation overshoot
    # dominates the residual and operator feel degrades. Per-arm tunable.
    max_lookahead_ms: float = 110.0

    # Predictor-corrector blend factor. 1.0 = pure leader-based prediction
    # (raw_shifted only). < 1.0 blends in a velocity-extrapolated predictor
    # from the action history → smoother motor stream. Validated value
    # on bi_so107 is 0.3 (70 % predictor + 30 % fresh leader-based shift).
    corrector_alpha: float = 1.0

    # Window over which leader velocity is estimated. Too short →
    # noise amplified; too long → phase lag (for centered estimators).
    velocity_window_ms: float = 70.0

    # How the controller computes v_leader from the window:
    #   * ``"quad"``    — LSQ quadratic fit, slope evaluated at the
    #                     most-recent sample's time. Unbiased estimate of
    #                     v(now) under acceleration. Backtest winner on
    #                     real teleop motion (scripts/backtest_velocity_
    #                     estimators.py). Default.
    #   * ``"linear"``  — LSQ linear slope (original prototype behaviour).
    #                     Centered velocity estimator → biased under
    #                     acceleration. Kept selectable for parity with
    #                     prior measurements / regression-debugging.
    #   * ``"forward_diff"`` — Two-sample causal difference at the window
    #                          tail. Lowest bias but highest noise
    #                          sensitivity at the control rate; fine for
    #                          smooth high-rate sources, marginal at
    #                          30 Hz dict-push sources.
    velocity_estimator: Literal["quad", "linear", "forward_diff"] = "quad"

    # Control-thread rate. The bus comfortably sustains 170+ Hz at P=48
    # under sync_write-only load. 200 Hz is the prototype default.
    control_rate_hz: float = 200.0

    # Whether to adapt ``lookahead_ms`` online from cross-correlation of
    # observed (intent, state). Set False to use the fixed lookahead.
    adaptive: bool = True

    # Safety clamp on the controller's per-step Δ-position vs the last
    # sent action. Extrapolation can spike during sharp leader reversals;
    # this caps the worst-case motor jolt. Normalized units (or degrees
    # when ``use_degrees=True``) per control tick. Conservative default:
    # 3 units/tick at 200 Hz ≈ 600 units/s.
    max_step_deg: float = 3.0
