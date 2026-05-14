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

"""Config fields for the predictive lookahead controller.

A dataclass mixin: any robot config that wants the predictive
controller inherits both its base ``RobotConfig`` subclass AND this
class, e.g.::

    @RobotConfig.register_subclass("so107_follower_predictive")
    @dataclass
    class SO107FollowerPredictiveRobotConfig(PredictiveControllerConfig, SOFollowerConfig):
        pass

All fields here carry validated defaults; subclasses can override
individual fields without touching the rest. Single source of truth for
predictive controller tunables — bi-arm configs inherit from the same
class instead of redeclaring fields (which previously meant new fields
added to one config silently went missing from the other).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class PredictiveControllerConfig:
    """Controller tunables. Inherit alongside the base robot config."""

    # Initial lookahead in ms. The controller refines this online via
    # amplitude-gated cross-correlation when ``adaptive=True``. For
    # bi_so107 white profile at P=16 the converged value is ~80-110 ms.
    lookahead_ms: float = 80.0

    # Cap on adaptive lookahead. Past this, extrapolation overshoot
    # dominates the residual and operator feel degrades. Empirical motor
    # τ on bi_so107 + Feetech STS3215 is ~147 ms — set cap above this so
    # the adaptive loop can fully compensate. Below this and the loop
    # saturates with residual state-vs-intent lag.
    max_lookahead_ms: float = 150.0

    # Predictor-corrector blend factor. 1.0 = pure leader-based prediction
    # (raw_shifted only). < 1.0 blends in a velocity-extrapolated predictor
    # from the action history → smoother motor stream. Validated value
    # on bi_so107 is 0.3 (70 % predictor + 30 % fresh leader-based shift).
    corrector_alpha: float = 1.0

    # Window over which leader velocity is estimated. Too short →
    # noise amplified; too long → phase lag (for centered estimators).
    velocity_window_ms: float = 70.0

    # How the controller computes v_leader from the window / publish stream:
    #
    #   * ``"quad"``    — LSQ quadratic fit, slope evaluated at the
    #                     most-recent sample's time. Unbiased estimate of
    #                     v(now) under acceleration.
    #   * ``"linear"``  — LSQ linear slope (original prototype behaviour).
    #                     Centered velocity estimator → biased under
    #                     acceleration. Kept selectable for parity with
    #                     prior measurements / regression-debugging.
    #   * ``"forward_diff"`` — Two-sample causal difference at the window
    #                          tail. Lowest bias but highest noise
    #                          sensitivity.
    #   * ``"amp_gated_lp"`` — EMA-lowpass V_est with per-joint amplitude
    #                          gating, rebuilt per tick over the velocity
    #                          window. WARNING: degrades to ungated raw
    #                          forward_diff when the window holds <3
    #                          samples (e.g. 30 Hz publish × 70 ms
    #                          window → 90% of ticks fall into the
    #                          fallback). Designed for the teleop pull
    #                          path where ~14 samples are available;
    #                          use ``stateful_lp`` for low-rate dict
    #                          publishes instead.
    #   * ``"stateful_lp"``  — Per-publish 1st-order EMA velocity held as
    #                          controller state, with α keyed on the
    #                          actual publish dt so the cutoff is
    #                          ``velocity_lowpass_hz`` regardless of
    #                          publish rate. Rate-INVARIANT by design:
    #                          the tick just reads the stored velocity,
    #                          there's no per-tick window recomputation,
    #                          so HF amplification in motor_cmd doesn't
    #                          balloon at low publish rates. Recommended
    #                          for any policy/teleop pushing dicts; the
    #                          chunk path bypasses velocity estimation
    #                          entirely so it doesn't care which option
    #                          is selected here.
    velocity_estimator: Literal["quad", "linear", "forward_diff", "amp_gated_lp", "stateful_lp"] = (
        "stateful_lp"
    )

    # Knobs for ``amp_gated_lp`` / ``stateful_lp``. Only consulted when
    # the corresponding estimator is selected.
    #
    # ``velocity_lowpass_hz`` — cutoff of the 1st-order EMA applied to
    #     the velocity estimate before extrapolation. Default 4 Hz
    #     attenuates the human 8-12 Hz tremor band while passing real
    #     motion (≤2 Hz dominant). Lower → more smoothing → more phase
    #     lag in V_est → less effective lookahead.
    # ``amp_gate_lo`` / ``amp_gate_hi`` — peak-to-peak motion amplitude
    #     (in joint units — degrees for arm joints, RANGE_0_100 for
    #     gripper) over the velocity window below which the controller
    #     becomes pure pass-through (no lookahead). Linear ramp between
    #     lo and hi. Defaults validated on bi_so107 gripper sweep at
    #     ±3-unit deliberate motion with 0.5-unit tremor. ONLY consulted
    #     by ``amp_gated_lp``; ``stateful_lp`` has no amplitude gate.
    velocity_lowpass_hz: float = 4.0
    amp_gate_lo: float = 1.0
    amp_gate_hi: float = 3.0

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
