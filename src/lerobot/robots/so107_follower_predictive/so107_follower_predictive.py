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

"""SO-107 follower with the predictive-lookahead controller built in.

This is treated as a SEPARATE EMBODIMENT from the plain ``so107_follower``
even though the hardware is identical, because the trained-policy contract
is materially different:

  * ``so107_follower``: ``state(t) ≈ leader(t − τ)`` — motor lag visible in
    every observation. Policy implicitly compensates τ via action chunks.

  * ``so107_follower_predictive``: ``state(t) ≈ intent(t)`` — controller
    transparently compensates motor τ. Dataset has aligned action/state
    pairs at matching timestamps. Policy never observes τ.

Mixing datasets across the two regimes in training is a real hazard:
training-time state-action alignment would be inconsistent and the policy
would output actions that lurch when deployed. The robot_type
separation makes this prevention automatic — recording / training /
inference all key off the same ``robot_type`` string.

Implementation: this class is now a thin composition over
:class:`PredictiveLookaheadMixin` and :class:`SO107Follower`. All
controller logic + lifecycle plumbing lives in the mixin / controller
modules under ``lerobot.robots.predictive``. Adding a new
``XXFollowerPredictive`` variant for different motor counts (e.g. SO-100
6-motor) follows the same pattern with no code duplication — see
``SOFollowerPredictive``.
"""

from __future__ import annotations

from ..predictive.mixin import PredictiveLookaheadMixin
from ..so_follower.so_follower import SO107Follower
from .config_so107_follower_predictive import SO107FollowerPredictiveRobotConfig


class SO107FollowerPredictive(PredictiveLookaheadMixin, SO107Follower):
    """SO-107 follower with predictive-lookahead controller always on.

    Treated as a distinct embodiment (own ``name`` / ``config_class``) so
    the dataset / policy contract is unambiguous. All behavior comes from
    :class:`PredictiveLookaheadMixin`; this class only nails the variant
    name + config type to the SO-107 motor layout.
    """

    config_class = SO107FollowerPredictiveRobotConfig
    name = "so107_follower_predictive"
