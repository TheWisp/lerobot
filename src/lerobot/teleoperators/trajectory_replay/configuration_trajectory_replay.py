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

"""Config for the trajectory-replay teleoperator.

Replays a previously hand-recorded safe trajectory as if it were a leader
arm. Useful for unattended latency experiments and automated dataset
recording — no human at the leader is required.
"""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("trajectory_replay")
@dataclass
class TrajectoryReplayTeleopConfig(TeleoperatorConfig):
    # Path to a ``<name>.trajectory.json`` produced by the Safe Trajectory
    # recorder on the Robot tab. The file is loaded once at ``connect()``
    # time. The trajectory's joint set must match the follower robot's
    # ``action_features``.
    trajectory_path: str = ""

    # Predictive lookahead: when the loop asks for an action at wall-time t,
    # return the trajectory's value at t + lookahead_s instead. Used to
    # cancel motor response lag (~60 ms on Feetech STS3215 at P=48).
    lookahead_s: float = 0.0

    # Chunk simulation: pretend we only "received" the trajectory in chunks
    # of N consecutive frames (like a chunked policy producing N future
    # actions every N control steps). When the lookahead query lands past
    # the current chunk's last frame, the source must extrapolate from the
    # chunk's own velocity rather than peek at the next chunk — modelling
    # what a real chunked policy faces. ``None`` (default) disables the
    # simulation, exposing the full trajectory for interpolation.
    simulate_chunk_size: int | None = None
