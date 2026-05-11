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
