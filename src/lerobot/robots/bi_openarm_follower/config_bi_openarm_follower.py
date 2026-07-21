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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..openarm_follower import OpenArmFollowerConfigBase


@RobotConfig.register_subclass("bi_openarm_follower")
@dataclass(kw_only=True)
class BiOpenArmFollowerConfig(RobotConfig):
    """Configuration class for Bi OpenArm Follower robots."""

    id: str | None = "bi_openarm_follower"

    left_arm_config: OpenArmFollowerConfigBase
    right_arm_config: OpenArmFollowerConfigBase

    # Top-level cameras not attached to a specific side. Keys are kept as-is in
    # observations (no `left_`/`right_` prefix). Per-arm cameras (declared on
    # `{left,right}_arm_config.cameras`) are prefixed.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Per-arm Cartesian-IK tuning. Only consulted when a Cartesian teleop
    # is attached (Quest VR); a joint-space leader never builds the IK
    # kinematics so these are inert. Two arms share one setting because the
    # IK behavior is per-arm-geometry — both arms are the same URDF
    # (mirrored). Mirrors BiSO107FollowerConfig's knobs.
    ik_posture_cost: float = field(
        default=0.05,
        metadata={
            "description": (
                "PostureTask weight relative to FrameTask (which is 1.0). "
                "Default 0.05 makes posture a null-space tiebreaker. Raise "
                "(e.g. 0.3) for stronger 'stay near previous pose' — tighter "
                "joint continuity near reach limits / singularities, at the "
                "cost of small EE tracking lag."
            ),
        },
    )
    ik_max_iters: int = field(
        default=50,
        metadata={
            "description": (
                "QP iteration budget per IK call. Pink's default is 10; 50 "
                "closes per-call lag on a moving teleop target (mm-scale at "
                "typical speeds). Lower to 10–20 for more 'stick to seed' "
                "feel at the cost of moving-target tracking accuracy."
            ),
        },
    )
