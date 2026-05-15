#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_so107_follower")
@dataclass
class BiSO107FollowerConfig(RobotConfig):
    left_arm_port: str
    right_arm_port: str

    # Optional
    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: float | dict[str, float] | None = None
    left_arm_use_degrees: bool = False
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_use_degrees: bool = False

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # How to read each camera frame in get_observation:
    #   "latest"       — return whatever is in the grab thread's buffer right
    #                    now (cam.read_latest). Never blocks; may return a
    #                    duplicate of the previous frame when the consumer
    #                    is faster than the camera. Default — keeps the
    #                    control loop at its target FPS even if a camera lags.
    #   "wait_for_new" — wait until a fresh frame arrives (cam.async_read).
    #                    Never duplicates; the loop period stretches to the
    #                    slowest camera's effective FPS. Use when the policy
    #                    was trained on a strict-no-duplicate dataset and you
    #                    want to mirror that contract.
    #
    # Why "latest" is the default: in a multi-camera setup the first camera
    # iterated under "wait_for_new" pays the full async_read wait while
    # every subsequent camera's grab thread has already cached a fresh
    # frame — the result is one super-fresh camera and the others
    # randomly stale (large cross-camera time skew), plus the loop period
    # stretching to the slowest camera. Empirical (white profile, 4 cams,
    # FlowMatchingS1): the swap dropped get_observation p50 from 66 ms →
    # 14 ms and doubled the effective control rate from 15 Hz to 30 Hz,
    # with overrun ratio falling from 99.5% to 0.0%. Cross-camera
    # staleness also tightened.
    #
    # Only affects cameras that go through async_read. RealSense aligned
    # color+depth (read_color_and_aligned_depth) keeps its existing
    # behaviour because it has no non-blocking variant; in practice that
    # method only blocks when the consumer is faster than the camera,
    # which the loop-rate fix above already addresses.
    camera_read_strategy: Literal["latest", "wait_for_new"] = "latest"

    # When True, ``send_action`` is a no-op: the motors never receive a
    # command, but every other side effect (connect, calibrate, motor
    # reads, camera reads, latency profiling) still runs. Lets automated
    # tooling exercise the full control-loop stack — including this
    # robot's get_observation timing and any policy under test — without
    # physically driving the arms.
    #
    # Note: torque is still enabled on connect (the motors hold position
    # so they don't drop under gravity), but no goal positions are sent.
    dry_run: bool = False
