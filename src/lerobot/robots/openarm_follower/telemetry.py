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

"""Aggregated follower telemetry (one-line summary every `period_secs`).

Arm joints 0..6 only (the gripper runs POS_FORCE, not MIT). Per cycle:
max |q_cmd - q_meas| per joint, tau_ext = qtorque - tff (signed mean,
mean abs, max abs) and max motor MOS temperature.
"""

import logging
import time

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# Period [s] between one-line telemetry summaries (always enabled).
TELEMETRY_PERIOD_SEC = 30.0


def _fmt(values: ArrayLike, precision: int) -> str:
    """Format a vector as '[v1 v2 ...]' for the telemetry line."""
    return "[" + " ".join(f"{v:.{precision}f}" for v in values) + "]"


class FollowerTelemetry:
    """Aggregates per-cycle follower stats into a one-line summary."""

    def __init__(self, name: str, n_joints: int = 7, period_secs: float = TELEMETRY_PERIOD_SEC):
        """Start a fresh aggregation window for the arm `name`."""
        self.name = name
        self.n = n_joints
        self.period_secs = period_secs
        self.reset()

    def reset(self) -> None:
        """Zero all accumulators and restart the window clock."""
        self.err_max = np.zeros(self.n)
        self.tau_sum = np.zeros(self.n)
        self.tau_abs_sum = np.zeros(self.n)
        self.tau_abs_max = np.zeros(self.n)
        self.tmax = 0.0
        self.count = 0
        self.t0 = time.monotonic()

    def update(
        self,
        q_cmd: ArrayLike,
        q_pos: ArrayLike,
        q_torque: ArrayLike,
        t_mos: ArrayLike,
        tff: ArrayLike | None = None,
    ) -> None:
        """Accumulate one control cycle (a few vector ops, no I/O).

        Args:
            q_cmd: Commanded joint positions (any unit, at least n_joints long).
            q_pos: Measured joint positions (same unit as q_cmd).
            q_torque: Measured joint torques [Nm].
            t_mos: MOS temperatures per motor.
            tff: Feedforward torque actually sent [Nm], subtracted from q_torque.
        """
        n = self.n
        self.err_max = np.maximum(
            self.err_max,
            np.abs(np.asarray(q_cmd, dtype=float)[:n] - np.asarray(q_pos, dtype=float)[:n]),
        )
        tau_ext = np.asarray(q_torque, dtype=float)[:n]
        if tff is not None:
            tau_ext = tau_ext - np.asarray(tff, dtype=float)[:n]
        self.tau_sum += tau_ext
        self.tau_abs_sum += np.abs(tau_ext)
        self.tau_abs_max = np.maximum(self.tau_abs_max, np.abs(tau_ext))
        self.tmax = max(self.tmax, float(np.max(np.asarray(t_mos, dtype=float)[:n])))
        self.count += 1

    def maybe_report(self, now: float | None = None) -> bool:
        """Log the one-line summary and reset if the window elapsed."""
        now = time.monotonic() if now is None else now
        if self.count == 0 or now - self.t0 < self.period_secs:
            return False
        logger.info(
            f"[{self.name}] telem n={self.count}"
            f" err_max={_fmt(self.err_max, 3)}"
            f" tau_ext_mean={_fmt(self.tau_sum / self.count, 2)}"
            f" tau_ext_absmean={_fmt(self.tau_abs_sum / self.count, 2)}"
            f" tau_ext_absmax={_fmt(self.tau_abs_max, 2)}"
            f" tmax={self.tmax:.0f}"
        )
        self.reset()
        return True
