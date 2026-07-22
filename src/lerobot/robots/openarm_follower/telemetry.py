#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Low-overhead, hardware-independent OpenArm follower telemetry.

The seven arm joints are aggregated. J8 is deliberately excluded because the
standard OpenArm configuration drives the gripper in POS_FORCE mode instead of
MIT mode. When gravity feed-forward is active, callers should pass the torque
that was actually sent. That torque must come from the verified MJCF path; this
module has no URDF dependency or fallback.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

TELEMETRY_PERIOD_SEC = 30.0


def _finite_head(values: Sequence[float], size: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < size:
        raise ValueError(f"{name} must be a one-dimensional sequence with at least {size} values")
    result = array[:size]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _format_vector(values: NDArray[np.float64], precision: int) -> str:
    return "[" + " ".join(f"{value:.{precision}f}" for value in values) + "]"


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Immutable aggregate for one telemetry window."""

    count: int
    position_error_max: NDArray[np.float64]
    external_torque_mean: NDArray[np.float64]
    external_torque_abs_mean: NDArray[np.float64]
    external_torque_abs_max: NDArray[np.float64]
    mos_temperature_max: float


class FollowerTelemetry:
    """Aggregate follower statistics without performing hardware I/O."""

    def __init__(
        self,
        name: str,
        n_joints: int = 7,
        period_secs: float = TELEMETRY_PERIOD_SEC,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not name:
            raise ValueError("name must not be empty")
        if isinstance(n_joints, bool) or not isinstance(n_joints, int) or n_joints <= 0:
            raise ValueError("n_joints must be a positive integer")
        if not math.isfinite(period_secs) or period_secs <= 0.0:
            raise ValueError("period_secs must be finite and positive")

        self.name = name
        self.n = n_joints
        self.period_secs = float(period_secs)
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        """Clear all aggregates and start a new reporting window."""
        self._position_error_max = np.zeros(self.n, dtype=np.float64)
        self._external_torque_sum = np.zeros(self.n, dtype=np.float64)
        self._external_torque_abs_sum = np.zeros(self.n, dtype=np.float64)
        self._external_torque_abs_max = np.zeros(self.n, dtype=np.float64)
        self._mos_temperature_max = 0.0
        self._count = 0
        self._started_at = self._clock()

    @property
    def count(self) -> int:
        return self._count

    @property
    def t0(self) -> float:
        """Compatibility view of the current window start time."""
        return self._started_at

    def update(
        self,
        q_cmd: Sequence[float],
        q_pos: Sequence[float],
        q_torque: Sequence[float],
        t_mos: Sequence[float],
        tff: Sequence[float] | None = None,
    ) -> None:
        """Accumulate one validated control-cycle sample.

        Positions must use the same unit. Torques use N.m and temperatures use
        degrees Celsius. Inputs may include J8; only the first ``n_joints``
        values are aggregated.
        """
        commanded = _finite_head(q_cmd, self.n, "q_cmd")
        measured = _finite_head(q_pos, self.n, "q_pos")
        torque = _finite_head(q_torque, self.n, "q_torque")
        temperature = _finite_head(t_mos, self.n, "t_mos")

        external_torque = torque
        if tff is not None:
            external_torque = torque - _finite_head(tff, self.n, "tff")

        absolute_external_torque = np.abs(external_torque)
        self._position_error_max = np.maximum(self._position_error_max, np.abs(commanded - measured))
        self._external_torque_sum += external_torque
        self._external_torque_abs_sum += absolute_external_torque
        self._external_torque_abs_max = np.maximum(self._external_torque_abs_max, absolute_external_torque)
        self._mos_temperature_max = max(self._mos_temperature_max, float(np.max(temperature)))
        self._count += 1

    def snapshot(self) -> TelemetrySnapshot | None:
        """Return a copy of the current aggregate, or ``None`` if empty."""
        if self._count == 0:
            return None
        return TelemetrySnapshot(
            count=self._count,
            position_error_max=self._position_error_max.copy(),
            external_torque_mean=self._external_torque_sum / self._count,
            external_torque_abs_mean=self._external_torque_abs_sum / self._count,
            external_torque_abs_max=self._external_torque_abs_max.copy(),
            mos_temperature_max=self._mos_temperature_max,
        )

    def maybe_report(self, now: float | None = None) -> bool:
        """Log and reset the aggregate after the configured interval."""
        current_time = self._clock() if now is None else now
        if not math.isfinite(current_time):
            raise ValueError("now must be finite")
        snapshot = self.snapshot()
        if snapshot is None or current_time - self._started_at < self.period_secs:
            return False

        logger.info(
            "[%s] telem n=%d err_max=%s tau_ext_mean=%s tau_ext_absmean=%s tau_ext_absmax=%s tmax=%.0f",
            self.name,
            snapshot.count,
            _format_vector(snapshot.position_error_max, 3),
            _format_vector(snapshot.external_torque_mean, 2),
            _format_vector(snapshot.external_torque_abs_mean, 2),
            _format_vector(snapshot.external_torque_abs_max, 2),
            snapshot.mos_temperature_max,
        )
        self.reset()
        return True


__all__ = ["FollowerTelemetry", "TELEMETRY_PERIOD_SEC", "TelemetrySnapshot"]
