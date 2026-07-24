# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned machine-readable records embedded in human training logs."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

TRAINING_LOG_JSON_MARKER = "LEROBOT_TRAINING_JSON:"
TRAINING_LOG_JSON_VERSION = 1


def format_training_log_record(
    *,
    step: int,
    total_steps: int,
    **values: int | float,
) -> str:
    """Encode one validated training progress/metrics record."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"Training log step must be a non-negative integer, got {step!r}")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError(f"Training log total_steps must be a positive integer, got {total_steps!r}")

    payload: dict[str, int | float] = {
        "version": TRAINING_LOG_JSON_VERSION,
        "step": step,
        "total_steps": total_steps,
    }
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Training log field {key!r} must be numeric, got {type(value).__name__}")
        if not math.isfinite(value):
            raise ValueError(f"Training log field {key!r} must be finite, got {value}")
        payload[key] = value
    return TRAINING_LOG_JSON_MARKER + json.dumps(payload, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class TrainingHealthSample:
    """One window of custom-trainer health metrics ready for human and machine logs."""

    values: dict[str, float]
    record: str
    omitted_fields: tuple[str, ...]


class TrainingHealthTracker:
    """Measure a custom LeRobot training loop over each logging window.

    Standard policies trained by ``lerobot-train`` already use
    :class:`lerobot.utils.logging_utils.MetricsTracker`. This helper is for
    policy-specific training entry points that bypass that loop but still need
    to emit the same dashboard health fields without duplicating timing,
    checkpoint-exclusion, and structured-record logic.

    Call :meth:`step` after each optimizer update, :meth:`sample` at a log
    boundary, and :meth:`reset` after logging. Wrap checkpoint writes in
    :meth:`exclude_time` so an in-window save does not distort throughput.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        total_steps: int,
        clock: Callable[[], float] = time.perf_counter,
        peak_memory_gb: Callable[[], float] | None = None,
        reset_peak_memory: Callable[[], None] | None = None,
        ema_alpha: float = 0.2,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError(f"total_steps must be a positive integer, got {total_steps!r}")
        if not 0 < ema_alpha <= 1:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha!r}")

        self.batch_size = batch_size
        self.total_steps = total_steps
        self._clock = clock
        self._peak_memory_gb = peak_memory_gb
        self._reset_peak_memory = reset_peak_memory
        self._ema_alpha = ema_alpha
        self._ema_step_time_s: float | None = None
        self._window_started = 0.0
        self._window_data_s = 0.0
        self._window_steps = 0
        self.reset()

    @contextmanager
    def measure_data_loading(self) -> Iterator[None]:
        """Accumulate data-fetch and host-to-device time for the current window."""
        started = self._clock()
        try:
            yield
        finally:
            self._window_data_s += max(0.0, self._clock() - started)

    @contextmanager
    def exclude_time(self) -> Iterator[None]:
        """Exclude checkpoint or other non-training work from the current window."""
        started = self._clock()
        try:
            yield
        finally:
            self._window_started += max(0.0, self._clock() - started)

    def step(self) -> None:
        """Record one completed optimizer update."""
        self._window_steps += 1

    def sample(self, *, step: int, values: Mapping[str, int | float]) -> TrainingHealthSample:
        """Build one finite metrics record without resetting the current window."""
        if self._window_steps <= 0:
            raise RuntimeError("Cannot sample training health before an optimizer step")

        elapsed_s = max(0.0, self._clock() - self._window_started)
        average_step_s = elapsed_s / self._window_steps
        average_data_s = self._window_data_s / self._window_steps
        average_update_s = max(0.0, average_step_s - average_data_s)
        self._ema_step_time_s = (
            average_step_s
            if self._ema_step_time_s is None
            else self._ema_alpha * average_step_s + (1 - self._ema_alpha) * self._ema_step_time_s
        )

        measured: dict[str, int | float] = {
            **values,
            "eta_seconds": max(0, self.total_steps - step) * self._ema_step_time_s,
            "updt_s": average_update_s,
            "data_s": average_data_s,
            "samples_per_s": self.batch_size / max(average_step_s, 1e-9),
            "step_time_ms": average_step_s * 1000,
        }
        if self._peak_memory_gb is not None:
            measured["mem_gb"] = self._peak_memory_gb()

        finite: dict[str, float] = {}
        omitted: list[str] = []
        for name, value in measured.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                omitted.append(name)
            else:
                finite[name] = float(value)

        return TrainingHealthSample(
            values=finite,
            record=format_training_log_record(step=step, total_steps=self.total_steps, **finite),
            omitted_fields=tuple(sorted(omitted)),
        )

    def reset(self) -> None:
        """Start a fresh logging window while preserving the ETA moving average."""
        self._window_started = self._clock()
        self._window_data_s = 0.0
        self._window_steps = 0
        if self._reset_peak_memory is not None:
            self._reset_peak_memory()
