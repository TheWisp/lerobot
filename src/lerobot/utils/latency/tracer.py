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

"""Per-iteration latency tracer for real-time loops.

Each iteration: call ``start()``, wrap timed regions in ``span()``, record
camera consumption with ``cam_consume()``, and finalize with ``commit()``.
The tracer is designed to be cheap (~5–10 µs/iter) and is a no-op when its
aggregator is None.

See ``src/lerobot/gui/docs/latency_monitoring.md`` for the contract; in
particular, the residual cross-check formula and which stages are
considered "post-consume".
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from lerobot.utils.latency.aggregator import LatencyAggregator

# Stages that happen AFTER all inputs have been assembled in the iteration.
# These contribute additively to the e2e latency residual. Camera staleness
# already includes the time the loop spent reading motors before reading the
# cached camera frame, so motor_read_ms is intentionally NOT in this set.
_POST_CONSUME_KEYS: tuple[str, ...] = (
    "process_obs_ms",
    "process_action_ms",
    "action_send_ms",
    "infer_total_ms",
    "dataset_write_ms",
    "video_encode_ms",
)


class LatencyTracer:
    """Build one latency record per loop iteration.

    Preconditions:
      - One tracer instance per loop. Not thread-safe across iterations.
      - ``start()`` is called at the top of each iteration before any
        ``span()`` / ``cam_consume()`` calls.
      - ``commit()`` is called exactly once at the end of each iteration.

    Postconditions:
      - Each ``commit()`` produces a fresh dict and (if ``aggregator`` is
        provided) appends it to the aggregator's bounded deque.
      - When ``input_oldest`` was set during the iteration (via
        ``cam_consume`` or ``mark_input``), ``commit()`` adds
        ``e2e_obs_to_action_ms`` and ``residual_ms`` to the record.
    """

    def __init__(
        self,
        aggregator: LatencyAggregator | None = None,
        loop_kind: str = "teleop",
        target_fps: float | None = None,
    ):
        self.aggregator = aggregator
        self.loop_kind = loop_kind
        self._target_period_ms: float | None = (1000.0 / float(target_fps)) if target_fps else None
        self._record: dict[str, Any] = {}
        self._iter_start: float = 0.0
        # Per-camera last latest_ts seen, for period jitter computation.
        self._cam_last_ts: dict[str, float] = {}
        # Oldest input timestamp (perf_counter domain) for E2E latency.
        self._input_oldest: float | None = None
        self._step: int = 0
        self._started: bool = False

    @property
    def step(self) -> int:
        """Step counter; incremented on each ``commit()``."""
        return self._step

    def start(self, ep: int | None = None) -> None:
        """Begin a new iteration. Resets the per-iter record."""
        self._record = {
            "loop_kind": self.loop_kind,
            "t": time.time(),
            "step": self._step,
        }
        if ep is not None:
            self._record["ep"] = ep
        self._iter_start = time.perf_counter()
        self._input_oldest = None
        self._started = True

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time a code block; result stored as ``f'{name}_ms'`` in the record.

        Reentrant within an iteration: nested spans are supported, but each
        ``name`` should be unique within an iteration (the second write
        overwrites the first).
        """
        assert self._started, "LatencyTracer.start() must be called before span()"
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._record[f"{name}_ms"] = (time.perf_counter() - t0) * 1000.0

    def cam_consume(self, cam_key: str, latest_ts: float) -> None:
        """Record camera staleness and period at the moment of consumption.

        Preconditions:
          - ``latest_ts`` is in the same time domain as ``time.perf_counter()``
            (which is what cameras' grab threads use; see ``camera_opencv.py``).
          - Called at the moment the consumer reads the cached frame.
        """
        assert self._started, "LatencyTracer.start() must be called before cam_consume()"
        now = time.perf_counter()
        stale_ms = (now - latest_ts) * 1000.0
        self._record[f"cam_{cam_key}_stale_ms"] = stale_ms
        prev_ts = self._cam_last_ts.get(cam_key)
        if prev_ts is not None:
            self._record[f"cam_{cam_key}_period_ms"] = (latest_ts - prev_ts) * 1000.0
        self._cam_last_ts[cam_key] = latest_ts
        if self._input_oldest is None or latest_ts < self._input_oldest:
            self._input_oldest = latest_ts

    def mark_input(self, t: float | None = None) -> None:
        """Mark a non-camera input timestamp (e.g. just-read motor sample) for E2E.

        When ``t`` is None, uses ``time.perf_counter()`` (i.e., "right now").
        """
        assert self._started, "LatencyTracer.start() must be called before mark_input()"
        ts = time.perf_counter() if t is None else t
        if self._input_oldest is None or ts < self._input_oldest:
            self._input_oldest = ts

    def set_field(self, key: str, value: Any) -> None:
        """Add a non-timing field to the record (e.g. dataset write counts)."""
        assert self._started, "LatencyTracer.start() must be called before set_field()"
        self._record[key] = value

    def commit(self) -> dict[str, Any]:
        """Finalize the record. Computes loop_dt, e2e, residual, overrun.

        Postconditions:
          - Returns the finalized record dict.
          - When an aggregator was provided, also appends to its deque.
          - The tracer is reset; the next iteration must call ``start()``.
        """
        assert self._started, "LatencyTracer.commit() requires a prior start()"
        end = time.perf_counter()
        loop_dt_ms = (end - self._iter_start) * 1000.0
        self._record["loop_dt_ms"] = loop_dt_ms

        if self._input_oldest is not None:
            e2e_ms = (end - self._input_oldest) * 1000.0
            self._record["e2e_obs_to_action_ms"] = e2e_ms
            self._record["residual_ms"] = self._compute_residual(e2e_ms)

        if self._target_period_ms is not None:
            self._record["overrun"] = loop_dt_ms > self._target_period_ms

        if self.aggregator is not None:
            self.aggregator.ingest(self._record)

        record = self._record
        self._step += 1
        self._started = False
        return record

    def _compute_residual(self, e2e_ms: float) -> float:
        """e2e − (max stale + post-consume stages).

        Camera staleness already absorbs motor-read time in our typical loop
        (motors are read before cameras), so motor_read is not in the sum.
        See ``latency_monitoring.md`` for the rationale.
        """
        stale_keys = [k for k in self._record if k.startswith("cam_") and k.endswith("_stale_ms")]
        max_stale = max((self._record[k] for k in stale_keys), default=0.0)
        post_consume = sum(self._record.get(k, 0.0) for k in _POST_CONSUME_KEYS)
        return e2e_ms - max_stale - post_consume
