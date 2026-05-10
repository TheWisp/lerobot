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

"""Inline 1 Hz JSON snapshot writer for the GUI to read cross-process.

Mirrors the RLT pattern (``policies/hvla/rlt/metrics.py`` →
``outputs/rlt_online/metrics.json``): the loop process writes a small
``latency_snapshot.json`` to a known location at ~1 Hz; the GUI's
``/api/latency-metrics`` endpoint reads it and serves to the frontend.

Single-threaded by design — the write happens on the loop's own thread,
gated by an elapsed-time check so it costs nothing on most iterations.
At 1 Hz with ~5 KB JSON, amortized cost is well under 1 µs/iter.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from lerobot.utils.latency.aggregator import LatencyAggregator

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 1.0
DEFAULT_SERIES_WINDOW_S = 30.0
DEFAULT_SERIES_MAX_POINTS = 60


class LatencySnapshotWriter:
    """Writes a JSON snapshot of the aggregator to disk on a fixed interval.

    Preconditions:
      - ``maybe_write()`` is called from the loop thread, e.g. after
        ``LatencyTracer.commit()``.
      - The output directory is writable; created on first write.

    Postconditions:
      - Writes to ``<output_dir>/latency_snapshot.json`` atomically (write to
        a temp file, then ``os.replace``) so the GUI never reads a partial
        file. Stale data on read is fine — the GUI polls again.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        interval_s: float = DEFAULT_INTERVAL_S,
        series_window_s: float = DEFAULT_SERIES_WINDOW_S,
        series_max_points: int = DEFAULT_SERIES_MAX_POINTS,
        loop_kind: str = "teleop",
        target_fps: float | None = None,
        process: str | None = None,
        track: str | None = None,
    ):
        assert interval_s >= 0, f"interval_s must be non-negative, got {interval_s}"
        self._dir = Path(output_dir)
        self._path = self._dir / "latency_snapshot.json"
        self._tmp_path = self._dir / "latency_snapshot.json.tmp"
        self._interval_s = interval_s
        self._series_window_s = series_window_s
        self._series_max_points = series_max_points
        self._loop_kind = loop_kind
        # ``process`` / ``track`` identify a snapshot in a multi-thread
        # process (e.g. HVLA writes one snapshot per thread: track=main vs
        # track=inference, both with process=hvla). Single-loop callers
        # can omit them — they default to ``loop_kind`` so existing
        # consumers see the same identifier in both fields.
        self._process = process if process is not None else loop_kind
        self._track = track if track is not None else loop_kind
        # The loop's nominal iteration budget in ms (1000/fps). Published in
        # every snapshot so the GUI can scale color thresholds to the actual
        # FPS instead of hardcoding for 60 Hz.
        self._target_period_ms: float | None = (1000.0 / float(target_fps)) if target_fps else None
        self._last_write_at: float = 0.0
        self._dir_ready: bool = False

    @property
    def path(self) -> Path:
        return self._path

    def maybe_write(self, aggregator: LatencyAggregator, now: float | None = None) -> bool:
        """Write a snapshot if ``interval_s`` has elapsed since the last write.

        Returns True when a write happened, False when skipped. Errors are
        logged at WARNING and swallowed — never crashes the loop over a
        snapshot failure.
        """
        t = time.time() if now is None else now
        if (t - self._last_write_at) < self._interval_s:
            return False
        self._last_write_at = t
        try:
            self._write(aggregator, t)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("latency snapshot write failed: %s", e)
            return False

    def _write(self, aggregator: LatencyAggregator, t: float) -> None:
        snap = aggregator.snapshot(
            percentiles=(50, 95, 99),
            series_window_s=self._series_window_s,
            series_max_points=self._series_max_points,
        )
        snap["t"] = t
        snap["loop_kind"] = self._loop_kind
        snap["process"] = self._process
        snap["track"] = self._track
        if self._target_period_ms is not None:
            snap["target_period_ms"] = self._target_period_ms
        # Two flavors of timeline data, both cheap to produce:
        #
        # - representative_iterations(): real captured iterations (median /
        #   p95 / p99 / max / latest sample). Per-stage values inside each
        #   reflect THAT specific iteration. Useful for outlier debugging
        #   because correlations are real.
        #
        # - aggregate_iteration(p): synthetic timeline whose per-stage bars
        #   each show the p-th percentile of that stage *independently*.
        #   Stable across snapshots; answers "what does a typical iteration
        #   look like?" without picking one specific record.
        snap["iterations"] = aggregator.representative_iterations()
        snap["iterations"]["aggregate_median"] = aggregator.aggregate_iteration(50)
        snap["iterations"]["aggregate_p95"] = aggregator.aggregate_iteration(95)

        if not self._dir_ready:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True

        # Atomic publish: write tmp file, then rename.
        with open(self._tmp_path, "w", encoding="utf-8") as f:
            json.dump(snap, f, separators=(",", ":"))
        os.replace(self._tmp_path, self._path)
