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

"""Bounded cyclic buffer of per-iteration latency records.

Backs every live query (current percentiles, time-series, histograms,
overrun ratio) from a single ``collections.deque(maxlen=N)`` of records.
See ``src/lerobot/gui/docs/latency_monitoring.md`` for the full contract.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

DEFAULT_MAXLEN = 4096


class LatencyAggregator:
    """In-memory store for live latency monitoring.

    Preconditions:
      - Records are dicts with ``"t"`` (wall-clock seconds) as a top-level key.
      - Latency fields end in ``_ms`` and are floats.
      - Percentile/series queries assume ``ingest()`` is called from one thread
        (the loop calling ``LatencyTracer.commit()``); read methods are safe to
        call from a different thread but may see a stale snapshot of the deque.

    Postconditions:
      - Memory is bounded by ``maxlen`` regardless of run length.
      - Queries on an empty aggregator return ``float("nan")`` or empty
        collections; they never raise.
    """

    def __init__(self, maxlen: int = DEFAULT_MAXLEN):
        assert maxlen > 0, f"maxlen must be positive, got {maxlen}"
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self.dropped_records: int = 0

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def __len__(self) -> int:
        return len(self._records)

    def ingest(self, record: dict[str, Any]) -> None:
        """Append a record. Oldest record is evicted when ``maxlen`` is reached."""
        assert isinstance(record, dict), f"record must be dict, got {type(record).__name__}"
        self._records.append(record)

    def clear(self) -> None:
        self._records.clear()
        self.dropped_records = 0

    def stage_keys(self) -> list[str]:
        """All ``*_ms`` keys seen across the buffer, sorted."""
        keys: set[str] = set()
        for r in self._records:
            keys.update(k for k in r if k.endswith("_ms"))
        return sorted(keys)

    def values(self, key: str) -> list[float]:
        """Raw values for one stage across the current buffer."""
        return [r[key] for r in self._records if key in r and r[key] is not None]

    def percentile(self, key: str, p: float | list[float]) -> float | list[float]:
        """p in [0, 100]. NaN when the key is absent."""
        vals = self.values(key)
        if not vals:
            return float("nan") if not isinstance(p, list) else [float("nan")] * len(p)
        result = np.percentile(np.asarray(vals, dtype=np.float64), p)
        if isinstance(p, list):
            return [float(x) for x in result]
        return float(result)

    def overrun_ratio(self) -> float:
        """Fraction of records flagged as overrun. 0.0 when empty."""
        flags = [bool(r.get("overrun")) for r in self._records if "overrun" in r]
        if not flags:
            return 0.0
        return sum(flags) / len(flags)

    def time_series(self, key: str, last_n_seconds: float | None = None) -> list[tuple[float, float]]:
        """Return ``[(t, value), ...]`` for ``key``, optionally trimmed to recent."""
        if last_n_seconds is None:
            return [(r["t"], r[key]) for r in self._records if key in r and r[key] is not None]
        if not self._records:
            return []
        latest_t = self._records[-1]["t"]
        cutoff = latest_t - last_n_seconds
        return [
            (r["t"], r[key]) for r in self._records if key in r and r[key] is not None and r["t"] >= cutoff
        ]

    def snapshot(
        self,
        percentiles: tuple[float, ...] = (50, 95, 99),
        series_window_s: float | None = None,
        series_max_points: int = 60,
    ) -> dict[str, Any]:
        """Compact dict for stderr summary / GUI overlay polling.

        Args:
          percentiles: which percentiles to compute per stage.
          series_window_s: when set, attach a downsampled time-series per stage
            (suitable for sparklines), trimmed to the last ``series_window_s``
            seconds and capped at ``series_max_points`` points.

        Postconditions:
          - Always contains ``n_records``, ``dropped_records``, ``overrun_ratio``,
            ``stages``.
          - ``stages`` maps each ``*_ms`` key to a dict of
            ``{f"p{int(p)}": ms, ..., "count": n}``.
          - When ``series_window_s`` is set, also contains ``series`` —
            ``stage_key -> [[t, value], ...]`` (lists, not tuples, so the dict
            is JSON-serializable).
        """
        snap: dict[str, Any] = {
            "n_records": len(self._records),
            "dropped_records": self.dropped_records,
            "overrun_ratio": self.overrun_ratio(),
            "stages": {},
        }
        keys = self.stage_keys()
        for key in keys:
            vals = self.values(key)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            stage_dict: dict[str, float | int] = {
                f"p{int(p)}": float(np.percentile(arr, p)) for p in percentiles
            }
            stage_dict["count"] = len(vals)
            snap["stages"][key] = stage_dict

        if series_window_s is not None:
            assert series_max_points > 0
            snap["series"] = {}
            for key in keys:
                ts = self.time_series(key, last_n_seconds=series_window_s)
                if not ts:
                    continue
                stride = max(1, len(ts) // series_max_points)
                downsampled = ts[::stride][-series_max_points:]
                snap["series"][key] = [[t, v] for t, v in downsampled]

        return snap

    def representative_iterations(self) -> dict[str, dict[str, Any]]:
        """Return a small set of "interesting" iterations for Gantt rendering.

        Picks one record near each of {p50, p95, p99, max} of ``loop_dt_ms``,
        plus the most recent record. The frontend lets the operator switch
        between them so they can compare a typical iteration against a
        worst-case one.

        Returns a dict ``{label: record_subset}`` where each subset contains
        only the timeline-relevant keys (``spans``, ``cam_events``,
        ``loop_dt_ms``, ``step``, ``t``). Returns an empty dict if the
        aggregator has no records.
        """
        if not self._records:
            return {}

        # Index records by loop_dt_ms for percentile selection.
        with_loop = [(r.get("loop_dt_ms", 0.0), r) for r in self._records if "loop_dt_ms" in r]
        if not with_loop:
            return {}
        with_loop.sort(key=lambda x: x[0])
        loops = [x[0] for x in with_loop]

        def pick_for(p: float) -> dict[str, Any]:
            target = float(np.percentile(np.asarray(loops), p))
            # Find the record whose loop_dt is closest to the target percentile.
            best_idx = min(range(len(with_loop)), key=lambda i: abs(with_loop[i][0] - target))
            return _gantt_record_subset(with_loop[best_idx][1])

        result = {
            "median": pick_for(50),
            "p95": pick_for(95),
            "p99": pick_for(99),
            "max": _gantt_record_subset(with_loop[-1][1]),
            "latest": _gantt_record_subset(self._records[-1]),
        }
        return result


def _gantt_record_subset(record: dict[str, Any]) -> dict[str, Any]:
    """Pull only the keys needed to render one iteration's Gantt timeline."""
    return {
        "step": record.get("step"),
        "t": record.get("t"),
        "loop_dt_ms": record.get("loop_dt_ms"),
        "spans": record.get("spans", {}),
        "cam_events": record.get("cam_events", {}),
    }
