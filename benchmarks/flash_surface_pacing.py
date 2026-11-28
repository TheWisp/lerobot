"""Recorded-time frame release and an independent 30 Hz consumer probe.

The producer takes the newest available RGB-D frame when it finishes its prior
inference. It never fabricates repeat observations. Every skipped source frame
stays an unfilled prediction in the benchmark array.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from benchmarks.flash_surface_benchmark import _quantiles


class PacedConsumer:
    def __init__(self, capture_s: np.ndarray, hz: float = 30.0):
        self.capture_s = np.asarray(capture_s, float)
        if hz <= 0 or not len(self.capture_s) or np.any(np.diff(self.capture_s) < 0):
            raise ValueError("invalid capture times or consumer rate")
        self.hz = hz
        self.start_s = time.perf_counter()
        self._latest = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._scheduled: list[float] = []
        self._actual: list[float] = []
        self._seen: list[int] = []
        self.publication_source: list[int] = []
        self.publication_s: list[float] = []
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_s

    def next_source(self, minimum_index: int) -> int:
        """Block until a source is available, then return the latest frame."""
        if minimum_index >= len(self.capture_s):
            raise IndexError("source stream ended")
        target = self.capture_s[minimum_index]
        until = target - self.elapsed()
        if until > 0:
            time.sleep(until)
        if minimum_index == 0:
            return 0  # Always initialize the tracker on its seeded frame.
        latest = int(np.searchsorted(self.capture_s, self.elapsed(), side="right") - 1)
        return max(minimum_index, min(latest, len(self.capture_s) - 1))

    def publish(self, index: int) -> None:
        stamp = self.elapsed()
        with self._lock:
            if index <= self._latest:
                raise ValueError("source publications must advance")
            self._latest = index
            self.publication_source.append(index)
            self.publication_s.append(stamp)

    def _consume(self) -> None:
        tick = 0
        while not self._stop.is_set():
            scheduled = tick / self.hz
            until = scheduled - self.elapsed()
            if until > 0:
                self._stop.wait(until)
                if self._stop.is_set():
                    break
            actual = self.elapsed()
            with self._lock:
                source = self._latest
            self._scheduled.append(scheduled)
            self._actual.append(actual)
            self._seen.append(source)
            tick += 1

    def finish(self) -> dict:
        # Include the final publication in at least one consumer sample.
        time.sleep(1 / self.hz)
        self._stop.set()
        self._thread.join(timeout=1)
        scheduled = np.asarray(self._scheduled)
        actual = np.asarray(self._actual)
        seen = np.asarray(self._seen, int)
        valid = seen >= 0
        period = 1 / self.hz
        age = actual[valid] - self.capture_s[seen[valid]]
        gaps = np.diff(np.asarray(self.publication_s))
        return {
            "consumer_hz": self.hz,
            "consumer_ticks": len(seen),
            "consumer_ticks_without_result": int((~valid).sum()),
            "consumer_deadline_misses": int(np.count_nonzero(actual - scheduled >= period)),
            "consumer_lateness_ms": _quantiles((actual - scheduled) * 1000),
            "observed_pose_age_ms": _quantiles(age * 1000),
            "publication_gap_ms": _quantiles(gaps * 1000),
            "processed_source_frames": len(self.publication_source),
            "skipped_source_frames": len(self.capture_s) - len(self.publication_source),
            "publication_source_index": self.publication_source,
            "publication_s": self.publication_s,
            "consumer_schedule_s": self._scheduled,
            "consumer_actual_s": self._actual,
            "consumer_source_index": self._seen,
        }
