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

"""Per-loop latency monitoring lifecycle.

``LatencySession`` bundles the ``LatencyTracer`` / ``LatencyAggregator`` /
``LatencySnapshotWriter`` trio plus the 1-Hz stderr summary into one
object whose API is shaped around the loop's iteration boundary. Replaces
the ~25 lines of per-loop boilerplate that teleop / record / record-with-
policy / eval / inference would otherwise each have to repeat.

Two key properties:

1. **Disabled-by-default is free**. ``LatencySession.disabled()`` returns
   a session object whose methods are all no-ops. Loop code stays
   identical whether monitoring is enabled or not — no ``if tracer is
   None`` branches at the call site.

2. **Iteration is a context manager**. ``with session.iteration():`` does
   start, then runs the body, then commits + (throttled) snapshot +
   (throttled) stderr summary on exit. If the body raises, the iteration
   is still committed and the next one starts cleanly.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from lerobot.utils.latency.aggregator import LatencyAggregator
from lerobot.utils.latency.snapshot import LatencySnapshotWriter
from lerobot.utils.latency.tracer import LatencyTracer

logger = logging.getLogger(__name__)


class LatencySession:
    """Owns one loop's tracer + aggregator + snapshot writer + summary loop.

    Preconditions:
      - ``iteration()`` is called exactly once per loop turn.
      - All ``span()`` / ``cam_consume()`` / ``add_span()`` calls occur
        inside an ``iteration()`` block.

    Postconditions:
      - On ``iteration()`` exit, the tracer commits, the snapshot writer
        (if configured) publishes when its interval has elapsed, and the
        stderr summary fires when its interval has elapsed.
      - When ``enabled=False`` (use ``disabled()``), every method is a
        cheap no-op; loop call sites need no conditional logic.
    """

    def __init__(
        self,
        aggregator: LatencyAggregator,
        writer: LatencySnapshotWriter | None,
        loop_kind: str,
        target_fps: float | None,
        summary_interval_s: float = 1.0,
    ):
        self.aggregator = aggregator
        self.writer = writer
        self.loop_kind = loop_kind
        self.target_fps = target_fps
        self._tracer = LatencyTracer(aggregator, loop_kind=loop_kind, target_fps=target_fps)
        self._summary_interval_s = summary_interval_s
        self._last_summary_at: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        enabled: bool,
        loop_kind: str,
        target_fps: float | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        summary_interval_s: float = 1.0,
    ) -> LatencySession:
        """Build either a real session or a no-op (when ``enabled=False``).

        ``output_dir`` is the directory for ``latency_snapshot.json``.
        Pass ``None`` to skip the snapshot writer entirely (still
        keeps the in-memory aggregator and stderr summary).
        """
        if not enabled:
            return _DisabledSession()
        agg = LatencyAggregator()
        writer = (
            LatencySnapshotWriter(output_dir, loop_kind=loop_kind, target_fps=target_fps)
            if output_dir is not None
            else None
        )
        return cls(
            aggregator=agg,
            writer=writer,
            loop_kind=loop_kind,
            target_fps=target_fps,
            summary_interval_s=summary_interval_s,
        )

    @classmethod
    def disabled(cls) -> LatencySession:
        """Return a no-op session for callers that want unconditional plumbing."""
        return _DisabledSession()

    # ------------------------------------------------------------------
    # Per-iteration API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return True

    @property
    def tracer(self) -> LatencyTracer | None:
        """Direct tracer access for callers needing ``add_span`` outside ``span()``."""
        return self._tracer

    @contextmanager
    def iteration(self, ep: int | None = None) -> Iterator[LatencySession]:
        """Context manager around one loop iteration.

        On entry: ``tracer.start(ep=ep)``.
        On exit: ``tracer.commit()`` + (throttled) snapshot publish +
        (throttled) stderr summary. Runs even if the body raises, so the
        next iteration starts from a clean state.

        Use this when the loop body has a clean structure that fits in
        a single ``with`` block. For loops with early ``break`` /
        ``continue`` paths that need to skip the commit, use the explicit
        ``start_iter()`` / ``end_iter()`` pair instead.
        """
        self.start_iter(ep=ep)
        try:
            yield self
        finally:
            self.end_iter()

    def start_iter(self, ep: int | None = None) -> None:
        """Start a new iteration explicitly. Pair with ``end_iter()``.

        Used by loops that can't cleanly wrap their body in
        ``with iteration():`` (e.g. record_loop has many ``continue``
        branches that should skip the commit). ``continue``-skipped
        iterations are simply discarded — the next ``start_iter()``
        resets the tracer's per-iteration state.
        """
        self._tracer.start(ep=ep)

    def end_iter(self) -> None:
        """Finalize the current iteration: commit + (throttled) publish/log."""
        self._tracer.commit()
        self._after_commit()

    def span(self, name: str):
        """Context manager around a timed region. Forwards to the tracer."""
        return self._tracer.span(name)

    def add_span(self, name: str, start_perf: float, end_perf: float | None = None) -> None:
        """Record a span from already-captured perf_counter values."""
        self._tracer.add_span(name, start_perf, end_perf)

    def cam_consume(self, cam_key: str, latest_ts: float) -> None:
        """Record a single camera frame consumption event."""
        self._tracer.cam_consume(cam_key, latest_ts)

    def cam_consume_all(self, cameras: dict[str, Any] | None) -> None:
        """Convenience: read ``latest_timestamp`` from each camera and record it.

        Skips cameras that don't expose ``latest_timestamp`` (e.g. mock
        cameras in tests, or cameras that haven't grabbed any frame yet).
        Safe to call with ``None`` for robots that have no cameras.
        """
        if not cameras:
            return
        for cam_key, cam in cameras.items():
            ts = getattr(cam, "latest_timestamp", None)
            if ts is not None:
                self._tracer.cam_consume(cam_key, ts)

    def set_field(self, key: str, value: Any) -> None:
        """Add a non-timing field to the current iteration's record."""
        self._tracer.set_field(key, value)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _after_commit(self) -> None:
        """Snapshot publish + stderr summary, both throttled by interval."""
        if self.writer is not None:
            self.writer.maybe_write(self.aggregator)
        now = time.time()
        if now - self._last_summary_at >= self._summary_interval_s:
            self._last_summary_at = now
            snap = self.aggregator.snapshot(percentiles=(50, 95))
            if snap["n_records"] > 0:
                # Lazy import to avoid pulling format_latency_summary
                # back into the latency package's public surface twice.
                from lerobot.utils.latency import format_latency_summary

                logger.info("[latency:%s] %s", self.loop_kind, format_latency_summary(snap))


class _DisabledSession(LatencySession):
    """No-op session for ``enabled=False`` paths.

    Every method either returns a no-op context manager or quietly
    discards its arguments. Call sites can use the same code regardless
    of whether monitoring is on; the only cost when disabled is one
    null context-manager enter+exit per iteration (~50 ns).
    """

    def __init__(self) -> None:
        # Skip the parent's __init__ — we don't allocate aggregator etc.
        self.aggregator = None  # type: ignore[assignment]
        self.writer = None
        self.loop_kind = ""
        self.target_fps = None

    @property
    def enabled(self) -> bool:
        return False

    @property
    def tracer(self) -> LatencyTracer | None:
        return None

    @contextmanager
    def iteration(self, ep: int | None = None) -> Iterator[LatencySession]:
        yield self

    def start_iter(self, ep: int | None = None) -> None:
        pass

    def end_iter(self) -> None:
        pass

    def span(self, name: str):
        return nullcontext()

    def add_span(self, name: str, start_perf: float, end_perf: float | None = None) -> None:
        pass

    def cam_consume(self, cam_key: str, latest_ts: float) -> None:
        pass

    def cam_consume_all(self, cameras: dict[str, Any] | None) -> None:
        pass

    def set_field(self, key: str, value: Any) -> None:
        pass

    def _after_commit(self) -> None:
        pass
