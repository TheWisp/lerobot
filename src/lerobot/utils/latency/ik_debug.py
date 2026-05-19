"""Verbose per-tick log of Cartesian-IK pipeline internals.

Complements the existing :class:`MotionLogger` (which records final intent
+ state per tick). This one captures the intermediate-stage signals that
let you tell *why* an IK output looked unstable: was the EE target
reasonable, did bounds clipping fire, did pink converge, etc.

Designed as a module-level singleton so the individual ProcessorSteps
in the Cartesian IK chain (EEReferenceAndDelta, EEBoundsAndSafety,
PinkInverseKinematicsEEToJoints) can record without explicit plumbing.
The run script (lerobot-teleoperate / -replay / -record) is responsible
for installing a recorder when --debug_teleop_log=true is set, calling
``tick_done()`` at the end of each loop iteration, and ``close()`` on
exit.

Output: ``<output_dir>/ik_debug_<YYYYMMDD_HHMMSS>.npz`` with one array
per (prefix, signal) key. Missing values (e.g., bounds clipping that
didn't fire on a given tick) are recorded as NaN, so the shape is
``(n_ticks, ...)`` per array — easy to load with numpy and analyze.

The recording is best-effort: any exception inside a step's record call
is swallowed (the loop never wedges because the debug log misbehaved).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_recorder: IKDebugRecorder | None = None


def get_recorder() -> IKDebugRecorder | None:
    """Return the active recorder, or None if disabled."""
    return _recorder


def set_recorder(recorder: IKDebugRecorder | None) -> None:
    """Install or clear the singleton."""
    global _recorder
    _recorder = recorder


class IKDebugRecorder:
    """Per-tick collector for Cartesian IK pipeline diagnostics.

    Preconditions:
        * ``output_dir`` is writable.
        * ``tick_done()`` is called from the loop thread once per tick.

    Postconditions:
        * On ``close()``, a file ``<output_dir>/ik_debug_<ts>.npz`` exists
          with one array per recorded key; values present in only some
          ticks become NaN on the others.
        * If ``close()`` is never called (crash), no file is written —
          we don't flush a half-finished log.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"ik_debug_{ts}.npz"
        self._records: list[dict[str, Any]] = []
        self._current: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()

    def record(self, prefix: str, key: str, value: Any) -> None:
        """Stash one (prefixed) signal value for the current tick.

        Safe to call from multiple ProcessorSteps within a single tick
        AND across threads (the Cartesian IK adapter runs steps on its
        own thread; the main loop calls ``tick_done`` on its thread).
        The lock guards against a tick boundary cutting through an
        in-progress write and splitting a step's signals across two
        records.
        """
        full_key = f"{prefix}{key}" if prefix else key
        with self._lock:
            self._current[full_key] = value

    def tick_done(self) -> None:
        """Finalize the current tick's record and start a new one."""
        with self._lock:
            if self._current:
                self._current["__t"] = time.perf_counter() - self._t0
                self._records.append(self._current)
            self._current = {}

    def close(self) -> None:
        if not self._records:
            logger.info("IKDebugRecorder.close: no records, not writing %s", self.path)
            return
        # Union of keys across all records; missing values become NaN.
        # For per-key array-shaped values, we pre-scan to find the shape so
        # we can pad missing ticks with NaN-filled arrays of the same shape.
        all_keys = sorted({k for r in self._records for k in r})
        arrays: dict[str, np.ndarray] = {}
        for k in all_keys:
            # Determine value shape from the first non-missing entry.
            sample = next((r[k] for r in self._records if k in r), None)
            if sample is None:
                continue
            if isinstance(sample, (list, tuple, np.ndarray)):
                arr_shape = np.asarray(sample, dtype=float).shape
                pad = np.full(arr_shape, np.nan, dtype=float)
                col = [np.asarray(r.get(k, pad), dtype=float) if k in r else pad for r in self._records]
            else:
                col = []
                for r in self._records:
                    v = r.get(k, np.nan)
                    try:
                        col.append(float(v))
                    except (TypeError, ValueError):
                        col.append(np.nan)
            arrays[k] = np.array(col)
        try:
            np.savez(self.path, **arrays)
            logger.info(
                "IKDebugRecorder wrote %s (%d ticks, %d signals)", self.path, len(self._records), len(arrays)
            )
        except OSError:
            logger.exception("IKDebugRecorder: failed to write %s", self.path)
