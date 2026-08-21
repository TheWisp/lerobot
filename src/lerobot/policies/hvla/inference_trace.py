# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Full-fidelity trace of what S1 planned and what the robot was told to do.

Reconstructing a rollout from summary statistics does not work. Deriving
"which chunk index was executed" from action deltas, or "how much the plan
moved" from a rough-chunk sample, produced four different wrong diagnoses of
one stuck rollout. This records the actual objects instead:

  * per inference — the observation it ran on, the RTC prefix it was given,
    the delay bookkeeping, and the chunk it produced;
  * per control step — which inference and which chunk index the action came
    from, the chunk's own value at that index, and the action finally sent
    (which differs when the ±30° jump clamp fires).

Joining those two on ``infer_id`` reproduces the rollout exactly, with no
modelling of the inference loop.

Design constraints, in priority order — a debugging tool that perturbs or
misreports the system under test is worse than none:

1. **Never changes behaviour.** Records are copies taken after the values are
   final. Nothing here touches RNG, tensors in flight, or control timing.
2. **Never raises into the caller.** Any failure disables the trace and lets
   the rollout continue.
3. **No I/O on the control path.** Records accumulate in memory and are written
   once at shutdown, so there is no writer thread, no partial-flush race, and
   no filesystem latency in the loop. A 60 s run at 15 Hz inference is a few MB.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class InferenceTrace:
    """Accumulates inference and control-step records; writes them at close.

    Pre: ``out_dir`` is writable, or writing is skipped with a logged warning.
    Post: after :meth:`close`, ``out_dir`` holds ``inferences.npz``,
    ``steps.npz`` and ``trace_meta.json``, or nothing if the trace was
    disabled by an earlier failure.
    """

    def __init__(self, out_dir: str | Path, *, max_records: int = 200_000) -> None:
        self.out_dir = Path(out_dir)
        self._max = max_records
        self._lock = threading.Lock()
        self._inferences: list[dict[str, Any]] = []
        self._steps: list[dict[str, Any]] = []
        self._failed = False
        self._dropped = 0

    # ── recording ────────────────────────────────────────────────────────────

    def record_inference(
        self,
        *,
        infer_id: int,
        t_obs: float,
        frame_index: int = -1,
        raw_state: np.ndarray | None,
        normalized_state: np.ndarray | None,
        prefix: np.ndarray | None,
        prefix_pre_inject: np.ndarray | None = None,
        prefix_drift: float = float("nan"),
        prefix_len: int = 0,
        expected_d: int,
        actual_d: int,
        exec_idx: int | None,
        chunk: np.ndarray,
    ) -> None:
        """One row per completed inference. ``chunk`` is copied, not retained."""
        if self._failed:
            return
        try:
            row = {
                "infer_id": int(infer_id),
                "t_obs": float(t_obs),
                "frame_index": int(frame_index),
                "prefix_len": int(prefix_len),
                "expected_d": int(expected_d),
                "actual_d": int(actual_d),
                "exec_idx": -1 if exec_idx is None else int(exec_idx),
                "chunk": np.asarray(chunk, dtype=np.float32).copy(),
                "raw_state": _copy_or_nan(raw_state),
                "normalized_state": _copy_or_nan(normalized_state),
                "prefix": _copy_or_nan(prefix),
                # What the model wanted at the pinned positions before the
                # stomp; chunk[0:D] alone cannot show this.
                "prefix_pre_inject": _copy_or_nan(prefix_pre_inject),
                "prefix_drift": float(prefix_drift),
            }
        except Exception:
            self._disable("could not build an inference record")
            return
        with self._lock:
            if len(self._inferences) >= self._max:
                self._dropped += 1
                return
            self._inferences.append(row)

    def record_step(
        self,
        *,
        step: int,
        episode_index: int,
        frame_index: int,
        chunk_t_obs: float,
        chunk_index: int,
        chunk_action: np.ndarray | None,
        sent_action: np.ndarray,
        jump_clamped: bool,
    ) -> None:
        """One row per control step.

        ``chunk_t_obs`` is the observation timestamp the executing chunk was
        computed from — the same float ``get_chunk()`` returns — so joining it
        against ``t_obs`` in the inference table identifies exactly which plan
        and which index produced this action. No new shared state, and no
        reconstructing the correspondence from timing afterwards.

        ``chunk_action`` is the plan's own value at ``chunk_index``;
        ``sent_action`` is what reached the robot. They differ when the jump
        clamp fires, and conflating them attributes the clamp to the model.
        """
        if self._failed:
            return
        try:
            row = {
                "step": int(step),
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "chunk_t_obs": float(chunk_t_obs),
                "chunk_index": int(chunk_index),
                "jump_clamped": bool(jump_clamped),
                "chunk_action": _copy_or_nan(chunk_action),
                "sent_action": np.asarray(sent_action, dtype=np.float32).copy(),
            }
        except Exception:
            self._disable("could not build a step record")
            return
        with self._lock:
            if len(self._steps) >= self._max:
                self._dropped += 1
                return
            self._steps.append(row)

    # ── output ───────────────────────────────────────────────────────────────

    def close(self, extra_meta: dict[str, Any] | None = None) -> Path | None:
        """Write the trace. Returns the directory, or None if nothing was written."""
        if self._failed:
            return None
        with self._lock:
            infers, steps, dropped = self._inferences, self._steps, self._dropped
            self._inferences, self._steps = [], []
        if not infers and not steps:
            return None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.out_dir / "inferences.npz", **_stack(infers))
            np.savez_compressed(self.out_dir / "steps.npz", **_stack(steps))
            meta = {
                "n_inferences": len(infers),
                "n_steps": len(steps),
                "dropped": dropped,
                "schema": 1,
                **(extra_meta or {}),
            }
            (self.out_dir / "trace_meta.json").write_text(json.dumps(meta, indent=2))
        except Exception:
            logger.exception("inference trace: write failed (rollout unaffected)")
            return None
        if dropped:
            logger.warning("inference trace: dropped %d records at the cap", dropped)
        logger.info("inference trace: %d inferences, %d steps → %s", len(infers), len(steps), self.out_dir)
        return self.out_dir

    # ── internals ────────────────────────────────────────────────────────────

    def _disable(self, why: str) -> None:
        if not self._failed:
            self._failed = True
            logger.exception("inference trace disabled: %s (rollout unaffected)", why)


def _copy_or_nan(a: np.ndarray | None) -> np.ndarray:
    """Copy, or a 0-length float32 array standing for "not available".

    A copy rather than a reference: the caller's arrays are reused and mutated
    in place by the control loop, so retaining them would silently rewrite
    history after the fact.
    """
    if a is None:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(a, dtype=np.float32).copy()


def _stack(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Column-major arrays for np.savez. Ragged fields are padded with NaN so a
    missing prefix stays distinguishable from a zero-length one."""
    if not rows:
        return {}
    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        if isinstance(vals[0], np.ndarray):
            shapes = {v.shape for v in vals}
            if len(shapes) == 1:
                out[key] = np.stack(vals)
            else:
                width = max(int(np.prod(v.shape)) if v.size else 0 for v in vals)
                padded = np.full((len(vals), width), np.nan, dtype=np.float32)
                for i, v in enumerate(vals):
                    flat = v.reshape(-1)
                    padded[i, : flat.size] = flat
                out[key] = padded
                out[f"{key}__shape"] = np.array(
                    [list(v.shape) + [0] * (2 - len(v.shape)) for v in vals], dtype=np.int32
                )
        else:
            out[key] = np.asarray(vals)
    return out
