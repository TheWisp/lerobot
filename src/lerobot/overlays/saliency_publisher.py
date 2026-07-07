"""Publish a policy's per-camera attention-map grids to the overlay aux buffer.

The GENERIC half of the policy-internal overlay contract (``gui/docs/policy_saliency.md`` §3):
an inference loop owns one ``SaliencyPublisher`` and calls ``publish(batch)`` once per
inference; the policy only needs a ``compute_input_saliency()`` method (and optionally
``compute_attention_rollout()``). Everything else — the demand gate, the cadence, the method
dispatch from the GUI control block, the key mapping, the pass timing the badge shows, the
optional on-disk dump — lives here, so adopting the overlay adds ~3 lines to a policy's loop.

DEMAND-GATED: nothing is computed unless an overlay worker is up (the off path is a single
shm-existence check, ~free). Latency when on: one gradient pass every ``every`` inferences
(~30 ms, cheaper than one inference — see the doc's measured numbers).
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


class SaliencyPublisher:
    """Owns the aux-buffer writer for one policy process.

    Precondition: ``image_keys`` are the policy's image-feature keys, in the same order as the
    obs-stream cameras the overlay worker reads (grid i maps to camera i after prefix-stripping).
    Postcondition of ``publish``: at most one compute per ``every`` inferences, and only while a
    worker is up; the policy's inference path is never touched. Call ``cleanup()`` on stop.
    """

    def __init__(
        self, policy, image_keys: list[str], mode: str | None = "saliency", every: int = 3, dump: bool = False
    ):
        self._policy = policy
        self._image_keys = list(image_keys)
        self._mode = mode  # "saliency" | None (off)
        self._every = max(1, every)
        self._dump = dump  # persist each grid set to outputs/saliency_dumps/ (offline analysis only)
        self._step = -1
        self._aux = None  # lazy SharedAuxBuffer — created on the first successful compute
        self._warned = False  # rate-limit the publish-failure warning to once per process
        self._ctrl = None  # lazy OverlayControlReader — the demand signal + the selected method
        self._dump_dir = None

    def publish(self, batch: dict) -> None:
        """The per-inference hook — DEMAND-GATED: skip entirely unless an overlay worker is up,
        then by cadence + mode. The off path is a single shm-existence check (~the pre-overlay
        baseline). Precondition: ``batch`` is the same raw batch the policy's action path consumes
        (what ``compute_input_saliency`` expects); the inference path is never touched."""
        if self._mode is None:
            return
        self._step += 1
        if self._step % self._every != 0:
            return
        cfg = self._config()  # None => no overlay worker up => no demand
        if cfg is None:
            return
        if self._mode == "saliency":
            method = cfg.get("method")
            self._publish_saliency(batch, method if method in ("gradient", "rollout") else "gradient")

    def cleanup(self) -> None:
        """Release the aux shm (idempotent; safe to call without a prior publish). Call on stop."""
        if self._aux is not None:
            with contextlib.suppress(Exception):
                self._aux.cleanup()
            self._aux = None

    # ---- internals ----

    def _config(self) -> dict | None:
        """The overlay worker's control config, or ``None`` when no worker is up — the DEMAND signal."""
        if self._ctrl is None:
            from lerobot.overlays.overlay_ipc import OverlayControlReader

            self._ctrl = OverlayControlReader()
        return self._ctrl.config()

    def _obs_keys(self) -> list[str]:
        # The policy keys by image feature ("observation.images.<cam>"); the obs-stream (and the
        # overlay worker) use the bare "<cam>" robot key. Map so the worker can match the grids.
        return [k.removeprefix("observation.images.") for k in self._image_keys]

    def _publish_saliency(self, batch: dict, method: str = "gradient") -> None:
        """Publish per-camera policy saliency. ``method``: "gradient" -> input-gradient
        (``compute_input_saliency``, a backward pass), "rollout" -> attention rollout
        (``compute_attention_rollout``, forward-only). No-op for a policy lacking the method."""
        attr = "compute_attention_rollout" if method == "rollout" else "compute_input_saliency"
        fn = getattr(self._policy, attr, None)
        if fn is None:
            return
        try:
            t0 = time.perf_counter()
            sal = fn(batch)  # {image_feature_key: HxW float grid}
            obs_keys = self._obs_keys()
            key_map = {k: obs_keys[i] for i, k in enumerate(self._image_keys) if i < len(obs_keys)}
            if not sal:
                logger.info("[saliency] %s returned {} this pass — nothing to publish (overlay blank)", attr)
                return
            written = {
                key_map[k]: f"{float(np.asarray(g).max()):.2e}" for k, g in sal.items() if k in key_map
            }
            if not written:
                logger.warning(
                    "[saliency] grids %s map to NO obs key (key_map=%s) — overlay blank", list(sal), key_map
                )
                return
            if self._aux is None:
                # Advertise ALL expected cameras, not just this pass's — a camera whose first pass
                # yields no grid (e.g. grad=None under a compiled forward) must still get a block,
                # or it would be dropped for the rest of the run (the camera set is frozen here).
                shape = tuple(next(iter(sal.values())).shape)
                cams = {key_map[k]: shape for k in self._image_keys if k in key_map}
                from lerobot.overlays.aux_ipc import SharedAuxBuffer

                self._aux = SharedAuxBuffer(cameras=cams, model="policy_saliency", create=True)
                logger.info("policy-saliency overlay engaged — per-camera grid map = %s", cams)
            for k, g in sal.items():
                if k in key_map:
                    self._aux.write_saliency(key_map[k], np.asarray(g, dtype=np.float32))
            # The net wall cost this publish added to the inference thread (compute + grid writes) —
            # the GUI badge shows it; the worker's own stats can't see policy-process work.
            self._aux.write_pass_ms((time.perf_counter() - t0) * 1000.0)
            logger.info("[saliency] published method=%s per-cam |grid|max=%s", method, written)
            # Optional ground-truth capture (dump, default off): persist every published grid set,
            # time-indexed, so the run's overlay content can be COMPARED frame-to-frame offline. An
            # unbounded write — only enabled when explicitly doing offline saliency analysis.
            if self._dump:
                if self._dump_dir is None:
                    self._dump_dir = os.path.join("outputs", "saliency_dumps", time.strftime("%Y%m%d_%H%M%S"))
                    os.makedirs(self._dump_dir, exist_ok=True)
                    self._dump_n = 0
                    logger.info("[saliency] recording grids -> %s (time-indexed)", self._dump_dir)
                np.savez(
                    os.path.join(self._dump_dir, f"{self._dump_n:06d}.npz"),
                    t=time.time(),
                    **{key_map[k]: np.asarray(g, dtype=np.float32) for k, g in sal.items() if k in key_map},
                )
                self._dump_n += 1
        except Exception:
            if not self._warned:
                logger.warning(
                    "policy-saliency overlay publish failed; overlay will not update", exc_info=True
                )
                self._warned = True
