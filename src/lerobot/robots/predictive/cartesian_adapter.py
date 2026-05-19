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

"""Adapter that presents a Cartesian teleop as a joint-space teleop.

Predictive followers' ``attach_teleop`` pull path expects the teleop to
return joint-space dicts (``{<motor>.pos: float}``) — that's the contract
:class:`SOLeaderHighRate` satisfies. A Cartesian teleop (Quest VR, phone,
...) emits ``{target_x, target_y, ..., gripper_pos}`` instead. Without a
shim, the controller's ``_action_to_array`` would either crash on missing
motor keys or treat ``target_x`` as a motor position and produce garbage.

This adapter wraps a Cartesian teleop, runs the robot's registered IK
pipeline (built by :func:`make_cartesian_ik_pipeline`) in a background
thread, and exposes the joint-space result through the same contract the
controller already understands. The robot owns the embodiment (URDF, motor
mapping, joint maps); the teleop stays embodiment-agnostic.

The IK runs at a fixed rate (default 90 Hz — Quest's native WebXR rate).
Both per-arm controllers (200 Hz each) poll the cached result lock-free;
IK does NOT re-run per poll. New Cartesian frames flowing in faster than
90 Hz are coalesced — only the latest is used.

Lifetime is owned by the wrapping robot's ``attach_teleop`` / detach:
``start()`` on attach, ``stop()`` on detach. The thread is daemonic so a
forgotten stop never blocks process exit.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


class _ObservationSource(Protocol):
    """Anything that can hand the adapter the latest joint observation.

    The robot is the natural owner: the main loop already calls
    ``robot.get_observation()`` at 30 Hz, and the predictive follower
    caches the motor portion of that observation for the adapter to read.
    """

    @property
    def last_observation_for_ik(self) -> dict[str, float] | None: ...


class BimanualCartesianIKAdapter:
    """Bimanual variant: wraps a unified bimanual Cartesian teleop and
    exposes ``.left_arm`` / ``.right_arm`` sub-teleops returning per-arm
    joint dicts (unprefixed, matching what
    :class:`BiSO107FollowerPredictive.attach_teleop` already passes to
    each per-arm controller).

    Threading model:
        * One background thread runs the pipeline at ``rate_hz`` (default
          90 Hz). It reads the wrapped Cartesian teleop and the
          observation source on each iteration, runs the pipeline, and
          atomically swaps ``_cached_action``.
        * Two reader threads (one per arm's predictive controller, each
          at 200 Hz) call ``left_arm.get_action()`` / ``right_arm.get_action()``;
          they take the lock briefly to copy the cached dict.

    Preconditions:
        * ``pipeline`` is the bimanual Cartesian IK chain from
          :func:`make_cartesian_ik_pipeline` (it produces prefixed
          ``left_<motor>.pos`` / ``right_<motor>.pos`` keys).
        * ``observation_source.last_observation_for_ik`` returns a dict
          containing every motor's ``.pos`` (prefixed for bimanual).
          ``None`` before the main loop has produced its first
          observation; in that case the adapter skips the tick (sub-arm
          ``get_action`` returns ``{}``, which the controller treats as
          "no fresh intent, hold last").
    """

    def __init__(
        self,
        teleop: Any,
        pipeline: Any,
        observation_source: _ObservationSource,
        rate_hz: float = 90.0,
    ) -> None:
        # rate_hz default: 90 Hz to match the Quest's native WebXR frame
        # rate. The per-arm controllers poll the cached result at 200 Hz
        # but new IK output only arrives at 90 Hz — matching the input
        # rate avoids running IK when nothing has changed.
        self._teleop = teleop
        self._pipeline = pipeline
        self._observation_source = observation_source
        self._rate_hz = float(rate_hz)

        self._cached_action: dict[str, float] | None = None
        self._cache_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._n_ticks = 0
        self._n_skipped = 0

        # Sub-teleops exposed to each per-arm controller. The controller
        # only calls .get_action(), so we just need that one method.
        self.left_arm = _PerArmFilter(self, "left_")
        self.right_arm = _PerArmFilter(self, "right_")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the IK background thread. Idempotent."""
        if self.is_running:
            return
        self._stop.clear()
        # Reset pipeline state so a previous attach/detach cycle doesn't
        # leak engage-snapshot or seed state into the new session.
        try:
            self._pipeline.reset()
        except Exception:
            logger.debug("BimanualCartesianIKAdapter: pipeline.reset() failed", exc_info=True)
        self._thread = threading.Thread(
            target=self._loop,
            name="bimanual_cart_ik_adapter",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "BimanualCartesianIKAdapter started: wrapping %r at %.0f Hz",
            type(self._teleop).__name__,
            self._rate_hz,
        )

    def stop(self) -> None:
        """Stop the IK thread and clear the cache. Idempotent."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("BimanualCartesianIKAdapter: thread didn't stop within 2s")
        self._thread = None
        with self._cache_lock:
            self._cached_action = None
        logger.info(
            "BimanualCartesianIKAdapter stopped (ran %d IK ticks, skipped %d)",
            self._n_ticks,
            self._n_skipped,
        )

    # ── Background loop ───────────────────────────────────────────────────

    def _loop(self) -> None:
        period = 1.0 / self._rate_hz
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("BimanualCartesianIKAdapter tick failed")
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                precise_sleep(sleep_for)
            else:
                # IK fell behind — reset target so we don't spin to catch up.
                next_tick = time.perf_counter()

    def _tick(self) -> None:
        cartesian = self._teleop.get_action()
        obs = self._observation_source.last_observation_for_ik
        if not cartesian or obs is None:
            # No teleop input yet, or main loop hasn't produced an
            # observation. The controller will see an empty dict from
            # the per-arm filters and hold its last command.
            self._n_skipped += 1
            return

        # Pipeline __call__ contract is the same one the main loop uses:
        # ((action_dict, observation_dict)) -> processed_action_dict.
        joint = self._pipeline((dict(cartesian), obs))

        with self._cache_lock:
            self._cached_action = joint
        self._n_ticks += 1

    # ── Filtered reads (per-arm sub-teleop API) ───────────────────────────

    def _get_filtered(self, prefix: str) -> dict[str, float]:
        """Return one arm's joint dict, unprefixed, from the cached pipeline output.

        Empty dict if the IK hasn't produced a sample yet — the per-arm
        controller will then skip its tick (see PredictiveController._tick).
        """
        with self._cache_lock:
            full = self._cached_action
        if full is None:
            return {}
        n = len(prefix)
        return {k[n:]: float(v) for k, v in full.items() if k.startswith(prefix) and k.endswith(".pos")}

    def get_full_joint_action(self) -> dict[str, float] | None:
        """Return the latest merged prefixed joint dict, or None if not yet warm.

        The main loop reads this to record the SAME joint action that
        actually reaches motors via the adapter→controller pull path,
        instead of running its own redundant IK pipeline with diverging
        state. Returns a copy so the caller can mutate freely.
        """
        with self._cache_lock:
            full = self._cached_action
        return dict(full) if full is not None else None


class _PerArmFilter:
    """Sub-teleop view returning one arm's joint dict from the parent adapter.

    The bimanual predictive follower's ``attach_teleop`` hands this object
    to each per-arm controller. Each per-arm controller's
    ``_action_to_array`` expects unprefixed motor keys — exactly what
    ``_get_filtered`` produces.
    """

    def __init__(self, parent: BimanualCartesianIKAdapter, prefix: str) -> None:
        self._parent = parent
        self._prefix = prefix

    def get_action(self) -> dict[str, float]:
        return self._parent._get_filtered(self._prefix)
