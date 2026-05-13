#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SO-107 leader with a background bus-read thread.

Drop-in replacement for :class:`SO107Leader`. Connect, calibrate,
intervention, feedback — all inherited unchanged. The only behavioral
difference is that ``get_action()`` no longer reads the bus; a
background thread (started in ``connect()``) reads at
``config.read_rate_hz`` and caches the latest pose. ``get_action()``
returns the cache.

Why this exists: the predictive follower's controller wants distinct
intent samples at its control rate (200 Hz) to estimate leader
velocity without the stair-stepping bias inherent in a 30 Hz publish
cadence. With this leader bound via ``robot.attach_teleop()``, the
controller's tick polls the cache directly and the LSQ-bias problem
goes away.

The cache is also useful for non-predictive consumers: the 30 Hz
record loop's ``get_action()`` becomes a cheap atomic read instead of
a serial round-trip (~3-5 ms on Feetech STS3215 × 7 motors), shaving
that time off the ``process_action`` span.
"""

from __future__ import annotations

import logging
import threading
import time

from lerobot.utils.robot_utils import precise_sleep

from ..so_leader.so_leader import SO107Leader
from .config_so107_leader_highrate import SO107LeaderHighRateConfig

logger = logging.getLogger(__name__)


class SO107LeaderHighRate(SO107Leader):
    """SO-107 leader that polls the leader bus in a background thread."""

    config_class = SO107LeaderHighRateConfig
    name = "so107_leader_highrate"

    def __init__(self, config: SO107LeaderHighRateConfig):
        super().__init__(config)
        # Latest cached pose (dict[str, float] or None before first read).
        # CPython reassignment is atomic at the reference level; we also
        # take ``_cache_lock`` for any read that's followed by use (the
        # consumer takes a snapshot under the lock to avoid the ref
        # being swapped mid-iteration, which is harmless in practice
        # but cheaper to just guard).
        self._cache_lock = threading.Lock()
        self._cached_pose: dict[str, float] | None = None
        self._read_stop = threading.Event()
        self._read_thread: threading.Thread | None = None

    def connect(self, calibrate: bool = True) -> None:
        # Parent does bus.connect() + calibration + configure() + gripper
        # bounce + (optionally) keyboard listener. We just need to start
        # the background read thread after the bus is open.
        super().connect(calibrate)
        self._start_read_thread()
        logger.info(
            "%s background read thread started at %.0f Hz",
            self,
            self.config.read_rate_hz,
        )

    def _start_read_thread(self) -> None:
        if self._read_thread is not None and self._read_thread.is_alive():
            return  # idempotent
        self._read_stop.clear()
        self._read_thread = threading.Thread(
            target=self._read_loop,
            name=f"{self}_leader_read",
            daemon=True,
        )
        self._read_thread.start()

    def _read_loop(self) -> None:
        period = 1.0 / self.config.read_rate_hz
        next_tick = time.perf_counter()
        while not self._read_stop.is_set():
            try:
                pose = self.bus.sync_read("Present_Position")
                pose = {f"{motor}.pos": float(val) for motor, val in pose.items()}
                with self._cache_lock:
                    self._cached_pose = pose
            except Exception:
                # A transient bus blip shouldn't kill the thread —
                # the cache stays at the last good value, the next
                # iteration tries again. Log so we hear about it.
                logger.exception("%s leader read failed", self)
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                precise_sleep(sleep_for)
            else:
                # Fell behind (bus retry, sleep jitter) — reset target
                # so we don't spin trying to catch up over many ticks.
                next_tick = time.perf_counter()

    def get_action(self) -> dict[str, float]:
        """Return the most recent cached leader pose.

        Lock-free in the steady state (the cache reference is swapped
        atomically by the read thread). The first call before the
        cache is warm — i.e. before the read thread has produced its
        first sample — falls through to the parent's blocking
        sync_read so callers don't see a None / empty dict.
        """
        with self._cache_lock:
            pose = self._cached_pose
        if pose is not None:
            # Return a copy so the caller can mutate without poisoning
            # the cache for the next consumer.
            return dict(pose)
        # Cache not warmed yet (very first tick). Block once on a
        # direct read so the caller has something to work with.
        return super().get_action()

    def disconnect(self) -> None:
        # Stop the read thread BEFORE the bus closes — otherwise the
        # thread's next sync_read would race with bus.disconnect() and
        # raise.
        self._read_stop.set()
        if self._read_thread is not None and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)
            if self._read_thread.is_alive():
                logger.warning("%s leader read thread did not stop within 2s", self)
        self._read_thread = None
        # Drop the cached pose so a subsequent reconnect can't return a
        # stale pose from the previous session before the new read thread
        # has produced its first sample. With the cache cleared,
        # get_action() falls through to the parent's blocking sync_read
        # until the new thread warms up — preserving the "no stale data"
        # invariant across reconnects.
        with self._cache_lock:
            self._cached_pose = None
        super().disconnect()
