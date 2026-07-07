# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Aux-GPU slot — the resource layer for heavy auxiliary GPU work.

Two layers:

* **The slot (this module)** is the *resource*: one exclusive aux-GPU slot per GPU
  (a single slot today). It does NOT gate a robot's own GPU work — policy inference
  during a run, or local training — only the resource-expensive *auxiliary* jobs
  the GUI spins up on demand.
* **An activity** is the *occupant*: exactly one at a time holds the slot. Today's
  activities are the SAM3 overlay (data tab or run tab) and a batch augmentation
  job; a future DepthAnything overlay / depth-export would be another. The slot
  doesn't care what the activity is — it holds an opaque ``key`` + a human ``label``
  and treats every requester the same (a plain mutex, no priority).

Interactive activities (overlays) ``heartbeat`` — a closed tab stops polling and its
lease lapses after ``timeout_s`` so the next requester can take over. Background
activities (a batch job) set ``heartbeat=False`` and hold the slot until they
explicitly release it (job done / cancelled). ``SLOT`` is the process-wide singleton.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Activity:
    """The current occupant of an aux-GPU slot."""

    key: str  # unique holder id, e.g. "overlay:data:<session>", "overlay:run", "process:<job_id>"
    label: str  # human-readable, shown to a blocked requester ("SAM3 overlay", "processing X (preview)")
    heartbeat: bool = True  # interactive (expires if silent) vs background (holds until released)
    seen: float = 0.0  # last acquire/heartbeat wall-clock time


class AuxGpuSlot:
    """A single exclusive aux-GPU slot held by at most one :class:`Activity`.

    Plain mutex: ``acquire`` succeeds iff the slot is free (or already this key, or
    the holder's heartbeat lapsed). No priority, no preemption — a would-be user
    stops the current activity (or lets it finish) first.
    """

    def __init__(self, timeout_s: float = 12.0):
        self._holder: Activity | None = None
        self._timeout = timeout_s

    def _lapsed(self, now: float) -> bool:
        h = self._holder
        return h is not None and h.heartbeat and (now - h.seen) > self._timeout

    def free(self, now: float) -> bool:
        return self._holder is None or self._lapsed(now)

    def blocks(self, key: str, now: float) -> bool:
        """True if an activity OTHER than ``key`` currently holds the slot."""
        return self._holder is not None and self._holder.key != key and not self._lapsed(now)

    def acquire(self, key: str, label: str, now: float, *, heartbeat: bool = True) -> bool:
        """Take the slot for ``key`` if free (or already ours). Returns False if
        another activity holds it. Re-acquiring with the same key refreshes it."""
        if self.blocks(key, now):
            return False
        self._holder = Activity(key=key, label=label, heartbeat=heartbeat, seen=now)
        return True

    def touch(self, key: str, now: float) -> bool:
        """Heartbeat: refresh ``key``'s lease. Returns True iff ``key`` holds it."""
        if self._holder is not None and self._holder.key == key and not self._lapsed(now):
            self._holder.seen = now
            return True
        return False

    def release(self, key: str) -> None:
        """Release the slot iff ``key`` holds it (no-op otherwise — you can't free
        another activity's slot)."""
        if self._holder is not None and self._holder.key == key:
            self._holder = None

    def holder(self, now: float) -> Activity | None:
        """The current (unlapsed) holder, or None if the slot is free."""
        return None if self.free(now) else self._holder


# Process-wide singleton — one aux-GPU slot for the whole GUI server (one GPU today).
SLOT = AuxGpuSlot()
