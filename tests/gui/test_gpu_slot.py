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
"""Aux-GPU slot mutex — the resource layer.

One exclusive slot; every activity (overlay data/run, batch job, future depth
export) requests it the same way. Plain mutex: no priority, no preemption. Time is
injected (``now``), so expiry is deterministic — no sleeps.
"""

from __future__ import annotations

from lerobot.gui.gpu_slot import AuxGpuSlot


def test_acquire_is_exclusive_and_reentrant():
    slot = AuxGpuSlot(timeout_s=10.0)
    assert slot.acquire("a", "A", now=0.0) is True
    # A different activity is refused while A holds it.
    assert slot.acquire("b", "B", now=1.0) is False
    assert slot.blocks("b", now=1.0) is True
    assert slot.blocks("a", now=1.0) is False  # the holder is never "blocked"
    # Re-acquiring with the same key just refreshes it.
    assert slot.acquire("a", "A", now=2.0) is True
    assert slot.holder(now=2.0).key == "a"


def test_release_only_frees_own_slot():
    slot = AuxGpuSlot(timeout_s=10.0)
    slot.acquire("a", "A", now=0.0)
    slot.release("b")  # someone else's release is a no-op
    assert slot.blocks("b", now=1.0) is True
    slot.release("a")
    assert slot.free(now=1.0) is True
    assert slot.holder(now=1.0) is None
    assert slot.acquire("b", "B", now=1.0) is True  # now free for the next activity


def test_heartbeat_lapse_frees_interactive_holder():
    slot = AuxGpuSlot(timeout_s=10.0)
    slot.acquire("a", "A", now=0.0, heartbeat=True)
    # Still held just under the timeout.
    assert slot.blocks("b", now=9.0) is True
    assert slot.acquire("b", "B", now=9.0) is False
    # Past the timeout with no touch → lapses; the next activity takes over.
    assert slot.free(now=11.0) is True
    assert slot.acquire("b", "B", now=11.0) is True


def test_touch_extends_the_lease():
    slot = AuxGpuSlot(timeout_s=10.0)
    slot.acquire("a", "A", now=0.0)
    assert slot.touch("a", now=8.0) is True  # heartbeat before expiry
    assert slot.blocks("b", now=17.0) is True  # extended: 17 - 8 < 10
    assert slot.touch("b", now=17.0) is False  # a non-holder can't heartbeat


def test_background_activity_never_lapses():
    slot = AuxGpuSlot(timeout_s=10.0)
    # A batch job holds the slot with no heartbeat — it must survive arbitrary silence.
    assert slot.acquire("job", "processing X (full)", now=0.0, heartbeat=False) is True
    assert slot.free(now=10_000.0) is False
    assert slot.blocks("a", now=10_000.0) is True
    assert slot.holder(now=10_000.0).label == "processing X (full)"
    slot.release("job")  # only an explicit release (job done) frees it
    assert slot.free(now=10_000.0) is True


def test_holder_returns_label_for_a_blocked_requester():
    slot = AuxGpuSlot(timeout_s=10.0)
    slot.acquire("overlay:run", "SAM3 overlay (run)", now=0.0)
    assert slot.holder(now=1.0).label == "SAM3 overlay (run)"
