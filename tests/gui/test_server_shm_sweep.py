# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the GUI's lifecycle hooks are allowed to delete from /dev/shm.

The sweep matches every segment named `lerobot_obs_*` in a directory the whole
host shares. The GUI called it at startup and at shutdown on the assumption
that either moment is writer-quiescent -- true on a machine with nothing else
running, and false for a teleop started outside the GUI, for a second GUI, and
for every other pytest-xdist worker when the suite starts servers of its own.

The defect was not hypothetical: a live-video suite reported
`no tap at lerobot_obs_t5191_meta; /dev/shm holds []` -- its tap, and every
other, gone.

These drive the hooks rather than the helper. `tests/test_obs_stream.py` covers
what `respect_liveness` means; what is covered here is that the GUI asks for it,
because a correct helper nobody passes the flag to protects nothing.

The sweep's directory is a default argument bound when the module is imported,
so it cannot be redirected by patching `_SHM_DIR`. The sweep is therefore
wrapped rather than replaced: the wrapper redirects only the directory and
forwards whatever the call site asked for, so the real liveness logic runs
against a temporary directory and the flag under test is the hook's own.
"""

from __future__ import annotations

import time

import pytest

import lerobot.gui.server as gui_server
from lerobot.robots.obs_stream import _HDR, SHM_PREFIX, cleanup_stale_streams


@pytest.fixture
def sweeps_here(monkeypatch, tmp_path):
    """Point the hooks' sweep at a directory of this test's own, keeping the
    real implementation and whatever the hook chose to pass it."""
    calls: list[dict] = []

    def wrapper(shm_dir=None, **kwargs):
        calls.append(dict(kwargs))
        return cleanup_stale_streams(tmp_path, **kwargs)

    monkeypatch.setattr("lerobot.robots.obs_stream.cleanup_stale_streams", wrapper)
    monkeypatch.setattr("lerobot.overlays.overlay_ipc.unlink_stale_segments", lambda: 0)
    return calls


def _segment(tmp_path, suffix: str, *, written_at: float):
    path = tmp_path / f"{SHM_PREFIX}{suffix}"
    path.write_bytes(_HDR.pack(5, 5, written_at))
    return path


async def _run(hook):
    """The hooks do a great deal besides sweeping, none of it available here.
    The sweep happens regardless and is what is under test."""
    try:
        await hook()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_startup_leaves_a_tap_that_is_being_written(sweeps_here, tmp_path):
    live = _segment(tmp_path, "meta", written_at=time.time())

    await _run(gui_server.startup_event)

    assert sweeps_here, "startup no longer sweeps at all; this test is now vacuous"
    assert live.exists(), (
        "starting the GUI deleted a tap another process is writing, which freezes "
        "its readers: an external teleop, a second GUI, or another test worker"
    )


@pytest.mark.asyncio
async def test_startup_still_removes_one_nobody_is_writing(sweeps_here, tmp_path):
    """The complement. Without it, a sweep that had been disabled outright
    would satisfy the test above, and the orphan it exists to remove -- a tap
    left by a crashed run, which a reader attaches to and serves frozen -- would
    stay forever."""
    orphan = _segment(tmp_path, "meta", written_at=time.time() - 3600)

    await _run(gui_server.startup_event)

    assert not orphan.exists(), "a tap nobody has written to in an hour survived startup"


@pytest.mark.asyncio
async def test_shutdown_leaves_a_tap_that_is_being_written(sweeps_here, tmp_path):
    live = _segment(tmp_path, "meta", written_at=time.time())

    await _run(gui_server.shutdown_event)

    assert sweeps_here, "shutdown no longer sweeps at all; this test is now vacuous"
    assert live.exists(), "stopping the GUI deleted a tap belonging to someone else"
