# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""/training/image-status must not do network I/O, and must not block the loop.

Every full GUI load hits this endpoint. It used to call ``get_image_status()``
inline from an ``async def``: roughly six ``git`` subprocesses, a
``docker image inspect``, and a ``git fetch`` — each with a 5–10 second timeout,
all on the event loop, blocking static files and websockets for the duration.

The rule this pins is "a passive GET returns cached state; refreshing is
explicit". Asserting it rather than describing it, because the failure is
invisible: the endpoint returns correct data either way, it just takes the
whole GUI down with it while it does.
"""

from __future__ import annotations

import pytest

from lerobot.gui.api import training


@pytest.fixture(autouse=True)
def _clear_cache():
    training._image_status_cache = None
    yield
    training._image_status_cache = None


@pytest.fixture
def spy(monkeypatch):
    """Record how get_image_status is called, without shelling out."""
    calls: list[bool] = []

    def _fake(allow_fetch: bool = False):
        calls.append(allow_fetch)
        return {"image": "test", "git": None}

    monkeypatch.setattr(training, "get_image_status", _fake)
    return calls


@pytest.mark.asyncio
async def test_the_get_never_permits_the_network_fetch(spy):
    """The `git fetch` is the specific thing that made page loads stall."""
    await training.image_status()

    assert spy == [False]


@pytest.mark.asyncio
async def test_refresh_is_where_the_fetch_lives(spy):
    """Capability preserved, but as something the operator asks for."""
    await training.image_status_refresh()

    assert spy == [True]


@pytest.mark.asyncio
async def test_a_second_load_within_the_ttl_is_served_from_cache(spy):
    """A reload should not re-run six subprocesses."""
    await training.image_status()
    await training.image_status()

    assert len(spy) == 1, "second GET re-ran the probe instead of using the cache"


@pytest.mark.asyncio
async def test_refresh_bypasses_the_cache(spy):
    await training.image_status()
    await training.image_status_refresh()

    assert spy == [False, True], "refresh must re-probe even when the cache is warm"


@pytest.mark.asyncio
async def test_an_expired_cache_re_probes(spy, monkeypatch):
    await training.image_status()
    stamp, value = training._image_status_cache
    training._image_status_cache = (stamp - training._IMAGE_STATUS_TTL_S - 1, value)

    await training.image_status()

    assert len(spy) == 2


@pytest.mark.asyncio
async def test_the_work_runs_off_the_event_loop(monkeypatch):
    """Correct output from the wrong thread is still a frozen GUI."""
    import threading

    loop_thread = threading.get_ident()
    ran_on: list[int] = []

    def _fake(allow_fetch: bool = False):
        ran_on.append(threading.get_ident())
        return {}

    monkeypatch.setattr(training, "get_image_status", _fake)
    await training.image_status()

    assert ran_on and ran_on[0] != loop_thread


def test_it_uses_a_named_pool_not_the_shared_default():
    """A stalled docker inspect must not starve decode or camera teardown."""
    assert training._image_status_executor._max_workers == 1
    assert "image-status" in training._image_status_executor._thread_name_prefix
