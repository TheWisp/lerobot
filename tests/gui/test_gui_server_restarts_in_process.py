# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GUI app can be started again in the process that stopped it.

The test suites start the real app in-process, one server per module, and
under `pytest -n auto` a worker runs many modules in turn. The app's shutdown
hook closed the module-level decode pool for good, so in every server after
the first the JPEG frame endpoint raised "cannot schedule new futures after
shutdown" -- 74 tracebacks in one CI run, every Full Quality tile a timeout,
and nothing of it visible where each module was run in its own process.
"""

from __future__ import annotations

import pytest

pytest.importorskip("av")

import requests  # noqa: E402

from tests.gui.chunk_fixtures import CAM_WIDE, GuiServer, build_dataset  # noqa: E402


def _frame(base, ds_id, idx=1):
    return requests.get(
        f"{base}/api/datasets/{requests.utils.quote(ds_id, safe='')}/episodes/0/frame/{idx}",
        params={"camera": CAM_WIDE},
        timeout=60,
    )


def test_a_second_server_in_the_same_process_still_decodes_frames(tmp_path_factory):
    root = build_dataset(tmp_path_factory.mktemp("restart") / "restart")
    first = GuiServer(tmp_path_factory.mktemp("config1"), tmp_path_factory.mktemp("cache1"))
    try:
        ds_id = first.open_dataset(root)
        r = _frame(first.base, ds_id)
        assert r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"), (
            r.status_code,
            r.text[:200],
        )
    finally:
        first.stop()
    second = GuiServer(tmp_path_factory.mktemp("config2"), tmp_path_factory.mktemp("cache2"))
    try:
        ds_id = second.open_dataset(root)
        r = _frame(second.base, ds_id)
        assert r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"), (
            "the second server's frame endpoint failed",
            r.status_code,
            r.text[:300],
        )
    finally:
        second.stop()


def test_a_shutdown_that_lands_late_does_not_close_the_pool_under_a_server(tmp_path_factory):
    """The previous server's shutdown can land after the next one has started.

    ``GuiServer.stop()`` waits for it, but with a deadline, and a loaded runner
    has outlived that deadline. Repairing the pools at startup cannot cover
    this: the close arrives afterwards. Every frame request then fails with
    "cannot schedule new futures after shutdown", which reaches the page as a
    Full Quality tile that never loads and a wait that spends its whole budget.
    """
    from lerobot.gui.api.datasets import shutdown_decode_executor

    root = build_dataset(tmp_path_factory.mktemp("late") / "late")
    srv = GuiServer(tmp_path_factory.mktemp("config-late"), tmp_path_factory.mktemp("cache-late"))
    try:
        ds_id = srv.open_dataset(root)
        before = _frame(srv.base, ds_id, 1)
        assert before.status_code == 200, ("the pool was not live to begin with", before.status_code)

        # Exactly what a straggling shutdown does to a server already serving.
        shutdown_decode_executor()

        # A frame the first request did not cache, or this never reaches the
        # pool at all and passes with the bug fully in place.
        after = _frame(srv.base, ds_id, 2)
        assert after.status_code == 200 and after.headers.get("content-type", "").startswith("image/"), (
            "a shutdown landing after startup closed the pool under the server",
            after.status_code,
            after.text[:300],
        )
    finally:
        srv.stop()


def test_the_test_server_keeps_its_state_in_its_own_directory(tmp_path_factory):
    """Opening a dataset records it in the server's config directory, not the
    developer's -- on both channels, since the in-process writer computed its
    path at import. Pinned here because the real-state guard only sees the
    escape on a machine whose import order differs (CI's did)."""
    from lerobot.gui.api import datasets as datasets_api

    root = build_dataset(tmp_path_factory.mktemp("own") / "own")
    config = tmp_path_factory.mktemp("config-own")
    srv = GuiServer(config, tmp_path_factory.mktemp("cache-own"))
    try:
        assert config / "opened_datasets.json" == datasets_api.OPENED_FILE
        srv.open_dataset(root)
        assert (config / "opened_datasets.json").exists(), sorted(p.name for p in config.iterdir())
    finally:
        srv.stop()
    assert config / "opened_datasets.json" != datasets_api.OPENED_FILE, "the redirect outlived the server"
