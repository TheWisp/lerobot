# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the server log says about the player, so a rig report can be read.

A wedge on the rig left sixteen seconds of silence in the log: the last chunk
line, then nothing, and nothing about what the page held or why it stopped
asking. Two channels close that: every chunk request carries the player's
state and the chunk line echoes it, and the player's own events -- a decoder
error, a chunk given up on, a rule violation -- are posted and logged as
warnings, so the log alone tells what the page saw.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("av")

import requests  # noqa: E402

from tests.gui.chunk_fixtures import GuiServer, build_dataset, chunk_url  # noqa: E402


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("events") / "events")


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    srv.ds_id = srv.open_dataset(dataset_root)
    yield srv
    srv.stop()


def test_the_chunk_line_echoes_the_players_state(server, caplog):
    caplog.set_level(logging.INFO, logger="lerobot.gui.api.chunk_playback")
    r = requests.get(
        chunk_url(server.base, server.ds_id, 0, 0),
        headers={"X-Player": "cur=7 held=0 inflight=20 painted=12 holds=1 errors=0"},
        timeout=60,
    )
    assert r.status_code == 200
    lines = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("chunk-playback ")]
    assert lines, "no chunk line was logged"
    assert "player cur=7 held=0 inflight=20 painted=12 holds=1 errors=0" in lines[-1], lines[-1]


def test_a_chunk_request_without_state_still_logs(server, caplog):
    caplog.set_level(logging.INFO, logger="lerobot.gui.api.chunk_playback")
    assert requests.get(chunk_url(server.base, server.ds_id, 0, 0), timeout=60).status_code == 200
    lines = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("chunk-playback ")]
    assert lines and "player" not in lines[-1].split("build=")[-1], lines[-1]


def test_the_players_events_are_logged_as_warnings(server, caplog):
    caplog.set_level(logging.WARNING, logger="lerobot.gui.api.chunk_playback")
    url = f"{server.base}/api/datasets/{requests.utils.quote(server.ds_id, safe='')}/episodes/0/player-event"
    r = requests.post(
        url,
        json={
            "kind": "never-ready",
            "detail": "chunk 60 held 4001 ms without a frame; dropped, asking again (1 of 3)",
        },
        timeout=30,
    )
    assert r.status_code == 204, r.text
    warnings = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING and "chunk-playback client" in rec.getMessage()
    ]
    assert warnings, [rec.getMessage() for rec in caplog.records]
    msg = warnings[-1].getMessage()
    assert "ep=0" in msg and "never-ready" in msg and "chunk 60 held 4001 ms" in msg, msg


def test_an_event_for_an_unknown_dataset_is_refused(server):
    url = (
        f"{server.base}/api/datasets/{requests.utils.quote('/nowhere/none', safe='')}/episodes/0/player-event"
    )
    assert requests.post(url, json={"kind": "x", "detail": "y"}, timeout=30).status_code == 404


def test_an_event_needs_a_kind_and_a_detail(server):
    url = f"{server.base}/api/datasets/{requests.utils.quote(server.ds_id, safe='')}/episodes/0/player-event"
    assert requests.post(url, json={"detail": "no kind"}, timeout=30).status_code == 422
