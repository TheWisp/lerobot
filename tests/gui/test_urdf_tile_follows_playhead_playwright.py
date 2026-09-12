# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The URDF tile follows the playhead from the episode it already fetched.

The viewer cached the whole episode's poses only when the trajectory toggle
was on; with it off -- the default -- every playhead tick was a round trip for
one frame's joints, so the tile moved at the rate the link answered. That was
invisible while the tab painted stills at two or three frames a second and
plain once the video path painted thirty: the pictures moved and the robot
beside them did not.

The measurement that motivated this (2026-09-11, this workstation, a
four-camera 1280x720 30 fps episode over loopback): playing at Low Bandwidth
the tab painted 20 fps while the tile updated 2.2 times a second, each update
a request whose network cost was 4 ms -- the rate was not the link, it was one
serialized round trip per tick. With the toggle on, the same playback issued
no per-frame request at all.

So the contract is the absence of those requests, plus the tile actually
following: a cache that is never read would satisfy the first assertion alone.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("av")

from playwright.sync_api import sync_playwright  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    BROWSER_LOG_ARGS,
    GuiServer,
    MediaLog,
    build_dataset,
    start_trace,
    wait_for_player,
    wait_while_decoding,
)

pytestmark = pytest.mark.requires_playwright

MODE_KEY = "lerobot.cameraVideoMode"
FRAMES = 60
# The vendored SO-101's motor set: enough for the tab to resolve a robot and
# show the tile at all.
SO101 = [
    f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = build_dataset(tmp_path_factory.mktemp("urdf") / "ds", frames=FRAMES, state_names=SO101)
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    ds_id = srv.open_dataset(root)
    yield srv, ds_id
    srv.stop()


def _page(p, media_out):
    """A browser whose media pipeline logs, and a recorder for what it says."""
    browser = p.chromium.launch(args=BROWSER_LOG_ARGS)
    ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
    pg = ctx.new_page()
    media_out.append(MediaLog(ctx, pg))
    return browser, pg


def _open(pg, srv, ds_id, mode):
    pg.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, {json.dumps(mode)});")
    pg.goto(srv.base)
    pg.wait_for_function("typeof openDataset === 'function'", timeout=30_000)
    pg.evaluate("(ds) => openDataset(ds)", ds_id)
    pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=120_000)
    pg.evaluate(f"selectEpisode({json.dumps(ds_id)}, 0, {FRAMES})")
    if mode == "low-bandwidth":
        wait_for_player(pg, "window.__chunkPlayer && window.__chunkPlayer.ready()")
    pg.wait_for_function(
        "() => document.getElementById('urdf-viz-panel')"
        " && getComputedStyle(document.getElementById('urdf-viz-panel')).display !== 'none'",
        timeout=60_000,
    )
    frame = _tile(pg)
    frame.wait_for_function("() => window.__urdfApplied", timeout=60_000)
    start_trace(pg)
    return frame


def _tile(pg):
    """The tile's iframe, once the parent has pointed it at the viewer."""
    pg.wait_for_function(
        "() => { const f = document.getElementById('urdf-viz-iframe');"
        " return f && f.src.includes('urdf_viz'); }",
        timeout=60_000,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for f in pg.frames:
            if "urdf_viz" in (f.url or ""):
                return f
        pg.wait_for_timeout(100)
    raise AssertionError("the URDF tile's iframe never appeared in the page")


@pytest.mark.parametrize("mode", ["full-quality", "low-bandwidth"])
def test_playback_moves_the_tile_without_a_request_per_frame(server, mode):
    srv, ds_id = server
    with sync_playwright() as p:
        media: list = []
        browser, pg = _page(p, media)
        frame = _open(pg, srv, ds_id, mode)

        per_frame = []
        pg.on(
            "request",
            lambda r: per_frame.append(r.url) if "/urdf-viz?" in r.url and "horizon=1" in r.url else None,
        )
        start = frame.evaluate("() => window.__urdfApplied")
        pg.evaluate("togglePlay()")
        # The tile follows the picture -- onPaint publishes the playhead -- so
        # if the picture did not move there is nothing to follow and asserting
        # on the tile measures the runner. Wait for exactly what the assertions
        # below need and no more: one painted frame past the start, which is
        # what makes "the tile moved" and "it moved without asking" observable.
        # Paints are paced by whole chunks, not by frames -- the transport keeps
        # its own clock, so a runner that cannot decode in real time laps the
        # decoder and paints as each chunk lands -- so a count here is a
        # stopwatch on the runner, and four of them cost four chunks.
        wait_while_decoding(
            pg,
            media[0],
            "() => window.__chunkPlayer"
            " ? window.__chunkPlayer.metrics.painted.some((q) => q.frame > 0)"
            " : window.currentFrame > 0",
            "the tab never advanced, so the tile had nothing to follow",
        )
        # Long enough that a round trip per painted frame would be dozens of them.
        pg.wait_for_timeout(int(3 * 1000))
        pg.evaluate("togglePlay()")
        end = frame.evaluate("() => window.__urdfApplied")

        assert not per_frame, f"the tile asked for {len(per_frame)} single frames: {per_frame[:3]}"
        assert end["frame"] > start["frame"], (start, end)
        assert end["joints"] != start["joints"], "the tile held one pose through the whole playback"


def test_a_scrub_lands_the_tile_on_that_frame(server):
    """The complement of the count above: the cache is read, and read at the
    frame asked for -- a cache indexed at the wrong row would keep the request
    count at zero just as well."""
    srv, ds_id = server
    with sync_playwright() as p:
        media: list = []
        browser, pg = _page(p, media)
        frame = _open(pg, srv, ds_id, "low-bandwidth")
        asked = []
        pg.on(
            "request",
            lambda r: asked.append(r.url) if "/urdf-viz?" in r.url and "horizon=1" in r.url else None,
        )
        for target in (FRAMES - 1, 7, FRAMES // 2):
            pg.evaluate(f"loadAllFrames({target})")
            # Wait for the picture first. The tile shares a renderer thread
            # with the tab, so while the seek's chunk is decoding it gets
            # little of it -- and the contract here is that the tile lands on
            # the frame asked for, not that it beats the picture to it. On CI
            # this is the difference between a real assertion and a stopwatch.
            # The precondition, not the property: a scrub into a chunk nobody
            # has decoded yet costs that whole chunk, which is the machine's
            # business and not this test's. The assertion below stays tight.
            wait_while_decoding(
                pg,
                media[0],
                "(t) => window.__chunkPlayer.metrics.painted.some((q) => q.frame === t)",
                f"the tab never painted frame {target} after a scrub to it",
                arg=target,
            )
            frame.wait_for_function(
                "(t) => window.__urdfApplied && window.__urdfApplied.frame === t",
                arg=target,
                timeout=30_000,
            )
        assert not asked, f"the scrub went to the network: {asked[:3]}"


def test_the_episode_is_fetched_once_for_the_whole_episode(server):
    """One whole-episode fetch per episode, not one per tick -- and it carries
    every frame, which is what makes the scrub above free."""
    srv, ds_id = server
    with sync_playwright() as p:
        media: list = []
        browser, pg = _page(p, media)
        whole = []
        pg.on(
            "request",
            lambda r: whole.append(r.url) if "/urdf-viz?" in r.url and "horizon=99999" in r.url else None,
        )
        frame = _open(pg, srv, ds_id, "low-bandwidth")
        pg.evaluate("togglePlay()")
        # Playing for a fixed couple of seconds asserts the runner painted
        # something in them, which on a loaded one it does not: paints come a
        # chunk at a time. Wait for the picture to have moved instead.
        wait_while_decoding(
            pg,
            media[0],
            "() => window.__chunkPlayer.metrics.painted.some((q) => q.frame > 0)",
            "the tab never painted a frame past the first, so the tile had nothing to follow",
        )
        pg.evaluate("togglePlay()")
        assert len(whole) == 1, f"the episode was fetched {len(whole)} times: {whole}"
        assert frame.evaluate("() => window.__urdfApplied.frame") > 0
