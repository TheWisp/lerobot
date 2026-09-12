# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the rig does when a chunk's decode never finishes (design: R1, R5).

Two operators on fc500t, 2026-09-11, on two different four-camera datasets:
playback stopped at frame 15 of episode 8 and never moved again. The server
log carries the whole sequence, which is why these tests exist:

    chunk-playback ... ep=8 start=0 frames=60 ... cameras=4
    ... player cur=15 held=60,120,180 inflight=0 painted=15 holds=1 errors=12
    client ... error: chunk 0 never ready after 3 tries; held 4004 ms ...

Three facts in that: the page had painted 15 of chunk 0's 60 frames and held
at the sixteenth; chunks 60, 120 and 180 were decoded and waiting behind it;
and the retry budget had been spent, yet the same chunk was re-requested every
four seconds for the next hour (errors 12, 13, 14 ... 36). Alongside it,
Chrome was saying `Codec reclaimed due to inactivity` for sixteen decoders --
four cameras times four buffered chunks -- which is where the missing frames
went.

So: four contracts. The page must not hold a decoder per camera per buffered
chunk; a chunk whose frames never arrive must not stop playback for good; the
retry budget must be terminal, because a budget that is spent and then ignored
is not a budget; and a scrub back into the frames it gave up on must ask for
them again, since that is the only way back to them. Alongside those, the
quality selector must keep the playhead -- the operators reached for it when
the picture was wrong, and it restarted the episode. The existing suites all
run two cameras and never stall a decoder, which is exactly why none of them
saw any of it.
"""

from __future__ import annotations

import json
import re
import struct
import time

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("av")

import requests  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    CAM_WIDE,
    CAMS_RIG,
    GuiServer,
    build_dataset,
    chunk_url,
    parse_chunk,
    wait_for_player,
    wait_while_decoding,
)

pytestmark = pytest.mark.requires_playwright

MODE_KEY = "lerobot.cameraVideoMode"
FRAMES = 120  # six 2 s chunks at 10 fps: enough that the buffer runs several ahead

# Counts the decoders the page has configured and not yet closed. The page
# makes one per camera per chunk it decodes, so this is the question "how many
# chunks is it decoding at once" in the units Chrome runs out of.
COUNT_DECODERS = """
window.__dec = { live: 0, max: 0, made: 0, open: new Set() };
// How long the oldest decoder the page has not closed has been open. A count
// sampled at an instant says nothing -- decoders come and go every chunk --
// but one that is still open seconds after it was configured is one the page
// has forgotten.
window.__dec.oldest = 0;
setInterval(() => {
  for (const d of window.__dec.open) {
    window.__dec.oldest = Math.max(window.__dec.oldest, performance.now() - d.__at);
  }
}, 50);
const V = window.VideoDecoder;
window.VideoDecoder = class extends V {
  constructor(init) { super(init); window.__dec.made++; this.__counted = false; }
  configure(c) {
    if (!this.__counted) {
      this.__counted = true;
      this.__at = performance.now();
      window.__dec.open.add(this);
      window.__dec.live++;
      window.__dec.max = Math.max(window.__dec.max, window.__dec.live);
    }
    return super.configure(c);
  }
  close() {
    if (this.__counted) { this.__counted = false; window.__dec.open.delete(this); window.__dec.live--; }
    return super.close();
  }
};
"""

# A decode that runs past the hold deadline and then succeeds -- a loaded
# machine, not a broken chunk. The page must not call that "never ready".
SLOW_FIRST_DECODE = """
window.__slow = { ms: 5000 };
const V = window.VideoDecoder;
let made = 0;
window.VideoDecoder = class extends V {
  constructor(init) {
    const mine = ++made;
    const queue = [];
    super({
      ...init,
      output: (f) => { if (mine === 1) queue.push(f); else init.output(f); },
    });
    this.__mine = mine; this.__queue = queue; this.__init = init;
  }
  flush() {
    const done = super.flush();
    if (this.__mine !== 1) return done;
    return done.then(() => new Promise((r) => setTimeout(() => {
      for (const f of this.__queue) this.__init.output(f);
      this.__queue.length = 0;
      r();
    }, window.__slow.ms)));
  }
};
"""

KEEP = 15  # frames of chunk 0 that survive, as on the rig
# Two fetches stand at once (the player's MAX_INFLIGHT) and a third chunk can
# be decoding from an earlier arrival, since decode starts on arrival and is
# not gated by the fetch limit. So three chunks' worth of decoders is the
# honest ceiling. What must never return is the rig's shape: one per camera
# per *buffered* chunk, which was sixteen to twenty and grows with the buffer.
DECODING_CHUNKS = 3


def _truncated_chunk(body: bytes, camera: str, keep: int) -> bytes:
    """The rig's chunk 0: every camera's frames are there in the header, and one
    camera's encoded bytes stop part way, so its decoder produces ``keep``
    frames and no more. Built from the server's own body, so everything else
    about the chunk -- the header, the other cameras, the masks -- is real."""
    header, parts = parse_chunk(body)
    hl = struct.unpack("<I", body[:4])[0]
    part, _data = parts[("video", camera)]
    kept = sum(part["frame_sizes"][:keep])
    at = 4 + hl + part["offset"]
    out = bytearray(body)
    # Zeroed bytes are not a decodable access unit; the decoder stops there,
    # which is what "held 4004 ms without a frame" was.
    out[at + kept : at + part["length"]] = b"\x00" * (part["length"] - kept)
    assert len(out) == len(body)
    return bytes(out)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = build_dataset(tmp_path_factory.mktemp("rig") / "ds", frames=FRAMES, cams=CAMS_RIG)
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    ds_id = srv.open_dataset(root)
    yield srv, ds_id
    srv.stop()


def _break_chunk_zero(pg, srv, ds_id):
    """Serve a chunk 0 whose first camera stops decoding at frame 15, every
    time it is asked for, and count the asks."""
    body = requests.get(chunk_url(srv.base, ds_id, 0, 0), timeout=120).content
    broken = _truncated_chunk(body, CAM_WIDE, KEEP)
    asks: list[float] = []

    def handler(route):
        asks.append(time.monotonic())
        route.fulfill(
            status=200,
            body=broken,
            headers={"Content-Type": "application/octet-stream", "Cache-Control": "no-store"},
        )

    pg.route(re.compile(r".*/chunk\?.*start=0&.*"), handler)
    return asks


# The give-up is three tries at a four-second hold by default: half a minute
# of sleeping per test that has to reach it, on every suite and every runner.
# The logic is the same at a tenth of the wait, and the production values are
# what the page uses -- nothing here changes them for anyone else.
FAST_DEADLINES = (
    "window.__chunkPlayerDeadlines = { holdRetryMs: 400, decodeBudgetMs: 400, decodeQuietMs: 250 };"
)


def _open(pg, srv, ds_id, mode="low-bandwidth"):
    pg.add_init_script(FAST_DEADLINES)
    pg.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, {json.dumps(mode)});")
    pg.goto(srv.base)
    pg.wait_for_function("typeof openDataset === 'function'", timeout=30_000)
    pg.evaluate("(ds) => openDataset(ds)", ds_id)
    pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=120_000)
    pg.evaluate(f"selectEpisode({json.dumps(ds_id)}, 0, {FRAMES})")
    if mode == "low-bandwidth":
        wait_for_player(pg, "window.__chunkPlayer && window.__chunkPlayer.ready()")


def test_the_page_does_not_open_a_decoder_per_camera_per_buffered_chunk(server):
    """Four cameras and a buffer several chunks deep was sixteen decoders on the
    rig, which Chrome reclaimed under the page's feet mid-stream.

    The page decodes a chunk when it arrives and closes each camera's decoder
    on flush, so what is open at once is bounded by the transfers in flight --
    not by how far ahead the buffer has run. This is the invariant, not a
    reproduction: this workstation decodes a 320-wide chunk in software fast
    enough that the count barely leaves one chunk's worth, and the rig's
    sixteen have never been seen here. It is here so a change that starts
    decoding eagerly, or stops closing on flush, is caught."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        pg.add_init_script(COUNT_DECODERS)
        _open(pg, srv, ds_id)
        pg.evaluate("togglePlay()")
        # Play until the page has opened decoders worth counting, rather than
        # for a fixed spell that assumes how fast this machine decodes. The
        # ceiling below is sampled across however long that took.
        wait_while_decoding(
            pg,
            None,
            f"() => window.__dec.made > {len(CAMS_RIG)}",
            "the page decoded nothing worth counting",
        )
        pg.evaluate("togglePlay()")
        dec = pg.evaluate("() => window.__dec")
        assert dec["made"] > len(CAMS_RIG), f"the page decoded nothing worth counting: {dec}"
        assert dec["max"] <= DECODING_CHUNKS * len(CAMS_RIG), (
            f"the page held {dec['max']} decoders open at once for {len(CAMS_RIG)} cameras: {dec}"
        )


def test_a_chunk_that_never_becomes_ready_is_given_up_on_and_playback_goes_on(server):
    """The rig's frame 15, with the rig's consequences: the chunk arrives and
    decodes part way, the chunks behind it are decoded and waiting, and the
    page sits at the sixteenth frame. The budget must end, and the frames
    behind the bad chunk must play."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        asks = _break_chunk_zero(pg, srv, ds_id)
        _open(pg, srv, ds_id)
        pg.evaluate("togglePlay()")
        # Three tries at the hold, then the give-up, then the frames behind it.
        # How long that takes is the machine's business; that it happens at all
        # is the product's, and a player that stops getting anywhere fails here
        # rather than running out a clock.
        wait_while_decoding(
            pg,
            None,
            "() => window.currentFrame >= 60",
            "playback never reached the frames behind the chunk it gave up on",
        )
        reached = pg.evaluate("() => window.currentFrame")
        assert len(asks) > 1, "chunk 0 was never re-asked: the stall did not reproduce"
        assert reached >= 60, (
            f"playback stopped at frame {reached}: one chunk that never finished decoding held "
            f"the episode, with chunks 60 and beyond decoded and waiting (asked {len(asks)} times)"
        )
        # And it is the budget that ended it, not luck: no more asks after.
        settled = len(asks)
        pg.wait_for_timeout(3_000)
        assert len(asks) == settled, (
            f"the page asked for the dead chunk {len(asks) - settled} more times in 10 s "
            f"after spending its budget"
        )
        assert settled <= 5, f"the page asked for one chunk {settled} times before stopping"


def test_changing_the_quality_keeps_the_playhead(server):
    """Switching Low Bandwidth off and on is what an operator reaches for when
    the picture is wrong, and it restarted the episode from frame 0 -- losing
    the place they were looking at, which on a 3000 frame episode is the whole
    reason they were scrubbing."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        _open(pg, srv, ds_id)
        pg.evaluate("loadAllFrames(80)")
        pg.wait_for_function("() => window.currentFrame === 80", timeout=30_000)
        for target in ("full-quality", "low-bandwidth"):
            pg.select_option("#video-mode-select", target)
            pg.wait_for_timeout(1500)
            now = pg.evaluate("() => window.currentFrame")
            assert now == 80, f"switching to {target} moved the playhead from 80 to {now}"


def test_scrubbing_back_into_a_skipped_gap_asks_for_it_again(server):
    """`Recover failed` in the report, and there is no Recover for playback:
    the only thing an operator does with a part of the episode they did not
    see is scrub back to it. That has to mean "ask again", or the gap is
    permanent for as long as the episode stays open."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        asks = _break_chunk_zero(pg, srv, ds_id)
        _open(pg, srv, ds_id)
        pg.evaluate("togglePlay()")
        wait_while_decoding(
            pg,
            None,
            "() => window.__chunkPlayer.state().dead.length > 0",
            "chunk 0 was never given up on",
        )
        assert pg.evaluate("() => window.__chunkPlayer.state().dead") == [0], "chunk 0 was never given up on"
        pg.evaluate("togglePlay()")

        # The transient is over -- the chunk is servable again, as it would be
        # after whatever made the decode fail has passed.
        pg.unroute(re.compile(r".*/chunk\?.*start=0&.*"))
        before = len(asks)
        pg.evaluate("loadAllFrames(5)")
        wait_for_player(pg, "() => window.__chunkPlayer.metrics.painted.some((p) => p.frame === 5)")
        assert pg.evaluate("() => window.__chunkPlayer.state().dead") == []
        assert len(asks) == before, "the scrub went through the broken route, so this proves nothing"


# `dec.decode()` throwing where it is called, which is what CI hit
# ("A key frame is required after configure()"). Injected rather than provoked
# with bad bytes: whether a given Chromium validates synchronously or reports
# through the error callback is the browser's business, and the contract here
# is what the page does when it throws.
THROWING_DECODE = """
window.__threw = 0;
const V = window.VideoDecoder;
let made = 0;
window.VideoDecoder = class extends V {
  constructor(init) { super(init); this.__mine = ++made; }
  decode(chunk) {
    if (this.__mine === 1) { window.__threw++; throw new DOMException('injected: a key frame is required', 'DataError'); }
    return super.decode(chunk);
  }
};
"""


def test_a_camera_that_cannot_be_decoded_closes_its_decoder_where_it_fails(server):
    """CI caught this one: `('decoders of the dropped chunk left open',
    {'live': 2, ...})`.

    `dec.decode()` throws synchronously when the stream has no key frame after
    configure. The throw left every decoder already created for that chunk
    open until the give-up closed them four seconds later -- four seconds of
    held decoders per attempt, on the exact path where decoders are already
    scarce. They must be closed where the throw happens, and the cameras after
    it must still decode.
    """
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        pg.add_init_script(COUNT_DECODERS)
        pg.add_init_script(THROWING_DECODE)
        _open(pg, srv, ds_id)
        pg.evaluate("togglePlay()")
        wait_for_player(pg, "() => window.__chunkPlayer.metrics.errors.some((e) => e.includes('injected'))")
        assert pg.evaluate("() => window.__threw") > 0
        wait_while_decoding(
            pg,
            None,
            "() => window.__dec.made > 1",
            "only one decoder was ever made: the throw was not on the path under test",
        )
        made = pg.evaluate("() => window.__dec.made")
        oldest = pg.evaluate("() => window.__dec.oldest")
        assert made > 1, f"only {made} decoders were made: the throw was not on the path under test"
        # Measured in the page from load, because the leak window closes when
        # the give-up drops the chunk -- sampling only after the player is
        # ready misses it entirely, which is how two earlier versions of this
        # test passed with the bug in place.
        assert oldest < 2000, (
            f"a decoder stayed open {oldest:.0f} ms: a chunk's decoders are left to the give-up"
        )


def test_a_decode_that_runs_long_is_not_a_chunk_that_never_arrives(server):
    """CI caught this one too, through its consequences.

    The deadline measured wall time since the chunk arrived, so on a loaded
    machine a chunk that was still decoding got called "never ready", dropped
    and asked for again -- and once a spent budget means the frames are
    skipped, a slow decode turns into a hole in the episode. The deadline has
    to measure the chunk's chance to be shown: its decode settling.
    """
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        pg.add_init_script(SLOW_FIRST_DECODE)
        _open(pg, srv, ds_id)
        pg.evaluate("togglePlay()")
        # Until the slow chunk paints. A fixed spell here is a bet that this
        # machine finishes a deliberately slowed decode inside it; waiting for
        # the paint carries past the window a wrong give-up would have fired in,
        # whatever the speed.
        wait_while_decoding(
            pg,
            None,
            "() => window.__chunkPlayer.metrics.painted.some((q) => q.frame < 60)",
            "the slow chunk never painted at all",
        )
        retries = pg.evaluate("() => window.__chunkPlayer.metrics.retries")
        painted = pg.evaluate("() => window.__chunkPlayer.metrics.painted.map((q) => q.frame)")
        assert any(f < 60 for f in painted), ("the slow chunk never painted at all", painted[:5])
        assert retries == [], f"a chunk that was still decoding was given up on: {retries}"
        assert pg.evaluate("() => window.__chunkPlayer.state().dead") == []


def _wait_dead(pg, start=0):
    """Play until the budget is spent on ``start``.

    Three tries at a hold apiece, so this is the slow part of both tests below
    -- and how slow depends on the machine, which is why it ends when the
    player stops getting anywhere rather than at a time of this helper's
    choosing.
    """
    wait_while_decoding(
        pg,
        None,
        f"() => JSON.stringify(window.__chunkPlayer.state().dead) === JSON.stringify([{start}])",
        f"chunk {start} was never given up on",
    )


def test_paused_the_player_holds_where_it_was_put(server):
    """The complement of stepping over a gap: stepping is a playback
    behaviour. An operator who scrubbed to a frame asked for that frame, and
    moving them elsewhere in the episode is not a recovery -- it is losing
    their place, which is what CI caught when a scrub to a frame in a chunk
    that would not decode ended up somewhere else entirely."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        _break_chunk_zero(pg, srv, ds_id)
        _open(pg, srv, ds_id)
        # Inside the broken chunk and past the frames that survive in it.
        pg.evaluate(f"loadAllFrames({KEEP + 2})")
        assert pg.evaluate("() => window.currentFrame") == KEEP + 2
        _wait_dead(pg)
        pg.wait_for_timeout(1200)
        assert pg.evaluate("() => window.currentFrame") == KEEP + 2, (
            "the playhead moved off the frame the operator scrubbed to"
        )
        assert pg.evaluate("() => window.__chunkPlayer.state().cur") == KEEP + 2


def test_a_mask_edit_asks_again_for_a_chunk_the_budget_gave_up_on(server):
    """A mask write rebuilds every chunk of the dataset on the server, so a
    chunk that could not be shown before is worth asking for again. Without
    this the gap outlived the edit that would have filled it, for as long as
    the episode stayed open."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        _break_chunk_zero(pg, srv, ds_id)
        _open(pg, srv, ds_id)
        pg.evaluate(f"loadAllFrames({KEEP + 2})")
        _wait_dead(pg)
        # The edit's rebuild, and the chunk servable again with it.
        pg.unroute(re.compile(r".*/chunk\?.*start=0&.*"))
        pg.evaluate("() => window.__chunkPlayer.masksChanged()")
        wait_for_player(
            pg, "(f) => window.__chunkPlayer.metrics.painted.some((q) => q.frame === f)", arg=KEEP + 2
        )
        assert pg.evaluate("() => window.__chunkPlayer.state().dead") == []


# CI's signature, injected: the decoder takes every access unit, hands back two
# frames, and then says nothing at all -- no further output, no error, and a
# flush that never settles. Nothing the page can ask of that decoder will
# finish; the chunk's bytes are still in hand, so a fresh one is the only move.
QUIET_FIRST_DECODER = """
window.__quiet = { after: 2, instances: 0 };
const V = window.VideoDecoder;
window.VideoDecoder = class extends V {
  constructor(init) {
    const mine = ++window.__quiet.instances;
    let out = 0;
    super({
      ...init,
      output: (frame) => {
        if (mine === 1 && out >= window.__quiet.after) { frame.close(); return; }
        out++;
        init.output(frame);
      },
    });
    this.__mine = mine;
  }
  flush() { return this.__mine === 1 ? new Promise(() => {}) : super.flush(); }
};
"""


def test_a_decoder_that_goes_quiet_is_thrown_away_and_the_chunk_decoded_again(server):
    """CI held a chunk sixteen seconds with `q=0`, two frames of twenty, no
    error and no flush: the decoder had taken everything it was given and
    stopped. Waiting on it cannot end well, and the retry budget only refetches
    bytes the page already has. The page keeps those bytes, so it can decode
    them again with a fresh decoder -- which also gives the browser back
    whatever the quiet one was holding.
    """
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        pg.add_init_script(QUIET_FIRST_DECODER)
        asks: list[str] = []
        pg.on("request", lambda r: asks.append(r.url) if "/chunk?" in r.url else None)
        _open(pg, srv, ds_id)
        pg.evaluate("togglePlay()")
        # Past the quiet decoder's chunk: the re-decode is the only way there.
        wait_for_player(pg, "() => window.__chunkPlayer.metrics.painted.some((q) => q.frame >= 25)")
        events = pg.evaluate("() => window.__chunkPlayer.metrics.events.map((e) => e.kind)")
        assert "decoder-quiet" in events, f"the chunk was recovered some other way: {events}"
        assert pg.evaluate("() => window.__chunkPlayer.state().dead") == []
        assert pg.evaluate("() => window.__quiet.instances") > len(CAMS_RIG), (
            "no fresh decoder was made for the chunk"
        )


# CI's actual failure, injected: the decoder works, it is just slow. Each frame
# arrives well inside the quiet window, so the decode never counts as stopped,
# but twenty of them take far longer than the give-up deadline -- which is the
# shape that had CI dropping a chunk two frames from done. The interval tracks
# the deadlines the suite runs with, so the relationship holds whatever they
# are set to: shorter than `decodeQuietMs`, twenty of them longer than
# `decodeBudgetMs + holdRetryMs`.
TRICKLING_FIRST_DECODER = """
window.__trickle = { ms: 120, delivered: 0 };
const V = window.VideoDecoder;
let made = 0;
window.VideoDecoder = class extends V {
  constructor(init) {
    const mine = ++made;
    let n = 0;
    super({
      ...init,
      output: (frame) => {
        if (mine !== 1) { init.output(frame); return; }
        setTimeout(() => { window.__trickle.delivered++; init.output(frame); }, (n++) * window.__trickle.ms);
      },
    });
  }
};
"""


def test_a_decode_that_trickles_is_not_given_up_on(server):
    """The failure CI actually had, once its log said what the decoder was
    doing: `a=17/20(q=2 configured)` at the moment the chunk was dropped -- two
    frames from done, and the drop threw twenty frames of work away and started
    again from the keyframe. Three times, and the frame was never shown.

    A decode that is still handing frames over is slow, not stuck. The deadline
    is for a chunk that has stopped.
    """
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        pg.add_init_script(TRICKLING_FIRST_DECODER)
        _open(pg, srv, ds_id)
        pg.evaluate("togglePlay()")
        wait_for_player(pg, "() => window.__chunkPlayer.metrics.painted.some((q) => q.frame >= 19)")
        assert pg.evaluate("() => window.__trickle.delivered") > 10, "the trickle never happened"
        assert pg.evaluate("() => window.__chunkPlayer.metrics.retries") == [], (
            "a decode that was still producing frames was given up on"
        )
        assert pg.evaluate("() => window.__chunkPlayer.state().dead") == []


# CI's compositor, injected: the robot tile's WebGL stalled on ReadPixels and
# took requestAnimationFrame down to about one and a half calls a second. The
# transport used to hang off that, so the picture stopped for a reason that had
# nothing to do with the picture.
# Applied *after* the tab is up, not before: the page's own setup uses
# animation frames, so stopping them from the start breaks the thing under
# test rather than isolating it. Stopping them once playback is running is the
# compositor going away, which is the case that matters.
STOP_RAF = (
    "window.__rafStopped = 0; window.requestAnimationFrame = () => { window.__rafStopped++; return 0; };"
)


def test_playback_survives_a_compositor_that_starves_animation_frames(server):
    """CI measured 45 ticks in thirty seconds -- rAF at one and a half calls a
    second, while the robot tile's WebGL stalled the compositor on ReadPixels
    -- with every chunk held and nothing to show for it. The playhead, the
    readiness checks and the fetch plan all hung off that callback.

    The injection here is harsher than CI's on purpose: rAF never calls back at
    all. A throttled one still paints eventually, so it cannot tell whether the
    transport depends on the compositor, which is the property that matters.
    """
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        _open(pg, srv, ds_id)
        # The compositor stops here, with the tab already running.
        pg.evaluate(STOP_RAF)
        pg.evaluate(f"loadAllFrames({KEEP + 25})")
        wait_for_player(
            pg, "(f) => window.__chunkPlayer.metrics.painted.some((q) => q.frame === f)", arg=KEEP + 25
        )
        state = pg.evaluate("() => window.__chunkPlayer.state()")
        assert state["cur"] == KEEP + 25, state
        # And the starvation was real, not something the page routed around.
        assert pg.evaluate("() => window.__rafStopped") > 0


def test_the_player_does_not_spin(server):
    """The transport is driven from two places -- animation frames and a timer
    -- and `tick` re-arms the animation-frame chain. When it re-armed for both
    callers, every timer beat added a chain and they multiplied: 1,131 ticks in
    the first second, 15,402 by the fourth, climbing. Nothing failed, because
    the work per tick is idempotent; it just burned more CPU the longer anyone
    watched, which is the kind of thing a rig session feels and no log shows.
    """
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1200})
        pg = ctx.new_page()
        _open(pg, srv, ds_id)
        pg.evaluate("togglePlay()")
        # What a chain per beat looks like is growth, not a rate: the ticks
        # climbed by an order of magnitude across the seconds that first caught
        # it. So compare the transport against itself over two windows -- a
        # rate in ticks a second would only say how fast this machine is, and a
        # loaded runner reaches the same number honestly.
        ticks = []
        for _ in range(3):
            ticks.append(pg.evaluate("() => window.__chunkPlayer.state().ticks"))
            pg.wait_for_timeout(1500)
        ticks.append(pg.evaluate("() => window.__chunkPlayer.state().ticks"))
        early, late = ticks[1] - ticks[0], ticks[3] - ticks[2]

        assert early > 0, "the transport never ticked: it is not running at all"
        assert late <= early * 3, (
            f"the transport ticked {early} times in the first window and {late} in the last:"
            " the callers are breeding chains"
        )
