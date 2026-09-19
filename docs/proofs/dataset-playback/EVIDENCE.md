# Dataset video playback, measured

What the chunk prototype on `feat/dataset-video-playback` does on this
workstation and over the tailnet link, against the targets in `src/lerobot/gui/docs/dataset_playback.md`.
Every number is read from `measurements/local.json`, written by the run itself;
nothing is quoted from a panel. Dated 2026-09-11, on the head that composites
through `mask_composite.js` (load 1.49 at the start of the run).

**Workload** — synthetic and throwaway, shaped like the rig's labelled dataset:
four cameras, two at 1280×720 and two at 960×600, 30 fps, one 12 s episode of
scrolling noise (the encoder's worst case, so bytes and build times are
pessimistic), two cameras carrying a moving-disc mask. The profile is the
design's starting point: 320-pixel target width, H.264 constant quality 26,
`veryfast`, 2 s chunks. Nothing of the operator's was read or written.

**Condition** — Local: server and browser on the same workstation (32 cores,
an RTX 5090 that this path does not use), headless Chromium.

| Requirement                  | Target (Local)                                             | Measured                                                                     | Met               |
| ---------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------- |
| R1, keeps time at 2×         | media time within 5% of 2× wall time, no stall over 500 ms | 9.97 s of media in 5.01 s of wall time, ratio 0.994; no stalls; 290 repaints | yes               |
| R2, first picture after open | ≤ 300 ms                                                   | 501 ms cold (497 by the page's clock)                                        | no                |
| R3, a seek paints the target | ≤ 150 ms                                                   | 402 ms cold into an unheld chunk; 13 ms warm into held media                 | no cold, yes warm |

**Why R2 and R3 miss, and by what.** Nothing paints before the first chunk is
built, and a 2 s chunk of four cameras took 377–601 ms to build on this content
(five misses, one hit at 1 ms). The page's own share is small: the decode
metric read 2.1–2.3 ms per frame per camera (main-thread time from a chunk's
first decode call to its last frame, so it includes the paints and composites
queued in between). So the cold latencies are the build time of
the first chunk, which is what the design's open measurement of the chunk
length is about: a shorter first chunk, or a faster preset for it, is the lever.
The reference branch built a 2 s window of four cameras in 161 ms on real
footage on the rig; scrolling noise is a harder encode than a recorded scene.

**Bytes.** 73.2 kB per second of media for four cameras with two mask tracks,
over six chunks — under the 86 kB/s the reference branch measured at the same
width on the rig's dataset, on content that should cost more. Masks are in that
figure.

**Errors.** None recorded by the player.

**What the composite costs.** `mask_composite.js` on one frame at the encoded
size (320×180, a disc mask of radius 37 — a 720p disc of radius 90 scaled — in
node's V8, 2026-09-11): tint 0.53 ms, solid 0.77 ms, random 0.80 ms, a random
background under a tinted object 2.4 ms, no treated region 0.01 ms, and blur
at strength 12 (sigma 3 at this size) 6.0 ms. The blur is the full-frame
Gaussian the library computes before cropping; at 2× with a blur on every
camera that alone is 4 × 60 × 6 ms of main-thread time per second, which the
frame budget does not have. Cropping the blur to the region's support, exact
for every pixel inside it, is the lever if a blur recipe is played at speed;
the tint recipe measured above keeps time.

## Over the link

Browser on this workstation against the GUI served by fc500t over the tailnet
(HTTPS, the tailnet's certificate), on the throwaway `claude_e2e/sam_e2e` on
the rig: three cameras, two at 960×600 and one at 2560×720, all three masked,
30 fps, 528 frames. Numbers from `measurements/link.json`, written by the run;
the run below is the fixed head (tiles at the camera's size, chrome by the mask
layer, one texture per episode) on a quiet workstation. Dated 2026-09-11.

**The link that day.** Direct IPv6 path, round trip 238–264 ms. Raw capacity for
a 3 MB stream over ssh: 189 kB/s on one run and 447 kB/s on the next. A ~90 kB
chunk fetched with curl: 54–61 kB/s with 0.75 s to the first byte. Within one
run of the page, chunks the server answered from its cache in 2–5 ms took
between 0.3 s and 9.6 s to arrive.

| Requirement                  | Target (Link)                   | Measured                                                                             | Met |
| ---------------------------- | ------------------------------- | ------------------------------------------------------------------------------------ | --- |
| R2, first picture after open | ≤ 2 s                           | 597 ms                                                                               | yes |
| R3, a seek paints the target | ≤ 1.5 s                         | 518 ms cold, 5 ms warm                                                               | yes |
| R1, keeps time at 1×         | within 5%, no stall over 500 ms | ratio 0.833; no single hold over 50 ms, the shortfall spread across chunk boundaries | no  |
| R1, keeps time at 2×         | within 5%, no stall over 500 ms | ratio 0.166; one stall of 11.1 s                                                     | no  |

Bytes: 49.6 kB per second of media for three cameras with three mask tracks.
The page's own cost per frame per camera, decode and paint together: 2.7–4.1 ms,
of which the chrome drawn at the cameras' declared sizes is now part.

**What this says.** On this link and day a fixed 320-pixel profile in 2 s chunks
meets the open and seek targets and does not keep time at either speed. The
mechanism is on record twice over: the bytes needed are steady at ~50 kB per
second of media, and the per-transfer rate for chunk-sized objects swung
thirtyfold within a minute while a 3 MB stream got 189–447 kB/s. A transfer
the size of a chunk is bounded by slow start and by whatever the path does
minute to minute, not by the link's capacity; a fixed profile cannot follow
that, which the design lists as the cost of not adapting. The operator's own
session on a four-camera dataset, from the rig's log, needed 158–210 kB per
2 s chunk — about 105 kB per second of media, twice this workload — and its
consecutive chunk requests arrived 13 s apart.

Not in the table: two runs on this head taken while this workstation was
loaded by test suites and commit hooks (load 5.7 and 7.6), which inflated the
page's decode figure two- to fivefold; and one earlier run on the previous
head. The emulated-link suite in
`tests/gui/test_low_bandwidth_smoothness_playwright.py` shows the fetcher
keeping time through fifteen chunks at 1× on a 250 ms link carrying 1.5× the
content's bytes, and at 2× on 3×, so what the tailnet run measures is the
link, not the fetcher.

## The composite, one definition

`static/mask_composite.js` is `overlays/effects.py` mirrored, arithmetic
included; `tests/gui/test_mask_composite_equivalence.py` runs the same 64×48
frame, masks, recipe and noise through both — nine recipes (tint at the default
and at a custom colour, solid, random, blur, none, a treated background under a
tinted object, a random background, two overlapping objects) at feather 0 and
5 — and requires zero difference on every pixel. Dated 2026-09-11. A float
Gaussian in place of cv2's 8-bit fixed-point one left two levels on the seams
and in the blur; mirroring the 1/256-quantized, error-diffused kernel and the
half-up rounding removed them. Blending in double in place of float32 moves one
pixel in 180 k on random input (measured against `cv2.blendLinear`), so the
blend is pinned on a triple where the two differ (2, 210, 236/255 → 194, not
195).

Mutations put back, each seen by a named test: the blend in double, the kernel
centre rounded instead of taking the residual, no error diffusion, a float
kernel, truncation instead of half-up rounding, the feather alpha in float, the
background alpha not 1 − union, overlaps to the later-listed mask, tint rounded
half up, chrome ignoring whether the camera has masks, chrome ignoring hidden
labels, the Low Bandwidth chrome bypassing `chromeOptions`, the player
compositing muted rows, the player applying its own enabled rule instead of the
layer's.

## The live overlay at production sizes

`tests/gui/test_low_bandwidth_overlay_stream_playwright.py`: two cameras at
1280×720 and 960×600, 30 fps, 120 frames, a stored disc mask with a tint
recipe, the real stream endpoint, ffmpeg, fragmented MP4 and MediaSource, the
SAM3 worker faked at the endpoint's seams, at a 2560×1400 viewport where the
tiles exceed the stream's atlas rects; every assertion runs in both modes.

Before the fix, measured 2026-09-11 on the JPEG path: the head camera's base
picture occupied 916×515 at (380, 305) and the stream canvas 640×360 at
(518, 383) — the atlas rect, centred — and the same rule applied at Low
Bandwidth. The base stayed visible under the stream in both modes, and at Low
Bandwidth the chunk player kept painting under it (31 to 54 paints in 0.8 s).
After: the stream canvas takes the base's rectangle and follows a resize, the
base and the still overlay's last PNG are hidden while it plays, the chunk
player pauses and lands on the frame the stream reached, and stored chrome is
not drawn while the live layer owns the tiles, whichever path asks for it.
Mutations put back, each red: `drawCamera` painting under the live layer, the
base left visible, the stream canvas left at the atlas size under the original
rule (both modes), the still PNG left under the stream, the chunk player
running under the stream, the still play loop fetching under the stream.

## A save burst, a chunk that never becomes ready, and what the log says

Reported from the rig, 2026-09-11 19:09 in its log: three treatment saves on
`eval_ball_0818_5` at Low Bandwidth, two of them 0.9 s apart, each dropping
the server's chunk cache twice (the save's own invalidation, then the
metadata-change reload the page's refresh triggers); after the last, the page
re-requested one chunk (19:09:40.6 miss, 19:09:43.5 hit) and made no request
for sixteen seconds until the episode was re-selected. The server log could
not say why: the page's side was in no log.

`tests/gui/test_low_bandwidth_edit_storm_playwright.py`, at the rig's three
cameras (1280×720 and two 960×600), 30 fps, a stored mask with a recipe, the
real edits pipeline driven through the inspector's own buttons, under an
emulated 250 ms / 150 kB/s link: two background saves 0.9 s apart while
playing recover here (a 3.2–3.5 s hold at the rebuild, then three more
seconds of media; 18 decoders made and closed, at most 6 alive), so the
sequence alone is not the wedge on this machine and link. What the code
shows is the shape that wedges: a chunk is registered as held before its
decoders run, and a held chunk that never becomes ready — its frames never
arrive from the decoder, its masks never decode — was held forever, because
the plan, seeing it held, never asked again. Reproduced by serving one chunk
once with its video bytes zeroed: the tile held on it with nothing in any log.

Now: a held chunk not ready 4 s after it arrived is dropped, its decoders
closed, and asked for again, up to three times, then reported as an error; a
seek keeps the transfers the plan from the new frame would ask for again,
where before every save's re-seek aborted the next chunk's transfer (the
miss-then-hit pairs 100–400 ms apart in the rig log); every chunk request
carries the player's state (`X-Player: cur= held= inflight= painted= holds=
errors=`) and the server's chunk line echoes it; the player's events —
never-ready, decoder and fetch errors, rule violations — are posted to
`player-event` and logged as warnings. Mutations put back, each red: no
give-up on a never-ready chunk, the dropped chunk's decoders left running,
events not posted, requests without the state, a seek aborting everything,
the chunk line without the state.

The CI failures on this branch had one cause, found the same way: the suites
start the real app in-process, one server per module, and under
`pytest -n auto` a worker runs many modules; the app's shutdown hook closed
the module-level decode pool, so every Full Quality frame request in a later
module raised "cannot schedule new futures after shutdown" — 74 tracebacks
in one run, every full-quality tile a timeout — and the test server's stop
returned before that shutdown ran. Startup now recreates shut-down pools and
the test server waits for its shutdown; `tests/gui/test_gui_server_restarts_in_process.py`
pins both.

## The rig's frame 15: four cameras, a chunk that decodes part way

Two operators on fc500t, 2026-09-11 22:23, two different four-camera datasets,
the same stop. The whole sequence is in the server log, because the page's
state now rides on every chunk request:

```
22:23:12.597  start=0   frames=60 miss 112898 B 172 ms cameras=4  player cur=0  held=-          painted=0
22:23:12.859  start=180 frames=60 miss 110976 B 141 ms cameras=4  player cur=0  held=0,60       painted=1  holds=1
22:23:17.859  start=240 frames=60 miss 112801 B 150 ms cameras=4  player cur=1  held=0,60,120,180 painted=1
22:23:17.964  start=0   frames=60 hit  112898 B   1 ms cameras=4  player cur=15 held=60,120,180 painted=15
...  (every 4 s, for the next hour)
23:27:55  client error: chunk 0 never ready after 3 tries; held 25463 ms without a frame
          [cur=15 held=60,120,180 inflight=- painted=15 holds=1 errors=36]
22:26:37  client error: chunk 60 observation.images.left_wrist decode:
          Codec reclaimed due to inactivity.          (×16: four cameras of four chunks)
```

Three things in it. The page painted 15 of chunk 0's 60 frames and held at the
sixteenth, with chunks 60, 120 and 180 decoded and waiting behind it. It spent
its three-try budget and then asked for the same chunk every four seconds for
an hour, reporting each time that it had already given up. And Chrome was
reclaiming sixteen decoders — one per camera per buffered chunk — which is
where the frames that never arrived went.

`tests/gui/test_low_bandwidth_recovery_playwright.py` reproduces it at the
rig's shape (four cameras, multi-resolution, six chunks per episode) by
serving a chunk 0 whose first camera's encoded bytes stop after frame 15 —
built from the server's own body, so the header, the other three cameras and
the masks are real — for every request of that chunk. Before the fix, as on
the rig: _playback stopped at frame 14 ... (asked 12 times)_ against a budget
of three. A first attempt that stalled one `VideoDecoder` instead passed,
because the retry then decoded cleanly; the rig's failure is permanent, so
the repro is too.

It was not only the four-camera dataset: the same wedge hit two three-camera
ones the same evening (`eval/eval_ball_0818_5`, once at `cur=887` with chunks
900 and 960 held behind it), always on the chunk under the playhead. Fifteen
frames of sixty is the shape of a decoder output pool running out rather than
of a slow decode -- a `VideoFrame` is a decoder output buffer, not a picture
in memory, and this player holds a whole chunk of them per camera, where a
hardware decoder's pool is far smaller. That hypothesis was right about the
buffers and wrong about the pool: the Media domain reports
`kIsPlatformVideoDecoder=false`, so no platform decoder is in use on these
machines and no hardware pool was ever involved. What the decoders were
starving on was their own output buffers, which the page held and did not give
back -- fixed by caching a picture of each frame and closing the frame at once.

Decoding one chunk at a time was tried on the pool theory and reverted: it
delayed every chunk behind the one decoding, carried a lost wakeup that could
strand a chunk undecoded, and addressed a cause that does not exist. Decoders
are opened per camera per chunk being decoded and closed when the flush
settles.

The rest is the consequence, fixed for any cause: a spent budget marks the
chunk dead,
says which frames are gone, steps playback to the next chunk and stops asking;
scrubbing back into a dead range asks for it again. The quality selector keeps
the playhead, which it did not — switching to look at the same frame at full
quality restarted the episode. Mutations put back, each red: the plan asking
for a dead chunk again, a scrub that does not clear it, the per-tick pose
fetch in the URDF tile.

## What the page holds, counted on the rig

The frame counter added for that question answered it within the hour. A
four-camera episode playing at Low Bandwidth on fc500t, 2026-09-12 00:53:

```
sam_e2e  (3 cameras)  player cur=301 held=180,240,300,360   frames=720/720   painted=11
GPU/...  (4 cameras)  player cur=5   held=0,120,180,240,300 frames=1200/1200 painted=21
```

1200 decoded frames alive at once -- four cameras times five buffered chunks
times sixty frames -- and 720 for the three-camera set, with live equal to
peak, so the buffer sits there. A `VideoFrame` is a decoder output buffer,
not a picture in memory, and a decoder stalls when its client does not give
them back; a hardware decoder's pool is a handful. That is the resource the
rig's fifteen-frame stall was starving on.

Fixed by caching a picture of each frame -- an `ImageBitmap` that owns its
pixels -- and closing the `VideoFrame` at once, so the decoder gets its
buffers back (Zhang Xinyu, `tests/gui/chunk_frame_lifetime.test.js`, which
drives the real player against a decoder with a fifteen-frame pool and fails
on the code above with fifteen frames held). Note that `frames=` now counts
cached pictures, not decoder buffers: the same 1200 is memory rather than a
stall, so a high count no longer means what it meant here.

The session measured was playing without a stall at 1200, so the number is
the pressure and not the failure: what varies between machines is the pool,
which is why the wedge follows the browser rather than the dataset. The fix
this points at -- hold the chunk's bytes, which are cheap, and decode a
window around the playhead rather than every buffered chunk -- is not in this
branch.

## Four rounds on one stalled decode, and what ended them

A seek straight into an episode's last chunk failed on CI and nowhere else --
not on the rig, not on this workstation, not pinned to two cores. Four rounds
produced four theories (a decoder pool running out, the dead-chunk skip, the
memory the cached pictures hold, a decoder gone silent), each consistent with
the evidence available, which was `TimeoutError: 30000ms exceeded`.

The evidence was wrong twice over. A give-up reported the chunk _after_
tearing it down, so every stall read `q=0 closed` -- the report describing its
own doing. And a decoder that fails fatally is closed by the browser, so its
`flush()` rejects on a closed decoder, which the catch skipped: the one event
that explains such a stall was the one the page refused to log.

With both fixed, one round answered it:

```
held  8685 ms  [a=1/20(q=18 configured)  b=1/20(q=18 configured)  masks=1 pending]
held  8314 ms  [a=17/20(q=2 configured)  b=18/20(q=1 configured)  masks=ready]
held 39008 ms  [a=5/20(q=14 configured)  b=5/20(q=14 configured)  masks=ready]
```

Configured, queue draining, and the second attempt two frames of twenty from
done when the deadline dropped it. Dropping a chunk discards its decode and
starts again from the keyframe, so on a machine slow enough to need twelve
seconds for twenty frames the retries only lose ground. The deadline was the
failure; it no longer applies to a decode that is handing frames over
(`tests/gui/test_low_bandwidth_recovery_playwright.py`, a decoder delivering a
frame every 600 ms).

The instrument also settled a question four rounds of guessing could not: the
`Media` domain reports `kVideoDecoderName=FFmpegVideoDecoder` and
`kIsPlatformVideoDecoder=false`, so no hardware decoder pool is involved on
these machines at all -- which was the first theory, and the one that cost a
regression.

## The URDF tile, and what a smooth picture exposed

The tile fetched one pose per playhead tick unless the trajectory toggle was
on, which caps it at one serialized round trip. Invisible while the JPEG path
painted 2.8 fps; plain at Low Bandwidth. Measured 2026-09-11 on this
workstation, four cameras at 1280×720, 30 fps, over loopback:

| path                      | tab paints | tile updates | request duration |
| ------------------------- | ---------- | ------------ | ---------------- |
| Full Quality, toggle off  | 2.8 fps    | 2.6 /s       | 6.1 ms           |
| Low Bandwidth, toggle off | 20.1 fps   | 2.2 /s       | 3.9 ms           |
| either, toggle on         | 18–20 fps  | 0 /s         | —                |

The endpoint costs 1.7 ms for one frame and 5.8 ms for a whole 438-frame
episode, so neither the server nor the link was the limit. The pose cache is
now filled once per episode whatever the toggle says;
`tests/gui/test_urdf_tile_follows_playhead_playwright.py` pins no per-frame
request, one whole-episode fetch, and the tile landing on the frame asked for.
In the same runs the JPEG path blocked the renderer's main thread for 8053 ms
of 8270 (67 long tasks, median 114 ms); Low Bandwidth had none.

## Captures

Taken on the test fixture (two cameras of different resolutions, 35 frames,
frame index carried as a band strip, a disc mask on camera `a`), not on the
measurement workload; they show the states, not the numbers.

- `states/1-selector-low-bandwidth.png` — the Camera Video selector beside the
  speed selector, set to Low Bandwidth, with the readout at frame 13 of 35.
- `states/2-tiles-from-chunks-with-mask.png` — both tiles painted from a chunk
  at the cameras' declared sizes: camera `a` (encoded at 320 wide, drawn at 480) with its tint in the pixels and the mask's outline and name over it,
  camera `b` narrower than the target and left at its own resolution.
- `states/4-tiles-jpeg-path-same-geometry.png` — the same tiles on the JPEG
  path, for the geometry the video tiles must match.
- `states/3-selector-full-quality.png` — the same controls at Full Quality, the
  JPEG path.

## Reproducing

The measurement script is not checked in (a single-use driver); the tests in
`tests/gui/test_data_tab_chunk_playback_playwright.py` drive the same tab the
same way, `tests/gui/chunk_fixtures.py` builds the fixture the captures were
taken on, and `tests/gui/test_low_bandwidth_overlay_stream_playwright.py`
builds the production-sized one.
