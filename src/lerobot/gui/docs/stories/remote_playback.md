# Playback over a WAN

**Story — not implemented.** Phase 1 (asset caching and compression) is
implemented in the same change that adds this file and is not merged yet;
phases 2–4 are a proposal. The measurements are real. See
[stories/README.md](README.md).

---

## Why it matters

The GUI increasingly runs on a robot host that is nowhere near the operator. On
one such host, reached over a direct (non-relayed) tailnet link:

| Measurement                         | Remote             | On the host |
| ----------------------------------- | ------------------ | ----------- |
| ICMP RTT                            | 248 ms             | —           |
| 84-byte API call                    | 486 ms             | 0.5 ms      |
| 1.9 MB transfer, single stream      | 525 KB/s           | —           |
| 1.9 MB transfer, 6 parallel streams | 453 KB/s aggregate | —           |

Two things follow, and they are independent constraints:

- **Latency is a tax on every request.** The server answers in half a
  millisecond; ~99.8% of the wall time is the wire.
- **Bandwidth is a hard ceiling of ~0.5 MB/s.** Six parallel streams moved
  _less_ than one, so the pipe is saturated — no amount of concurrency buys
  throughput. Concurrency can only hide latency.

## What the Data tab does today

`loadAllFrames` issues **one HTTP GET per camera, per frame**, as `img.src`. On
HTTP/1.1 a browser opens ~6 connections per origin, so at 248 ms the ceiling is
~24 images/second — with three cameras, ~8 fps, before a byte of JPEG moves.
Frames are served `no-store`, so scrubbing back re-fetches everything.

## The arithmetic that decides the design

A real dataset on that host (`LandR-66`, three cameras, AV1, 2560x720):

| Camera                           | Bytes/frame on disk |
| -------------------------------- | ------------------- |
| `observation.images.top`         | 85.9 KB             |
| `observation.images.left_wrist`  | 39.3 KB             |
| `observation.images.right_wrist` | 29.6 KB             |
| **total**                        | **155 KB/frame**    |

| Target                     | Needs    | Available |              |
| -------------------------- | -------- | --------- | ------------ |
| 30 fps, video-encoded      | 4.6 MB/s | 0.5 MB/s  | 9x short     |
| 30 fps, re-encoded as JPEG | ~15 MB/s | 0.5 MB/s  | ~30x short   |
| ~3 fps, video-encoded      | 0.5 MB/s | 0.5 MB/s  | at the limit |

**No protocol change makes full-resolution 30 fps work.** Bandwidth is the wall;
latency is the tax. Any design that does not reduce bytes per frame is treating
the symptom.

## What video streaming does that we don't

The comparison is apt, and unflattering: the frames are _already_ AV1 on disk,
and the GUI decodes them server-side to re-encode every frame as JPEG — discarding
inter-frame compression and then paying for it on the wire.

| Technique                         | Relevance here                                                                                                                                                                                |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Adaptive bitrate / renditions** | The decisive one. Streaming never ships the master; it ships the rendition that fits the pipe. 2560x720 -> 640x180 is 16x fewer pixels, ~10x fewer bytes, which puts 15–30 fps inside budget. |
| **Segmented delivery**            | One GET per 2–10 s segment instead of per frame. Kills the RTT tax.                                                                                                                           |
| **Client-side buffering**         | 10–30 s ahead makes latency invisible. Our prefetch window is sized for a LAN.                                                                                                                |
| **Inter-frame compression**       | We have it and throw it away.                                                                                                                                                                 |

## The two constraints that shape any video path

Serving the mp4 to a `<video>` element is the natural endpoint, and two
objections decide its shape.

**Frame-exact sync with action/state.** Resolved by data we already store.
Episode metadata records, per camera, `chunk_index`, `file_index`,
`from_timestamp` and `to_timestamp` — the exact time range of that episode
inside a specific mp4. All cameras share identical ranges, and frame timestamps
on the measured dataset are metronomic (dt = 0.033333 s, jitter 0.000000). So
`frame_index <-> mediaTime` is an exact linear map, and
`requestVideoFrameCallback` reports `mediaTime` per painted frame. This is a
lookup, not an estimate.

**Effects (SAM and friends) are applied to frames.** This is the real
constraint, and it argues for a change rather than against the design:
**composite masks in the browser, not on the server.** The overlay worker
already produces masks; a mask shipped as RLE or a small alpha PNG is far
smaller than the frame it covers. That _reduces_ bytes and decouples overlays
from frame re-encoding.

## Proposed shape

**Video is a playback fast-path; JPEG stays the authority.**

- **Playing** — `<video>` over byte ranges, browser decodes AV1, masks
  composited client-side.
- **Paused, scrubbing, editing, pixel-exact inspection** — the existing
  per-frame endpoint, for that one index.

Every current capability survives, and it ships incrementally.

## Phases

1. **Asset caching and compression.** _In this change_ — `gui/static_assets.py`.
   Immutable `Cache-Control` for vendored meshes and libraries, selective gzip
   that skips already-compressed payloads and never buffers SSE.
2. **Bytes per frame.** Expose `max_width` and `quality` on the frame endpoint
   (`encode_frame_to_jpeg` already takes `quality`, hardcoded to 85; the route
   exposes neither). The single highest-leverage change, and the only quick win
   that attacks bandwidth.
3. **Round trips.** Version-keyed frame URLs so scrub-back hits the browser
   cache; a prefetch window sized for the measured RTT; optionally the existing
   `/frames` batch endpoint as the prefetch transport.
4. **Video fast-path.** As above.

## Rejected, and why

- **The `/frames` batch endpoint as the playback path.** It works — pinned by
  `tests/gui/test_frames_batch_endpoint.py` — but it batches over _time_, not
  over _cameras_, so playback still issues one request per camera. It also
  inflates payloads >30% with base64, and decodes serially with no streaming,
  so a cold 100-frame batch returns nothing until the last frame is done. Useful
  as prefetch, wrong as the playback path.
- **More parallel connections / HTTP-2 multiplexing alone.** Measured: six
  streams moved less than one. Concurrency hides latency; it cannot create
  bandwidth.
