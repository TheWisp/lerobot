# Live video transport

How pixels reach a browser, and why the answer should not depend on which tab
you are looking at.

## The distinction everything rests on

There are two reasons this GUI moves images, and they have opposite
requirements:

|                | Who consumes it                  | Tolerates loss? | Bound by                    |
| -------------- | -------------------------------- | --------------- | --------------------------- |
| **Viewing**    | a human, watching                | yes — heavily   | round trips, then bandwidth |
| **Processing** | a model, a mask pass, an encoder | no              | correctness                 |

A human watching a wrist camera cannot tell 500 kbps from 6 Mbps on a 320px
tile. A mask pass at 224×224 that silently received a re-compressed frame
produces a model trained on pixels nobody chose.

**Every degradation in this document applies to the viewing path only.** The
processing path reads through `LeRobotDataset` and the decode pipeline and is
not addressed here, deliberately: the moment a transport decision can reach it,
the decision has to be made on correctness rather than on comfort.

## Where we are

Three viewers, three transports, none of them shared:

| Viewer                 | Transport                                       | Rate                                        |
| ---------------------- | ----------------------------------------------- | ------------------------------------------- |
| Robot page preview     | `GET /api/robot/camera-frame/<i>` per frame     | `setInterval(…, 100)` — 10 req/s per camera |
| Run tab observation    | `GET /api/run/obs-stream/image/<key>` per frame | `setInterval(…, 50)` — 20 req/s per camera  |
| Data viewer, scrubbing | `GET …/frame/<i>` per frame                     | one per scrub position                      |
| Data viewer, playback  | `GET …/video` — one transcoded H.264 clip       | one per episode                             |

Only the last is a video stream, and it exists only for **recorded episodes**.
Every live view is a JPEG-per-frame poll.

A browser-wide selector (`camera-video-mode`: auto / full-quality /
low-bandwidth) already exists and is read by all three viewers, so the
_intent_ is shared. The transport is not: each viewer builds its own URL,
appends its own query parameter, and interprets the mode itself.

## The cost, measured

**Bandwidth.** 60 consecutive frames from
`thewisp/intervention_cylinder_ring_assembly` (1280×720), encoded each way:

| Encoding                            | Total   | Per frame |
| ----------------------------------- | ------- | --------- |
| MJPEG, `low` profile (downscaled)   | 669 KB  | 11.1 KB   |
| MJPEG, `full` profile               | 2606 KB | 43.4 KB   |
| H.264 @ 500 kbps (full resolution)  | 122 KB  | 2.0 KB    |
| H.264 @ 1500 kbps (full resolution) | 490 KB  | 8.2 KB    |

H.264 at 500 kbps delivers **full resolution for one fifth the bytes** of
downscaled MJPEG. The ratio is what inter-frame compression buys: a robot
camera on a fixed mount is nearly static between frames, which is exactly the
redundancy JPEG cannot exploit and H.264 exists to.

**Round trips, which matter more off-LAN.** Four cameras on the Run tab at
`setInterval(…, 50)` is 80 requests per second. On a LAN that is invisible. On
a link with 60 ms RTT it is not a bandwidth problem at all — HTTP/1.1 allows
six connections per origin, so the frame rate is bounded by
`6 / RTT ≈ 100 frames/s` across _all_ cameras, and each frame costs a full
round trip before its first byte arrives. The viewer goes soft long before the
link is saturated.

## What the industry does

For interactive viewing, the field has converged on **one long-lived
connection carrying inter-frame-compressed video**, not per-frame requests:

- **WebRTC** is the default for robot teleoperation — 100–250 ms glass-to-glass,
  NAT traversal built in, native in every browser. It is also the heaviest
  thing to operate: ICE, STUN/TURN, and a peer connection per viewer.
- **fMP4 over MSE** (MediaSource Extensions) plays H.264 in a `<video>` element
  from a single streamed connection. Latency tracks fragment duration, so short
  fragments plus chunked transfer keep it near a second. Far less machinery
  than WebRTC and no signalling.
- **Low-latency HLS/DASH** targets broadcast fan-out; segment-oriented, seconds
  of latency. Wrong shape for one operator watching one robot.
- **MJPEG** remains the simplest thing that works and is what we do today. Its
  cost is precisely the two rows above: no inter-frame compression, and a
  request per frame.

For this GUI — one operator, a handful of viewers, self-hosted, already
running ffmpeg everywhere — **fMP4 over MSE is the proportionate choice**.
WebRTC's advantages are NAT traversal and sub-250 ms latency; the first is
already solved by Tailscale here, and the second is not required to watch a
camera. It should stay on the table for closed-loop teleoperation, which is a
different problem with a different tolerance.

## The gap

What this branch built is right in direction and narrow in reach:

- ✅ A browser-wide viewing-quality selector, read by all three viewers.
- ✅ Server-side H.264 transcode at named profiles, with a bounded clip cache.
- ✅ Quality folded into the frame cache key, so a low-bandwidth frame cannot
  be served where a full-quality one was asked for.
- ❌ **Video transport exists only for recorded episode playback.** Both live
  views — the reason non-LAN hurts — still poll JPEG per frame.
- ❌ **No shared abstraction.** Three viewers each construct their own URL and
  apply the mode themselves. A fourth viewer starts from nothing, and a change
  to the policy has to be made in three places.
- ❌ **The profile is a string threaded by hand** through cache keys, encoder
  calls and query parameters. `profile` appears 39 times in `datasets.py`
  alone; losing it in one call site is a silent quality regression, which is
  exactly the defect this branch's own cache-key commit was fixing.

## Where it should live

One module owning the answer to _"give me pixels for a human to look at"_:

```
gui/video_view.py        # server: named profiles, one encoder entry point,
                         # one cache key derivation, one fragment source
gui/static/video_view.js # client: attach(el, source, {mode}) and nothing else
```

with the three viewers reduced to naming a source and a element. The rules that
are currently spread — which profile "auto" resolves to, whether a mode change
invalidates a cache entry, what a frame costs — become properties of that
module rather than of each call site.

Two invariants worth stating up front, because both have already been violated
once:

1. **The processing path never consults it.** If a code path can feed a model
   or write a dataset, it does not get to ask what the viewer's bandwidth
   setting is. Today this is upheld by convention only.
2. **Quality is part of a cached artefact's identity.** A frame, a clip, and a
   fragment are all keyed by the profile that produced them, or a viewer who
   switches to full quality is served the low-bandwidth copy that is already
   in the cache.

## Not decided here

- Whether live fragments come from a per-camera encoder process or one mosaic
  encoder. The mosaic already exists for the low-bandwidth Run preview and is
  cheaper; per-camera is more flexible and multiplies encoder count by cameras.
- Whether `auto` should measure the link or infer from the request's origin.
  Today it is inferred, which is why a Tailscale address gets the low profile
  and a LAN address does not.
- WebRTC for closed-loop teleoperation, where 100–250 ms and the ability to
  drop frames rather than queue them start to matter.
