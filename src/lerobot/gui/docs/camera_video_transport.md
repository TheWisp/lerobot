# Camera video transport

How camera pixels reach the browser, for the three surfaces that show them:
the Data tab (stored episodes), the Run tab (live teleop and inference) and
the Robot tab (camera preview). A design under review, not a description of
shipped behaviour.

## The problem

Polling JPEG stills costs a full picture per request, so the pixels have to
be encoded as video at least once somewhere. The design question is how to
encode smartly:

- <a name="r1"></a>**R1. Little added latency where latency is the
  constraint.** Teleop. The branch's 0.4 s at low quality is accepted as
  workable; the design does not worsen it and takes back what is free.
- <a name="r2"></a>**R2. Little added compute where compute is the
  bottleneck.** The same host runs the policy, the recorder and,
  between runs, training and mask jobs.
- <a name="r3"></a>**R3. No duplicate work across consumers, or as little
  as possible.** Several viewers of one camera; several readers of one
  frame (policy, recorder, viewer); several readers of one file.
- <a name="r4"></a>**R4. Platform and hardware differences stay out of the
  architecture.** A host may have NVENC, VideoToolbox, or no GPU encoder
  at all.

Two things fix the setting. The stored side must scrub to any frame, play
at 2x and show the saved masks composited in; an upfront delay is
acceptable there. The Robot tab (camera preview, no run active) should ride
the same path as the Run tab so there is one thing to maintain.

**The proposal, in one paragraph.** Live cameras are encoded once per
camera and [profile](#g-profile) into raw H.264 frames ([Annex B](#g-annex-b),
no container) that each carry their capture time, and the same stream is
sent to every viewer. Stored episodes are transcoded once per camera,
profile and mask [recipe](#g-recipe) into a cached clip the browser plays
from a file, which is what the current branch already does. The policy, the
recorder and the jobs are not touched: the view reads what the run loop
already publishes into shared memory (the [tap](#g-tap)) and yields every
shared resource to them. Each stage that can use a GPU has a software
fallback resolved once per host. Part 2 draws this out; Part 1 is the
argument, one section per question above, with the evidence in the
appendix.

Two branches are named. `main` is what users run today.
`feat/camera-video-transport` is the branch this document lives on; it
carries a first implementation of streamed video for two of the three
surfaces, written before this design. Every R, Q, C or A mentioned is a
link to where it is stated. Terms with a fixed meaning are in the
[glossary](#glossary) and link to it at first use.

## Part 1: The four questions

### Q1. Where does the current path spend its latency, and what is free to take back?

<a name="q1"></a>

The branch's Run tab streams one H.264 mosaic per viewer over Tailscale at
1.18 Mbit/s; the picture is 0.4 s old at the median and 0.60 s at the 95th
percentile ([A1](#a1)). That was measured with the network round trip at
72 ms on the day of the measurement; the same Tailscale link to the rig
measured 237 ms by ping on 2026-09-06, and a fresh HTTP request to the GUI
took 0.48 s to first byte, so how much of the 0.4 s is the link's depends
on the day, and on today's link it is most of it. The pipeline's own terms,
in frame periods of 100 ms at the branch's 10 fps:

- Encoding: 1–5 ms on the CPU or the GPU ([A6](#a6)). Not a term.
- The 10 fps sampler: up to one period between a frame's capture and its
  pickup. The price of the frame rate, which buys bandwidth. A knob.
- The fragmented-MP4 container: one period. ffmpeg's MP4 muxer computes a
  frame's duration from the next frame's timestamp rather than from the
  declared rate, so it holds each frame until the next arrives: 102.5 ms
  per frame measured, against 2.3 ms for the same encoder writing raw
  H.264 at the same bitrate. Whether another MP4 option avoids the hold
  was not measured. Free to take back by changing the wire format.
- NVENC's default lookahead: two more periods when the GPU encoder runs
  with default settings. Free to take back with the encoder's low-latency
  setting.
- The player: [MSE](#g-mse) plays what it has been fed as it would a file,
  so delay accumulates after a burst, and the branch corrects it with a
  seek towards the live edge past 0.6 s. A receive buffer sized to the
  link's jitter buys smoothness; whatever the player holds beyond that is
  free to take back.

<a name="c1"></a>**Conclusion C1.** None of this is a reason on its own to
change the live path: 0.4 s at low quality is workable, and on today's link
the network is the larger part of what the operator waits for. It is a reason not
to lose the free terms when the path changes for the other three
questions. So the live encoder emits raw Annex B at the source's frame rate
with each encoder's low-latency flag applied inside the stage, the frame
rate stays a knob in the profile, and the player buffers for jitter and
nothing more. Each frame carries its capture time so that the picture's age
is measurable in the browser rather than by a separate reporting round
trip; whether the readouts beside the picture should also be shown at that
time is open ([Part 3](#part-3)).

### Q2. Where does compute go, and what is the bottleneck on each host?

<a name="q2"></a>

On the rig the recorder encodes every camera in real time on the CPU
(SVT-AV1) or on NVENC when `vcodec=auto` finds it; the branch's preview
encoders are libx264 pinned to one thread per viewer on the same CPU; and
the 5090 allows eight concurrent NVENC sessions ([A5](#a5), [A6](#a6)).
Between runs the same GPU runs training and the mask apply run, which
decode on NVDEC with the [aux-GPU slot](#g-aux-slot) as the only
arbitration. Encoder contention with a loaded policy and the recorder
running at once has not been measured and is the first measurement to make
([A8](#a8)).

<a name="c2"></a>**Conclusion C2.** The view is the consumer that yields:
it takes NVENC or NVDEC when free, drops to the software encoder when not,
and never holds a resource a real-time or job consumer needs. Each stage
resolves its backend once per host with the `auto`/`cpu`/`gpu` knob the
codebase already uses three times ([A8](#a8)); the budget of NVENC sessions
is written down per deployment (recorder cameras first, then previews by
profile).

### Q3. Who else reads the same pixels, and where is work duplicated?

<a name="q3"></a>

Ten consumers read the cameras or the files ([A5](#a5), table). By whether
a frame may be dropped they fall into three [classes](#g-class): real time
(policy, recorder: never), job (training, the mask [apply run](#g-apply-run):
never, wall time free), view (every preview and playback: yes). During a
[run](#g-run) the [run loop](#g-run-loop) already publishes each processed
observation into the tap, a memory write that waits for nothing ([A2](#a2),
[A3](#a3)); HVLA's [S2](#g-s1-s2) process receives the same pixels through a second
family of segments written by the same step ([A4](#a4)). Duplicated today:
one encoder process per open browser tab for the same frames; five
encoders and three stored decoders across the consumers; two
implementations of the shared-memory block ([A5](#a5)). Shared already,
with an equivalence test: the mask [compositor](#g-compositor), the GPU
decoder, the dataset reader.

<a name="c3"></a>**Conclusion C3.** Share the operation, never the
lifecycle (the rule `stereo.py` already states): consumers of different
classes may share a function, a model or a file format, never a queue, a
thread or a device handle, because a queue with a consumer that may drop
and one that may not eventually drops the wrong frame. So: one live encode
per (camera, profile) fanned out to every viewer; one stored transcode per
(episode, camera, profile, recipe) cached and shared; the view's decoder is
the dataset reader and its compositor is the one definition; the view's
only channel into the run is the tap, and nothing is added to the loop.
The S2 double copy and the two block classes are a policy-side refactor
(one block primitive, lifecycles kept apart), not this design's work.

### Q4. What differs by platform, and how does it stay out of the architecture?

<a name="q4"></a>

Encoders: NVENC on the rigs, VideoToolbox on a Mac, libx264 everywhere.
Decoders: NVDEC, or torchcodec on the CPU. Segmentation: CUDA, MPS or CPU.
The browser decodes H.264 in hardware on every platform, which AV1 cannot
promise ([A8](#a8)). The codebase resolves such differences three times
already, the same way: one interface, a software reference, accelerated
backends resolved once per host and logged, a knob whose forced mode
refuses rather than degrades. Each class tolerates a slower backend along
one axis: real time in frame rate, jobs in wall time, views in the wait
before first play.

<a name="c4"></a>**Conclusion C4.** Every view [stage](#g-stage) (decode,
composite, segment, encode) takes that pattern, and nothing past a stage
may see which [backend](#g-backend) ran: the [wire format](#g-wire-format)
is H.264 Annex B whichever encoder produced it, a profile is a resolution
and bitrate rather than an encoder preset, and the cache identity carries
no backend name. A host without a GPU runs the same architecture slower
along each class's own axis.

## Part 2: Proposed architecture

```mermaid
flowchart LR
  subgraph sources[Sources]
    dev["capture backends<br/>GUI-owned, no run"]
    shm["tap<br/>run-owned, or the data publisher"]
    vid[(video files + parquet ts)]
    masks[(saved masks + recipe)]
  end
  subgraph server[GUI server, stages with a resolved backend each]
    enc["encode once per source x profile<br/>resolved encoder, H.264 Annex B out"]
    xc["transcode + composite<br/>dataset reader, compositor, encoder"]
    seg["overlay worker<br/>SAM3 sidecar"]
  end
  dev -->|frame + capture ts| enc
  shm -->|frame + capture ts| enc
  shm -.->|skips frames| seg
  vid --> xc
  masks --> xc
  xc --> cache[(playback cache)]
  enc --> fan[fan-out to N viewers]
  fan -->|units + ts| live["live presenter<br/>follow-live cursor"]
  cache -->|file range requests| stored["stored presenter<br/>paced cursor, seek"]
  seg -.->|mask + ts| live
  live --> sync["sync by capture ts<br/>state, actions, URDF"]
```

**[Source adapters](#g-source-adapter)** ([C3](#c3)). One per origin, producing pixels plus
capture time at the source's own frame rate and owning nothing else: the
tap reader (Run tab, the run holds the cameras), the
[capture backend](#g-capture-backend) (Robot tab, the GUI holds them), the
dataset reader (Data tab). The first two produce the same form, so the rest
of the pipeline cannot tell which it is watching, which is what lets the
Robot tab ride the Run tab's path. The tap reader is the only place that
knows the tap's names, [sweeps](#g-sweep) and re-attach logic ([A3](#a3)).

**Encode once, fan out** ([C1](#c1), [C2](#c2), [C3](#c3)). A live source
is encoded once per (camera, profile) into H.264 Annex B
[units](#g-unit) carrying their capture time, and the same units go to
every viewer of that profile: viewers cost bandwidth, not encoder time. The
encoder is resolved per host and its low-latency flags stay inside the
stage. Stored video does not use the live encoder: it is a
[transcode](#g-transcode) whose product is a file in the
[playback cache](#g-playback-cache), the branch's Data-tab path unchanged.

**[Presenter](#g-presenter)** ([C1](#c1)). One client component that decodes units and
paints the newest one. Live surfaces follow live: newest frame, older ones
dropped, buffered only as far as the transport's jitter buffer asks.
Stored surfaces are paced by the playback rate, seekable, buffered freely.
Overlays are separate streams keyed by the same timestamps, painted over
the matching frame or dropped if late.

**Profiles** ([C4](#c4)). `low`, `medium`, `full` for live and stored,
chosen by the viewer: a resolution, a bitrate and, for live, a frame rate.
Part of a stream's or cache entry's identity; never consulted by a job.

**Backend resolution** ([C2](#c2), [C4](#c4)). Each stage: one interface,
a software reference, accelerated backends resolved once per host and
logged, the three-way knob. Today NVENC for live encode (floor libx264 at
2–5 ms per frame), NVDEC and CUDA for the jobs, the CPU path for the Data
tab until a measured need ([A8](#a8)).

**When the [robot host](#g-robot-host) and the [GUI server](#g-gui-server) split**, the live source adapters
and the encode stage move to the robot host, where the frames and the tap
originate, and the GUI server forwards the units. Nothing else changes,
which is why timestamps travel with the bytes from the start.

### Invariants

- One encode per (camera, profile), whatever the number of viewers; one
  transcode per (episode, camera, profile, recipe). ([C3](#c3))
- Processing never consults viewer settings; a recipe change is a new cache
  entry, never an invalidation. ([C3](#c3))
- Every frame carries its capture timestamp end to end. ([C1](#c1))
- Live buffers only for the link's jitter, never a fixed amount on top.
  ([C1](#c1))
- Overlays skip, never stall.
- Nothing is added to the run loop; no second channel into the run process.
  ([C3](#c3))
- The view yields every shared accelerator to real-time and job consumers.
  ([C2](#c2))
- No backend is visible past its stage. ([C4](#c4))

The teleop latency budget, term by term, is in [A9](#a9).

## Part 3: Open decisions

<a name="part-3"></a>

1. **Live transport.** _[WebRTC](#g-webrtc)_: the browser's [jitter buffer](#g-jitter-buffer)
   and hardware decode, works on plain http, needs a WebRTC peer on the
   server (aiortc or a small native relay). _MSE with in-browser remux_:
   JavaScript wraps the Annex B units into fragmented MP4 in the browser,
   removing the server-side container hold but keeping a buffered player.
   _[WebCodecs](#g-webcodecs) over WebSocket_: simplest sync, lowest
   latency, needs HTTPS ([A6](#a6)). Recommendation: measure WebRTC and
   WebCodecs-over-HTTPS side by side with the same source before choosing.
2. **Per-camera streams or a mosaic.** Per-camera costs more encoder
   sessions but lets the client lay out, pick and enlarge, and needs no
   per-robot layout table (the branch's only matches one robot's camera
   names, [A1](#a1c)). Recommendation: per-camera; the session budget
   allows it.
3. **Should the readouts beside the live picture show the robot at the
   picture's time?** Today the state, action and URDF readouts show the
   newest values (one request old) while the picture is a video pipeline
   old ([A1](#a1c)). With capture times on the stream the client could show
   the state recorded at the painted frame's time, as the Data tab does by
   frame index. Not confirmed as an operator need; if not needed, the
   timestamps serve only to measure age.
4. **Stored AV1: play directly or transcode.** Chrome and Firefox decode
   AV1, and `full` is already a re-wrap. Unverified: seeking with a
   keyframe every two frames, and hardware AV1 decode on the laptops in
   use.
5. **How `auto` picks a profile.** A manual selector exists; an RTT or
   throughput probe could choose. Not decided.

## Part 4: Order of work

Review happens on this file, in a draft PR on `feat/camera-video-transport`,
with line comments; decisions are written back here. Each step is a PR
small enough to read:

1. Capture times end to end on the existing branch path, and age measured
   in the browser. No transport change.
2. The live transport spike: WebRTC and WebCodecs-over-HTTPS, same source,
   same profile, source-to-display age measured the same way as on the
   branch. Pick one.
3. Per-camera streams, encode-once fan-out, Annex B from the resolved
   encoder. Delete the mosaic layout table.
4. Robot tab onto the same path.
5. Stored AV1 direct-play experiment for `full`.

## Glossary

Terms with a fixed meaning in this document, in alphabetical order.

<a name="g-age"></a>**Source-to-display age** — the time between a frame's
capture by the camera and its appearance on the operator's screen; the
number [R2](#r2) is about. Measured on the branch by a browser report
matched to the server's per-frame capture times (commit e0a76d076).

<a name="g-annex-b"></a>**Annex B** — the raw byte form of an H.264 stream:
encoded frames one after another, each preceded by a start code, with no
container around them. A decoder can consume a frame as soon as it arrives.
Named after the annex of the H.264 standard that defines it.

<a name="g-apply-run"></a>**Apply run** — the mask pass: a job that reads
every frame of an episode in order, segments it with SAM3 in the process
worker and writes the result to the mask store. Exact, never drops, holds
the aux-GPU slot while it runs.

<a name="g-atlas"></a>**Atlas** — one image holding several cameras' frames
side by side, so that one encoder and one decoder handle all of them. The
overlay worker's output on `main` is an atlas; the branch's Run-tab mosaic
is the same idea.

<a name="g-aux-slot"></a>**Aux-GPU slot** — a process-wide mutex in the GUI
server (`lerobot.gui.gpu_slot`) over the one heavy GPU activity allowed
beside a run. The overlay worker and the process worker take it in turn; a
second activity is refused, not queued.

<a name="g-backend"></a>**Backend** — one implementation of a stage: NVENC
or libx264 for encode, NVDEC or torchcodec for decode, `GpuMaskComposite` or
`composite_from_store` for the compositor. Resolved once per host, logged,
and never visible past its stage.

<a name="g-capture-backend"></a>**Capture backend** — the class that opens a
camera device and decodes its frames: `OpenCVCamera` (V4L2 on Linux,
AVFoundation on macOS) or `RealSenseCamera` (librealsense); `zmq` and
`reachy2` for network cameras.

<a name="g-class"></a>**Class** — the property that sorts every consumer of
camera pixels: _real time_ (in the loop; never drops; degrades in frame
rate), _job_ (reads every frame in order; never drops; degrades in wall
time), _view_ (shows pixels to a person; may drop or pace; degrades in the
delay before first play).

<a name="g-compositor"></a>**Compositor** — the one definition of "masks
applied to a frame under a recipe": `composite_from_store` on the CPU,
`GpuMaskComposite` on the device, pinned equal by
`tests/datasets/test_gpu_composite_equivalence.py`.

<a name="g-data-publisher"></a>**Data publisher** — the GUI process's writer
into the tap when no run is active: decoded episode frames, for the stored
overlay preview.

<a name="g-fmp4"></a>**Fragmented MP4** — the MP4 file format cut into small
self-contained fragments so that it can be streamed and fed to a player
piece by piece. The form MSE accepts. The format writes each frame's
duration in front of the frame, so a live encoder must hold a frame until
the next one arrives ([A6](#a6)).

<a name="g-gui-server"></a>**GUI server** — the FastAPI process the browser
talks to (`lerobot.gui.server`). It owns the camera previews when no run is
active, the Run-tab sampler, the transcodes and the caches, and it launches
the run, the workers and the jobs as subprocesses.

<a name="g-jitter-buffer"></a>**Jitter buffer** — a receiver's short queue
that absorbs uneven packet arrival so that frames can be shown at a steady
rate. Its depth is delay; WebRTC sizes it from the network's observed
variation.

<a name="g-lock-step"></a>**Lock-step** — processing every frame in order,
one after another, never skipping and never running ahead; the apply run's
mode, where each mask depends on the previous frame's.

<a name="g-mask-store"></a>**Mask store** — a dataset's saved masks, written
per episode and camera by the apply run (`mask_store.write_episode`) and
read by the training loader and by composited playback.

<a name="g-mosaic"></a>**Mosaic** — the branch's Run-tab atlas: a fixed
640x380 canvas with a rectangle per camera, drawn from a layout table
(`_PREVIEW_MOSAIC_RECT_OPTIONS`) keyed on camera names.

<a name="g-mse"></a>**MSE** — Media Source Extensions, the browser interface
that lets JavaScript feed video segments (fragmented MP4) into a `<video>`
element instead of giving it a file URL. The element buffers and plays them
at 1x, as it would a file: a natural fit for stored media, a buffered one
for live.

<a name="g-nvenc"></a>**NVENC, NVDEC** — the dedicated video encoder and
decoder engines on an NVIDIA GPU, separate from the CUDA cores. An encode
"session" is one stream being encoded; consumer drivers cap concurrent
sessions ([A6](#a6)).

<a name="g-overlay-worker"></a>**Overlay worker** — the SAM3 sidecar
subprocess that reads the tap, skips frames, and publishes RGBA overlays and
masks into `lerobot_overlay_*` for the live and stored overlay previews.
Best-effort by contract.

<a name="g-playback-cache"></a>**Playback cache** — the directory of
transcoded clips the Data tab plays, keyed by episode, camera, profile and
recipe fingerprint; filled by a transcode on a miss, pruned after each build.

<a name="g-presenter"></a>**Presenter** — the proposed client component that
decodes units, keeps the capture timestamp of the frame it last painted,
and paints: one for live, following the newest frame, and one for stored,
paced by the playback rate.

<a name="g-process-worker"></a>**Process worker** — the subprocess that runs
an apply run (`lerobot.gui.process_worker`): decodes through
`GpuFrameSource`, segments in lock step, writes the mask store.

<a name="g-profile"></a>**Profile** — a resolution and a bitrate under a name
(`low`, `medium`, `full`). Part of a stream's or a cache entry's identity;
never an encoder preset, never consulted by a job.

<a name="g-recipe"></a>**Recipe** — the per-region treatments a dataset's
masks are composited with. Its fingerprint is part of a composited clip's
identity; a changed recipe is a new entry, not an invalidation.

<a name="g-robot-host"></a>**Robot host** — the machine the cameras and the
robot are plugged into. Today the same machine as the GUI server; the
design keeps the two separable.

<a name="g-run"></a>**Run, run subprocess** — a teleop, record, replay or
inference session, launched by the GUI server as a subprocess
(`lerobot-teleoperate`, `lerobot-record`, and so on). It holds every camera
for its lifetime.

<a name="g-run-loop"></a>**Run loop** — the control loop inside the run
subprocess: read the cameras, feed the policy and the dataset writer, copy
into the tap, at the robot's control rate. The only real-time path.

<a name="g-s1-s2"></a>**S1, S2** — HVLA's two systems. S2 is a
vision-language model in its own process, producing a latent and a subtask
at its own rate; S1 is the policy in the run loop, conditioned on S2's
latest latent and its age through the `hvla_*` shared-memory channel.

<a name="g-source-adapter"></a>**Source adapter** — the proposed component
at each origin that produces pixels plus capture timestamp at the source's
own frame rate and owns nothing else: the tap reader (Run tab), the capture
backend (Robot tab), the dataset reader (Data tab). The first two are the
live source adapters.

<a name="g-stage"></a>**Stage** — one step of the view pipeline with a fixed
interface and resolvable backends: decode, composite, segment, encode.

<a name="g-sweep"></a>**Sweep** — removing shared-memory segments left behind
by a writer that exited without cleaning up. The GUI sweeps `lerobot_obs_*`
and `lerobot_overlay_*`; nobody sweeps `hvla_*`.

<a name="g-tap"></a>**Tap** — the `lerobot_obs_*` shared-memory segments
(`ObservationStream`): the run loop's latest-value copy of the processed
observation, one block per key, stamped at write, best-effort by contract.
Written by the loop during a run and by the data publisher otherwise; read
by the Run-tab view and the overlay worker. [A3](#a3).

<a name="g-torn-read"></a>**Torn read** — a copy taken while the writer was
in the middle of replacing the value, so that it holds part of the old
frame and part of the new. The tap's sequence counters let a reader detect
this and report it instead of returning the mixed frame.

<a name="g-transcode"></a>**Transcode** — decoding a stored video and
re-encoding it in another codec, resolution or bitrate. A re-wrap (remux)
changes only the container around the encoded frames and does not touch the
pixels; the `full` profile is a re-wrap.

<a name="g-unit"></a>**Unit** — one encoded access unit, a frame's worth of
H.264 Annex B, carrying its capture timestamp. What the live encoder emits
and the presenter decodes.

<a name="g-webcodecs"></a>**WebCodecs** — the browser interface that gives
JavaScript direct access to the hardware video decoder: a unit goes in with
a timestamp, a decoded frame comes out with the same one. Available only on
a secure context (HTTPS or `localhost`).

<a name="g-webrtc"></a>**WebRTC** — the browser's built-in real-time media
stack (used by video calls): packets over UDP, a jitter buffer, hardware
decode, timestamps carried by the transport, no container. Works on plain
http, but needs a WebRTC peer on the server side.

<a name="g-wire-format"></a>**Wire format** — what crosses the network: H.264
Annex B units for live surfaces, byte ranges of a clip in the playback cache
for stored ones. The same whichever backend produced the bytes.

## Appendix: the evidence behind each observation

### A1. The two current paths

<a name="a1"></a>

<a name="o1a"></a>**On `main`, every surface polls JPEG stills over HTTP.**
The browser asks for a picture, the server encodes the newest frame as a
JPEG and answers, and the browser asks again.

```mermaid
flowchart LR
  subgraph server[GUI server]
    dev["capture backends<br/>GUI-owned, no run"] -->|frame| rp["/api/robot/camera-frame/i"]
    shm["tap<br/>run active"] -->|frame| ru["/api/run/obs-stream/image/key"]
    vid[(video files)] -->|decode + JPEG| dt["/frame/i?camera="]
    shm --> ov[overlay worker] -->|H.264 atlas fMP4| os["/api/overlays/data/stream.mp4"]
  end
  rp -->|10 Hz per camera| robot[Robot tab img]
  ru -->|20 Hz per camera| run[Run tab img]
  dt -->|flipbook, all cameras per tick| data[Data tab img]
  os -->|MSE| datao[Data tab overlay canvas]
```

Measured:

- Data tab, 3 cameras at 30 fps: 324 KB per tick, 78 Mbit/s sustained. No
  remote link carries that; over Tailscale, playback stalls and skips
  (commit a4b0db5c3).
- Run tab over Tailscale: each 20 Hz request pays a full network round
  trip before its picture can be shown, and the picture is a still, so the
  frame rate the operator sees is bounded by the round trip
  (commit e0a76d076).

One path on `main` does not poll: the Data-tab overlay preview streams a
server-composited H.264 [atlas](#g-atlas) as
[fragmented MP4](#g-fmp4) over MSE (`overlays.py`
`_stream_encoder_command`, `overlay_stream.js`), fed by the
[overlay worker](#g-overlay-worker), newest frame wins.

<a name="o1b"></a>**On `feat/camera-video-transport`, two surfaces stream
encoded video, each its own way.**

```mermaid
flowchart LR
  subgraph server[GUI server]
    shm["tap<br/>run active"] -->|10 Hz sample| mosaic[mosaic 640x380] -->|libx264 or NVENC, fMP4| ps["/api/run/preview.mp4"]
    vid[(video files)] -->|ffmpeg transcode, per profile| cache[(playback cache, 4 GiB LRU)]
    masks[(saved masks + recipe)] -->|composite_from_store| cache
    cache --> ve["/api/datasets/.../video?profile=&masks="]
  end
  ps -->|MSE, catch-up seek| run[Run tab video]
  ve -->|video element, playbackRate, seek| data[Data tab video per camera]
```

_Run tab._ The GUI server samples the tap ten times a second, draws the
cameras into one 640x380 [mosaic](#g-mosaic), pipes the mosaic frames into
an ffmpeg process that encodes H.264 into fragmented MP4, and streams the
result to the browser, which plays it through MSE. Measured over Tailscale
with a round-trip time of 72.2 ms on that day (the same link measured
237 ms by ping on 2026-09-06, 20 packets, 235–244 ms, and 0.48 s to first
byte for a fresh HTTP request to the GUI): 1.18 Mbit/s; the first frame appears
498 ms after the request; the [source-to-display age](#g-age) is about
0.4 s median and 0.60 s at the 95th percentile, and does not grow over a
session (commit e0a76d076).

_Data tab._ Each camera is a `<video>` element playing a clip
[transcoded](#g-transcode) from the stored file into a profile: `low` is
640 px wide at 500 kbit/s, `medium` is 1280 px at 1500 kbit/s, and `full`
is the stored file re-wrapped without re-encoding. Scrubbing and 2x speed
come from the element. Saved masks are composited into the clip on the
server when the viewer asks for them, and the result is stored in the
playback cache under the episode, camera, profile and the
[recipe](#g-recipe)'s fingerprint, so a second viewer of the same clip
reads the file. Measured: 6.7 KB per frame at `medium`, 1.61 Mbit/s, 48x
less than the flipbook on `main`; a `full` clip prepares in 0.06 s against
0.44 s for `medium` (commit a4b0db5c3).

<a name="a1c"></a><a name="o1c"></a>**What the branch's Run path gets
wrong, and why each is a problem.**

- _The picture and the readouts beside it disagree, and nothing can bring
  them together._ The stream carries no capture timestamps: the server
  knows when each mosaic frame's source images were captured (it keeps that
  per frame in a session table, used only by a measurement endpoint), but
  the bytes sent to the browser do not say, and the browser shows whatever
  MSE has decoded. The state and action readouts and the URDF tile fetch
  the newest values independently, every 33 ms (`urdf_viz.html`
  `_pollLive`), so each readout is as old as one request: a network round
  trip plus at most one poll period, which is not measured but is bounded
  by one round trip plus 33 ms: about 100 ms at the 72 ms round trip of
  the branch's measurement, about 270 ms at the 237 ms measured on
  2026-09-06. The picture is
  0.4–0.6 s old (measured). The gap between the two is what the readouts
  lead the picture by. On `main` the gap is smaller, because a polled JPEG
  is one round trip old, not a video pipeline old. Whether the gap matters
  to an operator is open decision 3 in [Part 3](#part-3). A latest-only source does mean the
  video shows its newest frame regardless of any timestamp; the timestamp
  is not for choosing which picture to show. It is for the readouts: with
  the frame's capture time known, the client can show the state that was
  recorded at that time (the Data tab already does this, because a stored
  frame's index is its timestamp and the URDF tile in dataset mode is
  driven by the scrubber's frame), and the age of the picture becomes
  measurable in the browser rather than by a separate reporting round trip.
- _The 10 Hz resample and the container each cost one frame period, and
  only one of them buys anything._ Ten frames a second is an acceptable
  picture for a preview, and a lower frame rate is a legitimate way to fit
  a link: fewer frames, fewer bytes, and a smooth late picture beats one
  that stalls. That trade stays, as the profile's frame rate. But every
  term of the pipeline that waits for "the next frame" waits one frame
  period, and at 10 fps a period is 100 ms. The sampler's wait (a frame
  captured just after a sample waits up to 100 ms for the next one; this
  follows from the sampling and is not separately measured) is the price of
  the frame rate. The fragmented-MP4 container's wait is not: it holds each
  encoded frame until the next one arrives, because the MP4 format writes a
  frame's duration in front of it, measured at 102.5 ms per frame with this
  encoder at 10 fps against 2.3 ms for the same encoder writing raw H.264
  ([A6](#a6)), and it saves no bytes. The design removes the second and
  keeps the first as a knob.
- _Every open browser tab starts its own ffmpeg process._ One request to
  `/api/run/preview.mp4` is one `_preview_video_stream` call, and each call
  composes its own mosaic and spawns its own encoder for the same frames.
  Two operators watching means the same picture encoded twice: two libx264
  threads on the CPU, or two of the GPU's eight NVENC sessions, per
  profile. The per-frame encode cost is measured (2.3 ms median at this
  size on the CPU); the per-tab CPU load has not been.
- _The mosaic layout is written for one robot._ The layout table
  (`_PREVIEW_MOSAIC_RECT_OPTIONS`) only matches cameras named `top`,
  `left_wrist` and `right_wrist` (or `top_l`, `top_r`); a robot with other
  camera names gets no video preview and falls back to JPEG polling. The
  Robot tab is untouched and still polls.
- _MSE buffers, and the branch bounds the delay with a seek._ The
  `<video>` element plays what MSE has been fed at 1x from wherever its
  playhead is, and buffers what has arrived ahead. If the network delivers
  in bursts or the decoder pauses, the playhead falls behind the newest
  segment and stays behind, so the operator's delay grows. The branch's
  player watches the gap between the playhead and the end of the buffer
  and, when it exceeds 0.6 s, seeks to 0.15 s before the end (`run.js`, the
  `end - video.currentTime > 0.6` rule). A seek skips the frames in
  between. Whether that is visible in practice has not been observed: the
  seek is not logged, and the branch's measurement over Tailscale (age not
  growing over a session, 0.60 s at the 95th percentile) says the gap
  rarely reaches the threshold on that link. The point that stands is
  smaller: a buffered player needs a correction rule at all, where a
  player fed timestamped frames does not.

### A2. Who holds the cameras, and what the run loop does

<a name="a2"></a>

A camera is opened through a capture backend: `OpenCVCamera` for any UVC
device through V4L2 on Linux (the Arducam wrists and the ZED-M top camera
on the OpenArm2 rig, where the ZED's side-by-side frame is halved by
`split_stereo_frame`; the wrist and front cameras on the SO-107 bench), or
`RealSenseCamera` through librealsense (the SO-107 bench's top camera).
`zmq` and `reachy2` are network cameras behind the same interface. A device
handle belongs to one process at a time.

```mermaid
flowchart LR
  cams[("cameras<br/>OpenCVCamera, RealSenseCamera")]
  files[("dataset files")]
  subgraph run["run active: the run subprocess holds every handle"]
    loop[run loop]
    pol[policy]
    wr[dataset writer]
  end
  subgraph idle["no run: the GUI process holds them"]
    prev[Robot-tab preview]
    pub["data publisher<br/>decoded episode frames"]
  end
  subgraph shm["/dev/shm: lerobot_obs_*, the tap"]
    tap["the tap<br/>one latest-value block per key, stamped at write"]
  end
  cams --> loop
  loop --> pol
  loop --> wr
  loop -.->|"last processor step"| tap
  cams --> prev
  files --> pub
  pub -.->|"when no run holds it"| tap
  tap --> view[Run-tab view]
  tap --> ovl[overlay worker]
```

- <a name="o2a"></a>While a run is active, the run subprocess holds every
  camera. The GUI never touches the device: `/api/robot/detect-cameras`
  refuses to open previews while the run is alive. While no run is active,
  the GUI process opens the devices itself for the Robot-tab previews
  (`_preview_cameras` in `gui/api/robot.py`) and releases them before a run
  starts.
- <a name="o2b"></a>The run loop reads the cameras and feeds the policy and
  the dataset writer in the same process. Each observation passes through a
  chain of processing steps before the policy sees it, and the last step of
  that chain (`ObservationStreamWriterStep`) copies the processed
  observation into the tap. That copy is a write into memory: no lock, no
  reader to wait for, failures suppressed, the observation returned
  unchanged. The tap module's own contract says that no policy, control,
  safety or recording path may depend on the copy succeeding; [A4](#a4)
  explains what that contract rests on.
- <a name="o2c"></a>For the Data tab's overlay preview, the GUI process
  starts a [data publisher](#g-data-publisher) that writes decoded episode frames into the same
  tap, so the overlay worker reads one place whether the frames are live or
  stored. The publisher refuses to start while a run is alive, and the
  launch path stops it before a run's `connect()` would remove the segments
  from under it.

### A3. How the tap is managed

<a name="a3"></a>

- <a name="o3a"></a>_Names and contents._ POSIX shared-memory segments under
  `/dev/shm`: `lerobot_obs_meta` (a JSON descriptor of the keys and image
  sizes), `lerobot_obs_obs` (the scalar observation), `lerobot_obs_act` (the
  last action sent) and one `lerobot_obs_img_<camera>` per camera. Each
  holds one value, the newest, behind a 24-byte header: two sequence
  counters and the wall-clock time of the write. A reader compares the
  counters before and after its copy, and if they differ the writer was in
  the middle of an update; the reader reports a [torn read](#g-torn-read) rather than
  returning a mixed frame. There is no queue and no history.
- <a name="o3b"></a>_Ownership._ The names carry no run id and no server
  id, so there is one tap per host. Its writer creates it: the run
  subprocess when the robot connects (the GUI launches every run with
  `LEROBOT_OBS_STREAM=1`) and unlinks it when the robot disconnects; the
  data publisher creates the same names in the GUI process. Either creation
  removes a same-named leftover first. Readers attach by name; the GUI's
  reader notices a recreated stream because the meta segment is a new file
  (its inode changes) and re-attaches.
- <a name="o3c"></a>_Sweeps._ A writer that dies uncleanly leaves its
  segments behind, and a reader attached to them serves a frozen picture as
  if the run were alive. So the GUI sweeps `lerobot_obs_*` unconditionally
  at its own startup and shutdown, and before every launch only if nothing
  has written to the segments in the last two seconds, so that a writer the
  GUI does not know about (a teleop started from a terminal) is left alone.
- <a name="o3d"></a>_Isolation is by time, not by name._ A run and the data
  publisher write the same segments, never at once. That is the whole
  reason the two writers share names: the overlay worker and the Run-tab
  view read one place.
- <a name="o3e"></a>_Guarantees._ The tap module states its own contract: a
  best-effort, single-slot, latest-value channel for display and debugging,
  with no delivery or freshness guarantee. A reader can miss samples, can
  get a torn read, and can stay attached to stale data after an unclean
  writer exit until a sweep or a reconnect catches up. The GUI can remove
  the segments at its own startup and shutdown regardless of who is
  writing.
- _A second GUI server on the same host_ (a test instance on another port)
  reads the same stream as the first, and its unconditional startup sweep
  removes a live run's segments; a later attach by name then fails until
  the run reconnects.

### A4. The HVLA channel

<a name="a4"></a>

```mermaid
flowchart LR
  subgraph runp["run subprocess"]
    loop["run loop<br/>last processor step"]
    s1["S1 policy"]
  end
  tap[("lerobot_obs_*<br/>created by the writer, swept by the GUI")]
  img[("hvla_img_*<br/>created by S2")]
  lat[("hvla_s2_latent*<br/>created by S2")]
  ovl[("lerobot_overlay_*<br/>created by the worker, swept by the GUI")]
  subgraph s2p["S2 process, persistent across runs"]
    s2["S2 VLM"]
  end
  subgraph ovp["overlay worker"]
    ov["SAM3 sidecar"]
  end
  subgraph gui["GUI server"]
    view["Run-tab view"]
    badge["subtask badge"]
  end
  loop --> tap --> view
  tap --> ov --> ovl --> view
  loop --> img --> s2 --> lat --> s1
  lat --> badge
```

- <a name="o4a"></a>_The image blocks hold the same pixels as the tap._ The
  same processor step that writes the tap also writes `hvla_img_<view>`
  when `LEROBOT_S2_IMAGE_BUFFER=1`, which the GUI sets for a run started
  with S2 loaded. It copies from the same processed-observation dict, at
  the same moment. The differences are bookkeeping: only the cameras in
  `DEFAULT_S2_CAM_KEY_MAP` are copied, under S2's view names (`front` →
  `base_0_rgb`, and so on); a frame whose shape differs from the block is
  skipped; and the joint state goes into `hvla_img__state` as a float
  array. For a camera in that map, `hvla_img_<view>` and
  `lerobot_obs_img_<camera>` are the same array written twice.
- <a name="o4b"></a>_The latent block is different in kind._
  `hvla_s2_latent`, with its `_subtask` text and `_conf` companions, carries
  S2's output (a latent vector, the model's summary of the scene and the
  current subtask) back to S1. S1 reads it inside the loop together with
  its age, and the age is an input to the model, clamped to the 0.15 s the
  model was trained with (`s1_process.py`). The GUI attaches read-only for
  the subtask badge.
- <a name="o4c"></a>_The owner and the contract differ, not the protocol._
  S2 creates all of the `hvla_*` blocks when it runs standalone, which is
  how the GUI keeps one S2 process loaded across runs (the "debug model"
  process); an S1 launcher that finds no S2 creates them and spawns S2
  itself. The run subprocess only attaches, and retries until S2's blocks
  exist. The GUI never sweeps `hvla_*`. The block implementation is the
  same design as the tap's: the same 24-byte header, the same torn-read
  protocol, in a second class (`SharedBlock` in `policies/hvla/ipc.py`
  against `_Block` in `robots/obs_stream.py`; the tap's module says "same
  pattern as policies.hvla.ipc"). Both classes silence Python's
  `resource_tracker` so that a segment is not removed when a process that
  merely attached to it exits; `SharedBlock`'s comment gives the case: S2's
  memory must survive S1 exiting.
- _Merging the channels_ would mean giving the tap the lifecycle S2 relies
  on (no GUI removal while a run is alive, and a freshness guarantee); the
  tap's docstring names that future as an observation bus, to be built
  after the critical-path contract is added and tested. The writer step
  carries a TODO to move the S2 mirror into a policy-owned processor step.

### A5. Every reader of the cameras and of the files

<a name="a5"></a>

| Consumer                    | Reads                | Class     | Keeps up by      | May drop |
| --------------------------- | -------------------- | --------- | ---------------- | -------- |
| Policy inference            | cameras, in the loop | real time | the control rate | never    |
| Dataset writer              | cameras, in the loop | real time | encode threads   | never    |
| Run-tab view                | tap                  | view      | newest wins      | yes      |
| Overlay preview, live       | tap                  | view      | skips            | yes      |
| Robot-tab preview           | cameras, GUI process | view      | newest wins      | yes      |
| Training loader             | files, saved masks   | job       | throughput       | never    |
| Apply run (mask pass)       | files                | job       | lock-step        | never    |
| Playback, plain/composited  | files, saved masks   | view      | paced, seeks     | no       |
| Overlay preview, stored     | files, via the tap   | view      | skips            | yes      |
| Hub transfer, merge, export | files as bytes       | job       | —                | —        |

"Keeps up by" says what each consumer does when the source is faster than
it is: the loop runs at the robot's control rate and everything in it must
finish within a step; the dataset writer encodes on its own threads and
never skips a frame; a view shows the newest frame and forgets the rest;
the apply run works in [lock-step](#g-lock-step); playback is paced by the
browser from a file. The three mask modes the UI names are three rows: the
overlay preview (live and stored: approximate, skips, never written), the
apply run (a job that reads the files exactly like training, down to
decoding through training's GPU video reader `GpuFrameSource`; its product
is the [mask store](#g-mask-store)) and composited playback (a view that
reads what the apply run wrote and recomputes nothing).

_What each class runs on today._ Real time decodes through the capture
backend; the dataset writer encodes with `VideoEncodingManager` (PyAV:
SVT-AV1, H.264, HEVC, or a hardware codec under `vcodec=auto`). Jobs decode
through `GpuFrameSource` ([NVDEC](#g-nvenc)) with the CPU dataset reader
(torchcodec) as the fallback; training composites with `GpuMaskComposite`
or `composite_from_store`; the apply run segments in the SAM3
[process worker](#g-process-worker) and writes the mask store. Views each
have their own encoder: the Run tab an ffmpeg H.264 mosaic on the branch
and JPEG per poll on `main`; the two overlay previews the overlay worker's
ffmpeg H.264 atlas; the Robot tab JPEG per poll; playback ffmpeg H.264 into
the playback cache after `composite_from_store`. The stored overlay preview
decodes with torchcodec in the GUI process and publishes into the tap.

_Shared today, by design._ The compositor: one definition, two backends,
`composite_from_store` on the CPU and `GpuMaskComposite` batched on the
device (4.6–7.3 ms per 720p frame on the CPU against about 0.2 ms batched,
per its module note), pinned equal on real rows by
`tests/datasets/test_gpu_composite_equivalence.py`; the playback clip calls
the CPU definition at display scale (5–18 ms per frame, commit 8f040758c).
The GPU decoder `GpuFrameSource` serves training and the apply run. The
dataset reader `LeRobotDataset` over `decode_video_frames` (torchcodec,
PyAV fallback) is the CPU training loader, the apply run's fallback and the
Data-tab frame endpoint. The tap has two writers and two readers, never
more than one writer at a time. The stereo split `split_stereo_frame` is
called by the live camera and by the offline dataset transform, and its
module states the rule: only the naming and the split are shared, the
lifecycles are not.

_Duplicated today._ Encoders: PyAV in the recorder, three ffmpeg command
builders in the GUI (`_preview_encoder_command`, `_stream_encoder_command`,
`_transcode_episode*`), and JPEG behind every polling endpoint, each with
its own flags, latency profile and bug surface. Stored decoders: torchcodec
in the dataset reader, NVDEC in the jobs, ffmpeg in the transcodes (the
last only because the transcode wants a pipe rather than frames). The
shared-memory block: `_Block` and `SharedBlock` ([A4](#a4)).

_Separate on purpose._ Two SAM3 processes, the overlay worker (skips
frames, reads the tap) and the process worker (lock-step, reads files),
load the same weights and take turns through the
[aux-GPU slot](#g-aux-slot). They cannot share a queue, because a queue
with a consumer that may drop and one that may not eventually drops the
wrong frame. They share the model call and the mask codec.

_Where the classes collide._ The encoder budget: the recorder encodes every
camera in real time in its own threads (SVT-AV1 on the CPU on the rig
today, NVENC when `vcodec=auto` finds it); the branch's preview encoders
are libx264 pinned to one thread per viewer on the same CPU; moving
previews to NVENC frees the CPU but spends sessions, and three recorded
cameras plus three preview cameras at one profile is six of the eight. The
decode engines: training and the apply run both decode on NVDEC, with
nothing arbitrating between them beyond the aux-GPU slot; the view decodes
in software and is not a third contender today. The clock: dataset
timestamps are episode-relative (frame index over fps), tap stamps are wall
clock at write, and on the live side the image block's stamp is the key
that joins a frame to the state and action of the same loop iteration. The
process boundary: the tap is the view's only channel into the run, and
`hvla_*` is the policy's.

_The three classes at run time_, each diagram grouping its participants by
process, left to right.

During a run, four places: the run subprocess holds the cameras, the loop,
the policy and the dataset writer (the writer's encoders are threads of
that process); the tap is the segments in `/dev/shm`; the GUI server
samples the tap from an asyncio task and pipes each frame into a child
ffmpeg; the browser is on the operator's machine. The two loops are in two
processes that share only the tap, and the first never waits for the
second.

```mermaid
sequenceDiagram
  box transparent run subprocess
    participant C as cameras
    participant L as run loop
    participant P as policy
    participant W as dataset writer
  end
  box transparent /dev/shm
    participant T as tap
  end
  box transparent GUI server
    participant G as sampler + ffmpeg child
  end
  box transparent operator's machine
    participant B as browser
  end
  loop run loop: every control step
    C->>L: frame
    L->>P: observation
    L->>W: observation (encoder threads)
    L-->>T: processed observation + stamp
  end
  loop GUI task: at the tap's cadence
    G->>T: read newest
    G->>B: unit + capture ts
  end
```

A job: the GUI server starts it and polls it; the job is a subprocess (the
process worker for the apply run, the training container for training)
that owns the decoder, the model and the output for its lifetime.

```mermaid
sequenceDiagram
  box transparent GUI server
    participant U as job API
  end
  box transparent job subprocess
    participant J as job (apply run, training)
    participant D as GpuFrameSource (NVDEC)
    participant M as model (SAM3, policy)
  end
  box transparent disk
    participant O as output (masks, checkpoints)
  end
  U->>J: start
  loop job: every frame, in order
    J->>D: next chunk
    D-->>J: decoded batch
    J->>M: frames (+ composited masks when training)
    M-->>J: result
    J->>O: write
  end
  U->>J: poll progress
```

A view of stored video: the browser asks for a clip by its identity (the
episode, the camera, the profile and the recipe fingerprint together). A
request handler in the GUI server answers from the playback cache, or runs
the transcode on a worker thread (a decoding ffmpeg child, the compositor's
thread pool, an encoding ffmpeg child) under a lock held per identity, so
two requests for the same clip build it once.

```mermaid
sequenceDiagram
  box transparent operator's machine
    participant B as browser
  end
  box transparent GUI server
    participant S as request handler
    participant X as transcode thread + ffmpeg
  end
  box transparent disk
    participant K as playback cache
  end
  B->>S: episode, camera, profile, masks
  S->>K: entry for that identity?
  alt miss
    S->>X: decode, composite_from_store, encode
    X->>K: clip
  end
  K-->>B: byte ranges, paced cursor
```

### A6. Transport measurements

<a name="a6"></a>

<a name="o6a"></a>_Encoder latency._ Per-frame latency from frame-in to the
encoded unit out, idle RTX 5090, 150 frames per row, medians (script
`enc_latency2.py`; the full table with p95 and max is in [A10](#a10)).

- libx264 (software), fragmented MP4: 102.5 ms at 640x380@10, 35.4 ms at
  30 fps, 38.0 ms at 720p30, one frame period each time. The MP4 format
  writes a frame's duration in front of the frame, so the container holds
  each packet until the next one arrives.
- libx264, Annex B (`-f h264 -flush_packets 1`): 2.3 / 2.2 / 4.6 ms.
- h264_nvenc (GPU), fragmented MP4, default settings:
  300.8 / 101.0 / 101.6 ms, three frame periods: NVENC's default two-frame
  output delay plus the container hold.
- h264_nvenc, fragmented MP4, `-delay 0`: 101.1 / 34.6 / 35.5 ms, one
  period.
- h264_nvenc, Annex B, `-delay 0`: 1.1 / 1.2 / 2.1 ms median; p95 up to
  about 101 ms and max about 150–163 ms, so NVENC has an occasional long
  frame that the software encoder does not.

<a name="o6b"></a>_Browser capabilities depend on the origin._ Browsers
expose some interfaces only on a "secure context" (HTTPS or `localhost`).
Probed with headless Chromium 151 (`codec_probe2.py`): over plain `http://`
on the LAN IP or the Tailscale IP, `isSecureContext` is false and WebCodecs
(`VideoDecoder`) and `WebTransport` are undefined; MSE,
`requestVideoFrameCallback`, WebRTC (`RTCPeerConnection`) and `WebSocket`
are available. On `localhost` everything is available. Tailscale can issue
a certificate for the machine's `ts.net` name.

<a name="o6c"></a>_GPU inventory._ RTX 5090: three NVENC engines (9th
generation), two NVDEC engines; GeForce drivers ≥ 550.54.14 allow eight
concurrent NVENC sessions. ffmpeg on both rigs exposes `h264_nvenc`,
`hevc_nvenc`, `av1_nvenc` and CUDA hardware decode.

### A7. What the industry does

<a name="a7"></a>

- Foxglove `CompressedVideo`: H.264 Annex B, one frame per message, its
  timestamp beside it, the decoder configuration (SPS/PPS) repeated with
  every keyframe, no frames that depend on a later frame (B-frames), so any
  frame can be decoded as soon as it arrives. Decoded with WebCodecs.
- Rerun `VideoStream` (0.24+): H.264 Annex B samples with presentation
  timestamps, same constraints, same decoder.
- Teleoperation products default to WebRTC: the browser's own jitter buffer
  and hardware decoder, timestamps carried by the transport (RTP), no
  container, works on insecure origins. Latency is bounded by the jitter
  buffer, which the sender controls through pacing and keyframe policy.
- MSE is the right tool for stored media and acceptable for live with a
  catch-up policy; it is buffered by design.
- WebCodecs is the lowest-latency browser decoder and the simplest to
  synchronise, but requires a secure context.
- ROS 2: one image topic, and each consumer subscribes with its own quality
  of service, the recorder reliable and complete, the visualiser
  best-effort with a history depth of one, while `image_transport` plugins
  put the viewer's compression in the subscriber's path, never the
  publisher's.

### A8. Accelerators, floors, and hosts without them

<a name="a8"></a>

_The three existing resolutions._ `resolve_vcodec` walks `HW_VIDEO_CODECS`
(VideoToolbox, NVENC, VA-API, QSV) and falls to `libsvtav1`. The training
`data_path` knob is `auto`, `cpu` or `gpu`: `auto` checks facts (a CUDA
device, a decodable codec, a dataset it can composite, this dataset's own
frames verified against the CPU decoder), and `gpu` refuses rather than
silently training on the other path, "a wrong measurement rather than a
slow one". The apply run's `_gpu_frame_sources` returns the CPU read with
the reason logged. The compositor's CPU definition is the reference and the
GPU one is pinned equal to it. The encoder stage's knob replaces the
`LEROBOT_PREVIEW_ENCODER` environment override on the branch.

_Each stage's accelerator today, and the floor under it._

- Live encode: NVENC, three engines and eight sessions on the 5090; two
  sources (Run, Robot preview) at two or three profiles fit with room, and
  an inference run that records on NVENC spends one session per camera.
  Floor: libx264, one thread per stream, 2.3 ms median at 640x380 and 4.6 ms
  at 720p30 into Annex B on the rig's CPU. VideoToolbox on a Mac is the
  first entry in `HW_VIDEO_CODECS` and is unmeasured; `enc_latency2.py`
  runs there unchanged.
- Stored decode and composite: the jobs use NVDEC through `GpuFrameSource`
  and CUDA through `GpuMaskComposite`. The Data tab does not: its transcode
  is the dataset reader (torchcodec, PyAV where torchcodec has no wheel),
  `composite_from_store` at display scale (5–18 ms per frame, commit
  8f040758c; 7.8 → 1.9 ms per frame with the composite thread pool on a
  24-core box, per the note in `gui/api/datasets.py`) and the resolved
  encoder. Moving the view onto NVDEC or CUDA is taken only with a measured
  need, and it yields to the jobs.
- Segmentation: SAM3 in both workers runs wherever torch puts it; on a host
  without CUDA that is MPS or the CPU, at a speed nobody has measured. The
  overlay preview drops to whatever rate the model sustains; the apply run
  takes longer and stays exact.
- The client decodes H.264 in hardware in every browser on every platform.
  That universality is a reason H.264 is the wire format: AV1 decode in
  Safari depends on the machine's hardware, so playing a stored AV1 file
  directly is never the only path.

_Unmeasured._ VideoToolbox as the live encoder; SAM3 on MPS or the CPU;
encoder contention on the rig with the recorder and a loaded policy running
at once, which is the first measurement to make.

_Blockers on a Mac that are not this design's._ `ObservationStream` sweeps
stale segments through `/dev/shm` paths (`multiprocessing.shared_memory`
itself works on macOS, the sweep does not); camera discovery is V4L2
(`_linux_video_capture_candidates`), while OpenCV opens devices through
AVFoundation there; librealsense on macOS is partial; PyNvVideoCodec is
absent, which the `auto` knobs already handle.

### A9. Latency budget, teleop

<a name="a9"></a>

Where the time goes between the camera and the operator's eye, with the
measured terms filled in and the rest named as unmeasured.

```mermaid
flowchart LR
  cap[capture] --> shm["tap write<br/>unmeasured"] --> samp["sample<br/>source cadence, not 10 Hz"] --> enc["encode<br/>1–5 ms median, NVENC tail to 160 ms"] --> mux["container<br/>0 with Annex B"] --> net["network<br/>RTT 72 ms at the branch measurement; 237 ms ping on 2026-09-06"] --> jb["jitter buffer<br/>transport-dependent"] --> dec["decode<br/>hardware, unmeasured"] --> paint[paint + sync]
```

The two terms the redesign controls are sampling (the encoder runs at the
source's frame rate, not a 10 Hz resample) and the container plus player
buffer (Annex B into a decoder that presents immediately). Both are measured
at one frame period each on the branch ([A6](#a6)); together they are the
pipeline's own budget over the network's, and [R1](#r1) says a term stays
only if it buys bandwidth or smoothness.

### A10. Encoder latency table

<a name="a10"></a>

Frame-in to unit-out, idle RTX 5090, 150 frames, `enc_latency2.py`.

```
encoder     shape    size@fps       bitrate   median   p95     max
libx264     mp4      640x380@10     1200k    102.5   103.0   103.2
libx264     mp4      640x380@30     1200k     35.4    68.7   102.3
libx264     mp4      1280x720@30    1500k     38.0    39.0    71.5
libx264     annexb   640x380@10     1200k      2.3     2.7    20.1
libx264     annexb   640x380@30     1200k      2.2    35.4    36.2
libx264     annexb   1280x720@30    1500k      4.6     5.4    66.7
h264_nvenc  mp4      640x380@10     1200k    300.8   301.4   301.6
h264_nvenc  mp4      640x380@30     1200k    101.0   167.5   201.2
h264_nvenc  mp4      1280x720@30    1500k    101.6   135.4   168.4
h264_nvenc  mp4 -delay 0  640x380@10  1200k  101.1   101.5   160.7
h264_nvenc  mp4 -delay 0  640x380@30  1200k   34.6   101.5   165.5
h264_nvenc  mp4 -delay 0  1280x720@30 1500k   35.5    69.4   167.4
h264_nvenc  annexb   640x380@10     1200k      1.1     1.3   159.1
h264_nvenc  annexb   640x380@30     1200k      1.2   101.0   163.2
h264_nvenc  annexb   1280x720@30    1500k      2.1    35.5   150.4
```

`mp4` is the branch's `_preview_encoder_command` shape (fragment per frame,
zero-latency tuning). `annexb` is `-f h264 -flush_packets 1`. Milliseconds.
