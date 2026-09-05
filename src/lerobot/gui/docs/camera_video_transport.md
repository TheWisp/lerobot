# Camera video transport

How camera pixels reach the browser, for the three surfaces that show them:
the Data tab (stored episodes), the Run tab (live teleop and inference) and
the Robot tab (camera preview). A design under review, not a description of
shipped behaviour.

**The proposal, in one paragraph.** Live cameras are encoded once per
camera and quality level into a stream of H.264 frames that each carry
their capture time, and the same stream is sent to every viewer; the
browser always shows the newest frame and, if the readouts beside the
picture must agree with it (an unconfirmed part of R2, see Part 1), looks
them up at that frame's time. Stored episodes are transcoded
once per camera, quality level and mask recipe into a cached clip the
browser plays from a file, which is what the current branch already does.
The policy, the recorder and the training and mask jobs are not touched:
the view reads what the run loop already publishes and yields every shared
resource to them. Every stage that can use a GPU has a software fallback
resolved once per host. Part 4 draws this out; Parts 1–3 are the argument
for it.

The document is meant to be read without any other context. Part 1 states
the requirements R1–R6. Part 2 is eight observations O1–O8 about the system
as it is, each reduced to what bears on the design and each closing with
the conclusion C1–C8 it forces; the evidence behind each observation is in
the appendix, linked from it. Part 3 adds the conclusions up into the
constraints F1–F5 the design may not move and the freedoms it has. Part 4
is the architecture, each element naming what it comes from. Every R, O, C
or F mentioned is a link to where it is stated. Terms with a fixed meaning
are in the [glossary](#glossary) and link to it at first use; a word not
in the glossary is meant in plain English.

Two branches are named. `main` is what users run today.
`feat/camera-video-transport` is the branch this document lives on; it
carries a first implementation of streamed video for two of the three
surfaces, written before this design.

Client is a desktop browser on an arbitrary laptop. Server is the
[GUI server](#g-gui-server)'s host, which usually has an RTX 5090 but may
have no NVIDIA GPU or no GPU at all. Until the [robot host](#g-robot-host)
and the GUI server are split, that host sits next to the robot. Phone and
tablet clients are out of scope.

## Part 1: Requirements

- <a name="r1"></a>**R1. Works over a remote link** (LAN, Tailscale, or
  worse).
- <a name="r2"></a>**R2. Live video spends delay only where it buys
  something.** Two trades are legitimate and stay as knobs: frame rate
  against bitrate, which the viewer chooses through a profile, and a
  receive buffer sized to the link's jitter, which the transport chooses,
  because a smooth picture that is a little late beats one that stalls.
  Delay that buys neither (a container that holds a frame until the next
  one, an encoder's default lookahead, a fixed player buffer on top of the
  jitter buffer) is removed. _Unconfirmed second half:_ the state, action
  and robot-model (URDF) readouts show the robot at the time of the
  picture rather than at the present instant. This was written into the
  first draft by the author and has not been confirmed as a need; if it is
  dropped, timestamps on the stream serve only to measure the picture's
  age.
- <a name="r3"></a>**R3. Stored video scrubs to any frame, plays at 2x, and
  shows the saved masks composited in.** An upfront delay is acceptable if
  playback is then smooth.
- <a name="r4"></a>**R4. The Robot tab and the Run tab share one path.**
- <a name="r5"></a>**R5. Nothing the view does may slow, block or corrupt
  the policy, the recorder, or a training or mask job.**
- <a name="r6"></a>**R6. Every stage runs on a host without NVIDIA, and
  without any GPU, with no change of architecture.**

The Data tab's source is stored video (whatever codec the recorder chose:
SVT-AV1 by default, or H.264, HEVC, or a hardware encoder) plus a per-frame
timestamp in parquet. The Run tab's source is the running process's latest
frame in shared memory (the [tap](#g-tap)). The Robot tab's source is the
camera device, which the GUI opens only while no run is active. What
differs between the surfaces is the source and the clock, not the pixels.

## Part 2: Observations, each reduced to what it forces

### O1. What the two current paths cost

<a name="o1"></a>

On `main` every surface polls JPEG stills: the Data tab needs 78 Mbit/s
for three cameras at 30 fps, and the Run tab pays a network round trip per
frame (measured, [A1](#a1)). On the branch, the Run tab streams one H.264
mosaic per viewer at 1.18 Mbit/s and the Data tab plays cached transcoded
clips at 1.61 Mbit/s (measured, [A1](#a1)). The Data-tab path meets
[R1](#r1) and [R3](#r3) as it stands. The Run-tab path meets [R1](#r1) as well, and fails
[R2](#r2) and [R4](#r4) in five ways, each detailed in [A1](#a1c):

- The stream carries no capture times. The picture is 0.4–0.6 s old
  (measured); the readouts beside it are as old as one request, a network
  round trip plus at most one 33 ms poll period (not measured). Whether
  that gap matters is the unconfirmed half of [R2](#r2).
- It resamples the cameras at 10 Hz and wraps the frames in fragmented
  MP4. The frame rate is a bandwidth trade and stays a choice. The
  container is not: it holds every frame until the next one arrives, one
  frame period (102.5 ms measured at 10 fps), and saves no bytes.
- Every open browser tab spawns its own encoder for the same frames.
- The mosaic's layout table only matches one robot's camera names; other
  robots fall back to JPEG polling. The Robot tab is untouched.
- The player ([MSE](#g-mse)) buffers what has arrived, and the branch bounds
  the resulting delay with a seek towards the live edge once the buffer
  runs 0.6 s ahead. How often that seek fires, and whether it is visible,
  has not been observed; the branch's own measurement (age not growing
  over a session, 0.60 s at the 95th percentile) suggests it rarely does.

<a name="c1"></a>**Conclusion C1.** Stills cannot meet [R1](#r1): a full picture
per request and a round trip per frame is structural. An encoded stream
can. The Data-tab path is taken as it is. The Run-tab path is taken as an
idea and redone so that frames carry their capture time, the source is
encoded per camera at a frame rate the profile chooses rather than a fixed
10 Hz, one encode serves every viewer, and the player buffers only for the
link's jitter with nothing fixed on top. The Robot tab joins that path.

### O2. Who holds the cameras

<a name="o2"></a>

A camera handle belongs to one process at a time. While a run is active
the [run subprocess](#g-run) holds every camera; its [loop](#g-run-loop)
feeds the policy and the dataset writer in-process and, as the last step
of its observation processing, copies the processed observation into the
tap, a memory write that waits for nothing. While no run is active the GUI
process holds the cameras for the Robot-tab preview, and a
[data publisher](#g-data-publisher) in the GUI writes decoded episode
frames into the same tap for the stored overlay preview. Details and
diagram in [A2](#a2).

<a name="c2"></a>**Conclusion C2.** The Run and Robot tabs see the same
cameras under two owners, so the view needs two live
[source adapters](#g-source-adapter), one reading the tap and one reading
a [capture backend](#g-capture-backend), producing the same form: pixels
plus capture time. That is what makes [R4](#r4) possible. The tap is the
view's only channel into the run, and the loop already publishes what the
view needs, so nothing is added to the loop ([R5](#r5)).

### O3. What the tap is

<a name="o3"></a>

The tap is a set of shared-memory segments with fixed names
(`/dev/shm/lerobot_obs_*`), one per key, each holding only the newest
value with its write time. The names carry no run or server id, so there
is one tap per host. Its writer creates it and removes it; the GUI
[sweeps](#g-sweep) leftovers at its own startup and shutdown. The run and
the data publisher share the names by writing at different times, never
together. The tap promises nothing: a reader may miss frames, get a
[torn read](#g-torn-read), or hold stale data until a sweep. Details in
[A3](#a3).

<a name="c3"></a>**Conclusion C3.** The tap's source adapter is one per
host, so "encode once" is per host. A second GUI server on the same host
sweeps a live run's tap away at startup, so a test server never runs
beside a live GUI. The tap's names, sweeps and re-attach logic live only in
its source adapter. And the tap is same-host, so when the robot host and
the GUI server split, the adapter and encoder move with the cameras.

### O4. HVLA already shares the same pixels under a different owner

<a name="o4"></a>

HVLA's [S1 and S2](#g-s1-s2) exchange observations and a latent through a
second family of segments, `hvla_*`. The same processor step that writes
the tap also writes S2's image blocks, from the same observation at the
same moment: for the mapped cameras the two segments hold the same array.
What differs is the owner: S2 creates `hvla_*` and the GUI never touches
it, whereas the GUI creates, removes and reuses the tap at will. The block
implementation exists twice (`_Block`, `SharedBlock`) with one header and
one protocol. So "the tap must not feed the policy" is a statement about
the tap's lifecycle, not its pixels: a policy reading it would lose the
cameras whenever the GUI restarted. Details in [A4](#a4).

<a name="c4"></a>**Conclusion C4.** Share the block primitive (one class,
an equivalence test on the header), keep the lifecycles in their own
modules, and do not merge the channels: merging means giving the tap a
policy-grade lifecycle, which the view does not need. The double copy of
S2's frames is the policy's own TODO. This design adds no channel and no
writer to the loop ([R5](#r5)).

### O5. Every reader of the cameras and files, in three classes

<a name="o5"></a>

Ten consumers read the cameras or the files ([A5](#a5), table). They fall
into three [classes](#g-class) by whether a frame may be dropped: _real
time_ (policy, recorder: never), _job_ (training, the mask
[apply run](#g-apply-run), transfers: never, but wall time is free), _view_
(every preview and playback: yes). Today five separate encoders and three
separate stored decoders serve them, while the [compositor](#g-compositor),
the GPU decoder and the dataset reader are shared with an equivalence test
pinning the compositor's two backends equal. The classes meet on the
encoder budget (NVENC sessions), the decode engines, the clock (wall clock
live, episode-relative stored) and the process boundary.

<a name="c5"></a>**Conclusion C5.** Share the operation, never the
lifecycle (the rule `stereo.py` already states): consumers of different
classes may share a function, a model or a file format, never a queue, a
thread or a device handle. So the five encoders become one live encoder
plus the recorder's own; the three stored decoders become the dataset
reader; the view uses the one compositor definition ([R3](#r3)); the view
carries the source's own clock and never re-stamps ([R2](#r2)); and
wherever the classes meet a budget, the view yields ([R5](#r5)).

### O6. Latency is design choices, not codec work

<a name="o6"></a>

Encoding a frame costs 1–5 ms on either encoder; the branch's 100 ms per
frame comes from the fragmented-MP4 container holding each frame until the
next arrives, and NVENC's default settings add two more frame periods
(measured, [A6](#a6)). Neither of those saves bandwidth; the frame rate
does, and is a separate choice. Browsers expose the direct hardware decoder
([WebCodecs](#g-webcodecs)) only on HTTPS or `localhost`; the GUI is
reached over plain http, where [WebRTC](#g-webrtc) and MSE work. The 5090
allows eight concurrent NVENC sessions.

<a name="c6"></a>**Conclusion C6.** The live path emits raw H.264
([Annex B](#g-annex-b), no container) at the source's frame rate, with each
encoder's low-latency flag applied inside the encoder stage. Decoding in
JavaScript needs HTTPS first; WebRTC does not. Which to choose is
[open](#part-5).

### O7. What the industry converged on

<a name="o7"></a>

Foxglove and Rerun stream timestamped H.264 Annex B frames and decode
them in the browser; teleoperation products use WebRTC; MSE is for stored
media. ROS 2 serves the policy, recorder and viewer from one topic with a
per-subscriber quality of service, and puts the viewer's compression on the
subscriber's side ([A7](#a7)).

<a name="c7"></a>**Conclusion C7.** The capture timestamp travels with the
bytes and the receiver synchronises on it ([R2](#r2)). The ROS arrangement
is the tap by another name: the loop publishes once, best-effort readers
take the newest, and the viewer's encoder lives on the reader's side, never
in the loop ([R5](#r5)).

### O8. Hosts without NVIDIA, or without a GPU

<a name="o8"></a>

The codebase already resolves accelerated backends three times the same
way: one interface, a software reference, GPU backends resolved once per
host and logged, an `auto`/`cpu`/`gpu` knob whose forced mode refuses
rather than degrades ([A8](#a8)). Each class tolerates a slower backend
along one axis: real time in frame rate, jobs in wall time, views in the
wait before first play.

<a name="c8"></a>**Conclusion C8.** [R6](#r6) costs no architecture: every
view [stage](#g-stage) (decode, composite, segment, encode) takes that
pattern, and nothing past a stage may see which [backend](#g-backend) ran.
The live player's rule of buffering only for jitter ([C1](#c1)) turns a
slow encoder into fewer frames per second rather than a growing lag.

## Part 3: Constraints, freedoms, and the shape they leave

**Fixed**, because a consumer that must not drop owns it:

- <a name="f1"></a>**F1. The run loop.** Its cadence and order, and that
  nothing is added to it. ([C2](#c2), [C4](#c4), [R5](#r5))
- <a name="f2"></a>**F2. The recorder's product.** The view adapts to the
  file; the file never adapts to the view. (Part 1, [C1](#c1))
- <a name="f3"></a>**F3. The jobs' exactness.** Same artefact on every
  backend; one compositor definition. ([C5](#c5), [C8](#c8))
- <a name="f4"></a>**F4. The process boundary.** The tap is the view's only
  way into the run; `hvla_*` is the policy's; neither is merged. ([C2](#c2),
  [C4](#c4))
- <a name="f5"></a>**F5. The tap's scope.** One per host, same host as the
  cameras, removable by the GUI. ([C3](#c3))

**Free**, because a view may drop: which [profile](#g-profile) a viewer
watches and how many viewers share an encode ([C1](#c1)); where the encode
runs and which backend each stage resolves ([C3](#c3), [C8](#c8)); what the
[playback cache](#g-playback-cache) holds ([C1](#c1)); whether the view gets
the accelerators at all ([C5](#c5)).

**The shape.** Adapters at the two origins (the cameras, through the tap or
a capture backend; the files, through the dataset reader) producing one
form, pixels plus capture time. Behind them one chain of stages, decode,
composite, encode, each with a software reference and a resolved backend.
At the client one [presenter](#g-presenter) keyed on the source's time. On
the stored side a cache, because there the product is a file. Everything
between an adapter and the presenter is view-class: it drops, yields and
resolves, and none of it is visible from the loop or the jobs.

## Part 4: Proposed architecture

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

**Frame model** ([C7](#c7), [R2](#r2)). Every [unit](#g-unit) that leaves
the server carries the capture timestamp of the frame it encodes: the tap's
stamp for live frames, the parquet timestamp for stored ones. Nothing
downstream invents a time.

**Source adapters** ([C2](#c2), [C5](#c5), [F5](#f5)). One per source,
producing pixels plus capture time at the source's own frame rate: the
capture backend (Robot tab), the tap reader (Run tab), the dataset reader
(Data tab). The first two produce the same form, so the presenter cannot
tell which it is watching. The tap reader is the only place that knows the
tap's names, sweeps and re-attach logic. Adapters own nothing else.

**Encode once, fan out** ([C1](#c1), [C3](#c3), [C5](#c5), [C6](#c6)). A
live source is encoded once per (source, profile) into H.264 Annex B units,
and the same units go to every viewer of that profile: viewers cost
bandwidth, not encoder time. The encoder is resolved per host; its
low-latency flags stay inside the stage. Stored video does not use the live
encoder: it is a transcode whose product is a file in the playback cache,
the branch's Data-tab path unchanged.

**Cursor policies** ([C1](#c1), [C7](#c7), [C8](#c8)). A cursor is the rule
that decides which frame is on screen. Live surfaces _follow live_: show
the newest decoded frame, drop older ones, and buffer only what the
transport's jitter buffer asks for, never a fixed amount on top, so a slow
link shows fewer frames rather than older ones and no catch-up seek is
needed.
Stored surfaces are _paced_: rate times wall clock, seekable, buffered
ahead freely. A surface picks one.

**Presenter** ([C7](#c7), [R2](#r2)). One client component that decodes
units, remembers the capture time of the frame it last painted, and paints.
Overlays are separate streams keyed by the same timestamps, painted over
the matching frame or dropped if late. State, actions and the URDF pose are
read at the painted frame's time if the unconfirmed half of [R2](#r2)
stands; the client keeps the last few hundred milliseconds of state for
that lookup, which the existing 33 ms state poll already delivers.

**Profiles** ([F2](#f2), [F3](#f3)). `low`, `medium`, `full` for both live
and stored, chosen by the viewer. A profile is a resolution and a bitrate,
never an encoder preset, so it means the same picture on every host. It is
part of a stream's or cache entry's identity and is never consulted by a
job.

**Backend resolution** ([C8](#c8), [R6](#r6)). Each stage has one interface,
a software reference, and accelerated backends resolved once per host and
logged, with the three-way knob the codebase already uses. Today: NVENC for
live encode (floor: libx264 at 2–5 ms per frame), NVDEC and CUDA for the
jobs while the Data tab stays on the CPU path until a measured need
([A8](#a8) lists each stage's accelerator and floor, and the Mac blockers
outside this design).

**When the robot host and the GUI server split** ([C3](#c3), [F5](#f5)),
the live source adapters and the encode stage move to the robot host, where
the frames and their timestamps originate, and the GUI server forwards the
units. Nothing else changes, which is why timestamps travel with the bytes
from the start.

### Invariants

- Processing never consults viewer settings. ([F3](#f3))
- Quality is part of the identity: two profiles are two streams or two
  cache entries; a recipe change is a new entry, never an invalidation.
  ([C1](#c1))
- One clock: every frame carries its capture timestamp end to end, and
  anything shown beside it is looked up by that timestamp. ([C7](#c7))
- Live buffers only for the link's jitter, never a fixed amount on top.
  ([C1](#c1), [C8](#c8))
- Overlays skip, never stall. (Part 1)
- One encode per source and profile, whatever the number of viewers.
  ([C1](#c1), [C3](#c3))
- Nothing is added to the run loop; no second channel into the run process.
  ([F1](#f1), [F4](#f4))
- No backend is visible past its stage; the [wire format](#g-wire-format)
  is H.264 Annex B whichever encoder produced it. ([C8](#c8))

The teleop latency budget, term by term with the measured ones filled in,
is in [A9](#a9).

## Part 5: Open decisions

<a name="part-5"></a>

1. **Live transport.** _WebRTC_: the browser's [jitter buffer](#g-jitter-buffer)
   and hardware decode, works on plain http, needs a WebRTC peer on the
   server (aiortc or a small native relay). _MSE with in-browser remux_:
   JavaScript wraps the Annex B units into fragmented MP4 in the browser,
   which removes the server-side container hold but keeps a buffered player
   and its catch-up seek. _WebCodecs over WebSocket_: simplest sync, lowest
   latency, needs HTTPS ([O6](#o6)). Recommendation: measure WebRTC and
   WebCodecs-over-HTTPS side by side with the same source before choosing;
   either beats the branch by two frame periods, and the difference is
   operational (certificates against a relay).
2. **Per-camera streams or a mosaic.** Per-camera costs more encoder
   sessions but lets the client lay out, pick and enlarge, and needs no
   per-robot layout table. Recommendation: per-camera; the session budget
   allows it and the layout table is the part of the branch that does not
   generalise.
3. **Stored AV1: play directly or transcode.** Chrome and Firefox decode
   AV1, and `full` is already a re-wrap. Unverified: seeking in the browser
   when the file has a keyframe only every two frames, and hardware AV1
   decode on the laptops in use.
4. **How `auto` picks a profile.** A manual selector exists; an RTT or
   throughput probe could choose. Not decided.

## Part 6: Order of work

Review happens on this file, in a draft PR on `feat/camera-video-transport`,
with line comments; decisions are written back here. Each step is a PR
small enough to read:

1. Timestamps end to end on the existing branch path, and state read at the
   painted frame's time. No transport change; it makes [R2](#r2) testable.
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
by the Run-tab view and the overlay worker. [O3](#o3) and [A3](#a3).

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
with a round-trip time of 72.2 ms: 1.18 Mbit/s; the first frame appears
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
  by about 100 ms on the Tailscale link (72 ms round trip). The picture is
  0.4–0.6 s old (measured). The gap between the two is what the readouts
  lead the picture by. On `main` the gap is smaller, because a polled JPEG
  is one round trip old, not a video pipeline old. Whether the gap matters
  to an operator is the unconfirmed half of [R2](#r2). A latest-only source does mean the
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
  starts a data publisher that writes decoded episode frames into the same
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
  the middle of an update; the reader reports a torn read rather than
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
  cap[capture] --> shm["tap write<br/>unmeasured"] --> samp["sample<br/>source cadence, not 10 Hz"] --> enc["encode<br/>1–5 ms median, NVENC tail to 160 ms"] --> mux["container<br/>0 with Annex B"] --> net["network<br/>RTT 72 ms Tailscale measured; 400 ms reported"] --> jb["jitter buffer<br/>transport-dependent"] --> dec["decode<br/>hardware, unmeasured"] --> paint[paint + sync]
```

The two terms the redesign controls are sampling (the encoder runs at the
source's frame rate, not a 10 Hz resample) and the container plus player
buffer (Annex B into a decoder that presents immediately). Both are measured
at one frame period each on the branch ([A6](#a6)); together they are the
pipeline's own budget over the network's, and [R2](#r2) says a term stays
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
