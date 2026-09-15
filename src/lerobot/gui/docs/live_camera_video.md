# Live camera video

Status: proposed
State of the work: [https://github.com/TheWisp/lerobot/pull/226](https://github.com/TheWisp/lerobot/pull/226) until the tracking issue is opened

The Run tab shows a run's cameras as one JPEG per camera per tick, polled
twenty times a second. Over the link to the rig each picture is a round trip
old before it is shown, and every picture costs a full frame of bytes, so the
operator sees a slideshow that lags the robot. That operator is the person
this design is for: a remote operator whose commands reach the robot by their
own path, and whose only feedback is this picture — the same picture an
observer watches to decide when to press Stop. Today neither can act on it.

**Proposal.** The server pushes one encoded frame per camera, at the camera's
own rate, the moment it exists: no container, nothing queued, the newest frame
always winning. Any overlay is drawn onto the frame on the server before
encoding, from whichever adapter produced it, and the frame never waits for it.
The robot's state and the commanded action travel on the same connection, one
message per cycle, so the joint readouts and the URDF tile are as fresh as the
picture with nothing polled. The transport is WebRTC; the whole pipeline runs
on the GPU where there is one, and the encode runs once per camera for every
viewer; the JPEG path stays as the other choice of the same control the Data
tab already has.

## Scope

In:

- The Run tab's camera tiles, the URDF tile and the state/action readouts
  during a run — teleop, record, replay, a policy — at the
  [Low Bandwidth](#g-profile) profile.
- Overlays on those tiles: the live SAM3 preview, policy saliency, and any
  future [adapter](#g-adapter), drawn on the server.
- One viewer at P0; several viewers of one run at P1.
- The link constant, shared with the Data tab: its `low` profile is re-anchored
  on the same constant in the same change ([O16](#o16)).

Out, and why:

- **The Robot tab.** Between runs the GUI process owns the cameras itself, so
  it is the same pipeline with a different frame source and no policy. After the
  Run tab, because the run case is the one the requirement is about.
- **Adapting to the link.** A fixed [profile](#g-profile) first, for the reason
  [`dataset_playback.md`](dataset_playback.md) gives: whether one is enough is
  the first thing to find out, and a mechanism that moves between rungs has to
  be understood before a stall can be explained. WebRTC's bandwidth estimate is
  where adaptation would attach later.
- **Removing the JPEG path.** It is the comparison this path is measured
  against and what the operator switches to when the stream fails; removing it
  is the step after,
  as for the Data tab.

Non-goals:

- **The remote command path.** Keyboard, space mouse or VR commands from the
  operator to the robot are low-dimensional and travel by their own transport;
  nothing in this design carries them or depends on how they travel. The two
  add up in the operator's loop, which is why this design states the picture's
  budget on its own.
- **A view for a headset.** Whether a VR client shows these cameras, and how, is
  decided later; a client on the operator's machine can receive the same stream.
- **Auditing what a policy is fed.** The picture is a transcode with an overlay
  drawn on it; what the policy sees is the tap's frame in the run process.

## Requirements

The picture is the remote operator's only feedback, and the observer's means to
stop a run in time, so **age and continuity come first and fidelity last**. The
host that serves the picture is the host running the policy and the recorder,
so not slowing the run is a P0 rather than a courtesy.

Conditions, named once:

- **Local** — server and browser on the same workstation.
- **Class link** — the link the design is built for: Lighthouse's and Chrome
  DevTools' _Slow 4G_ preset — 1.6 Mbit/s down, 750 kbit/s up, 150 ms round
  trip — which Lighthouse describes as roughly the bottom quarter of 4G
  connections and the top quarter of 3G ([E10](#e10)). One constant in code, shared with the Data tab,
  names it ([C11](#c11)); the profile, the budget, the shaper and the tests
  derive from it, and every target below is stated against it.
- **Rig link** — Tailscale to fc500t as measured: round trip 233–265 ms and
  2.2–2.8 Mbit/s at the browser on 2026-09-07 ([E2](#e2)); it varies by day.
  Slower in round trip and faster in throughput than the class. Runs over it
  are evidence, dated; they do not define a target.
- **Workload** — the rig: four cameras at 960×600 and 1280×720, 30 fps, one of
  them carrying a live overlay, during a teleop or policy run with the recorder
  on.

| #                       | Pri | Requirement                                                    | Target                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Why that target                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Checked by                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------- | --- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <a name="r1"></a>**R1** | P0  | The freshest observation through the lowest-latency pipeline   | Every stage forwards the newest frame it has and holds nothing; the pipeline runs on the GPU wherever it can; no camera waits for another and no readout waits for a picture. Measured two ways: median [age](#g-age) at the eye over a ten-minute session — Local ≤ 50 ms at 30 fps, Class link ≤ 125 ms, Rig link Local plus half that day's round trip — and no gap between painted frames longer than three periods except while the receiver reports loss | Decision 1 (2026-09-06, reaffirmed 2026-09-12): the picture's delay must be nearly the camera's exposure plus the network alone — one frame period of capture cadence, a pipeline share measured under a millisecond on the GPU ([E9](#e9)), half a display refresh. The two numbers are one mechanism seen twice, its level and its absence of stalls, and a right mechanism meets both (2026-09-14); the earlier branch's 0.4 s median ([E3](#e3)) is what a wrong one looks like. Holding a fast camera for a slow one, or a readout for a picture, would spend the budget on an alignment nobody asked for. | The capture time on every frame ([C3](#c3)); the page's clock offset estimated over the data channel; capture-to-paint per frame in the controls bar, and the painted-interval series. Mechanism tests: the [mailbox](#g-mailbox) never holds two; the stages run on both devices. In the suite, relatively: the WebRTC path's Local age against the Local instrument's on the same machine ([O10](#o10)). The numbers in dated runs; the rig's GPU and CPU share during a recording ([to measure](#to-measure)). |
| <a name="r2"></a>**R2** | P0  | The first picture follows the first frame                      | Every camera painted within two round trips of frames becoming available — 300 ms on the Class link — on a connection opened before the run's frames exist                                                                                                                                                                                                                                                                                                     | Once the connection is up, the first picture is one forced keyframe ([O17](#o17)), one way across the link, and a decode; nothing in that path justifies more than two round trips. The connection's own setup — signalling, ICE, DTLS — is several round trips ([O15](#o15)) and is paid when the tab is shown with Low Bandwidth selected or when Launch is pressed, behind the run's own start, which takes seconds.                                                                                                                                                                                         | Under the shaper at delays _d_, the time from the first tap frame to every tile painted grows by no more than 2 _d_; the connection's setup time measured and reported separately; the rig, dated.                                                                                                                                                                                                                                                                                                                |
| <a name="r3"></a>**R3** | P0  | The stream fits the shared link constant with margin           | All cameras together ≤ three quarters of the constant's downlink — 1,200 kbit/s — the per-camera bitrate derived as that budget divided by the camera count, the frame rate untouched                                                                                                                                                                                                                                                                          | One constant names the link for both tabs (2026-09-14), and the Data tab's profile is re-anchored on it in the same change ([O16](#o16)). A quarter is kept for the data channel, retransmissions and the rest of the page. The Data tab reached 30 fps for four cameras at 320 wide at 690 kbit/s ([`dataset_playback.md` E3](dataset_playback.md#e3)), so the width fits the budget. Resolution and bitrate are the knobs; a lower frame rate costs a frame period of age per frame ([O4](#o4)).                                                                                                              | The sender's byte count on the synthetic four-camera workload against the budget derived from the constant ([C11](#c11)); the shaper runs at the class rate; the Data tab's smoothness tests re-capped on the same constant.                                                                                                                                                                                                                                                                                      |
| <a name="r4"></a>**R4** | P0  | State and action ride the stream                               | The data channel carries every [cycle](#g-cycle)'s state and action; the readouts and the URDF tile show the newest received, as fresh as the pictures; no `obs-stream/state`, `urdf-viz` or `obs-stream/image` request runs while streaming                                                                                                                                                                                                                   | Today the URDF tile polls thirty times a second and the pictures twenty times per camera, each as old as its own request ([O11](#o11)); one connection carries all of it with nothing to poll. Pairing a readout to one particular frame is not required (2026-09-14): each shows its newest.                                                                                                                                                                                                                                                                                                                   | A synthetic tap writer with the cycle encoded in the state: the readouts show the newest cycle written; the request log shows no poll while the stream is up.                                                                                                                                                                                                                                                                                                                                                     |
| <a name="r5"></a>**R5** | P0  | An overlay never delays the picture                            | With the adapter slowed tenfold, the painted-frame interval distribution is unchanged within one frame period; the overlay drawn is the newest; the reported lag equals the injected delay in cycles                                                                                                                                                                                                                                                           | Overlays are computed, not stored, and expensive; they may skip frames. A picture that waits for one inherits its rate — which is what the Data tab's stream does today and what [#225](https://github.com/TheWisp/lerobot/issues/225) records as wrong.                                                                                                                                                                                                                                                                                                                                                        | A test adapter with an injected delay; the documented rule in [`overlays.md`](overlays.md), "latest-wins, no frame pairing", holds on this path.                                                                                                                                                                                                                                                                                                                                                                  |
| <a name="r6"></a>**R6** | P0  | The run is not slowed by being watched                         | The loop's cycle-time median and 95th percentile with one viewer lie within the spread between two runs without a viewer; the run process's change is the tap header alone                                                                                                                                                                                                                                                                                     | The same host runs the policy and the recorder. The tap's contract already says no control path may depend on the write ([O3](#o3)); a viewer that costs the loop time would defeat the purpose of watching.                                                                                                                                                                                                                                                                                                                                                                                                    | Three runs on the virtual robot — two without a viewer, one with — reading the run's own latency report; the diff of the run process.                                                                                                                                                                                                                                                                                                                                                                             |
| <a name="r7"></a>**R7** | P0  | Full Quality is untouched, and Low Bandwidth reports its state | At Full Quality the tab behaves as today and no stream is opened. At Low Bandwidth the controls bar shows connecting, streaming with the age, or failed with the reason; a failure leaves the last frame on the tiles with the reason beside them, never a blank tile, and the operator switches paths by the control                                                                                                                                          | The JPEG path is the comparison this path is measured against. An automatic switch between paths mid-run is a second state machine to get right; a visible state and one control are enough (2026-09-13).                                                                                                                                                                                                                                                                                                                                                                                                       | The Run tab's existing tests at Full Quality pass unchanged and make no signalling request; with the encoder unavailable or the connection refused, the reason is shown, the last frame stays, and no request leaves for the JPEG path.                                                                                                                                                                                                                                                                           |
| <a name="r8"></a>**R8** | P1  | Several people can watch one run                               | One encoder per camera for any number of viewers; a second viewer on a link with injected loss leaves the first viewer's median age within one frame period of its single-viewer value; a joiner's first picture within R2's bound, from a keyframe forced on join ([O17](#o17))                                                                                                                                                                               | Two operators watching should not cost the host two encodes ([O14](#o14)); one bad link must not become everyone's. Agreed as P1 on 2026-09-13.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Two browser contexts, one with injected loss: encoder count is one; the other's age unchanged; join-to-first-picture within R2's bound.                                                                                                                                                                                                                                                                                                                                                                           |

R4, R5 and R7 are behaviours; the rest are measurements,
met or missed in dated runs on the rig and the workstation, recorded under
`docs/proofs/`; the guardrails in the test suite compare against a baseline on
the same machine — the Local instrument, a run without a viewer, the profile's
own setting — rather than against these numbers, so a slow CI machine cannot
fail them and a fast one cannot pass them for the wrong reason.

## Observations

Each is sourced to `main` at `ef6f5bf93` unless it says otherwise, and each
closes with what it forces.

<a name="o1"></a>**O1 — Both live tabs poll a full-resolution JPEG per camera.**
The Run tab ticks every 50 ms and sets one `img.src` per camera
([run.js L2412–L2438](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/static/run.js#L2412-L2438));
the Robot tab every 100 ms
([robot.js L768](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/static/robot.js#L768)).
The endpoint encodes the newest tap frame at JPEG quality 80, and its own note
puts the conversion and encode at 5–10 ms for a 720p frame
([run.py L1506](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/api/run.py#L1506)).
Over the Link each picture is one round trip old at best, and each is a full
picture of bytes: the same content as video measured 48× smaller
([E4](#e4)). → Pictures are pushed, not polled, and compressed as video.

<a name="o2"></a>**O2 — The [tap](#g-tap) is per-cycle and latest-value, with
no cycle identity and no capture time.** Every block carries a 24-byte header
of two sequence counters and the wall-clock write time
([obs_stream.py L113–L115](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/robots/obs_stream.py#L113-L115));
`write_obs` writes the scalar block and then each image block, each with its
own counter, and `write_action` is a third write
([L279–L300](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/robots/obs_stream.py#L279-L300)).
The camera's capture time exists — `OpenCVCamera` keeps `latest_timestamp` and
`read_latest()` returns it
([camera_opencv.py L437](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/cameras/opencv/camera_opencv.py#L437))
— but `read()` and `async_read()` return the array alone, so it is gone before
the observation is built. The writer is the last step of the observation
processor
([obs_stream.py L399](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/robots/obs_stream.py#L399),
[lerobot_record.py L968–L976](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/scripts/lerobot_record.py#L968-L976)).
→ Age needs a capture time and the overlay's lag needs a cycle number, both
written on every block of a cycle. That is a change to the tap's header, not
to the loop.

<a name="o3"></a>**O3 — The tap write is best-effort by contract, and the run
process owns the cameras.** The writer's docstring: no policy, control, safety
or recording path may depend on the publication succeeding
([obs_stream.py L399–L410](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/robots/obs_stream.py#L399-L410)).
During a run the run subprocess holds every camera and uploads each frame to
the GPU for the policy; the GUI never touches the device (the earlier design's
[A2](https://github.com/TheWisp/lerobot/blob/e7dfef2fed87ba11a17673f2fbe5ba4068aca3b6/src/lerobot/gui/docs/camera_video_transport.md#a2)).
→ The stream's work runs outside the run process, on a frame the view uploads
a second time — 1.7–2.8 MB per frame per camera at the Workload's sizes, by
arithmetic; the cost is unmeasured and not on any control path.

<a name="o4"></a>**O4 — Encoding is cheap; containers, resampling and buffered
players each cost a frame period.** Idle host with a 5090, 150 frames per row
([E1](#e1)): raw H.264 out of libx264 costs 2.2–4.6 ms at the median, out of
NVENC 1.1–2.1 ms; the fragmented-MP4 container holds each frame one period
(35 ms at 30 fps, 100 ms at 10) because it writes a frame's duration before the
frame; NVENC's default settings add two more periods; a 10 fps resample waits
up to 100 ms for the next sample. NVENC has the better median and the worse
tail in every row — worst frames 150–163 ms against libx264's 20–67. → No
container, no resample, no buffered player; the encoder runs at the camera's
rate; encode cost is a host-cost term, not an age term, at the Link's round
trip.

<a name="o5"></a>**O5 — The Link's round trip dominates, and its loss is
unmeasured.** 233–265 ms and 2.2–2.8 Mbit/s at the browser on 2026-09-07; 237
ms on 2026-09-06; 72 ms on the earlier branch's day ([E2](#e2)). Loss and
jitter during a session have never been measured. → Four cameras must fit two
to three megabits; R1's target is stated as an addition to the network because
the network is the largest term; whether packets are lost decides how the
transport must behave under loss ([O9](#o9)) and is [to measure](#to-measure).

<a name="o6"></a>**O6 — The Data tab's composited overlay stream is the closest
existing code.** It reads frames, blends the worker's RGBA overlay in NumPy,
resizes into an atlas, and pipes raw frames to an ffmpeg child that encodes
libx264 (ultrafast, zerolatency, baseline, a keyframe every second, no
B-frames) into fragmented MP4 for an MSE player
([overlays.py L737–L788](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/api/overlays.py#L737-L788),
[L912–L975](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/api/overlays.py#L912-L975),
[overlay_stream.js](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/static/overlay_stream.js)).
Three of its choices are the ones O4 and R5 exclude: the container, the
buffered player, and a wait for each frame's overlay
([#225](https://github.com/TheWisp/lerobot/issues/225)). Its input is a pipe
with back-pressure — a queue, so a slow encoder ages every frame behind it.
→ The shape carries over — frames, blend, encode, stream — with raw H.264 out,
a mailbox in, and no wait.

<a name="o7"></a>**O7 — Overlays are produced by [adapters](#g-adapter) on the
GPU, and leave it as a picture.** The worker builds its adapter on CUDA
([process_worker.py L107](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/process_worker.py#L107)),
reads the tap
([standalone.py L630](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/overlays/standalone.py#L630)),
and publishes one RGBA overlay per camera through shared memory
([overlay_ipc.py L169](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/overlays/overlay_ipc.py#L169));
the tensor becomes NumPy at
[adapters.py L201](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/overlays/adapters.py#L201).
Adapters include SAM3 tracking and policy saliency
([L519](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/overlays/adapters.py#L519),
[L1303](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/overlays/adapters.py#L1303)),
and one overlay — depth edges — runs inside the run process as a processor
step, so the tap already carries it drawn
([depth_edge_processor.py L198](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/processor/depth_edge_processor.py#L198)).
The Run tab today layers the RGBA as a PNG over each tile
([overlays.py L1709–L1712](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/api/overlays.py#L1709-L1712)).
The documented rule for compositing is "latest-wins, no frame pairing"
([overlays.md L224](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/docs/overlays.md#L224)).
→ The stream takes an overlay from whichever adapter produced it, without
knowing which. An adapter reads a stamped frame, so the cycle its overlay was
computed for is known and can travel with the frame it is drawn on.

<a name="o8"></a>**O8 — The GPU pieces exist; only the encoder is unused.**
`GpuMaskComposite` reproduces the saved-mask composite on the device for
training, ~0.2 ms per frame batched against 4.6–7.3 on the CPU, pinned to the
CPU result within two levels ([E7](#e7)); the training pipeline resizes on the
device; `PyNvVideoCodec` ≥ 2.2.2 is a declared dependency on Linux and Windows
with no macOS wheel
([pyproject.toml L187–L190](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/pyproject.toml#L187-L190))
and only its decoder is called
([gpu_data_pipeline.py L166](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/datasets/gpu_data_pipeline.py#L166));
PyAV is under the `av-dep` extra; aiortc is not declared. → Nothing new is needed
on the device but an encoder invocation: the overlay stays there past
`adapters.py:201`, the blend and the encode run there, and the resize and the
binding already present are reused; the training pipeline is not refactored
for it. Without a GPU the same stages run on CPU tensors, with libx264 in the
encoder's place.

<a name="o9"></a>**O9 — WebRTC in Python sends what it is handed, and shares
the encoded frame across viewers but not the packets.** In aiortc at
`8a28646` (read 2026-09-13, [E5](#e5)) the sender takes what the track hands
it: a raw frame is encoded, an already-encoded packet is split into RTP
payloads as-is — the path its own `MediaPlayer(decode=False)` uses. Each peer
connection has its own sender, sequence numbers and encryption keys, so packets
are built per viewer while the encoded frame is shared. A receiver's keyframe
request sets a flag that only the encode path reads; on the pre-encoded path it
is ignored. → One encoder per camera feeds a track of packets; fan-out is at
the encoded-frame level ([R8](#r8)); the keyframe cadence is the server's to
set — one per second, as O6's command already does — or the request is wired
through by hand.

<a name="o10"></a>**O10 — What the browser offers depends on the origin, and
the rig's origin is now HTTPS.** WebCodecs needs a secure context; WebRTC and
MSE do not (the earlier design's
[A6](https://github.com/TheWisp/lerobot/blob/e7dfef2fed87ba11a17673f2fbe5ba4068aca3b6/src/lerobot/gui/docs/camera_video_transport.md#a6),
probed 2026-09-04). The rig serves over HTTPS (verified 2026-09-12). Chromium
151 exposes `jitterBufferTarget` and `playoutDelayHint` on a WebRTC receiver
([`camera_video_pipelines.md`](https://github.com/TheWisp/lerobot/blob/594fe772f12f9923a397bd6fbab1f8ccb3f1b2c1/src/lerobot/gui/docs/camera_video_pipelines.md)),
and `requestVideoFrameCallback` reports a painted frame's RTP timestamp, which
is what the page measures a painted frame's age from ([to verify](#to-measure)
on the Chromium we ship). A WebCodecs H.264 decoder and per-camera paint exist for the
Data tab
([chunk_player.js L326–L352](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/static/chunk_player.js#L326-L352)).
→ WebRTC is the product transport ([C6](#c6)); the same encoded frames over a
streamed HTTP response into WebCodecs is the Local measuring instrument, with
no transport in the way.

<a name="o11"></a>**O11 — State, action and the URDF tile are polled apart from
the pictures.** `/obs-stream/state` returns the newest observation and action
with their write times
([run.py L1397](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/api/run.py#L1397));
the URDF tile is an iframe
([run.js L2375](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/static/run.js#L2375))
that polls `/urdf-viz?source=state` every 33 ms
([urdf_viz.html L640–L651](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/static/urdf_viz.html#L640-L651));
pictures are polled every 50 ms (O1). Each is as old as its own request and
nothing pairs them. On the Data tab the tile follows the playhead's frame. →
State and action ride the stream per cycle; the tile and the readouts show the
newest cycle received, as fresh as the pictures; the polls stop while the
stream is up.

<a name="o12"></a>**O12 — The Data tab already has the control.** A dropdown in
its controls bar, Full Quality or Low Bandwidth, kept per browser in
`localStorage` under the earlier prototype's key
([index.html L137–L141](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/static/index.html#L137-L141),
[app.js L28–L60](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/static/app.js#L28-L60)).
→ One setting describes the link, not a tab: the Run tab reads the same key and
shows the same control in the same position.

<a name="o13"></a>**O13 — No teleop input crosses the network today.** Leader
arms are USB on the rig; the Quest opens `https://<LAN-IP>:8443`
([configuration_quest_vr.py L27–L32](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/teleoperators/quest_vr/configuration_quest_vr.py#L27-L32));
the phone is found by `hebi.Lookup()` on the LAN
([teleop_phone.py L98–L100](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/teleoperators/phone/teleop_phone.py#L98-L100)).
Remote commands are planned and are a separate design (decided 2026-09-12). →
The stream is designed for an operator whose commands arrive by another path;
its budget is the picture's share of that operator's loop.

<a name="o14"></a>**O14 — What the earlier Run-tab branch got wrong.**
`feat/camera-video-transport` tiled the cameras into one mosaic whose layout
knew three camera names, resampled to 10 fps, muxed into fragmented MP4 for
MSE with a catch-up seek, and started one ffmpeg per open browser tab; measured
over the Link at a 72 ms round trip, 1.18 Mbit/s and an age of 0.4 s median,
0.60 s at the 95th percentile ([E3](#e3)). → One stream per camera, so the
browser lays out and enlarges as it already does; one encoder per camera shared
by viewers; no resample; no buffered player.

<a name="o15"></a>**O15 — How the transport moves a frame.** In aiortc ([E5](#e5))
an encoded frame is cut into RTP packets of at most 1,300 bytes — a 2 KB
P-frame at the profile is two packets, a 30 KB keyframe about twenty-five —
and sent as soon as the frame is handed over. The sender keeps a history of
sent packets and retransmits on the receiver's NACK; a PLI or FIR sets a
keyframe flag that only an internal encoder reads; the receiver's bandwidth
estimate is applied only to an internal encoder. Each video track is its own
RTP stream with its own sequence numbers and its own receive-side jitter
buffer, so frames of different cameras arrive and are presented independently.
Setting a connection up is several round trips — signalling, ICE, DTLS — before
the first packet. → Per-camera tracks do not present the same cycle at the
same instant ([R1](#r1)); loss recovery and keyframe requests are per track
and, on the pre-encoded path, the server's to honour ([O17](#o17)); the connection
is opened before it is needed ([R2](#r2)).

<a name="o16"></a>**O16 — The Data tab's profile is anchored on width and
quality, not on a link.** `PROFILES["low"]` is 320 wide at constant quality 26
([chunk_playback.py L62](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/gui/api/chunk_playback.py#L62));
no bandwidth constant exists in the GUI package, and the smoothness test caps
its emulated link at a multiple of the content's own bytes
([test_low_bandwidth_smoothness_playwright.py L159](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/tests/gui/test_low_bandwidth_smoothness_playwright.py#L159)).
→ One constant names the link both tabs are built for; the live path derives
its profile, budget, shaper and tests from it, and the Data tab's profile is
re-anchored on it in the same change ([C11](#c11)).

<a name="o17"></a>**O17 — The GPU encoder holds three frames unless flushed, and
gives keyframes on demand.** `PyNvVideoCodec`'s encoder on the workstation's
5090 ([E9](#e9)): a submitted frame's bitstream comes back three submissions
later — 100 ms at 30 fps — while a flush after each submission returns it in
0.24 ms at the median (0.32 for a four-tile atlas), keeps the session, and
leaves the keyframe cadence as configured. `NV_ENC_PIC_FLAG_FORCEIDR` on a
submission yields an IDR frame, with the parameter sets when also flagged. →
The encoder is driven with a flush per frame; a keyframe is forced when a
viewer's stream starts or a viewer asks for one, which is what stands in for
the request aiortc drops on the pre-encoded path ([O9](#o9)).

## Constraints and freedoms

<a name="c1"></a>**C1** One encoded frame is pushed per camera at the camera's own rate, in raw H.264 with no container
([O4](#o4), [O14](#o14), [R1](#r1), [R3](#r3)).

<a name="c2"></a>**C2** Between the tap and each encoder sits a
[mailbox](#g-mailbox): one value, a new frame replacing an unread one. The
encoder takes the newest frame when it is free and nothing is ever queued
behind a slow frame. The page paints a frame on arrival and drops any decoded
frame older than the newest ([O6](#o6), [R1](#r1)).

<a name="c3"></a>**C3** Every tap block written in one [cycle](#g-cycle)
carries the cycle number and the capture time. That is the only change to the
run process ([O2](#o2), [O3](#o3), [R1](#r1), [R5](#r5), [R6](#r6)).

<a name="c4"></a>**C4** The overlay is drawn on the server onto the frame it is
newest for, and the frame never waits; the cycle the overlay was computed for
travels with the frame ([O7](#o7), [R5](#r5)).

<a name="c5"></a>**C5** The stream's work — upload, adapter, resize, blend,
encode — runs in the [pipeline process](#g-pipeline-process) on the GPU, never
in the run loop; the GUI process holds only the senders ([O3](#o3),
[O8](#o8), [R6](#r6), [R1](#r1)).

<a name="c6"></a>**C6** The transport is WebRTC: one peer connection per
viewer, opened before frames exist, carrying one video track per camera and one [data channel](#g-data-channel) for the cycle's state and
action. The encoded frame is shared across viewers; packets are built per
viewer; tracks present independently ([O9](#o9), [O15](#o15), [R1](#r1),
[R2](#r2), [R8](#r8)).

<a name="c7"></a>**C7** One encoder per camera per
[profile](#g-profile), at the width and the bitrate the constant derives, a
keyframe every second and none held back: flushed per frame, and forced on a
viewer's start or request ([O4](#o4), [O6](#o6), [O17](#o17), [R2](#r2),
[R3](#r3), [R8](#r8)).

<a name="c8"></a>**C8** One pipeline of device-agnostic stages: on the GPU where
there is one, on CPU tensors where there is not and in CI. The encoder is the
only backend switch — NVENC or libx264 — resolved per host as the other
backends are ([O8](#o8), [R1](#r1)).

<a name="c9"></a>**C9** The JPEG path is untouched. Low Bandwidth opens the
stream and reports its state; nothing switches paths on its own ([O12](#o12),
[R7](#r7)).

<a name="c10"></a>**C10** Stop is the HTTP request it is today and has no part
in the stream; nothing in the stream is on its path.

<a name="c11"></a>**C11** One constant names the [class link](#g-class-link) for both tabs.
The profile's per-camera bitrate, the stream budget, the shaper's rate and
round trip and every test's cap derive from it ([O16](#o16), [R3](#r3)).

Free, within those:

<a name="c12"></a>**C12** The profile's width, against the per-camera bitrate
the constant derives — [to measure](#to-measure). 320 wide, as the Data tab's
profile, is the starting point.

<a name="c13"></a>**C13** How a viewer's keyframe request reaches the encoder:
a message on the data channel, or the sender's RTCP feedback read by the
pipeline ([O15](#o15), [O17](#o17)). Decided by the loss measurement.

## Architecture

**Today** ([O1](#o1), [O7](#o7), [O11](#o11)). Three readers of the tap, and
a page that polls each of them on its own clock.

```mermaid
flowchart LR
  subgraph run["run subprocess"]
    loop["run loop<br/>cameras → policy → action"]
    tap["the tap<br/>latest value per block:<br/>images, state, action"]
    loop -->|"last CPU step"| tap
  end
  subgraph worker["overlay worker (GPU)"]
    ad["adapter"] --> ov["overlay RGBA<br/>per camera"]
  end
  subgraph gui["GUI server"]
    jpg["JPEG per camera<br/>quality 80, cached per sequence"]
    png["overlay PNG per camera"]
    st["state and action<br/>joint angles for the URDF tile"]
  end
  subgraph page["Run tab"]
    img["img per camera<br/>polled every 50 ms"]
    ovimg["overlay img per camera<br/>same tick"]
    urdf["URDF tile<br/>polls every 33 ms"]
  end
  tap --> ad
  tap --> jpg
  tap --> st
  ov --> png
  jpg -->|"one request per camera per tick"| img
  png -->|"one request per camera per tick"| ovimg
  st -->|"one request per poll"| urdf
```

**After** — the addition beside the JPEG path ([reuse](#reuse),
[beside the JPEG path](#beside-jpeg)). Yellow is new, blue is changed, plain
is unchanged or reused as is.

```mermaid
flowchart LR
  subgraph run["run subprocess"]
    loop["run loop"]
    tap["the tap<br/>+ cycle number, capture time"]
    loop -->|"last CPU step"| tap
  end
  subgraph worker["pipeline process (GPU) — today's overlay worker"]
    up["upload once per frame"]
    ad["adapter, optional<br/>overlay + its cycle"]
    mb["mailbox per camera<br/>newest frame wins"]
    bl["resize to profile<br/>blend newest overlay"]
    enc["encoder, flushed per frame<br/>NVENC, or libx264 without a GPU"]
    up --> ad
    up --> mb --> bl --> enc
    ad --> bl
  end
  subgraph gui["GUI server"]
    jpg["JPEG, overlay PNG, state endpoints<br/>Full Quality, unchanged"]
    pc["peer connection per viewer<br/>video tracks + data channel"]
  end
  subgraph page["Run tab"]
    ctl["control<br/>Full Quality or Low Bandwidth"]
    img["img tiles and their polls<br/>Full Quality"]
    vid["video tile per camera<br/>readouts + URDF tile from the data channel<br/>Low Bandwidth"]
  end
  tap --> up
  tap --> jpg
  ad -->|"overlay RGBA, as today"| jpg
  enc -->|"encoded frames + stamps"| pc
  tap -->|"state, action, per cycle"| pc
  jpg -->|"polled, as today"| img
  pc -->|"WebRTC"| vid
  ctl --> img
  ctl --> vid
  classDef new fill:#fff4d6,stroke:#b8860b,stroke-width:2px
  classDef changed fill:#e8f1fb,stroke:#3b6ea5,stroke-width:2px
  class up,mb,bl,enc,pc,vid,ctl new
  class tap,ad changed
```

**The tap header** ([C3](#c3), [O2](#o2)). Each cycle the writer stamps every
block it writes with the cycle number and one capture time. The capture time
comes from the camera when its backend provides one and from the moment the
observation was assembled when it does not, and the header says which. Readers
that ignore the two fields are unaffected; the pipeline process reads them so
each overlay knows its cycle ([C4](#c4)).

**The pipeline process** ([C5](#c5), [C8](#c8), [R1](#r1)). Today's overlay
worker, which already holds the GPU and reads the tap, becomes the process
that also produces the stream: it uploads each camera's frame once, runs the
adapter on it when one is loaded, and carries the frame through resize, blend
and encode on the device. Encoded frames leave it with their stamps to the GUI
process, which holds the senders and nothing per pixel. The stages are written
on tensors without a device in them, so the same code runs on CPU tensors in
CI and on a host without an NVIDIA GPU, with libx264 in the encoder's place;
that is the one place a backend is chosen. Two consequences for the worker's
life: it must exist for the stream whether or not an adapter is loaded, and
its hold on the GPU is shared with the Data tab's batch jobs, which today may
take the worker over — a run's stream must not be what they evict.

**The mailbox** ([C2](#c2), [O6](#o6)). One per camera in the pipeline
process, filled from the tap at the tap's rate, emptied by the encoder at the
encoder's. A frame that arrives while the previous is unread replaces it. The
encoder is the only reader, so the rule is one compare-and-swap, and it is
what makes [R1](#r1) a property rather than a hope: no stage can hold more
than one frame.

**The overlay and its lag** ([C4](#c4), [O7](#o7), [R5](#r5)). The frame is
resized to the profile first and the overlay resized to the same size; only
then is the newest overlay the adapter has produced for that camera drawn on
it, as a tensor operation. The order is not cosmetic: blending at the source
resolution costs about sixteen times more per frame than at the profile's on
the CPU ([E8](#e8)) and is measured in microseconds on the device at the
profile's size ([E9](#e9)). The overlay's cycle number rides in the data
channel message beside the frame's; the page shows the difference when it is
not zero. A frame with no overlay yet is sent bare. The adapter is whatever
the worker is running — SAM3, saliency, a future one — and the blend does not
know which.

**The encoder and the profile** ([C7](#c7), [C11](#c11), [C12](#c12),
[O17](#o17)). Per camera: resize to the profile's width, keeping
the aspect ratio and never upscaling; encode with a keyframe every second and
no B-frames, at the bitrate the constant derives — the budget divided by the
camera count — flushing after every submission so no frame is held; emit raw
H.264 with the [stamp](#g-stamp), the capture time in the transport's units.
A keyframe is forced when a viewer's stream starts and when a viewer asks for
one. On a host without NVENC the same stage calls libx264 through the existing
command with the container flags removed.

**Warm connection and the first picture** ([C6](#c6), [R2](#r2), [O15](#o15)).
The page opens its peer connection when the Run tab is shown with Low
Bandwidth selected, or at Launch, so signalling, ICE and DTLS are done while
the run itself is starting. When the tap appears, the pipeline starts, the
first frame is a forced keyframe, and the first picture is one way across the
link plus a decode away.

**Transport** ([C6](#c6), [O9](#o9), [O15](#o15), [R4](#r4)). A viewer's
connection carries one video track per camera, each fed by that camera's
encoder through a track that hands aiortc encoded packets, and one data
channel. Per cycle the server sends one message on the channel: the cycle
number, the stamp, the state, the action, and each camera's overlay cycle. The
readouts and the URDF tile update from each message as it arrives; nothing
waits for a picture, and no picture waits for a message. The stamp is what
the page measures age from: it reads the painted frame's stamp from
`requestVideoFrameCallback`, recovers the capture time it encodes, and
subtracts it with the clock offset applied. The cycle number can be shown
beside each tile; a difference between tiles is information, not a fault.

**The page** ([O11](#o11), [R4](#r4)). Each camera tile is a video element on its
track, with the receiver's buffer set to its minimum.
The overlay is already in the pixels, so nothing is drawn on top; the subtask
text overlay stays a DOM layer. The URDF tile and the readouts update from
the data channel instead of polling; their polls do not run while the stream
is up. Enlarging one camera is the tile grid's existing behaviour. Stop is the
button it is today ([C10](#c10)).

<a name="reuse"></a>**What is reused and what is new** ([O6](#o6), [O7](#o7),
[O8](#o8), [O11](#o11), [O12](#o12)).

| Part                                                              | Status                            | What                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Run loop, cameras, policy, recorder                               | unchanged                         | The stream never touches them                                                                                                                                                                                                          |
| The tap                                                           | changed                           | Two header fields per block — cycle number and capture time — and the writer stamping them ([C3](#c3)); a reader that ignores the fields is unaffected                                                                                 |
| The tap reader                                                    | reused                            | The same reader the JPEG endpoints and the worker use, re-attaching when a run recreates the tap                                                                                                                                       |
| Overlay worker, adapters, overlay buffer                          | reused, extended                  | The worker becomes the pipeline process; the adapter's overlay stays on the device past `adapters.py:201` for the blend and is still copied out for the PNG path; the worker publishes each overlay's cycle beside its sequence number |
| Resize on the device                                              | reused                            | The training pipeline's on-device resize                                                                                                                                                                                               |
| Encoder settings                                                  | reused                            | The Data tab stream's libx264 settings for the CPU backend, with raw H.264 out in place of fragmented MP4                                                                                                                              |
| `PyNvVideoCodec`                                                  | present, first use of its encoder | A declared dependency whose decoder the training pipeline already calls                                                                                                                                                                |
| Mailbox, pipeline stages, broadcaster                             | new                               | The stages in the pipeline process; the broadcaster in the GUI ([C2](#c2), [C5](#c5))                                                                                                                                                  |
| Peer connections, signalling, data channel, the pre-encoded track | new                               | aiortc, a new dependency, behind one signalling endpoint ([C6](#c6))                                                                                                                                                                   |
| The class-link constant                                           | new                               | One value the profile, the budget, the shaper and the tests derive from ([C11](#c11))                                                                                                                                                  |
| Per-cycle joint angles                                            | reused                            | The function the URDF endpoint calls today, run per cycle for the message                                                                                                                                                              |
| Run tab tile grid, focus, layout                                  | reused                            | A cell holds a video element instead of an `<img>` while streaming                                                                                                                                                                     |
| Run tab stream client                                             | new                               | Opens the connection early, feeds the readouts and the URDF tile from the data channel, reports age and state                                                                                                                          |
| The control                                                       | reused                            | The Data tab's mode module and its per-browser key, bound to a second dropdown                                                                                                                                                         |
| Overlay on or off per camera                                      | changed                           | Today a client-side gate on the overlay's URL; while streaming, a flag the pipeline blends by                                                                                                                                          |
| URDF tile                                                         | reused                            | Fed from the data channel instead of its own poll, as the Data tab feeds it from the playhead                                                                                                                                          |
| JPEG endpoints, JPEG cache, `obs-stream/meta`, `obs-stream/state` | unchanged                         | The JPEG path as it is                                                                                                                                                                                                                 |
| Local instrument                                                  | new, small                        | A streamed-response endpoint and the Data tab's decoder                                                                                                                                                                                |
| Synthetic tap writer, link shaper                                 | new                               | Test instruments                                                                                                                                                                                                                       |

<a name="beside-jpeg"></a>**Beside the JPEG path** ([C9](#c9), [R7](#r7),
[O1](#o1)). One tab, one viewer, one path at a time, chosen by the control.

- **Full Quality.** The viewer as today: image tiles polled every 50 ms, an
  overlay PNG per camera, the URDF tile's own poll. No signalling request is
  made.
- **Low Bandwidth.** The connection is opened when the tab is shown or Launch
  is pressed; when `obs-stream/meta` reports the tap, the pipeline starts and
  the tiles show the tracks. The overlay image layer stays hidden because the
  overlay is in the pixels; the subtask text layer stays a DOM layer; the
  readouts and the URDF tile update from the data channel; no poll runs.
- **Switching.** The control changed mid-run stops one viewer and starts the
  other, in either direction. Nothing switches on its own.
- **Server side.** The pipeline runs only while a peer connection is open: the
  first offer starts it, the last close stops it. The worker serves both paths
  from its one overlay: the PNG for one, the blend for the other. Stop, the run
  lifecycle and its status stream are untouched on both paths.

<a name="the-control"></a>**The control** ([O12](#o12), [C9](#c9)). The Data
tab's dropdown, shown on the Run tab in the same position: below the split
handle, above the bottom tabs, in a controls bar of its own. Both read and
write the one per-browser key, so changing it on either tab changes both.
Beside it, the stream's state: connecting; streaming, with the profile in use
and the measured age; or failed, with the reason.

```
Data tab (today)                              Run tab (proposed)
┌────────────────────────────────────┐        ┌────────────────────────────────────┐
│  [top]     [left_wrist] [right_w.] │        │  [top]     [left_wrist] [right_w.] │
│                       [URDF tile]  │        │                       [URDF tile]  │
├── drag handle ─────────────────────┤        ├── drag handle ─────────────────────┤
│ ▶ Play  [1x ▾]  [Full Quality ▾] … │        │ [Low Bandwidth ▾]  320 wide · age …│  ← controls bar
├────────────────────────────────────┤        ├────────────────────────────────────┤
│ timeline lanes                     │        │ Output │ Latency │                 │
│                                    │        │ …process log…                      │
└────────────────────────────────────┘        └────────────────────────────────────┘
```

**Failure** ([C9](#c9), [R7](#r7)). If the connection does not reach connected
or drops, or the server has no encoder, the controls bar says so with the
reason and the tiles keep the last frame they painted. The operator switches
the control to Full Quality if they want pictures now; the tab does not switch
for them.

**Several viewers** ([R8](#r8), [C6](#c6), [O17](#o17); P1). The encoder's
output goes to a broadcaster; each viewer's track subscribes and aiortc
packetizes the same encoded frame per viewer. A viewer that falls behind drops
to newest in its own queue. A joiner gets a forced keyframe.

**The Local instrument** ([O10](#o10)). The same encoded frames over a streamed
HTTP response, decoded with the Data tab's WebCodecs setup and painted on
arrival, with the stamp inline. It measures the pipeline's own age with no
transport in the way and is the baseline every WebRTC number is compared to.
It is not a product mode.

## Alternatives, and what this costs

What else would meet the requirements, and why it is not the proposal:

- **Keep polling JPEGs.** Fails R1 and R3 over the Link by O1 and O5: a round
  trip per picture, a full picture per frame. Stays as the fallback.
- **Push JPEGs instead of polling them.** Meets R1's structure — encode, one
  way, decode — and fails R3 on bytes: no inter-frame compression, so 48× the
  bytes of video at the same picture ([E4](#e4)), against a link of two to
  three megabits for four cameras.
- **Chunks, as the Data tab does.** A chunk is as old as its length; the Data
  tab's design says so itself — a live path encodes what a camera just produced
  and caches nothing. Fails R1.
- **Fragmented MP4 over MSE**, the earlier branch's form. One frame period in
  the container and a buffered player that needs a catch-up rule (O4, O14);
  measured at 0.4 s median age ([E3](#e3)). Fails R1.
- **A streamed HTTP response into WebCodecs as the product.** Meets R1 on a
  clean link and carries the stamp inline, and it is the Local instrument here.
  Over TCP a lost packet stalls every frame behind it for a round trip — 233–265
  ms on the Link — and it does so precisely when the link is bad. Kept as the
  case to revisit if the loss measurement comes back low.
- **One atlas of every camera in one track.** One encoder session and one
  stream for any number of cameras, and every tile the same cycle by
  construction — which mattered until synchronization was dropped
  (2026-09-14). Against it: a lost packet stalls every camera rather than one,
  every viewer receives every camera at one rate, and each tile is a copy out
  of the decoded frame rather than a video element on its own track. Revisit
  if encoder sessions run out with several profiles and viewers.
- **A CPU pipeline first, the GPU later.** Measured cheap on the workstation
  ([E8](#e8)), and rejected on 2026-09-13: it makes two paths of one, and it
  downloads the frame the adapter already holds on the device to blend it
  again. The device-agnostic stages give the CPU case without a second path.
- **Switch to the JPEG path by itself when the stream fails.** A second state
  machine to get right in the middle of a run, and a tab that changes its own
  picture source. A visible state and the one control are enough (R7).
- **Hold the fast cameras for the slow one.** Every tile then waits for the
  slowest track's jitter, which is R1 spent on alignment.
- **Encode in the run process from the policy's GPU tensor**, as the
  `camera_video_pipelines.md` blueprint proposed. Saves the view's second upload
  (O3) and puts work in the control loop that the tap's contract forbids
  depending on. Revisited if the second upload measures as a cost.
- **Send the overlay to the page as data**, as the Data tab does with saved
  masks. Saved masks are stored rows; a live overlay is computed, expensive and
  skips frames (O7). Baking it in on the server is one picture instead of two
  and no client work.
- **Do nothing.** A remote operator has no picture they can act on, and the
  observer's Stop is a guess.

What the proposal makes harder or more expensive:

- A new dependency with a protocol behind it — aiortc, signalling, ICE — where
  the JPEG path has none.
- The readouts cannot ride inside the video packets, so they need the data
  channel beside the tracks, and the age needs the stamp lookup on the page; a
  page that shows the video alone shows the picture and nothing else.
- Keyframe requests from a viewer are the server's to honour, not the
  library's (O9).
- A second upload of every frame while the run process keeps its own (O3).
- Two picture paths in the Run tab until the JPEG path is removed.
- A host without an NVIDIA GPU gets Low Bandwidth through the same stages on
  CPU tensors with libx264, measured only on the workstation ([E8](#e8)).
- The pipeline process must run for every stream, adapter or not, and its GPU hold has to be arbitrated with the Data tab's batch jobs.
- The Local instrument is a second, small sender to keep working.

## Open questions

None remain for the reader; the forks this document carried are decided
above — one track per camera and the tile's surface
([Alternatives](#alternatives-and-what-this-costs)), synchronization
([R1](#r1)), the fallback ([R7](#r7)), and the link constant ([C11](#c11)).

<a name="to-measure"></a>To measure — settled by a number, not by the reader:

- **Age at the eye**, Local, on the Class link through the shaper, and over the
  Rig link, from the carried capture time, for the Workload ([R1](#r1)). The
  Local number is the pipeline's share.
- **The rig's GPU and CPU share** for the stream during a recording, with the
  recorder's encoder and the policy running ([R6](#r6), [R1](#r1)).
- **Loss and jitter on the Rig link during a session**, dated ([O5](#o5),
  [C13](#c13)). It decides how a keyframe request travels and whether the
  streamed-HTTP alternative deserves a second look.
- **The profile's width** against the per-camera bitrate the constant derives
  ([C12](#c12)); and the encoder's overshoot at that bitrate, measured at
  about fifteen percent over the target on the workstation ([E9](#e9)), which
  the budget must absorb.
- **Packetization cost per viewer** in aiortc for four tracks at 30 fps
  ([R8](#r8); P1).

To verify, once: that the Chromium we ship reports the RTP timestamp in
`requestVideoFrameCallback` metadata for a WebRTC track ([O10](#o10)) — the
age measurement depends on it, and the fallback is the data channel's stamps
matched to painted frames by arrival order, which is approximate.

## Glossary

<a name="g-age"></a>**Age** — The time between a frame's capture by the camera
and its appearance on the viewer's screen. Measured in the page from the
[stamp](#g-stamp).

<a name="g-cycle"></a>**Cycle** — One pass of the run loop: read the cameras and
the state, run the policy or the teleoperator, send the action, write the tap.
The cycle number is the loop's counter, written on every tap block of that
pass.

<a name="g-stamp"></a>**Stamp** — A frame's capture time in the transport's
units, carried on the encoded frame and repeated in the data channel message
for the same cycle. The number the page measures age from.

<a name="g-tap"></a>**Tap** — The shared-memory blocks the run subprocess
writes each cycle — images, state, action — one latest value per block, read
by the GUI and the overlay worker. Defined in `robots/obs_stream.py`.

<a name="g-adapter"></a>**Adapter** — A model wrapped to produce an overlay
from a camera frame in the overlay worker: SAM3 tracking, policy saliency, and
whatever is added next. Defined in `overlays/adapters.py`.

<a name="g-profile"></a>**Profile** — A named quality setting for the stream: a
target width per camera and a bitrate. Low Bandwidth is the stream at the
profile; Full Quality is the JPEG path.

<a name="g-mailbox"></a>**Mailbox** — A one-value handoff between two stages in
which a new value replaces an unread one. The stage after it always takes the
newest, and nothing waits behind a slow value.

<a name="g-keyframe-group"></a>**Keyframe group** — A keyframe and the frames
that depend on it until the next keyframe; one second long here. A decoder can
start only at a keyframe.

<a name="g-data-channel"></a>**Data channel** — WebRTC's message pipe beside
the video tracks on the same connection; here it carries one message per
cycle.

<a name="g-jpeg-path"></a>**JPEG path** — The existing per-frame JPEG endpoints
and what the tabs do with them.

<a name="g-viewer"></a>**Viewer** — One browser page watching a run over one
peer connection.

<a name="g-class-link"></a>**Class link** — The link the design is built for, a
named public preset rather than a measurement of ours: Slow 4G, 1.6 Mbit/s
down, 750 kbit/s up, 150 ms round trip. One constant in code for both tabs; the profile, the
budget, the shaper and the tests derive from it.

<a name="g-pipeline-process"></a>**Pipeline process** — The process that holds
the GPU and carries a frame from the tap through adapter, resize, blend and
encode: today's overlay worker, extended. The GUI process receives its encoded
frames.

## Appendix: evidence

Numbers taken on a superseded branch are attributed to it so nobody re-measures
what is known, and so nobody mistakes them for a measurement of this design.

<a name="e1"></a>**E1 — Encoder and container costs.** Frame in to encoded
bytes out, idle host with an RTX 5090, 150 frames per row, script
`enc_latency2.py` on `feat/camera-video-transport`, recorded in that design's
[A10](https://github.com/TheWisp/lerobot/blob/e7dfef2fed87ba11a17673f2fbe5ba4068aca3b6/src/lerobot/gui/docs/camera_video_transport.md#a10)
(2026-09-04). Milliseconds.

```
encoder     shape         size@fps     median   p95     max
libx264     annexb        640x380@10      2.3     2.7    20.1
libx264     annexb        640x380@30      2.2    35.4    36.2
libx264     annexb        1280x720@30     4.6     5.4    66.7
libx264     fMP4          640x380@10    102.5   103.0   103.2
libx264     fMP4          640x380@30     35.4    68.7   102.3
h264_nvenc  annexb -delay 0  640x380@10   1.1     1.3   159.1
h264_nvenc  annexb -delay 0  640x380@30   1.2   101.0   163.2
h264_nvenc  annexb -delay 0  1280x720@30  2.1    35.5   150.4
h264_nvenc  fMP4 default  640x380@10    300.8   301.4   301.6
h264_nvenc  fMP4 -delay 0 640x380@30     34.6   101.5   165.5
```

<a name="e2"></a>**E2 — The Link.** Round trip 233–265 ms; 3.8–4.1 Mbit/s in a
single server-side stream, 2.2–2.8 Mbit/s as the browser measured it
(2026-09-07, [`dataset_playback.md` E1](dataset_playback.md#e1)). 237 ms by
ping on 2026-09-06, 20 packets, 235–244 ms, and 0.48 s to first byte for a
fresh HTTP request to the GUI; 72.2 ms on the day of the earlier branch's
measurement (that design's
[A1](https://github.com/TheWisp/lerobot/blob/e7dfef2fed87ba11a17673f2fbe5ba4068aca3b6/src/lerobot/gui/docs/camera_video_transport.md#a1)).
Loss and jitter: not measured.

<a name="e3"></a>**E3 — The earlier Run-tab branch, over the Link.**
`feat/camera-video-transport`, commit e0a76d076, at a 72.2 ms round trip: a
640×380 mosaic at 10 fps, libx264 into fragmented MP4 over MSE; 1.18 Mbit/s;
first frame 498 ms after the request; age 0.4 s median and 0.60 s at the 95th
percentile, not growing over a session (that design's
[A1](https://github.com/TheWisp/lerobot/blob/e7dfef2fed87ba11a17673f2fbe5ba4068aca3b6/src/lerobot/gui/docs/camera_video_transport.md#a1)).

<a name="e4"></a>**E4 — Bytes: JPEG against video.** On the rig's three-camera
dataset, the Data tab's JPEG flipbook needed 324 KB per tick, 78 Mbit/s at 30
fps — about 108 KB per camera picture; the same content as H.264 at 1280 wide
and 1500 kbit/s was 6.7 KB per frame, 48× less
(`feat/camera-video-transport`, commit a4b0db5c3, that design's
[A1](https://github.com/TheWisp/lerobot/blob/e7dfef2fed87ba11a17673f2fbe5ba4068aca3b6/src/lerobot/gui/docs/camera_video_transport.md#a1)).

<a name="e5"></a>**E5 — aiortc's sender.** Read on 2026-09-13 at commit
[`8a28646`](https://github.com/aiortc/aiortc/commit/8a2864630bf417d977a06dda8d0254b1c1501bfb)
(2026-07-17):
[`rtcrtpsender.py` L294–L332](https://github.com/aiortc/aiortc/blob/8a2864630bf417d977a06dda8d0254b1c1501bfb/src/aiortc/rtcrtpsender.py#L294-L332)
— a `Frame` from the track is encoded with the pending keyframe flag, anything
else is passed to the codec's `pack`;
[`codecs/h264.py` L298–L302](https://github.com/aiortc/aiortc/blob/8a2864630bf417d977a06dda8d0254b1c1501bfb/src/aiortc/codecs/h264.py#L298-L302)
— `pack` splits the Annex B bitstream into NAL units and packetizes them, with
the timestamp from the packet's `pts`;
[L351–L355](https://github.com/aiortc/aiortc/blob/8a2864630bf417d977a06dda8d0254b1c1501bfb/src/aiortc/rtcrtpsender.py#L351-L355)
— a PLI or FIR from the receiver sets the flag the encode path reads. Each
`RTCRtpSender` belongs to one peer connection.

<a name="e6"></a>**E6 — The overlay preview's rate today.** With the live SAM3
preview on and three cameras at the 672 preset, the Data tab's stream shows
about 3 pictures a second on the 5090, 5.5–6.5 with one camera
([#134](https://github.com/TheWisp/lerobot/issues/134)); the stream holds each
frame for that frame's overlay
([#225](https://github.com/TheWisp/lerobot/issues/225)).

<a name="e7"></a>**E7 — The GPU composite.** `GpuMaskComposite` on real 720p
rows: about 0.2 ms per frame batched against 4.6–7.3 ms for the CPU composite,
identical on about 97% of pixels and within two levels elsewhere, pinned by
`tests/datasets/test_gpu_composite_equivalence.py`
([gpu_mask_composite.py](https://github.com/TheWisp/lerobot/blob/ef6f5bf933f8b6c061aa36c85adca1710035425a/src/lerobot/datasets/gpu_mask_composite.py)).

<a name="e8"></a>**E8 — The CPU path's stages, per frame, on the workstation.**
Measured 2026-09-13 on an AMD Ryzen 9 9950X, idle, moving-noise frames, 150
iterations per row, medians; script kept out of the tree. The JPEG path's
cost is the endpoint's own `cvtColor` + `imencode` at quality 80 at the
source resolution. The encode row is frame-in to bytes-out through a pipe to
an ffmpeg child (`-threads 1`, ultrafast, zerolatency, baseline, keyframe per
second, raw H.264, flushed per packet), with the child's CPU share over the
run at 30 fps.

```
stage                                        median   p95   (ms per frame)
JPEG path: imencode q80 1280x720               2.40   2.57
JPEG path: imencode q80 960x600                1.53   1.56
resize 1280x720 -> 320x180 (INTER_AREA)        0.72   0.74
resize 960x600 -> 320x200                      0.43   0.45
NumPy blend, 30% coverage, 320x200             1.03   1.04
NumPy blend, 30% coverage, 1280x720           15.93  16.13
libx264 320x200@30, 375 kbit/s, in -> out     5.76   5.85   ffmpeg 2.3% of one core, 382 kbit/s out
libx264 640x380@30, 1200 kbit/s, in -> out    6.62   7.01   ffmpeg 4.8% of one core, 1222 kbit/s out
```

Per camera at the profile, scale then blend then encode is about two to three
milliseconds of CPU per frame; four cameras at 30 fps is about a third of one
core by arithmetic, against the JPEG path's 1.5–2.4 ms per camera frame plus
its per-request handling. The in-to-out figure includes the pipe and the
reading thread, not encode alone; compare [E1](#e1) for the encoder by itself.

<a name="e9"></a>**E9 — The GPU path's stages and the encoder, on the
workstation.** Measured 2026-09-13 on an RTX 5090 with torch 2.11 and
`PyNvVideoCodec` 2.2.2, idle, moving-noise frames, medians over 150 iterations
with device synchronization around each stage; scripts kept out of the tree.

```
stage                                              median    p95    max   (ms per frame)
upload 1280x720x3 uint8 from NumPy to the GPU        0.17   0.21   0.55
resize 1280x720 -> 320x180 on the GPU (area)         0.09   0.12  23.16   (max is the first call)
blend RGBA overlay, 30% coverage, 320x200            0.04   0.04  12.21   (max is the first call)
pack to ARGB 320x200                                 0.01   0.01   2.30
```

The encoder, `preset P1, ultra_low_latency, CBR, keyframe every 30 frames, no
B-frames`, fed ARGB tensors at 30 fps pacing, 300 frames per row:

```
configuration                              submit -> bitstream          keyframes  out
320x200, 300 kbit/s, no flush              100.0 ms median (3 frames held)  10/300   325 kbit/s
1280x200 atlas, 1200 kbit/s, no flush      100.0 ms median (3 frames held)  10/300  1305 kbit/s
640x380, 1200 kbit/s, no flush             100.0 ms median (3 frames held)  10/300  1313 kbit/s
320x200, 300 kbit/s, flush per frame         0.24 median, 0.33 p95, 1.04 max  10/300   349 kbit/s
1280x200 atlas, 1200 kbit/s, flush per frame 0.32 median, 0.40 p95, 1.35 max  10/300  1411 kbit/s
```

With no flush the bitstream of a frame is returned by the third later
submission, whatever the size; a flush (`EndEncode`) after each submission
returns it at once, keeps the session, and leaves the keyframe cadence as
configured. `NV_ENC_PIC_FLAG_FORCEIDR` (value 2) on a submission produced an
IDR frame at that position, with the parameter sets when `OUTPUT_SPSPPS` (4)
was also set. CBR overshot its target by about fifteen percent at these sizes.
The earlier ffmpeg-driven measurement of the same hardware encoder
([E1](#e1)) showed a 95th percentile near a frame period at 30 fps; driven
directly with a flush per frame it did not.

<a name="e10"></a>**E10 — The class link.** Lighthouse's throttling guide
([docs/throttling.md](https://github.com/GoogleChrome/lighthouse/blob/main/docs/throttling.md),
read 2026-09-13): latency 150 ms, throughput 1.6 Mbps down / 750 Kbps up,
"roughly the bottom 25% of 4G connections and top 25% of 3G connections",
currently called _Slow 4G_ and formerly _Fast 3G_. Chrome DevTools' presets
([NetworkManager.ts](https://github.com/ChromeDevTools/devtools-frontend/blob/main/front_end/core/sdk/NetworkManager.ts),
same day): _Slow 4G_ 1.6 Mbit/s down and 750 kbit/s up, each applied at 0.9;
_Fast 4G_ 9 Mbit/s down and 1.5 Mbit/s up; _Slow 3G_ 500 kbit/s each way.
