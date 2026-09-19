// Unit tests for the Run tab's live-video client (static/live_video.js),
// run from pytest via test_live_video_client_js.py or directly with
// `node tests/gui/live_video_client.test.js`.
//
// What is worth pinning here is everything the page must get right before a
// single picture can appear, and which a browser test would only report as
// "no video": the offer has to describe one receive-only stream per camera
// with H.264 first and open the cycle channel, because an answer can add
// none of those; the tracks arrive in the order the answer names the
// cameras, and nothing else says which tile is which; and the age at the eye
// is a reading plus a constant the server never sends, which the page has to
// work out from the capture times it is told.

const assert = require("assert");

// A loop's own jitter, deterministic but aperiodic: what tells one cycle
// from another when the page has to work out which reading is which.
function jitter(seed) {
  let x = seed;
  return () => {
    x = (x * 1103515245 + 12345) % 2147483648;
    return (x / 2147483648) * 0.004;
  };
}
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "../../src/lerobot/gui/static/live_video.js"),
  "utf8",
);

// ---------------------------------------------------------------------------
// A peer connection that records what it was asked for.
// ---------------------------------------------------------------------------
function makeFakeRTC(opts = {}) {
  const made = [];
  class FakeDataChannel {
    constructor(label) {
      this.label = label;
      this.readyState = "connecting";
      this.onmessage = null;
      this.onopen = null;
    }
    deliver(obj) {
      if (this.onmessage) this.onmessage({ data: JSON.stringify(obj) });
    }
  }
  class FakeTransceiver {
    constructor(kind, init) {
      this.kind = kind;
      this.direction = (init && init.direction) || "sendrecv";
      this.preferred = null;
    }
    setCodecPreferences(list) {
      this.preferred = list;
    }
  }
  class FakePeerConnection {
    constructor(config) {
      this.config = config;
      this.transceivers = [];
      this.channels = [];
      this.localDescription = null;
      this.remoteDescription = null;
      this.closed = false;
      this.ontrack = null;
      this.onconnectionstatechange = null;
      this.connectionState = "new";
      made.push(this);
    }
    createDataChannel(label) {
      const ch = new FakeDataChannel(label);
      this.channels.push(ch);
      return ch;
    }
    addTransceiver(kind, init) {
      const t = new FakeTransceiver(kind, init);
      this.transceivers.push(t);
      return t;
    }
    getTransceivers() {
      return this.transceivers.slice();
    }
    async createOffer() {
      if (opts.offerFails) throw new Error("no offer");
      return { sdp: "the-offer", type: "offer" };
    }
    async setLocalDescription(d) {
      this.localDescription = d;
    }
    async setRemoteDescription(d) {
      this.remoteDescription = d;
    }
    close() {
      this.closed = true;
    }
    // Test-side: pretend a track arrived on the nth transceiver.
    fireTrack(index, track) {
      if (this.ontrack) {
        this.ontrack({ track, transceiver: this.transceivers[index] });
      }
    }
  }
  return { FakePeerConnection, made };
}

function makeFetch(routes) {
  const calls = [];
  return {
    calls,
    fn: async (url, init) => {
      calls.push({ url, init });
      const route = routes[url];
      if (!route) return { ok: false, status: 404, text: async () => "no route" };
      return typeof route === "function" ? route(init) : route;
    },
  };
}

function load(extra = {}) {
  const listeners = {};
  const context = vm.createContext({
    console,
    window: {},
    document: {
      addEventListener: (t, f) => {
        listeners[t] = f;
      },
      getElementById: () => null,
      createElement: () => ({ style: {}, appendChild() {}, addEventListener() {} }),
    },
    setTimeout,
    clearTimeout,
    setInterval: () => 0,
    clearInterval: () => {},
    performance: { now: () => 0 },
    Date,
    JSON,
    Math,
    MediaStream: class {
      constructor(tracks) {
        this.tracks = tracks;
      }
    },
    ...extra,
  });
  vm.runInContext(source, context);
  return context.window.LiveVideo;
}

// The server as it answers: an offer that is not attaching carries one
// placeholder video stream and is answered with no cameras at all, and only
// an attaching offer is answered with the run's.
function answerFor(cameras) {
  return (init) => {
    const attaching = JSON.parse(init.body).attach;
    return {
      ok: true,
      status: 200,
      json: async () => ({
        sdp: "the-answer",
        type: "answer",
        cameras: attaching ? cameras : [],
        session: "s1",
      }),
    };
  };
}

// ---------------------------------------------------------------------------
// The offer
// ---------------------------------------------------------------------------
async function testTheOfferDescribesWhatTheAnswerCannotAdd() {
  const { FakePeerConnection, made } = makeFakeRTC();
  const fetcher = makeFetch({
    "/api/run/live-video/offer": answerFor(["front", "top"]),
  });
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  await client.open();
  await client.attach(["front", "top"]);

  const pc = made[0];
  assert.strictEqual(pc.config.iceServers.length, 0, "no ICE server is asked for");
  assert.strictEqual(pc.channels.length, 1);
  assert.strictEqual(pc.channels[0].label, "cycles");
  assert.strictEqual(pc.transceivers.length, 2, "one stream per camera");
  for (const t of pc.transceivers) {
    assert.strictEqual(t.kind, "video");
    assert.strictEqual(t.direction, "recvonly");
    assert.strictEqual(t.preferred.length, 1, "H.264 alone");
    assert.strictEqual(t.preferred[0].mimeType, "video/H264", "H.264 first");
  }
  const posts = fetcher.calls.filter((c) => c.url === "/api/run/live-video/offer");
  assert.strictEqual(posts.length, 2, "one offer to open, one to attach");
  const opening = JSON.parse(posts[0].init.body);
  const attaching = JSON.parse(posts[1].init.body);
  assert.strictEqual(opening.attach, false, "the first offer is not carrying cameras");
  assert.strictEqual(opening.session, null);
  assert.strictEqual(attaching.attach, true);
  assert.strictEqual(attaching.session, "s1", "the second offer names the connection");
  assert.strictEqual(pc.remoteDescription.sdp, "the-answer");
  assert.strictEqual(pc.remoteDescription.type, "answer");
  client.close();
  assert.ok(pc.closed, "closing the client closes the connection");
}

async function testTheConnectionOpensBeforeTheRunHasCameras() {
  // What R2 rests on: the connection is up, with one placeholder video
  // stream, before anything is streaming — so the first picture is a
  // keyframe away rather than a connection away.
  const { FakePeerConnection, made } = makeFakeRTC();
  const fetcher = makeFetch({
    "/api/run/live-video/offer": answerFor([]),
  });
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  const opened = await client.open();
  assert.strictEqual(opened, true);
  const pc = made[0];
  assert.strictEqual(pc.transceivers.length, 1, "one placeholder stream");
  assert.strictEqual(pc.channels.length, 1);
  // The connection is up and carrying nothing, which is what the operator
  // is told: a bar still reading "connecting" would describe a fault that
  // is not there, and stay there until a run was launched.
  assert.strictEqual(client.state.name, "idle");
  const body = JSON.parse(fetcher.calls.find((c) => c.url.endsWith("/offer")).init.body);
  assert.strictEqual(body.attach, false);

  // A second open() is not a second connection.
  await client.open();
  assert.strictEqual(made.length, 1);
}

async function testAttachingAddsOnlyTheStreamsStillMissing() {
  const { FakePeerConnection, made } = makeFakeRTC();
  const fetcher = makeFetch({
    "/api/run/live-video/offer": answerFor(["front", "top", "wrist"]),
  });
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  await client.open();
  const pc = made[0];
  assert.strictEqual(pc.transceivers.length, 1);
  await client.attach(["front", "top", "wrist"]);
  assert.strictEqual(pc.transceivers.length, 3, "the placeholder is one of the three");
  assert.deepStrictEqual(client.cameras.slice(), ["front", "top", "wrist"]);
  // The tracks land on the tiles the answer names, in its order.
  pc.fireTrack(2, "third");
  pc.fireTrack(0, "first");
  assert.deepStrictEqual(Object.keys(client.tracks).sort(), ["front", "wrist"]);
  assert.strictEqual(client.tracks.front, "first");
  assert.strictEqual(client.tracks.wrist, "third");
}

async function testAStreamWithNoH264InTheBrowserIsRefusedBeforeOffering() {
  const { FakePeerConnection, made } = makeFakeRTC();
  const fetcher = makeFetch({});
  const LiveVideo = load();
  const states = [];
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/VP8" }],
    onState: (s) => states.push(s),
  });
  await client.open();
  await client.attach(["front", "top"]);
  assert.strictEqual(client.state.name, "failed");
  assert.match(client.state.reason, /H\.264/);
  assert.ok(!fetcher.calls.some((c) => c.url.endsWith("/offer")), "nothing was offered");
  assert.strictEqual(made.length, 0, "no connection was built");
}

// ---------------------------------------------------------------------------
// Tracks and tiles
// ---------------------------------------------------------------------------
async function testTracksAreNamedByTheAnswerNotByArrivalOrder() {
  const { FakePeerConnection, made } = makeFakeRTC();
  const fetcher = makeFetch({
    // The answer's order is the session's, and may differ from the status
    // the page asked before offering.
    "/api/run/live-video/offer": answerFor(["top", "front"]),
  });
  const LiveVideo = load();
  const seen = [];
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
    onTrack: (camera, track) => seen.push([camera, track]),
  });
  await client.open();
  await client.attach(["front", "top"]);
  const pc = made[0];
  pc.fireTrack(1, "second-track");
  pc.fireTrack(0, "first-track");
  assert.deepStrictEqual(seen, [
    ["front", "second-track"],
    ["top", "first-track"],
  ]);
  assert.deepStrictEqual(client.cameras, ["top", "front"]);
}

async function testTheStateGoesConnectingThenStreamingOnTheFirstPaintedFrame() {
  const { FakePeerConnection, made } = makeFakeRTC();
  const fetcher = makeFetch({
    "/api/run/live-video/offer": answerFor(["front", "top"]),
  });
  const LiveVideo = load();
  const states = [];
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
    onState: (s) => states.push(s.name),
  });
  await client.open();
  await client.attach(["front", "top"]);
  assert.strictEqual(client.state.name, "connecting");
  client.notePainted("front", 90000, 1000.0);
  assert.strictEqual(client.state.name, "streaming");
  // A second frame is not a second transition.
  client.notePainted("front", 93000, 1000.033);
  // The whole sequence the operator sees: the connection comes up carrying
  // nothing, the run's cameras are attached, the first frame paints.
  assert.deepStrictEqual(states, ["connecting", "idle", "connecting", "streaming"]);
  made[0].connectionState = "failed";
  if (made[0].onconnectionstatechange) made[0].onconnectionstatechange();
  assert.strictEqual(client.state.name, "failed");
  assert.match(client.state.reason, /connection/i);
}

async function testNothingToWatchIsNotAFailure() {
  // The tab is opened at Low Bandwidth before anything is launched. The
  // connection comes up and carries no camera, which is a state of its own:
  // not a failure, and not a connection still being made. The words shown
  // for it belong to the page, so the client names the state and no reason.
  const { FakePeerConnection, made } = makeFakeRTC();
  const routes = { "/api/run/live-video/offer": answerFor([]) };
  const fetcher = makeFetch(routes);
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  await client.open();
  assert.strictEqual(client.state.name, "idle");
  assert.strictEqual(client.state.reason, "");
  assert.ok(!made[0].closed, "the connection is up, waiting for a run");

  // And the run starting moves it on, without a reconnection.
  routes["/api/run/live-video/offer"] = answerFor(["front", "top"]);
  await client.attach(["front", "top"]);
  assert.strictEqual(client.state.name, "connecting");
  client.notePainted("front", 90000, 1000.0);
  assert.strictEqual(client.state.name, "streaming");
  assert.strictEqual(made.length, 1, "the same connection carried it");
}

async function testARefusedOfferIsShownWithTheServersReason() {
  const { FakePeerConnection } = makeFakeRTC();
  const fetcher = makeFetch({
    "/api/run/live-video/offer": {
      ok: false,
      status: 400,
      json: async () => ({ detail: "This run has 4 cameras and the offer describes 2 video streams." }),
      text: async () => "unused",
    },
  });
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  await client.open();
  await client.attach(["front", "top"]);
  assert.strictEqual(client.state.name, "failed");
  assert.match(client.state.reason, /4 cameras/);
}

// ---------------------------------------------------------------------------
// The age at the eye
// ---------------------------------------------------------------------------
function testTheAgeIsLearnedFromTheCaptureTimesTheChannelAnnounces() {
  const LiveVideo = load();
  const tracker = LiveVideo.createAgeTracker();
  const t0 = 1789344182.0;
  const captures = [];
  const noise = jitter(7);
  for (let i = 0; i < 40; i++) captures.push(t0 + i / 30 + noise());
  captures.forEach((c) => tracker.noteCapture(c));

  const origin = 123456789;
  // Every frame painted 40 ms after its capture.
  for (let i = 5; i < 35; i++) {
    const wire = (Math.round(captures[i] * 90000) + origin) % 2 ** 32;
    tracker.notePainted("front", wire, captures[i] + 0.04);
  }
  const age = tracker.medianAgeMs("front");
  assert.ok(Math.abs(age - 40) < 2, `expected about 40 ms, got ${age}`);
}

function testAnAgeItCannotWorkOutIsReportedAsUnknown() {
  const LiveVideo = load();
  const tracker = LiveVideo.createAgeTracker();
  const t0 = 1789344182.0;
  for (let i = 0; i < 30; i++) tracker.noteCapture(t0 + i / 30);
  // Readings from somewhere else entirely: no constant explains them.
  for (let i = 0; i < 30; i++) tracker.notePainted("front", (i * 7919 + 13) % 2 ** 32, t0 + i / 30);
  assert.strictEqual(tracker.medianAgeMs("front"), null);
  assert.strictEqual(tracker.medianAgeMs("never-painted"), null);
}

function testTheAgeFollowsTheRecentPastNotTheWholeSession() {
  const LiveVideo = load();
  const tracker = LiveVideo.createAgeTracker();
  const t0 = 1789344182.0;
  const captures = [];
  const noise = jitter(11);
  for (let i = 0; i < 400; i++) {
    const c = t0 + i / 30 + noise();
    captures.push(c);
    tracker.noteCapture(c);
  }
  const origin = 7;
  // A first half that was slow, a second half that is fast: what the bar
  // shows must be the second.
  for (let i = 0; i < 400; i++) {
    const wire = (Math.round(captures[i] * 90000) + origin) % 2 ** 32;
    tracker.notePainted("front", wire, captures[i] + (i < 200 ? 0.3 : 0.02));
  }
  const age = tracker.medianAgeMs("front");
  assert.ok(Math.abs(age - 20) < 5, `expected about 20 ms, got ${age}`);
}

function testTheOriginIsSearchedOnceNotOnEveryPaintedFrame() {
  // The search tries every announced capture time against every kept
  // reading, so it costs the page far more than a painted frame does. It
  // ran on every frame: the readings array is capped at the same number the
  // cadence was taken modulo of, so that count never changed again.
  let lookups = 0;
  class CountingSet extends Set {
    has(value) {
      lookups++;
      return super.has(value);
    }
  }
  const LiveVideo = load({ Set: CountingSet });
  const tracker = LiveVideo.createAgeTracker();
  const ORIGIN = 4242;
  const t0 = 1000.0;
  const wireFor = (ts) => (LiveVideo.rtpFromCapture(ts) + ORIGIN) % LiveVideo.WRAP;

  // Irregular, as real capture times are: a perfectly regular cadence has
  // no single answer, and the tracker says so rather than guessing.
  const noise = jitter(23);
  for (let i = 0; i < 400; i++) {
    const c = t0 + i / 30 + noise();
    tracker.noteCapture(c);
    tracker.notePainted("front", wireFor(c), c + 0.02);
  }
  assert.ok(tracker.medianAgeMs("front") !== null, "the constant was found at all");

  const settled = lookups;
  const frames = 60;
  for (let i = 400; i < 400 + frames; i++) {
    const c = t0 + i / 30 + noise();
    tracker.noteCapture(c);
    tracker.notePainted("front", wireFor(c), c + 0.02);
  }
  const perFrame = (lookups - settled) / frames;
  assert.ok(perFrame < 5, `${perFrame} lookups per painted frame once the constant is known`);

  const age = tracker.medianAgeMs("front");
  assert.ok(Math.abs(age - 20) < 5, `expected about 20 ms, got ${age}`);
}

function testTheOriginIsSearchedAgainWhenTheTrackIsReoffered() {
  // A re-offered track carries a new random origin. The readings stop
  // agreeing with the constant, and that is what asks for the search again.
  const LiveVideo = load();
  const tracker = LiveVideo.createAgeTracker();
  const t0 = 2000.0;
  const wireFor = (ts, origin) => (LiveVideo.rtpFromCapture(ts) + origin) % LiveVideo.WRAP;
  const noise = jitter(29);
  let i = 0;
  const feed = (origin, n, lag) => {
    for (let k = 0; k < n; k++, i++) {
      const c = t0 + i / 30 + noise();
      tracker.noteCapture(c);
      tracker.notePainted("front", wireFor(c, origin), c + lag);
    }
  };
  feed(11, 200, 0.02);
  assert.ok(Math.abs(tracker.medianAgeMs("front") - 20) < 5);
  feed(900001, 200, 0.05);
  const age = tracker.medianAgeMs("front");
  assert.ok(Math.abs(age - 50) < 10, `expected about 50 ms after the re-offer, got ${age}`);
}

async function testAFailureIsStillReportedAfterAnEarlierOneClosedTheClient() {
  // The first open fails, which closes the client. A connection built after
  // that is a live connection, and its failures are the operator's only
  // warning that the pictures have stopped.
  const { FakePeerConnection, made } = makeFakeRTC();
  const routes = {
    "/api/run/live-video/offer": { ok: false, status: 503, json: async () => ({ detail: "no run" }), text: async () => "" },
  };
  const fetcher = makeFetch(routes);
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  assert.strictEqual(await client.open(), false);
  assert.strictEqual(client.state.name, "failed");

  routes["/api/run/live-video/offer"] = answerFor(["front", "top"]);
  await client.attach(["front", "top"]);
  const pc = made[made.length - 1];
  client.notePainted("front", 90000, 1000.0);
  assert.strictEqual(client.state.name, "streaming");

  pc.connectionState = "failed";
  pc.onconnectionstatechange();
  assert.strictEqual(client.state.name, "failed", "the second connection reports its own failure");
}

async function testALinkThatGoesQuietIsNotAFailedStream() {
  // The rig's link drops for seconds at a time and comes back. A browser
  // reports that as "disconnected"; treating it as terminal would leave the
  // bar reading failed for the rest of a run that recovered.
  const { FakePeerConnection, made } = makeFakeRTC();
  const fetcher = makeFetch({ "/api/run/live-video/offer": answerFor(["front", "top"]) });
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  await client.open();
  await client.attach(["front", "top"]);
  client.notePainted("front", 90000, 1000.0);
  assert.strictEqual(client.state.name, "streaming");

  const pc = made[0];
  pc.connectionState = "disconnected";
  pc.onconnectionstatechange();
  assert.strictEqual(client.state.name, "connecting", "quiet, not failed");

  pc.connectionState = "connected";
  pc.onconnectionstatechange();
  client.notePainted("front", 93000, 1000.1);
  assert.strictEqual(client.state.name, "streaming", "a painted frame says it is back");
}

async function testARunThatRestartsWithOtherCamerasGetsAConnectionThatFitsIt() {
  // A connection's video streams only grow, and their order names them. A
  // run relaunched with a camera unplugged would otherwise offer more
  // streams than the run has and be refused for the rest of the session.
  const { FakePeerConnection, made } = makeFakeRTC();
  const routes = { "/api/run/live-video/offer": answerFor(["front", "top", "wrist"]) };
  const fetcher = makeFetch(routes);
  const LiveVideo = load();
  const seen = [];
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
    onTrack: (camera, track) => seen.push([camera, track]),
  });
  await client.open();
  await client.attach(["front", "top", "wrist"]);
  assert.strictEqual(made.length, 1);
  assert.strictEqual(made[0].transceivers.length, 3);

  routes["/api/run/live-video/offer"] = answerFor(["front", "wrist"]);
  await client.attach(["front", "wrist"]);
  assert.strictEqual(made.length, 2, "the connection was built again");
  assert.ok(made[0].closed, "and the one that could not carry it was closed");
  assert.strictEqual(made[1].transceivers.length, 2, "two streams for two cameras");
  assert.deepStrictEqual(client.cameras, ["front", "wrist"]);

  made[1].fireTrack(1, "wrist-track");
  assert.deepStrictEqual(seen[seen.length - 1], ["wrist", "wrist-track"]);
}

async function testACameraThatStopsEncodingIsReported() {
  // The server can fail in a way nothing on the wire shows: the connection
  // is fine and the frames just stop. The cycle channel is still running, so
  // it carries the reason, and the bar says it instead of sitting on
  // "streaming" with an age that never moves.
  const { FakePeerConnection, made } = makeFakeRTC();
  const fetcher = makeFetch({ "/api/run/live-video/offer": answerFor(["front", "top"]) });
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  await client.open();
  await client.attach(["front", "top"]);
  client.notePainted("front", 90000, 1000.0);
  assert.strictEqual(client.state.name, "streaming");

  const channel = made[0].channels[0];
  channel.onmessage({ data: JSON.stringify({ cycle: 5, failing: { front: "no encoder session" } }) });
  assert.strictEqual(client.state.name, "failed");
  assert.match(client.state.reason, /front/);
  assert.match(client.state.reason, /no encoder session/);

  // And it comes back when the encoder does, without a reconnection.
  channel.onmessage({ data: JSON.stringify({ cycle: 6, failing: {} }) });
  client.notePainted("front", 93000, 1000.033);
  assert.strictEqual(client.state.name, "streaming");
  assert.strictEqual(made.length, 1);
}

// ---------------------------------------------------------------------------

async function main() {
  await testTheOfferDescribesWhatTheAnswerCannotAdd();
  await testTheConnectionOpensBeforeTheRunHasCameras();
  await testAttachingAddsOnlyTheStreamsStillMissing();
  await testOpeningAndAttachingDoNotOverlap();
  await testAStreamWithNoH264InTheBrowserIsRefusedBeforeOffering();
  await testTracksAreNamedByTheAnswerNotByArrivalOrder();
  await testTheStateGoesConnectingThenStreamingOnTheFirstPaintedFrame();
  await testNothingToWatchIsNotAFailure();
  await testARefusedOfferIsShownWithTheServersReason();
  await testAFailureIsStillReportedAfterAnEarlierOneClosedTheClient();
  await testALinkThatGoesQuietIsNotAFailedStream();
  await testARunThatRestartsWithOtherCamerasGetsAConnectionThatFitsIt();
  await testACameraThatStopsEncodingIsReported();
  testTheAgeIsLearnedFromTheCaptureTimesTheChannelAnnounces();
  testAnAgeItCannotWorkOutIsReportedAsUnknown();
  testTheAgeFollowsTheRecentPastNotTheWholeSession();
  testAPerfectlyRegularCadenceHasNoAnswerRatherThanAWrongOne();
  testTheOriginIsSearchedOnceNotOnEveryPaintedFrame();
  testTheOriginIsSearchedAgainWhenTheTrackIsReoffered();
  console.log("live_video_client.test.js: all assertions passed");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

function testAPerfectlyRegularCadenceHasNoAnswerRatherThanAWrongOne() {
  // With capture times on an exact grid, a constant a whole period out
  // explains every reading just as well, so the age is one period wrong for
  // every frame -- and reads as a plausible number. It must come back
  // unknown instead.
  const LiveVideo = load();
  const tracker = LiveVideo.createAgeTracker();
  const t0 = 1789344182.0;
  const captures = [];
  for (let i = 0; i < 60; i++) {
    const c = t0 + i / 30;
    captures.push(c);
    tracker.noteCapture(c);
  }
  for (let i = 5; i < 40; i++) {
    const wire = (Math.round(captures[i] * 90000) + 4242) % 2 ** 32;
    tracker.notePainted("front", wire, captures[i] + 0.02);
  }
  assert.strictEqual(tracker.medianAgeMs("front"), null);
}

async function testOpeningAndAttachingDoNotOverlap() {
  // The tab opens the connection as soon as it is shown and the grid
  // attaches the cameras when the run's frames appear; nothing orders those
  // two, and a connection can only carry one offer at a time.
  const { FakePeerConnection, made } = makeFakeRTC();
  const seen = [];
  const fetcher = makeFetch({
    "/api/run/live-video/offer": async (init) => {
      const body = JSON.parse(init.body);
      seen.push(body.attach);
      await new Promise((r) => setTimeout(r, 20)); // the server takes a moment
      return {
        ok: true,
        status: 200,
        json: async () => ({
          sdp: "the-answer",
          type: "answer",
          session: "s1",
          cameras: body.attach ? ["front", "top"] : [],
        }),
      };
    },
  });
  const LiveVideo = load();
  const client = LiveVideo.createClient({
    fetch: fetcher.fn,
    RTCPeerConnection: FakePeerConnection,
    videoCodecs: () => [{ mimeType: "video/H264" }],
  });
  client.open(); // not awaited, as the tab does not await it
  await client.attach(["front", "top"]);
  assert.strictEqual(made.length, 1, "one connection");
  assert.deepStrictEqual(seen, [false, true], "the open finished before the attach began");
  assert.notStrictEqual(client.state.name, "failed", client.state.reason);
  assert.deepStrictEqual(client.cameras.slice(), ["front", "top"]);
}
