"""What the browser we ship actually does with the stream.

The page measures a frame's age from the timestamp it reads back for the
frame it just painted, so this checks that Chromium reports one for a WebRTC
track, that the offer the page has to make is one it can make, and that the
pictures arrive and are painted. It is the end of the path the server tests
cover: tap, pipeline, transport, endpoint, and now a real decoder and a real
video element.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import sync_playwright  # noqa: E402

from tests.gui.chunk_fixtures import GuiServer  # noqa: E402

pytestmark = pytest.mark.requires_playwright

# The page's side of the stream, as the Run tab will do it: ask what is
# streaming, offer one receive-only H.264 stream per camera plus the cycle
# channel, then paint each track and read back each painted frame.
PAGE_SCRIPT = """
async (seconds) => {
  const status = await (await fetch('/api/run/live-video/status')).json();
  if (!status.available) return {error: 'nothing to watch'};
  const pc = new RTCPeerConnection({iceServers: []});
  const report = {
    cameras: status.cameras,
    h264Offered: false,
    frames: {},
    messages: [],
    firstMessageAt: null,
    error: null,
    //: Which camera each arrival-ordered track carries, so a silent one can be
    //: named rather than reported as "track1".
    trackIds: {},
    //: Every camera the server said was failing, and why. The cycle message
    //: carries this for exactly the case where "the frames simply cease"; the
    //: messages list holds it already, but 200 of them do not survive a print.
    failing: {},
    //: Any element the browser refused to start.
    playFailed: {},
    //: Whether each element was still playing at the end. A track the server
    //: stopped feeding and a decoder the browser stalled look identical in a
    //: frame count and completely different here.
    videoState: {},
  };
  const channel = pc.createDataChannel('cycles');
  channel.onmessage = (e) => {
    if (report.firstMessageAt === null) report.firstMessageAt = performance.now();
    const message = JSON.parse(e.data);
    if (report.messages.length < 200) report.messages.push(message);
    Object.assign(report.failing, message.failing || {});
  };
  const h264 = RTCRtpSender.getCapabilities('video').codecs.filter(
    (c) => c.mimeType.toLowerCase() === 'video/h264');
  report.h264Offered = h264.length > 0;
  const videos = [];
  for (let i = 0; i < status.cameras.length; i++) {
    const tr = pc.addTransceiver('video', {direction: 'recvonly'});
    if (h264.length) tr.setCodecPreferences(h264);
  }
  pc.ontrack = (e) => {
    const v = document.createElement('video');
    v.autoplay = true; v.muted = true; v.playsInline = true;
    v.srcObject = new MediaStream([e.track]);
    document.body.appendChild(v);
    // Not `autoplay` alone: only the first of four elements reliably starts,
    // and a paused element still gets frame callbacks on an idle machine --
    // so this harness measured three players that were never playing, and
    // said nothing until the host was slow enough for them to starve.
    const play = v.play();
    if (play && play.catch) play.catch((err) => { report.playFailed['track' + index] = String(err); });
    const index = videos.length;
    videos.push(v);
    report.trackIds['track' + index] = e.track.id;
    const painted = [];
    report.frames['track' + index] = painted;
    const onFrame = (now, meta) => {
      painted.push({
        now,
        rtpTimestamp: meta.rtpTimestamp === undefined ? null : meta.rtpTimestamp,
        presentationTime: meta.presentationTime,
        expectedDisplayTime: meta.expectedDisplayTime,
        width: meta.width,
        height: meta.height,
        presentedFrames: meta.presentedFrames,
      });
      v.requestVideoFrameCallback(onFrame);
    };
    v.requestVideoFrameCallback(onFrame);
  };
  await pc.setLocalDescription(await pc.createOffer());
  const answer = await fetch('/api/run/live-video/offer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type}),
  });
  if (!answer.ok) { report.error = await answer.text(); return report; }
  const body = await answer.json();
  report.answerHasH264 = body.sdp.includes('H264');
  await pc.setRemoteDescription(body);
  const started = performance.now();
  await new Promise((r) => setTimeout(r, seconds * 1000));
  report.pageEpochMs = Date.now() - performance.now();
  report.observedForMs = performance.now() - started;
  videos.forEach((v, i) => {
    report.videoState['track' + i] = {
      readyState: v.readyState,
      paused: v.paused,
      ended: v.ended,
      currentTime: v.currentTime,
      error: v.error ? v.error.code : null,
    };
  });
  pc.close();
  return report;
}
"""


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def tap(server):
    """After the server: its startup sweeps every segment in our namespace,
    which would take this tap with it."""
    from tests.gui.test_live_video_pipeline import SyntheticTap

    t = SyntheticTap()
    t.start()
    deadline = time.time() + 5.0
    while t.cycles_written < 2 and time.time() < deadline:
        time.sleep(0.01)
    yield t
    t.stop()


@pytest.fixture(scope="module")
def watched(server, tap):
    """One session of watching, read back once and asserted on many times."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(server.base, wait_until="domcontentloaded")
        report = page.evaluate(PAGE_SCRIPT, 6)
        browser.close()
    # The two big lists last and bounded: 200 cycle messages swamped a 2000
    # character print, which is how the per-camera failure reason stayed
    # invisible while sitting in the report all along.
    summary = {k: v for k, v in report.items() if k not in ("frames", "messages")}
    summary["frameCounts"] = {name: len(f) for name, f in report.get("frames", {}).items()}
    summary["messageCount"] = len(report.get("messages", []))
    print(json.dumps(summary, indent=2)[:4000])
    return report


def test_the_browser_can_offer_what_this_stream_needs(watched):
    assert watched.get("error") is None, watched["error"]
    assert watched["h264Offered"] is True, "this Chromium cannot ask for H.264"
    assert watched["answerHasH264"] is True


def test_every_camera_is_painted(watched):
    assert len(watched["frames"]) == len(watched["cameras"])

    # Every track's count, not the first one that falls short. One track
    # delivering a single frame while the others stream is a different fault
    # from every track running slow, and a per-track assertion cannot tell
    # them apart: it stops at the first and reports one number.
    counts = {name: len(painted) for name, painted in watched["frames"].items()}
    starved = {name: n for name, n in counts.items() if n <= 30}
    assert not starved, (
        f"tracks below the floor: {starved}; all tracks: {counts}; "
        f"cameras the server reported failing: {watched.get('failing')}; "
        f"element state at the end: {watched.get('videoState')}"
    )

    for name, painted in watched["frames"].items():
        assert painted[0]["width"] == 320, name


def test_the_painted_frame_carries_the_timestamp_the_age_is_measured_from(watched):
    """The design's one open verification: rVFC's rtpTimestamp on a WebRTC
    track. Without it the page cannot say how old the picture it shows is."""
    for name, painted in watched["frames"].items():
        stamps = [f["rtpTimestamp"] for f in painted]
        assert all(s is not None for s in stamps), f"{name}: no rtpTimestamp in the metadata"
        assert len(set(stamps)) > len(stamps) // 2, f"{name}: the timestamp does not advance"


def test_the_age_at_the_eye_is_a_number_the_page_can_compute(watched):
    """R1's measurement, end to end: the page turns what it reads back for a
    painted frame into that frame's capture time, and the difference is an
    age rather than a sign error or a wrap."""
    import statistics

    from lerobot.gui.live_video.transport import WRAP, capture_ts_from_rtp, learn_origin

    messages = watched["messages"]
    assert messages, "no cycle messages arrived"
    captures = [m["capture_ts"] for m in messages]
    epoch_ms = watched["pageEpochMs"]
    ages: dict[str, float] = {}
    for name, painted in watched["frames"].items():
        stamps = [f["rtpTimestamp"] for f in painted]
        origin = learn_origin(stamps, captures)
        assert origin is not None, f"{name}: no constant explains these readings"
        per_frame = []
        for frame in painted:
            capture = capture_ts_from_rtp((frame["rtpTimestamp"] - origin) % WRAP, near=captures[0])
            if not (captures[0] - 1.0 <= capture <= captures[-1] + 1.0):
                continue  # a frame from before the messages started
            per_frame.append(((epoch_ms + frame["now"]) / 1000.0 - capture) * 1000.0)
        assert len(per_frame) > 20, (name, len(per_frame))
        ages[name] = statistics.median(per_frame)
    print("median age at the eye, ms:", {k: round(v, 1) for k, v in ages.items()})
    for name, age in ages.items():
        # Local, on one machine: the pipeline, the loopback and a decode.
        # The bound is what a wrong sign or a missed wrap would break, not
        # the design's Local target, which is measured in a dated run.
        assert 0 < age < 500, (name, age)


def test_the_readouts_arrive_with_the_pictures(watched):
    messages = watched["messages"]
    assert len(messages) > 30, len(messages)
    cycles = [m["cycle"] for m in messages]
    assert cycles == sorted(cycles)
    assert all(m["state"]["cycle"] == float(m["cycle"]) for m in messages)
    assert watched["firstMessageAt"] is not None


def test_every_camera_is_actually_playing(watched):
    """The frame count cannot see this, and that is how it was missed.

    Only the first of four elements reliably starts from `autoplay`. On an idle
    machine a paused element still receives frame callbacks, so all four report
    a full count while three sit at currentTime 0 -- the harness measured three
    players that were never playing and said nothing. On a loaded host those
    three are deprioritised and starve, which is what the CI failure was: one
    frame in six seconds, at the right size, then silence.

    The product had the same shape: `run.js` set autoplay and never called
    play(), so a four-camera Run tab could freeze three tiles with the stream
    delivering normally and nothing in the controls bar to explain it.
    """
    assert not watched.get("playFailed"), f"the browser refused to start: {watched['playFailed']}"

    state = watched["videoState"]
    assert len(state) == len(watched["cameras"]), state
    stalled = {name: st for name, st in state.items() if st["paused"] or st["currentTime"] == 0}
    assert not stalled, f"elements that never played: {stalled}; all: {state}"
