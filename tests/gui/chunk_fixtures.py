"""Shared scaffolding for the chunk-playback tests: a synthetic dataset whose
frames identify themselves, and a GUI server that serves it.

Frames are flat greys unique to (episode, frame), so a decoded frame says which
frame it is: a chunk cut one frame off, a camera painted a frame behind another,
or a chunk that ran into the next episode all show up as the wrong grey. Two
cameras of different resolutions, one wider than the profile's target width and
one narrower, so scaling is exercised in both directions. Two episodes, so the
second sits at a non-zero offset inside the packed video file. Camera ``a``
carries a saved mask.
"""

from __future__ import annotations

import io
import json
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

CAM_WIDE = "observation.images.a"  # 480x240: scaled down to the 320 target
CAM_NARROW = "observation.images.b"  # 200x100: narrower than the target, never upscaled
CAM_THIRD = "observation.images.c"
CAM_FOURTH = "observation.images.d"
CAMS = [CAM_WIDE, CAM_NARROW]
#: The rig records four cameras, and four is where the page ran out of decoders
#: (2026-09-11: sixteen VideoDecoders open at once, Chrome reclaiming them mid
#: chunk). Tests that are about how many decoders the page keeps alive, or about
#: what a chunk costs, take this set.
CAMS_RIG = [CAM_WIDE, CAM_NARROW, CAM_THIRD, CAM_FOURTH]
SIZES = {
    CAM_WIDE: (240, 480),
    CAM_NARROW: (100, 200),
    CAM_THIRD: (240, 480),
    CAM_FOURTH: (180, 320),
}  # (H, W)
FPS = 10
FRAMES = 35  # 2 s chunks at 10 fps are 20 frames: the last chunk of an episode is 15 frames
EPISODES = 2
LABELS = ["ball"]
# A disc, not a rectangle: a rectangle is four runs at any resolution, so its
# rows cannot show that resized masks are smaller. Centre (y=100, x=140), r=55.
BLOB_CENTER, BLOB_RADIUS = (100, 140), 55


def blob_mask(h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[:h, :w]
    cy, cx = BLOB_CENTER
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= BLOB_RADIUS**2


#: A frame identifies itself by a row of black/white bands along its top: one
#: band per bit of the frame index and one for the episode. Bands survive any
#: encode a chunk goes through -- the stored AV1 at crf 30 and the chunk's H.264
#: at crf 26 both move a flat grey by a few levels, which is enough to misread a
#: frame index carried as a grey level, and nowhere near enough to flip a band.
BANDS = 8  # bits 0..6 of the frame index, then the episode
BAND_ROWS = 16
BODY = 100  # the flat grey of everything below the bands


def frame_image(ep: int, i: int, h: int, w: int, texture: np.ndarray | None = None) -> np.ndarray:
    """``texture``, when given, is an HxW*2x3 noise sheet the body scrolls
    through by 4 pixels per frame, so the frames cost what real footage costs;
    without it the body is flat, so pixel assertions have a known ground."""
    assert 0 <= i < 128 and ep in (0, 1), (ep, i)
    if texture is None:
        img = np.full((h, w, 3), BODY, np.uint8)
    else:
        off = (i * 4) % w
        img = np.ascontiguousarray(texture[:, off : off + w])
    for k in range(BANDS):
        on = ep if k == BANDS - 1 else (i >> k) & 1
        img[:BAND_ROWS, k * w // BANDS : (k + 1) * w // BANDS] = 255 if on else 0
    return img


def read_ids(sample) -> tuple[int, int]:
    """(frame index, episode) from the 8 band samples (mean red at each band's
    centre), for a decoder or a canvas read to hand over."""
    bits = [1 if v > 128 else 0 for v in sample]
    return sum(b << k for k, b in enumerate(bits[: BANDS - 1])), bits[BANDS - 1]


def band_samples(frame: np.ndarray) -> list[float]:
    """Mean red at the centre of each band, from rows well inside the band strip
    at any scale the frame was encoded at."""
    h, w = frame.shape[:2]
    rows = slice(2, max(3, min(6, h * BAND_ROWS // 240)))  # inside the strip after scaling
    return [
        float(frame[rows, (2 * k + 1) * w // (2 * BANDS) - 1 : (2 * k + 1) * w // (2 * BANDS) + 2, 0].mean())
        for k in range(BANDS)
    ]


def frame_ids(frames) -> list[tuple[int, int]]:
    """(frame index, episode) each decoded frame identifies itself as -- so a
    chunk is judged by what any decoder gets out of it, and a frame off by one
    or from the other episode is caught."""
    return [read_ids(band_samples(f)) for f in frames]


#: The fixture's recipe: the ball tinted red. Pass ``treatment=None`` for none.
TINT_RECIPE = {"key": "tint", "params": {"color": [255, 0, 0]}}


def _state_row(names: list[str], a: int, b: int) -> np.ndarray:
    """The two-motor fixture's row unchanged; a longer motor set gets one ramp
    per motor, so no two frames share a pose."""
    if names == ["a", "b"]:
        return np.array([a, b], np.float32)
    return np.array([0.1 * j + 0.005 * b for j in range(len(names))], np.float32)


def build_dataset(
    root: Path,
    frames: int = FRAMES,
    treatment: dict | None = TINT_RECIPE,
    noise: bool = False,
    state_names: list[str] | None = None,
    cams: list[str] | None = None,
):
    """``frames`` per episode (both episodes); ``treatment`` is the recipe's
    treatment for the ``ball`` label on camera a, or None for no treatment.

    ``state_names`` names the state and action motors -- give a vendored
    robot's motor set and the tab shows the URDF tile, whose pose then moves
    frame by frame because each motor is a ramp of its own slope. ``cams``
    defaults to the two-camera set; pass ``CAMS_RIG`` for the rig's four."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    names = state_names or ["a", "b"]
    dim = len(names)
    cams = cams or CAMS
    ds = LeRobotDataset.create(
        repo_id="tests/chunks",
        fps=FPS,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (dim,), "names": names},
            "action": {"dtype": "float32", "shape": (dim,), "names": names},
            **{
                c: {"dtype": "video", "shape": (*SIZES[c], 3), "names": ["height", "width", "channels"]}
                for c in cams
            },
        },
        use_videos=True,
    )
    rng = np.random.default_rng(0)
    textures = (
        {c: rng.integers(0, 256, (SIZES[c][0], SIZES[c][1] * 2, 3), np.uint8) for c in cams} if noise else {}
    )
    for ep in range(EPISODES):
        for i in range(frames):
            ds.add_frame(
                {
                    "observation.state": _state_row(names, ep, i),
                    "action": _state_row(names, i, ep),
                    "task": "chunks",
                    # The strip carries 7 bits of index; a long episode cycles it.
                    **{c: frame_image(ep, i % 128, *SIZES[c], textures.get(c)) for c in cams},
                }
            )
        ds.save_episode()
    ds.finalize()
    ds = LeRobotDataset("tests/chunks", root=root)
    h, w = SIZES[CAM_WIDE]
    adopt(ds, [CAM_WIDE], LABELS, (h, w), treatments={"ball": treatment} if treatment else {})
    blob = blob_mask(h, w)
    for ep in range(EPISODES):
        write_episode(ds, ep, CAM_WIDE, [{"ball": blob} for _ in range(frames)])
    return root


#: Chromium's own media logging. The decoder stalls this suite chases happen
#: inside the browser, where the WebCodecs API shows only that frames stopped
#: arriving; these two flags make the media pipeline say why on stderr, which
#: pytest shows for a failing test and swallows for a passing one.
BROWSER_LOG_ARGS = ["--enable-logging=stderr", "--vmodule=*video_decoder*=2,*media*=1"]


class MediaLog:
    """What Chrome's media pipeline reported, for a test that has to explain a
    decoder rather than just fail on it.

    The DevTools ``Media`` domain carries the events and errors behind
    chrome://media-internals -- decoder selection, fallbacks to software, codec
    reclaims, and the errors WebCodecs never surfaces to the page. Attach it
    before opening the tab; read it when something times out.

    Pre: ``context`` is a Playwright browser context and ``page`` belongs to it.
    """

    def __init__(self, context, page):
        self.events: list[str] = []
        self.page_errors: list[str] = []
        # An exception thrown inside the player's paint aborts it before any
        # metric is written, so the page reads healthy and says nothing. This
        # is the one silence the player's own instrumentation cannot cover.
        page.on("pageerror", lambda e: self.page_errors.append(f"pageerror: {e}"))
        page.on(
            "console",
            lambda m: self.page_errors.append(f"console.{m.type}: {m.text}")
            if m.type in ("error", "warning")
            else None,
        )
        try:
            cdp = context.new_cdp_session(page)
            cdp.send("Media.enable")
        except Exception as exc:  # the domain is not essential to any assertion
            self.events.append(f"(Media domain unavailable: {exc})")
            return
        cdp.on("Media.playerErrorsRaised", lambda m: self._add("error", m))
        cdp.on("Media.playerEventsAdded", lambda m: self._add("event", m))
        cdp.on("Media.playerPropertiesChanged", lambda m: self._add("property", m))

    def _add(self, kind: str, message) -> None:
        for item in (
            (message or {}).get("errors", [])
            or (message or {}).get("events", [])
            or ((message or {}).get("properties", []))
        ):
            self.events.append(f"{kind}: {item}")

    def dump(self, limit: int = 40) -> str:
        errors = (
            "page errors:\n  " + "\n  ".join(self.page_errors[-15:])
            if self.page_errors
            else "page errors: none"
        )
        if not self.events:
            return errors + "\nmedia log: nothing reported (the pipeline raised no event)"
        return errors + "\nmedia log (last {} of {}):\n  {}".format(
            min(limit, len(self.events)), len(self.events), "\n  ".join(self.events[-limit:])
        )


SAMPLER = """
window.__trace = [];
setInterval(() => {
  const p = window.__chunkPlayer;
  if (!p) return;
  const s = p.state();
  window.__trace.push(
    `${(performance.now() / 1000).toFixed(1)}s cur=${s.cur} painted=${p.metrics.painted.length}`
    + ` frames=${s.liveFrames} buffered=${s.buffered} ticks=${s.ticks}`
    + ` | ${(p.detail ? p.detail() : []).map((d) => d.start + ': ' + d.why).join(' | ')}`);
  if (window.__trace.length > 120) window.__trace.shift();
}, 1000);
"""


def start_trace(page):
    """Sample the player once a second. The evidence dump is a snapshot, and a
    snapshot cannot tell a decode that stopped from one still moving -- which
    is the difference between a defect and a test budget that is too short.
    """
    page.evaluate(SAMPLER)


def _evidence(page, media, what):
    """What the page and the browser were doing, for a failed wait's message --
    the difference between a CI failure that names its cause and one that costs
    another round trip."""
    state = page.evaluate("() => (window.__chunkPlayer ? window.__chunkPlayer.state() : null)")
    detail = page.evaluate(
        "() => (window.__chunkPlayer && window.__chunkPlayer.detail ? window.__chunkPlayer.detail() : null)"
    )
    trace = page.evaluate("() => (window.__trace || []).slice(-20)") or []
    metrics = page.evaluate(
        "() => (window.__chunkPlayer ? {"
        " events: window.__chunkPlayer.metrics.events.slice(-8),"
        " errors: window.__chunkPlayer.metrics.errors.slice(-8),"
        " retries: window.__chunkPlayer.metrics.retries,"
        " stalls: window.__chunkPlayer.metrics.stalls.slice(-5),"
        " painted: window.__chunkPlayer.metrics.painted.length } : null)"
    )
    return (
        f"{what}\nplayer state: {state}\nheld chunks: {detail}\n"
        + ("trace (last 20s):\n  " + "\n  ".join(trace) + "\n" if trace else "")
        + f"player metrics: {metrics}\n"
        + (media.dump() if media is not None else "")
    )


def wait_for_player(page, expression, arg=None, what=None):
    """`page.wait_for_function` for a condition gated on the player.

    Same shape as the Playwright call it replaces, minus the deadline: what
    these wait for is decoding, and how long that takes is the runner's to
    decide. See :func:`wait_while_decoding`.
    """
    return wait_while_decoding(
        page, None, expression, what or f"the page never reached: {expression}", arg=arg
    )


def wait_with_evidence(page, media, expression, what, arg=None, timeout=30_000):
    """``page.wait_for_function``, and on timeout an assertion carrying what the
    page and the browser were doing.

    For anything gated on decoding, prefer :func:`wait_while_decoding` -- a
    deadline there measures the runner.
    """
    try:
        page.wait_for_function(expression, arg=arg, timeout=timeout)
    except Exception as exc:
        raise AssertionError(_evidence(page, media, what)) from exc


# Only the player's own monotonic counters: frames it has ever held, and frames
# it has painted. Both move when decoding gets somewhere and stop when it does
# not. The tick count is deliberately absent -- the transport beats whether or
# not any work is happening, so it would report a wedged player as busy.
_PROGRESS = (
    "() => { const p = window.__chunkPlayer; if (!p) return null;"
    " const s = p.state();"
    " return {moved: [s.peakFrames, p.metrics.painted.length, p.metrics.chunks.length],"
    "         busy: s.inflight.length > 0}; }"
)


def wait_while_decoding(page, media, expression, what, arg=None, quiet_s=25.0, cap_s=420.0):
    """Wait for `expression`, giving up when the player stops getting anywhere
    rather than when a clock runs out.

    A deadline measures the machine. These suites decode video in the browser,
    so a wait that takes a second on a workstation takes a minute on a loaded
    four-vCPU runner doing it in software, and every fixed budget picked here
    has been both too short for CI and too slow to report a real hang. Waiting
    on the player's progress instead means a runner several times slower simply
    waits several times as long, while a player that has actually stopped is
    caught in `quiet_s` whatever the speed.

    Pre: `page` is the tab, not the tile's frame. Post: `expression` was true,
    or the assertion says what the page was doing when progress ceased.
    """
    start = last_change = time.monotonic()
    seen = None
    while True:
        if page.evaluate(expression, arg):
            return
        now = time.monotonic()
        progress = page.evaluate(_PROGRESS)
        # A request still outstanding is the page waiting on the link, not the
        # page stuck: counters only move once bytes have arrived, so a slow
        # fetch would otherwise read as silence.
        if progress is not None and (progress["moved"] != seen or progress["busy"]):
            seen, last_change = progress["moved"], now
        # A page with no player (the JPEG path) reports nothing to be quiet
        # about, so only the cap applies there.
        if progress is not None and now - last_change > quiet_s:
            raise AssertionError(
                _evidence(page, media, f"{what}\n(the player stopped getting anywhere: {seen})")
            )
        if now - start > cap_s:
            raise AssertionError(_evidence(page, media, f"{what}\n(still going nowhere at the cap)"))
        page.wait_for_timeout(250)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def redirect_gui_config(mp, config_dir: Path) -> None:
    """Point the GUI's state files at ``config_dir`` on BOTH channels: the env
    var, for anything that reads it later, and the module constants the
    in-process writers already computed from the environment at import.

    The per-test autouse fixture in conftest does the same for a test's own
    duration; a server that lives for a module opens datasets in its setup
    and writes its list on shutdown, outside any test -- which is how a
    module-scoped server on CI wrote the runner's real
    ``~/.config/lerobot/opened_datasets.json`` and tripped the real-state guard
    at the module's last teardown.
    """
    from lerobot.gui.api import datasets as datasets_api, models as models_api, robot as robot_api

    mp.setenv("LEROBOT_GUI_CONFIG_DIR", str(config_dir))
    mp.setattr(datasets_api, "OPENED_FILE", config_dir / "opened_datasets.json")
    mp.setattr(datasets_api, "SOURCES_FILE", config_dir / "dataset_sources.json")
    mp.setattr(models_api, "SOURCES_FILE", config_dir / "model_sources.json")
    mp.setattr(robot_api, "ROBOT_PROFILES_DIR", config_dir / "robots")
    mp.setattr(robot_api, "TELEOP_PROFILES_DIR", config_dir / "teleops")


class GuiServer:
    """The real GUI app on an ephemeral port, with its config and chunk cache in scratch dirs."""

    def __init__(self, config_dir: Path, cache_dir: Path):
        import pytest

        self._mp = pytest.MonkeyPatch()
        redirect_gui_config(self._mp, config_dir)
        self._mp.setenv("LEROBOT_CHUNK_CACHE_DIR", str(cache_dir))
        self.config_dir = config_dir
        self.cache_dir = cache_dir
        import uvicorn

        from lerobot.gui import server as gui_server_mod

        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self._srv = uvicorn.Server(
            uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._srv.run, daemon=True)
        self._thread.start()
        import requests

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if requests.get(self.base, timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            self._srv.should_exit = True
            raise RuntimeError("GUI server did not become ready")

    def open_dataset(self, root: Path) -> str:
        import requests

        r = requests.post(f"{self.base}/api/datasets", json={"local_path": str(root)}, timeout=120)
        assert r.status_code == 200, r.text
        return r.json()["id"]

    def stop(self):
        """Stop and WAIT: the app's shutdown hook closes process-wide pools, and
        the next server in this process must start after it, not beside it."""
        self._srv.should_exit = True
        self._thread.join(timeout=15)
        self._mp.undo()


def chunk_url(base: str, dataset_id: str, episode: int, start: int, profile: str = "low") -> str:
    from urllib.parse import quote

    return f"{base}/api/datasets/{quote(dataset_id, safe='')}/episodes/{episode}/chunk?start={start}&profile={profile}"


def parse_chunk(body: bytes):
    """The header and the parts of a chunk body: 4-byte LE header length, JSON header, parts back to back."""
    hl = struct.unpack("<I", body[:4])[0]
    header = json.loads(body[4 : 4 + hl])
    parts = {}
    for p in header["parts"]:
        parts[(p["kind"], p["camera"])] = (p, body[4 + hl + p["offset"] : 4 + hl + p["offset"] + p["length"]])
    return header, parts


def decode_h264(data: bytes) -> list[np.ndarray]:
    """Every frame of a raw Annex B H.264 stream as HxWx3 uint8, through PyAV -- an
    independent decoder, so a chunk is judged by what any decoder gets out of it."""
    import av

    with av.open(io.BytesIO(data), format="h264") as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def mean_greys(frames: list[np.ndarray]) -> list[int]:
    return [int(round(float(f.mean()))) for f in frames]


def nal_types(annexb: bytes) -> list[int]:
    """NAL unit types in order, for asserting what a chunk starts with."""
    out = []
    i = 0
    while True:
        j = annexb.find(b"\x00\x00\x01", i)
        if j < 0 or j + 3 >= len(annexb):
            break
        out.append(annexb[j + 3] & 0x1F)
        i = j + 3
    return out
