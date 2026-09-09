"""The page's adaptation rule, judged on emulated links rather than on any
one real one.

Chromium's network emulation caps the download rate and adds latency to
everything the page fetches. Each profile below states what a correct rule
must do on it: no holds once the buffer has formed on a link that carries
some rung, the rung settling where the link's rate allows, and a step down
without more than a brief hold when the link drops mid-play. The dataset
is synthetic noise, which is the worst case for a codec, so the byte rates
per rung are measured first and the link caps are set relative to them:
the test scales with the content, not with a fixed number.

Constant-bitrate H.264 keeps the bytes per rung predictable for this test.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time

import numpy as np
import pytest

pytest.importorskip("playwright")

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

CAMS = ["observation.images.a", "observation.images.b"]
H, W = 240, 480
FPS = 30
FRAMES = 600  # 20 s: long enough for the buffer to form and the rung to climb


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("adapt") / "adapt"
    ds = LeRobotDataset.create(
        repo_id="tests/adapt",
        fps=FPS,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["a"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["a"]},
            **{
                c: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
                for c in CAMS
            },
        },
        use_videos=True,
    )
    # Textured, moving content: a noise field scrolling one column per frame,
    # so every frame costs the codec something and no rung is free.
    field = rng.integers(0, 256, (H, W * 2, 3), dtype=np.uint8)
    for i in range(FRAMES):
        img = np.ascontiguousarray(field[:, i % W : i % W + W])
        ds.add_frame(
            {
                "observation.state": np.array([i], np.float32),
                "action": np.array([i], np.float32),
                "task": "adapt",
                **dict.fromkeys(CAMS, img),
            }
        )
    ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    import os

    from lerobot.gui import server as gui_server_mod

    prev = {k: os.environ.get(k) for k in ("LEROBOT_GUI_CONFIG_DIR", "LEROBOT_WINDOW_CACHE_DIR")}
    os.environ["LEROBOT_GUI_CONFIG_DIR"] = str(tmp_path_factory.mktemp("config"))
    os.environ["LEROBOT_WINDOW_CACHE_DIR"] = str(tmp_path_factory.mktemp("wcache"))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()

    import requests

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(base, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        srv.should_exit = True
        pytest.fail("GUI server did not become ready")
    r = requests.post(f"{base}/api/datasets", json={"local_path": str(dataset_root)}, timeout=60)
    assert r.status_code == 200, r.text
    yield base, r.json()["id"]
    srv.should_exit = True
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def bytes_per_second(server) -> dict[str, float]:
    """Measured bytes per second of media of a 2 s window at each rung, this content, both cameras."""
    import requests

    base, did = server
    out = {}
    for rung in ("160", "320", "640", "1280"):
        r = requests.get(
            f"{base}/api/datasets/{did}/episodes/0/window?start=60&len=2&rung={rung}&codec=h264&rc=cbr",
            timeout=120,
        )
        assert r.status_code == 200, r.text
        hl = struct.unpack("<I", r.content[:4])[0]
        header = json.loads(r.content[4 : 4 + hl])
        out[rung] = len(r.content) / (header["frames"] / FPS)
    # The ladder must be a ladder for this content, or the profiles below mean nothing.
    assert out["160"] < out["320"] < out["640"] < out["1280"], out
    return out


def _run_loads(base: str, did: str, loads) -> list[dict]:
    """Open the page once per load, each under a link profile: a list of
    (at_second, download_bytes_per_s, latency_ms). Returns the page metrics of
    each load.

    Every load gets a fresh context carrying the previous one's storage, so
    what travels between loads is the remembered rung and nothing else. Sharing
    one context carries the HTTP cache too, and window responses are cacheable
    for an hour: a second visit to the same episode then answers every window
    from the cache in a few milliseconds, no bytes cross the emulated link, and
    a link that has since collapsed looks perfectly fast."""
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        storage = None
        for profile, seconds in loads:
            context = browser.new_context(storage_state=storage)
            page = context.new_page()
            cdp = context.new_cdp_session(page)
            cdp.send("Network.enable")

            def set_link(rate, latency, cdp=cdp):
                cdp.send(
                    "Network.emulateNetworkConditions",
                    {
                        "offline": False,
                        "latency": latency,
                        "downloadThroughput": rate,
                        "uploadThroughput": -1,
                    },
                )

            at, rate, latency = profile[0]
            set_link(rate, latency)
            page.goto(
                f"{base}/static/window_playback.html?dataset={did}&episode=0&rung=auto&codec=h264&rc=cbr&autoplay=1"
            )
            t0 = time.monotonic()
            for at, rate, latency in profile[1:]:
                page.wait_for_timeout(max(0, int((t0 + at - time.monotonic()) * 1000)))
                set_link(rate, latency)
                page.evaluate(f"window.__playback.mark('link {rate}')")
            page.wait_for_timeout(max(0, int((t0 + seconds - time.monotonic()) * 1000)))
            out.append(page.evaluate("window.__metrics"))
            storage = context.storage_state()
            page.close()
            context.close()
        browser.close()
    return out


def _run(base: str, did: str, profile, seconds: float) -> dict:
    """Play episode 0 under one link profile in a fresh browser."""
    return _run_loads(base, did, [(profile, seconds)])[0]


def _holds_after(m: dict, seconds: float) -> list[int]:
    """Holds that began after the first `seconds` of painted playback."""
    t_first = m["painted"][0]["t"] if m["painted"] else 0
    return [
        s["ms"]
        for s in m["stalls"]
        if any(f["frame"] == s["frame"] and f["t"] > t_first + seconds * 1000 for f in m["painted"])
    ]


def test_slow_high_latency_link_plays_without_holds_at_a_low_rung(server, bytes_per_second):
    """1.5x the 320 rung's bytes, 250 ms latency: the rule must settle at 320
    or below and, once the buffer has formed, never hold."""
    base, did = server
    cap = 1.5 * bytes_per_second["320"]
    m = _run(base, did, [(0, cap, 250)], seconds=18)
    assert not m["errors"], m["errors"]
    assert m["firstPicture"] < 4000, m["firstPicture"]
    late_holds = _holds_after(m, 3.0)
    assert not late_holds, (late_holds, [w["len"] for w in m["windows"]])
    rungs = {f["rung"] for f in m["painted"][len(m["painted"]) // 2 :]}
    assert rungs <= {"160", "320"}, rungs
    # The buffer formed: windows grew past the shortest length.
    assert max(w["len"] for w in m["windows"]) >= 2, [w["len"] for w in m["windows"]]


def test_fast_link_climbs_the_ladder(server, bytes_per_second):
    """3x the 1280 rung's bytes, 30 ms latency: the rule must climb at least to 640 within the episode, without holds."""
    base, did = server
    m = _run(base, did, [(0, 3 * bytes_per_second["1280"], 30)], seconds=18)
    assert not m["errors"], m["errors"]
    assert not _holds_after(m, 3.0), _holds_after(m, 3.0)
    assert m["painted"][-1]["rung"] in ("640", "1280", "full"), [f["rung"] for f in m["painted"][::30]]


def test_link_drop_steps_the_rung_down_with_at_most_a_brief_hold(server, bytes_per_second):
    """Fast for 8 s, then 1.2x the 160 rung: the rung must come down and playback continue."""
    base, did = server
    m = _run(
        base,
        did,
        [(0, 3 * bytes_per_second["1280"], 30), (8, 1.2 * bytes_per_second["160"], 250)],
        seconds=20,
    )
    assert not m["errors"], m["errors"]
    drop_t = next(mk["t"] for mk in m["marks"] if mk["phase"].startswith("link"))
    after = [f for f in m["painted"] if f["t"] > drop_t + 6000]
    assert after, "nothing painted after the drop"
    assert after[-1]["rung"] == "160", [f["rung"] for f in after[::15]]
    long_holds = [s for s in m["stalls"] if s["ms"] > 3000]
    assert not long_holds, long_holds


def test_the_rung_is_remembered_per_server_across_page_loads(server, bytes_per_second):
    """Three loads in one browser. A fast link leaves a memory for this server;
    the next load asks for window 0 at the remembered rung instead of the
    bottom of the ladder; a slow link after that starts from the remembered
    rung and steps down without a long hold."""
    base, did = server
    fast = [(0, 3 * bytes_per_second["1280"], 30)]
    slow = [(0, 1.5 * bytes_per_second["320"], 250)]
    first, second, third = _run_loads(base, did, [(fast, 12), (fast, 8), (slow, 15)])
    for m in (first, second, third):
        assert not m["errors"], m["errors"]
    assert first["recalled"] is None, "the first load had nothing to remember"
    assert first["memory"] and first["memory"]["rung"] in ("640", "1280", "full"), first["memory"]
    # The second load starts where the first ended, and asks for its first
    # window there rather than climbing the ladder again.
    assert second["recalled"]["rung"] == first["memory"]["rung"], (second["recalled"], first["memory"])
    assert second["windows"][0]["rung"] == second["recalled"]["rung"], second["windows"][:3]
    assert not _holds_after(second, 3.0), _holds_after(second, 3.0)
    # The third starts there too, on a link that can no longer carry it: the
    # first window is given up rather than waited out, and the rung comes down.
    assert third["recalled"]["rung"] == second["memory"]["rung"], (third["recalled"], second["memory"])
    assert third["painted"][-1]["rung"] in ("160", "320"), [f["rung"] for f in third["painted"][::15]]
    long_holds = [s for s in third["stalls"] if s["ms"] > 3000]
    assert not long_holds, long_holds
