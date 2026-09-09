"""Windowed playback: the bundle, the window body, and the page's one clock.

Frames are recorded as distinct flat greys, so a decoded frame identifies
itself: a window cut one frame off, or a camera painted one frame behind
another, shows up as the wrong grey. Two episodes so that the second sits at a
non-zero offset inside the packed file.
"""

from __future__ import annotations

import gzip
import io
import json
import socket
import struct
import threading
import time

import numpy as np
import pytest

pytest.importorskip("playwright")

import av  # noqa: E402
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from lerobot.gui.api import window_playback  # noqa: E402

CAMS = ["observation.images.a", "observation.images.b"]
H, W = 240, 480  # wider than the 320 rung, so masks are scaled down for it
FPS = 10
FRAMES = 40
LABELS = ["ball"]


def grey(ep: int, i: int) -> int:
    return 30 + 60 * ep + 4 * i  # 0..39 -> 30..186 for episode 0, 90..246 for episode 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    root = tmp_path_factory.mktemp("window") / "window"
    ds = LeRobotDataset.create(
        repo_id="tests/window",
        fps=FPS,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            **{
                c: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
                for c in CAMS
            },
        },
        use_videos=True,
    )
    for ep in range(2):
        for i in range(FRAMES):
            ds.add_frame(
                {
                    "observation.state": np.array([ep, i], np.float32),
                    "action": np.array([i, ep], np.float32),
                    "task": "window",
                    **{c: np.full((H, W, 3), grey(ep, i), np.uint8) for c in CAMS},
                }
            )
        ds.save_episode()
    ds.finalize()
    ds = LeRobotDataset("tests/window", root=root)
    adopt(ds, [CAMS[0]], LABELS, (H, W))
    blob = np.zeros((H, W), bool)
    blob[40:160, 40:240] = True  # 120 x 200 of 240 x 480: 20.8% of the frame
    for ep in range(2):
        write_episode(ds, ep, CAMS[0], [{"ball": blob} for _ in range(FRAMES)])
    return root


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    """One server for the module; starting the app is the slow part."""
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


def _parse(body: bytes):
    hl = struct.unpack("<I", body[:4])[0]
    header = json.loads(body[4 : 4 + hl])
    parts = {}
    for p in header["parts"]:
        parts[(p["kind"], p["camera"])] = (p, body[4 + hl + p["offset"] : 4 + hl + p["offset"] + p["length"]])
    return header, parts


def _decode_greys(data: bytes, fmt: str = "h264") -> list[int]:
    """Mean luma of every frame of a raw H.264 (Annex B) or AV1 (OBU) stream, through PyAV."""
    out = []
    with av.open(io.BytesIO(data), format=fmt) as c:
        for frame in c.decode(video=0):
            out.append(int(frame.to_ndarray(format="gray").mean()))
    return out


def test_bundle_carries_the_overview_in_one_response(server):
    import requests

    base, did = server
    b = requests.get(f"{base}/api/datasets/{did}/episodes/1/bundle", timeout=30).json()
    assert b["length"] == FRAMES and b["fps"] == FPS
    assert set(b["cameras"]) == set(CAMS)
    assert b["series"]["observation.state"][7] == [1.0, 7.0]
    mask_key = next(iter(b["masks"]))
    assert b["masks"][mask_key]["labels"] == LABELS
    # Presence per frame is a bitset; every frame carries label 0.
    assert all(bits & 1 for bits in b["series"][mask_key])


@pytest.mark.parametrize(
    "encoder", ["codec=h264&rc=cbr", "codec=h264&rc=crf&q=26&preset=medium", "codec=av1&rc=crf&q=30"]
)
def test_window_is_frame_exact_for_every_camera_and_carries_masks(server, encoder):
    import requests

    base, did = server
    url = f"{base}/api/datasets/{did}/episodes/1/window?start=10&len=1&rung=320&{encoder}"
    r = requests.get(url, timeout=120)
    assert r.status_code == 200, r.text
    assert r.headers["x-window-cache"] == "miss"
    header, parts = _parse(r.content)
    assert header["first_frame"] == 10 and header["frames"] == FPS
    codec = header["enc"]["codec"]
    for cam in CAMS:
        p, data = parts[("video", cam)]
        assert p["codec"] == codec
        assert sum(p["frame_sizes"]) == len(data) and len(p["frame_sizes"]) == FPS
        greys = _decode_greys(data, "h264" if codec == "h264" else "obu")
        assert len(greys) == FPS
        # Each decoded frame is the recorded frame: 10..19 of episode 1, within codec noise.
        expected = [grey(1, 10 + i) for i in range(FPS)]
        assert all(abs(g - e) <= 6 for g, e in zip(greys, expected, strict=True)), (greys, expected)
    (mask_part, mask_bytes) = parts[("masks", next(k for kind, k in parts if kind == "masks"))]
    assert mask_part["encoding"] == "gzip"
    # Scaled to the rung's width, keeping the aspect; the blob still covers the same fraction of the frame.
    assert mask_part["size"] == [160, 320], mask_part
    masks = json.loads(gzip.decompress(mask_bytes))
    assert len(masks) == FPS and all(len(entry) == 1 for entry in masks)
    from lerobot.datasets.mask_codec import decode_mask

    small = decode_mask(masks[0][0][1], (160, 320))
    assert abs(small.mean() - (120 * 200) / (H * W)) < 0.01, small.mean()
    # Second request is served from the cache, byte for byte.
    r2 = requests.get(url, timeout=120)
    assert r2.headers["x-window-cache"] == "hit" and r2.content == r.content


def test_encoder_options_are_validated_and_keyed(server):
    import requests

    base, did = server
    w = f"{base}/api/datasets/{did}/episodes/1/window?start=0&len=0.5&rung=320"
    assert requests.get(f"{w}&codec=hevc", timeout=30).status_code == 400
    assert requests.get(f"{w}&codec=av1&preset=medium", timeout=30).status_code == 400
    assert requests.get(f"{w}&q=99", timeout=30).status_code == 422
    # Different options are different cache entries, not one overwriting the other.
    a = requests.get(f"{w}&rc=cbr", timeout=120)
    b = requests.get(f"{w}&rc=crf&q=34", timeout=120)
    assert a.headers["x-window-cache"] == "miss" and b.headers["x-window-cache"] == "miss"
    assert _parse(a.content)[0]["enc"]["rc"] == "cbr" and _parse(b.content)[0]["enc"]["rc"] == "crf"


def test_window_rejects_unknown_rung_and_length(server):
    import requests

    base, did = server
    assert (
        requests.get(
            f"{base}/api/datasets/{did}/episodes/1/window?start=0&len=1&rung=huge", timeout=30
        ).status_code
        == 400
    )
    assert (
        requests.get(
            f"{base}/api/datasets/{did}/episodes/1/window?start=0&len=3&rung=320", timeout=30
        ).status_code
        == 400
    )
    assert (
        requests.get(
            f"{base}/api/datasets/{did}/episodes/1/window?start=999&len=1&rung=320", timeout=30
        ).status_code
        == 404
    )


def test_cache_prune_drops_least_recently_used(tmp_path, monkeypatch):
    monkeypatch.setenv("LEROBOT_WINDOW_CACHE_DIR", str(tmp_path))
    old = tmp_path / "a.bin"
    new = tmp_path / "b.bin"
    old.write_bytes(b"x" * 100)
    new.write_bytes(b"y" * 100)
    import os

    os.utime(old, (1, 1))
    removed = window_playback.prune_cache(ceiling=150)
    assert removed == 100 and not old.exists() and new.exists()


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_page_paints_the_same_frame_on_every_camera_with_its_mask(server, codec):
    """The browser is the consumer. Each camera's canvas must show the grey of
    the frame the page says it painted, on both cameras at once, and the mask
    canvas must carry paint on the masked camera. Both codecs go through the
    browser's own decoder."""
    base, did = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(
            f"{base}/static/window_playback.html?dataset={did}&episode=1&autoplay=1&codec={codec}&rc=crf&q=30"
        )
        page.wait_for_function("window.__metrics && window.__metrics.painted.length >= 12", timeout=60_000)
        page.evaluate("window.__playback.pause()")
        page.wait_for_timeout(300)
        probe = page.evaluate(
            """() => {
              const cur = window.__playback.state().cur;
              const out = { cur, cams: [] };
              for (const box of document.querySelectorAll('.cam')) {
                const [video, mask] = box.querySelectorAll('canvas');
                const px = video.getContext('2d').getImageData(video.width >> 1, video.height >> 1, 1, 1).data;
                const md = mask.getContext('2d').getImageData(0, 0, mask.width, mask.height).data;
                let painted = 0; for (let i = 3; i < md.length; i += 4) if (md[i]) painted++;
                out.cams.push({ tag: box.querySelector('.tag').textContent, grey: px[0], maskPixels: painted });
              }
              return out;
            }"""
        )
        m = page.evaluate("window.__metrics")
        browser.close()

    assert not m["errors"], m["errors"]
    frames = [f["frame"] for f in m["painted"]]
    assert frames == sorted(frames) and len(set(frames)) == len(frames), (
        "the clock went backwards or repainted"
    )
    assert m["firstPicture"] is not None and m["firstPicture"] < 5000, m["firstPicture"]
    expected = grey(1, probe["cur"])
    for cam in probe["cams"]:
        assert abs(cam["grey"] - expected) <= 8, (probe, expected)
    by_tag = {c["tag"]: c for c in probe["cams"]}
    assert by_tag["a"]["maskPixels"] > 0 and by_tag["b"]["maskPixels"] == 0, probe
