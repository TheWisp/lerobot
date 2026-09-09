"""The static playback path: the stored file served as it is, and the page
that plays an episode's range of it with the masks painted per frame.

Two episodes are recorded so that the second one starts at a non-zero
offset inside the packed file; a manifest that reported the file's own
time base instead of the episode's range would show episode 0's frames
for episode 1.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from lerobot.gui.api.static_playback import moov_before_mdat

pytest.importorskip("playwright")

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

CAM = "observation.images.cam"
H, W = 64, 96
FPS = 10
FRAMES = 20
LABELS = ["ball", "tray"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    """Two episodes, each frame a different flat grey so a frame is identifiable."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    root = tmp_path_factory.mktemp("static") / "static"
    ds = LeRobotDataset.create(
        repo_id="tests/static",
        fps=FPS,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            CAM: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
        },
        use_videos=True,
    )
    for ep in range(2):
        for i in range(FRAMES):
            grey = 40 + 100 * ep + 5 * i
            ds.add_frame(
                {
                    "observation.state": np.array([ep, i], np.float32),
                    "action": np.array([i, ep], np.float32),
                    "task": "static",
                    CAM: np.full((H, W, 3), grey, np.uint8),
                }
            )
        ds.save_episode()
    ds.finalize()

    ds = LeRobotDataset("tests/static", root=root)
    adopt(ds, [CAM], LABELS, (H, W))
    blob = np.zeros((H, W), bool)
    blob[8:32, 8:48] = True
    for ep in range(2):
        write_episode(ds, ep, CAM, [{"ball": blob} for _ in range(FRAMES)])
    return root


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    """One server for the module; starting the app is the slow part."""
    import os

    from lerobot.gui import server as gui_server_mod

    prev = os.environ.get("LEROBOT_GUI_CONFIG_DIR")
    os.environ["LEROBOT_GUI_CONFIG_DIR"] = str(tmp_path_factory.mktemp("config"))
    port = _free_port()
    config = uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
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
    dataset_id = r.json()["id"]
    yield base, dataset_id
    srv.should_exit = True
    if prev is None:
        os.environ.pop("LEROBOT_GUI_CONFIG_DIR", None)
    else:
        os.environ["LEROBOT_GUI_CONFIG_DIR"] = prev


def test_moov_detector_reads_the_recorded_file(dataset_root):
    files = sorted((dataset_root / "videos" / CAM).rglob("*.mp4"))
    assert files, "no video written"
    assert moov_before_mdat(files[0]) in (True, False)


def test_manifest_and_range_requests(server):
    import requests

    base, dataset_id = server
    ep1 = requests.get(f"{base}/api/datasets/{dataset_id}/episodes/1/playback", timeout=30).json()
    assert ep1["length"] == FRAMES and ep1["fps"] == FPS
    cam = ep1["cameras"][CAM]
    # Episode 1 starts where episode 0 ended, inside the same file.
    assert cam["from_timestamp"] == pytest.approx(FRAMES / FPS)
    assert cam["to_timestamp"] == pytest.approx(2 * FRAMES / FPS)
    assert cam["codec"] and cam["pix_fmt"] and cam["file_bytes"] > 0

    # The file endpoint honours Range: a partial GET returns 206 with exactly
    # the bytes asked for, and they are the file's own bytes.
    whole = requests.get(f"{base}{cam['url']}", timeout=30)
    assert whole.status_code == 200 and whole.headers["accept-ranges"] == "bytes"
    assert len(whole.content) == cam["file_bytes"]
    part = requests.get(f"{base}{cam['url']}", headers={"Range": "bytes=100-199"}, timeout=30)
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 100-199/{cam['file_bytes']}"
    assert part.content == whole.content[100:200]


def test_page_presents_the_episodes_own_frames_with_masks(server):
    """The browser is the consumer: it must present frames whose index maps
    back into the episode, paint a mask on them, and not touch any endpoint
    that decodes on the server."""
    import requests

    base, dataset_id = server
    masks = requests.get(f"{base}/api/datasets/{dataset_id}/episodes/1/masks", timeout=30)
    assert masks.status_code == 200, masks.text
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"{base}/static/static_playback.html?dataset={dataset_id}&episode=1&autoplay=1")
        page.wait_for_function("window.__metrics && window.__metrics.frames.length >= 5", timeout=30_000)
        page.wait_for_timeout(500)
        m = page.evaluate("window.__metrics")
        browser.close()

    assert not m["errors"], m["errors"]
    # The features and masks both arrived; a wrong endpoint path fails silently in the page otherwise.
    assert set(m["timeline"]) >= {"manifest", "masks", "features"}, m["timeline"]
    frames = [f["frame"] for f in m["frames"]]
    assert all(0 <= f < FRAMES for f in frames), frames
    assert max(frames) > min(frames), "playback did not advance"
    # Every presented frame carries the one stored mask, and the page drew it.
    assert all(f["drawn"] == 1 for f in m["frames"]), (
        m["timeline"],
        m["errors"],
        [r for r in m["requests"] if "masks" in r["url"]],
        m["frames"][:2],
    )
    urls = [r["url"] for r in m["requests"]]
    assert not any("/frame/" in u or "/frames" in u for u in urls), urls
    assert any("/video-file/" in u for u in urls), urls
