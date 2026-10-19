"""The GUI's demo-and-act page (gui/api/manipulation/compare.py): a demo and one of its acts, each processed by the
perception pipeline, served side by side at the same moment of the demo; nothing outside the demos folder; and a
recording the pipeline as it is now has not processed, processed on request beside the server."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import manipulation
from lerobot.showservo import perception
from lerobot.showservo.recordings import open_recording
from tests.showservo.test_perception import BLOCK, FPS, T0, W, _demo, _groups_run, group_live  # noqa: F401

N = 60
LATER = 100.0  # the act was recorded this long after the demo


def _act_of(demo_dir, act_dir):
    """The demo's own frames recorded again as an act of it, LATER, in the act's layout: every tick captured, the arm,
    and targets that played the demo's time one to one."""
    cam = act_dir / "camera"
    for sub in ("rgb", "depth"):
        (cam / sub).mkdir(parents=True)
        for k in range(N):
            ext = "jpg" if sub == "rgb" else "png"
            (cam / sub / f"{k:06d}.{ext}").symlink_to(demo_dir / "recording" / sub / f"{k:06d}.{ext}")
    rows = [
        {
            "slot": k,
            "t_mono": 50.0 + k / FPS,
            "t_wall": T0 + LATER + k / FPS,
            "status": "captured",
            "image": k,
            "camera_mono": 50.0 + k / FPS,
        }
        for k in range(N)
    ]
    (cam / "timeline.json").write_text(json.dumps({"fps": FPS, "rows": rows}))
    z = np.load(demo_dir / "showservo_demo.npz")
    np.savez_compressed(
        act_dir / "arm.npz",
        t=T0 + LATER + z["t"],
        t_mono=50.0 + z["t"],
        q_obs=z["q_obs"],
        q_cmd=z["q_cmd"],
        tip_obs=z["tips"],
    )
    (act_dir / "frames").mkdir()
    mask = np.zeros((480, W), np.uint8)
    mask[BLOCK] = 255
    cv2.imwrite(str(act_dir / "frames" / "000000_mask.png"), mask)
    targets = [{"t": T0 + LATER + k / FPS, "step": "leg: key frame 1", "td": k / FPS} for k in range(N)]
    (act_dir / "act.json").write_text(
        json.dumps(
            {
                "intr": {"fx": 604.2, "fy": 604.2, "cx": 419.2, "cy": 250.6},
                "t_bc": np.eye(4).tolist(),
                "demo_root": str(demo_dir),
                "objects": ["block"],
                "frames": [{"i": 0, "t_capture": T0 + LATER, "others": []}],
                "targets": targets,
            }
        )
    )


def _process(group_live, folder, mask, t0=T0):  # noqa: F811
    rec = open_recording(folder)
    tracker, frames, pictures = _groups_run(group_live, N, 32, t0)
    d = perception.cache_dir(rec)
    d.mkdir(parents=True)
    group_live.write_groups_dump(
        d / "tracks.npz", frames, pictures, {"block": mask}, {"block": tracker.objects["block"].tracks}
    )
    perception.finish(rec, d)


@pytest.fixture
def demos(tmp_path, monkeypatch, group_live):  # noqa: F811
    root = tmp_path / "demos"
    demo = root / "insertion"
    mask = _demo(demo, N)
    (demo / "keypoints.json").write_text(
        json.dumps({"keypoints": [{"t": 1.0, "kind": "key", "frame": "block", "move": "line"}]})
    )
    _process(group_live, demo, mask)
    _act_of(demo, demo / "acts" / "20261019_101010")
    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: root)
    monkeypatch.setattr(manipulation.compare, "_loaded", {})
    monkeypatch.setattr(manipulation.compare, "_processing", {})
    return demo, mask


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(manipulation.router)
    return TestClient(app)


def test_a_demo_and_its_act_are_served_side_by_side_at_the_same_moment_of_the_demo(demos, client, group_live):  # noqa: F811
    demo, mask = demos
    _process(group_live, demo / "acts" / "20261019_101010", mask, T0 + LATER)
    page = client.get("/api/pregrasp/compare", params={"demo": "insertion", "act": "20261019_101010"})
    assert page.status_code == 200 and 'id="slider"' in page.text
    info = client.get(
        "/api/pregrasp/compare/info", params={"demo": "insertion", "act": "20261019_101010"}
    ).json()
    assert info["demo"]["processed"] and info["act"]["processed"] and info["version"] == perception.version()
    assert info["key_frames"][0]["t"] == 1.0 and info["span"][1] == pytest.approx((N - 1) / FPS, abs=1e-3)
    # at 1.5 s the groups have the block slid aside (from frame 32): the act at any other moment would look different
    r = client.get(
        "/api/pregrasp/compare/frame.jpg", params={"demo": "insertion", "act": "20261019_101010", "t": 1.5}
    )
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 2 * W + 6, "the demo and the act, side by side"
    # the act played the demo's time one to one: at the same moment both sides show the same frame, drawn alike
    differ = np.abs(img[:, :W].astype(int) - img[:, W + 6 :].astype(int))[40:440].max(axis=2) > 60
    assert differ.sum() < 20, differ.sum()


def test_nothing_outside_the_demos_folder_is_served(demos, client):
    for params in (
        {"demo": ".."},
        {"demo": "insertion", "act": "../.."},
        {"demo": "insertion", "act": "nope"},
    ):
        assert client.get("/api/pregrasp/compare/info", params=params).status_code == 404, params


def test_an_unprocessed_act_is_processed_on_request_beside_the_server(demos, client, monkeypatch):
    """The act has no perception by the pipeline as it is now: the page says so, and processing it is queued off the
    event loop, the demo already processed left alone."""
    calls = []
    monkeypatch.setattr(
        manipulation.compare.perception, "process", lambda rec, **_kw: calls.append(rec.root.name)
    )
    params = {"demo": "insertion", "act": "20261019_101010"}
    info = client.get("/api/pregrasp/compare/info", params=params).json()
    assert info["demo"]["processed"] and not info["act"]["processed"]
    assert client.post("/api/pregrasp/compare/process", json=params).json() == {"queued": ["20261019_101010"]}
    for _ in range(100):
        if manipulation.compare._processing.get(str(demos[0] / "acts" / "20261019_101010")) == "done":
            break
        time.sleep(0.02)
    assert calls == ["20261019_101010"]
