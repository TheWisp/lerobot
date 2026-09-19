"""The treatments' defaults live in one place -- the compositor -- and the page
receives them filled in, so a tint with no colour tints the compositor's
light blue in the page too, not a colour restated in JavaScript."""

from __future__ import annotations

from lerobot.overlays.effects import resolve_params


def test_defaults_are_filled_from_the_compositor():
    assert resolve_params("tint", {}) == {"color": [79, 195, 247], "strength": 0.55}
    assert resolve_params("tint", {"color": [1, 2, 3]}) == {"color": [1, 2, 3], "strength": 0.55}
    assert resolve_params("blur", {}) == {"strength": 12}
    assert resolve_params("solid", {}) == {"color": [0, 200, 0]}
    assert resolve_params("random", {}) == {}
    assert resolve_params("none", {}) == {}


def test_the_status_endpoint_serves_filled_params(tmp_path):
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lerobot.gui.api import datasets as datasets_module
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.state import AppState
    from tests.gui.chunk_fixtures import build_dataset

    root = build_dataset(tmp_path / "d", treatment={"key": "tint", "params": {}})
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("tests/chunks", root=root)
    app = FastAPI()
    app.include_router(datasets_module.router)
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    state.datasets["d"] = ds
    datasets_module.set_app_state(state)
    client = TestClient(app)
    r = client.get("/api/datasets/d/episodes/0/masks/status")
    assert r.status_code == 200, r.text
    cam = r.json()["cameras"]["masks.a"]
    assert cam["treatments"]["ball"]["params"] == {"color": [79, 195, 247], "strength": 0.55}
    del asyncio
