import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.overlays.saved_saliency import SavedSaliencyStore

CAM = "observation.images.top_l"


def write_maps(root, frames=(0, 50), length=100):
    directory = root / "diagnostics/policy_saliency/episode_000000"
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": 1,
        "episode_index": 0,
        "episode_length": length,
        "frames": list(frames),
        "cameras": {CAM: {"height": 72, "width": 128}},
        "method": "gradient",
    }
    (directory / "manifest.json").write_text(json.dumps(meta))
    grid = np.arange(len(frames) * 64 * 64, dtype=np.float32).reshape(len(frames), 64, 64)
    np.savez(directory / "grids.npz", **{CAM: grid})
    return directory


def test_missing_and_prediction_boundary(tmp_path):
    store = SavedSaliencyStore()
    assert store.load(tmp_path, 0, 100) is None
    write_maps(tmp_path, (10, 50))
    saved = store.load(tmp_path, 0, 100)
    assert [saved.index_at(f) for f in (0, 9, 10, 49, 50, 99, 100)] == [None, None, 0, 0, 1, 1, None]
    assert not saved.grids[CAM].flags.writeable


@pytest.mark.parametrize("frames", [(50, 0), (0, 0), (-1, 50), (0, 100), (0, 1.5)])
def test_invalid_frame_contract(tmp_path, frames):
    write_maps(tmp_path, frames)
    with pytest.raises(ValueError):
        SavedSaliencyStore().load(tmp_path, 0, 100)


def test_invalid_shape_values_and_length(tmp_path):
    directory = write_maps(tmp_path)
    with pytest.raises(ValueError):
        SavedSaliencyStore().load(tmp_path, 0, 99)
    for value in (np.full((2, 64, 64), np.nan), np.ones((3, 64, 64)), -np.ones((2, 64, 64))):
        np.savez(directory / "grids.npz", **{CAM: value})
        with pytest.raises(ValueError):
            SavedSaliencyStore().load(tmp_path, 0, 100)


def test_replacement_invalidates_cache_and_budget_is_bounded(tmp_path):
    store = SavedSaliencyStore(max_bytes=40000)
    write_maps(tmp_path / "a")
    first = store.load(tmp_path / "a", 0, 100)
    write_maps(tmp_path / "a", (1, 51))
    second = store.load(tmp_path / "a", 0, 100)
    assert first.revision != second.revision and second.frames == (1, 51)
    write_maps(tmp_path / "b")
    store.load(tmp_path / "b", 0, 100)
    assert len(store._cache) == 1


def test_sidecar_symlink_cannot_escape_dataset(tmp_path):
    directory = write_maps(tmp_path / "outside")
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "diagnostics").symlink_to(directory.parent.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        SavedSaliencyStore().load(root, 0, 100)


def test_saved_api_never_starts_worker_and_reuses_live_renderer(tmp_path, monkeypatch):
    from lerobot.gui.api import overlays
    from lerobot.overlays.adapters import PolicySaliencyAdapter

    def forbidden(*a, **kw):
        pytest.fail("Saved replay must not launch or consult a live worker")

    monkeypatch.setattr(overlays, "_get_live_reader", forbidden)
    monkeypatch.setattr(overlays, "_saved_saliency", SavedSaliencyStore())
    ds = SimpleNamespace(root=tmp_path, meta=SimpleNamespace(total_episodes=1, episodes=[{"length": 100}]))
    monkeypatch.setattr(overlays, "_app_state", SimpleNamespace(datasets={"eval/test": ds}))
    app = FastAPI()
    app.include_router(overlays.router)
    client = TestClient(app)
    base = "/api/overlays/saved/eval/test/episode/0"
    assert client.get(base).json()["state"] == "missing"
    write_maps(tmp_path)
    meta = client.get(base).json()
    assert meta["available"] and meta["frames"] == [0, 50]
    params = {"camera": CAM, "revision": meta["revision"]}
    response = client.get(base + "/frame/74", params=params)
    assert response.status_code == 200 and response.headers["x-heatmap-frame"] == "50"
    saved = overlays._saved_saliency.load(tmp_path, 0, 100)
    rgba = PolicySaliencyAdapter(device="cpu")._render(saved.grids[CAM][1], 128, 72)
    assert response.content == overlays._png(rgba)
    assert client.get(base + "/frame/74", params={**params, "revision": "old"}).status_code == 409
    assert client.get(base + "/frame/74", params={**params, "smooth": "nan"}).status_code == 422
    assert client.get(base + "/frame/74", params={**params, "camera": "other"}).status_code == 404
    assert client.get(base + "/frame/100", params=params).status_code == 404
    assert client.get(base.replace("episode/0", "episode/-1")).status_code == 404
