"""Does the batch-frames endpoint work, and how does it differ from the
per-frame endpoint the UI actually uses?

``GET /episodes/{ep}/frames`` has existed since the first GUI commit and has
never been called by the frontend or covered by a test. Before wiring playback
to it — the obvious fix for a high-latency remote host, where the UI's
one-request-per-camera-per-frame loop is round-trip bound — it needs to be
shown to work and its shape pinned down.
"""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMS = ("observation.images.top", "observation.images.wrist")
N_FRAMES = 6


@pytest.fixture(scope="module")
def two_camera_dataset(tmp_path_factory):
    """A tiny two-camera dataset on disk — never a real one from the cache."""
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        **{c: {"dtype": "image", "shape": (32, 32, 3), "names": ["h", "w", "c"]} for c in CAMS},
    }
    ds = LeRobotDataset.create(
        repo_id="test/frames_batch",
        fps=10,
        root=tmp_path_factory.mktemp("frames_batch") / "ds",
        features=features,
        use_videos=False,
    )
    rng = np.random.default_rng(0)
    for _ in range(N_FRAMES):
        ds.add_frame(
            {
                "action": np.zeros(2, dtype=np.float32),
                "observation.state": np.zeros(2, dtype=np.float32),
                "task": "probe",
                **{c: rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for c in CAMS},
            }
        )
    ds.save_episode()
    return ds.root


@pytest.fixture(scope="module")
def client(two_camera_dataset):
    """Module-scoped on purpose: ``datasets._decode_executor`` is a module
    global that the app's shutdown hook closes, so a second TestClient lifespan
    in the same process hits "cannot schedule new futures after shutdown"."""
    from lerobot.gui.server import app

    with TestClient(app) as c:
        res = c.post("/api/datasets", json={"local_path": str(two_camera_dataset)})
        assert res.status_code == 200, res.text
        body = res.json()
        # The open response has never advertised a stable id field; the id the
        # frame routes take is the repo_id the GUI registered it under.
        c.ds_id = body.get("dataset_id") or body.get("id") or body["repo_id"]  # type: ignore[attr-defined]
        yield c


def _batch(client, **params):
    return client.get(
        f"/api/datasets/{client.ds_id}/episodes/0/frames",  # type: ignore[attr-defined]
        params=params,
    )


def test_batch_returns_the_same_bytes_as_the_per_frame_endpoint(client):
    """The claim that makes it a drop-in for playback."""
    singles = []
    for i in range(N_FRAMES):
        r = client.get(
            f"/api/datasets/{client.ds_id}/episodes/0/frame/{i}",  # type: ignore[attr-defined]
            params={"camera": CAMS[0]},
        )
        assert r.status_code == 200
        singles.append(r.content)

    res = _batch(client, start=0, count=N_FRAMES, camera=CAMS[0])
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["count"] == N_FRAMES
    assert body["camera"] == CAMS[0]
    assert [f["frame_idx"] for f in body["frames"]] == list(range(N_FRAMES))
    assert [base64.b64decode(f["data"]) for f in body["frames"]] == singles


def test_batch_covers_one_camera_only(client):
    """The caveat: it batches over time, not over cameras.

    Playback needs every camera for one frame; this returns one camera for
    many frames. Wiring playback to it still means one request per camera.
    """
    res = _batch(client, start=0, count=2, camera=CAMS[1])
    assert res.json()["camera"] == CAMS[1]

    default = _batch(client, start=0, count=2)
    assert default.status_code == 200
    assert default.json()["camera"] in CAMS
    assert "frames" in default.json()
    # No parameter exists to ask for more than one camera at a time.
    assert "cameras" not in default.json()


def test_batch_clamps_to_the_episode_end(client):
    res = _batch(client, start=N_FRAMES - 2, count=50, camera=CAMS[0])
    body = res.json()
    assert body["count"] == 2
    assert body["total_frames"] == N_FRAMES


def test_batch_payload_is_larger_than_the_raw_jpegs(client):
    """base64 + JSON inflate the bytes; the win is round trips, not size."""
    res = _batch(client, start=0, count=N_FRAMES, camera=CAMS[0])
    raw = sum(len(base64.b64decode(f["data"])) for f in res.json()["frames"])
    assert len(res.content) > raw * 1.3


def test_unknown_dataset_is_a_404(client):
    res = client.get("/api/datasets/nope%2Fmissing/episodes/0/frames", params={"count": 1})
    assert res.status_code == 404
