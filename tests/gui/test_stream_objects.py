"""Objects designated on a demo's recorded stream: the worker's tracking pass and the server's flow.

A demo records the camera's colour and depth; after the demo, the operator clicks an
object on a frame of the playback, and the worker tracks it forward and backward
through the whole stream. These tests run the worker's pass with a fake segmenter
and a fake tracker, and the server's flow with the worker's answer played by the test.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import manipulation
from lerobot.showservo import motion as core

INTR = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0, "width": 848, "height": 480}
H, W = 480, 848  # the rig's camera


def fake_playback(monkeypatch, send) -> None:
    """The jog's playback (:func:`jog.play_joints`) on a fake arm: each sample goes out through ``send``, in order, once
    its time has come, as the act asks how far the playback has got."""
    from lerobot.gui.api import jog

    play: dict = {"samples": [], "due": [], "t0": 0.0, "sent": -1, "at": []}

    def play_joints(samples, due_s):
        play.update(samples=list(samples), due=list(due_s), t0=time.monotonic(), sent=-1, at=[])

    def playback_state():
        now = time.monotonic() - play["t0"]
        while play["sent"] + 1 < len(play["samples"]) and play["due"][play["sent"] + 1] <= now:
            play["sent"] += 1
            send(play["samples"][play["sent"]])
            play["at"].append(time.time())
        return play["sent"], list(play["at"])

    def update_next(expected, sample, due_s):
        if play["sent"] != expected:
            return False
        k = expected + 1
        assert len(play["samples"]) <= k + 1
        if k == len(play["samples"]):
            play["samples"].append(sample)
            play["due"].append(due_s)
        else:
            play["samples"][k], play["due"][k] = sample, due_s
        return True

    def append_next(expected_tail, sample, due_s):
        if expected_tail != len(play["samples"]) - 1:
            return False
        play["samples"].append(sample)
        play["due"].append(due_s)
        return True

    def append_chunk(expected_tail, samples, due_s):
        if expected_tail != len(play["samples"]) - 1:
            return False
        play["samples"].extend(samples)
        play["due"].extend(due_s)
        return True

    def replace_tail(expected_sent, samples, due_s):
        if play["sent"] != expected_sent:
            return False
        play["samples"][expected_sent + 1 :] = list(samples)
        play["due"][expected_sent + 1 :] = list(due_s)
        return True

    monkeypatch.setattr(jog, "update_playback_next", update_next)
    monkeypatch.setattr(jog, "append_playback_next", append_next)
    monkeypatch.setattr(jog, "append_playback_chunk", append_chunk)
    monkeypatch.setattr(jog, "replace_playback_tail", replace_tail)
    monkeypatch.setattr(jog, "play_joints", play_joints)
    monkeypatch.setattr(jog, "playback_state", playback_state)
    monkeypatch.setattr(jog, "playback_stop", lambda: play.update(samples=[]))


def write_stream(root: pathlib.Path, n: int, t0: float, hz: float = 30.0) -> pathlib.Path:
    """A recorded stream in the demo recorder's layout; frame j's pixels all read j, so a reader can tell frames apart."""
    import cv2

    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    for j in range(n):
        cv2.imwrite(
            str(root / "rgb" / f"{j:06d}.jpg"),
            np.full((H, W, 3), j, np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        cv2.imwrite(str(root / "depth" / f"{j:06d}.png"), np.full((H, W), 450, np.uint16))
    np.savetxt(root / "cam_K.txt", [[INTR["fx"], 0, INTR["cx"]], [0, INTR["fy"], INTR["cy"]], [0, 0, 1]])
    np.savetxt(root / "times.txt", t0 + np.arange(n) / hz, fmt="%.6f")
    return root


def box_mask() -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[200:280, 300:420] = True
    return m


@pytest.fixture(scope="module")
def worker():
    bench = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(bench))
    try:
        import manipulation_worker

        yield manipulation_worker
    finally:
        sys.path.remove(str(bench))


def test_face_consensus_on_large_two_plane_cloud(worker):
    """A large designation still selects the dominant face using full-cloud verification."""
    rng = np.random.default_rng(12)
    xy_main = rng.uniform(-0.05, 0.05, (30_000, 2))
    xy_other = rng.uniform(-0.05, 0.05, (20_000, 2))
    main = np.column_stack((xy_main, 0.4 + rng.normal(0, 0.0003, len(xy_main))))
    other = np.column_stack((xy_other, 0.45 + rng.normal(0, 0.0003, len(xy_other))))
    pts = np.concatenate((main, other))
    inliers = worker._consensus_plane(pts, np.random.default_rng(0), worker.FACE_TRIALS)
    assert inliers is not None
    assert inliers.sum() > 29_000
    assert abs(float(np.median(pts[inliers, 2])) - 0.4) < 0.001


def test_p2p_tracking_skips_unused_worker_geometry_but_keeps_flat_mode(worker, monkeypatch):
    """The normal live fit still has a mask and pose without running the expensive footprint pass."""
    from types import SimpleNamespace

    class P2P:
        def step(self, rgb, depth):
            return {
                "ok": True,
                "lost": False,
                "delta": np.eye(4),
                "mask": np.ones(depth.shape, bool),
                "live_uv": np.zeros((0, 2)),
                "n_visible": 50,
                "mean_residual_m": 0.001,
            }

    calls = []

    def geometry(*args):
        calls.append(1)
        return {"surface": None, "blob": None, "face_pts": None, "table_find": [0, 0, 1]}

    monkeypatch.setattr(worker, "_geometry", geometry)
    intr = worker.CameraIntrinsics(fx=600, fy=600, cx=4, cy=4)
    card = SimpleNamespace(xyz=np.array([[0.0, 0.0, 0.45]]))
    tracker = worker.Tracker(card, None, None, intr, p2p=P2P())
    frame = worker._Frame(np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 0.45), "live")
    normal = tracker.step(frame, "p2p", geometry=False)
    assert normal["ok"] and normal["card_points"] == 1 and normal["mask"].all()
    assert "table_find" not in normal and calls == []
    flat = tracker.step(frame, "p2p", geometry=True)
    assert flat["table_find"] == [0, 0, 1] and calls == [1]


class _FakeBridge:
    """Answers each step with a motion that names the frame it was given: x translation = the frame's pixel value in mm."""

    def __init__(self, lost_frame: int, fail_frame: int | None = None):
        self.inits, self.lost_frame, self.fail_frame = [], lost_frame, fail_frame

    def init(self, rgb, depth_m, mask, intr):
        self.inits.append((int(rgb[0, 0, 0]), int(mask.sum())))
        return {"ok": True}

    def step(self, rgb, depth_m):
        j = int(round(float(rgb.mean())))  # JPEG keeps a flat frame's value within a level
        if j == self.fail_frame:
            return {"ok": False, "reason": "OutOfMemoryError: CUDA out of memory"}
        delta = np.eye(4)
        delta[0, 3] = j / 1000.0
        return {"ok": True, "delta": delta, "lost": j == self.lost_frame, "mask": box_mask()}


def test_the_worker_tracks_a_clicked_object_forward_and_backward_through_the_stream(worker, tmp_path):
    n, k = 12, 5
    rec = write_stream(tmp_path / "recording", n, t0=1000.0)
    bridge = _FakeBridge(lost_frame=9)

    class Models:
        p2p_error: dict = {}

        def __init__(self, given=None):
            self.given, self.dropped = given or bridge, []

        def p2p_bridge(self, mode, key=None):
            assert key == "stream", "the stream is tracked in a process of its own, never the live one"
            return self.given

        def drop_bridge(self, key):
            self.dropped.append(key)

    class Sam:
        def mask_at(self, rgb, x, y):
            assert (x, y) == (360.0, 240.0)
            return box_mask()

    import cv2

    rgb_k = cv2.cvtColor(cv2.imread(str(rec / "rgb" / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB)
    frame = worker._Frame(rgb_k, np.full((H, W), 0.45, np.float32), "job")
    calls = []
    models = Models()
    out = worker._track_stream(
        {"recording": str(rec), "frame": k, "click": [360, 240]},
        frame,
        Sam(),
        models,
        None,
        lambda d, t: calls.append((d, t)),
    )
    z = np.load(io.BytesIO(out))
    meta = json.loads(str(z["meta"]))
    assert meta["ok"] and meta["frames"] == n
    deltas, seen, masks = z["deltas"], z["seen"], z["masks"]
    assert np.allclose(deltas[k], np.eye(4)), "the clicked frame is the reference"
    others = [j for j in range(n) if j != k]
    assert np.allclose([deltas[j][0, 3] for j in others], [j / 1000.0 for j in others], atol=1.5e-3), (
        "every frame read in its pass"
    )
    assert len(bridge.inits) == 2 and all(v == (k, int(box_mask().sum())) for v in bridge.inits), (
        "both passes start on the clicked frame"
    )
    assert not seen[9] and seen[[j for j in range(n) if j != 9]].all()
    assert masks.shape == (n, H // worker.STREAM_MASK_SCALE, W // worker.STREAM_MASK_SCALE) and masks[k].any()
    assert z["mask"].shape == (H, W) and calls[-1] == (n - 1, n - 1)
    assert models.dropped == ["stream"], (
        "the stream's process is closed after the job, freeing its GPU memory"
    )
    # A tracker step that fails ends the job with its reason, and the process is still closed.
    models = Models(_FakeBridge(lost_frame=-1, fail_frame=8))
    job = {"recording": str(rec), "frame": k, "click": [360, 240]}
    out = worker._track_stream(job, frame, Sam(), models, None, lambda d, t: None)
    meta = json.loads(str(np.load(io.BytesIO(out))["meta"]))
    assert not meta["ok"] and "frame 8" in meta["reason"] and "out of memory" in meta["reason"]
    assert models.dropped == ["stream"]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(manipulation.router)
    return TestClient(app)


class _FakeProc:
    def poll(self):
        return None

    def terminate(self):
        pass


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


class _GridTier:
    """Descriptors that name their spot: a grid inside the mask, each spot's descriptor the same in every frame."""

    def teach(self, rgb, mask):
        ys, xs = np.nonzero(mask)
        uv = np.array(
            [
                (x, y)
                for y in range(ys.min(), ys.max() + 1, 6)
                for x in range(xs.min(), xs.max() + 1, 6)
                if mask[y, x]
            ],
            dtype=float,
        )
        desc = np.stack([np.random.default_rng(int(x) * 1000 + int(y)).standard_normal(32) for x, y in uv])
        return uv, (desc / np.linalg.norm(desc, axis=1, keepdims=True)).astype(np.float32)


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_rotated_features_keep_their_original_camera_pixels(worker, turns):
    """Only appearance rotates; depth still belongs to the original image pixels."""
    rgb = np.zeros((7, 11, 3), np.uint8)
    mask = np.zeros((7, 11), bool)
    mask[1, 8] = mask[5, 2] = True
    rgb[1, 8] = [10, 20, 30]
    rgb[5, 2] = [40, 50, 60]

    class Pixels:
        def teach(self, image, region):
            assert image.flags.c_contiguous and region.flags.c_contiguous
            ys, xs = np.nonzero(region)
            return np.c_[xs, ys].astype(float), image[ys, xs]

    uv, desc = worker._QuarterTurnTier(Pixels(), turns).teach(rgb, mask)
    xy = uv.astype(int)
    assert mask[xy[:, 1], xy[:, 0]].all()
    np.testing.assert_array_equal(desc, rgb[xy[:, 1], xy[:, 0]])
    assert set(map(tuple, xy)) == {(8, 1), (2, 5)}


def test_reference_matching_uses_the_strongest_certified_image_orientation(worker, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lerobot.showservo.pose import CameraIntrinsics, Rigid3

    rec = write_stream(tmp_path / "rec", 1, t0=time.time())
    intr = CameraIntrinsics(**{k: INTR[k] for k in ("fx", "fy", "cx", "cy")})
    live = worker._Frame(np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.45), "live")
    tried = []

    def bind(card, frame, region, tier, intr):
        k = getattr(tier, "turns", 0)
        tried.append(k)
        return (
            SimpleNamespace(
                n_inliers=[21, 90, 30, 25][k],
                rms=0.001,
                transform=Rigid3(np.eye(3), np.array([0.01 * k, 0.0, 0.0])),
            ),
            None,
            None,
            None,
        )

    monkeypatch.setattr(worker, "_bind", bind)
    monkeypatch.setattr(worker, "_surface", lambda *args: np.empty((0, 3)))
    result = worker._find_reference(
        {"recording": str(rec), "frame": 0, "mask": box_mask()}, live, box_mask(), _GridTier(), intr
    )
    assert result["ref_ok"] and result["ref_inliers"] == 90
    np.testing.assert_allclose(result["ref_delta"][:3, 3], [0.01, 0, 0])
    assert tried == [0, 1, 2, 3]


def test_arbitrary_image_rotation_maps_features_back_to_original_depth_pixels(worker):
    import cv2

    rgb = np.zeros((80, 120, 3), np.uint8)
    mask = np.zeros((80, 120), bool)
    mask[5:20, 4:85] = True  # deliberately near two edges

    class Pixels:
        def teach(self, image, region):
            y, x = np.nonzero(region)
            return np.column_stack((x, y)).astype(float), np.ones((len(x), 1))

    uv, _ = worker._ImageTurnTier(Pixels(), 32.0).teach(rgb, mask)
    xy = np.round(uv).astype(int)
    assert mask[xy[:, 1], xy[:, 0]].all()
    # Expanded canvas must preserve the entire object, up to rasterization.
    assert len(uv) == pytest.approx(mask.sum(), rel=0.02)
    rotated = cv2.warpAffine(mask.astype(np.uint8), cv2.getRotationMatrix2D((60, 40), 32, 1), (120, 80)) > 0
    difference = (worker._outline_angle(rotated) - worker._outline_angle(mask) + 32 + 90) % 180 - 90
    assert abs(difference) < 1


@pytest.mark.parametrize("reverse", [False, True])
def test_surface_refinement_keeps_the_correspondences_that_established_identity(
    worker, tmp_path, monkeypatch, reverse
):
    from types import SimpleNamespace

    from lerobot.showservo.pose import CameraIntrinsics, Rigid3

    rec = write_stream(tmp_path / "rec", 1, t0=time.time())
    intr = CameraIntrinsics(**{k: INTR[k] for k in ("fx", "fy", "cx", "cy")})
    live = worker._Frame(np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.45), "live")

    def bind(card, *args):
        return (
            SimpleNamespace(n_inliers=len(card.xyz), rms=0.0, transform=Rigid3.identity()),
            card.uv,
            np.arange(len(card.xyz)),
            None,
        )

    shift = np.array([0.0001, 0, 0])

    def refit(reference, live, start, **kw):
        assert kw["turns"] == 1
        rotation = np.diag([-1.0, -1, 1]) if reverse else np.eye(3)
        centre = reference.mean(axis=0)
        return Rigid3(rotation, centre - rotation @ centre + shift), 0.0

    monkeypatch.setattr(worker, "_bind", bind)
    monkeypatch.setattr(worker, "fit_surface", refit)
    result = worker._find_reference(
        {"recording": str(rec), "frame": 0, "mask": box_mask()}, live, box_mask(), _GridTier(), intr
    )
    np.testing.assert_allclose(result["ref_delta"][:3, :3], np.eye(3))
    np.testing.assert_allclose(result["ref_delta"][:3, 3], np.zeros(3) if reverse else shift, atol=1e-12)


def test_a_tracked_frame_carries_what_the_rule_judges_the_object_placed_onto_by(worker):
    """The act's rule judges a view of the object placed onto as it judges the picked one: by its fit points against
    its middle. The worker sent only its motion and mask; it now sends its fit points and key points too, and leaves
    them out when the tracker has none."""
    share = {
        "delta": np.eye(4),
        "mask": np.ones((4, 4), bool),
        "fit_uv": np.zeros((5, 2), np.float32),
        "fit_inlier": np.ones(5, bool),
        "model": np.zeros((7, 3)),
    }
    arrays = worker._others_arrays(
        [("box", share), ("roll", {"delta": np.eye(4), "model": np.zeros((0, 3))})]
    )
    assert set(arrays) == {
        "other_delta_0",
        "other_mask_0",
        "other_fit_uv_0",
        "other_fit_inlier_0",
        "other_model_0",
        "other_delta_1",
    }
    assert arrays["other_model_0"].dtype == np.float32 and arrays["other_model_0"].shape == (7, 3)


def test_the_find_reports_how_many_points_its_match_is_a_share_of(worker, tmp_path):
    """Whether a find is strong is its matched points as a share of the demo view's card; without the card's size the
    server cannot tell, and an unknown strength is not refused, so a find that stopped reporting it would quietly
    switch the weak-find check off."""
    import cv2

    from lerobot.showservo.pose import CameraIntrinsics

    rec = write_stream(tmp_path / "rec", 3, t0=time.time())
    intr = CameraIntrinsics(fx=INTR["fx"], fy=INTR["fy"], cx=INTR["cx"], cy=INTR["cy"])
    rgb = cv2.cvtColor(cv2.imread(str(rec / "rgb" / "000001.jpg")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(rec / "depth" / "000001.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    live = worker._Frame(rgb, depth, "live")
    r = worker._find_reference(
        {"recording": str(rec), "frame": 1, "mask": box_mask()}, live, box_mask(), _GridTier(), intr
    )
    assert r["ref_ok"], r
    card = worker.Card(worker._Frame(rgb, depth, "reference"), box_mask(), _GridTier(), intr)
    assert r["ref_card_points"] == len(card.xyz) > 0
    assert r["ref_inliers"] <= r["ref_card_points"] and r["ref_turn_deg"] == pytest.approx(0.0, abs=1.0)


def test_an_object_clicked_on_the_recording_is_tracked_listed_drawn_saved_and_loaded(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path / "demos")
    t0 = time.time()
    n_samples, n_frames = 30, 15
    # The stream runs at half the arm's 30 Hz, as it does when the live tracker shares the camera.
    rec = write_stream(tmp_path / "demos" / ".recordings" / "work", n_frames, t0=t0, hz=15.0)
    t = np.arange(n_samples) / 30.0
    demo = manipulation.runtime._Demo(
        name="stacked",
        concept="demo",
        fps=30.0,
        t=t,
        tips=np.tile(np.eye(4), (n_samples, 1, 1)),
        grippers=np.zeros(n_samples),
        q_obs=np.zeros((n_samples, 7)),
        q_cmd=np.zeros((n_samples, 7)),
        deltas=np.tile(np.eye(4), (n_samples, 1, 1)),
        seen=np.zeros(n_samples, dtype=bool),
        delta0=np.eye(4),
        t0=t0,
        intr=dict(INTR),
        recording=str(rec),
    )
    manipulation.runtime._state.worker.proc = _FakeProc()
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()
        assert (
            client.post(
                "/api/pregrasp/demo/objects", json={"i": 6, "x": 360, "y": 240, "name": ""}
            ).status_code
            == 422
        )
        r = client.post("/api/pregrasp/demo/objects", json={"i": 6, "x": 360, "y": 240, "name": "gamepad"})
        assert r.status_code == 200 and r.json()["objects"][0]["status"] == "tracking"
        job = client.get("/api/pregrasp/worker/job", params={"wait": 0}).json()
        assert (
            job["kind"] == "stream_object"
            and job["recording"] == str(rec)
            and job["frame"] == 3
            and job["click"] == [360, 240]
        )
        frame = np.load(
            io.BytesIO(client.get("/api/pregrasp/worker/frame.npz", params={"id": job["id"]}).content)
        )
        assert int(frame["rgb"][0, 0, 0]) in (2, 3, 4) and frame["depth"].max() == pytest.approx(0.45), (
            "the clicked frame, in metres"
        )
        assert (
            client.post(
                "/api/pregrasp/worker/progress", params={"id": job["id"], "done": 7, "total": 14}
            ).status_code
            == 200
        )
        assert client.get("/api/pregrasp/demo/objects").json()["objects"][0]["progress"] == pytest.approx(0.5)
        seen = np.ones(n_frames, dtype=bool)
        seen[10:] = False
        masks = np.zeros((n_frames, H // 4, W // 4), dtype=bool)
        masks[:, 50:70, 75:105] = True
        result = _npz(
            meta=json.dumps({"ok": True, "frames": n_frames, "seen_fraction": float(seen.mean())}),
            deltas=np.tile(np.eye(4), (n_frames, 1, 1)),
            seen=seen,
            masks=masks,
            mask=np.zeros((H, W), dtype=bool),
        )
        assert (
            client.post("/api/pregrasp/worker/result", params={"id": job["id"]}, content=result).status_code
            == 200
        )
        listed = client.get("/api/pregrasp/demo/objects").json()["objects"][0]
        assert (
            listed["status"] == "done"
            and listed["seen_fraction"] == pytest.approx(10 / 15)
            and listed["t"] == pytest.approx(0.2, abs=1e-5)  # stamps are written to the microsecond
        )
        assert client.get("/api/pregrasp/demo/frame.jpg", params={"i": 6}).status_code == 200, (
            "the outline is drawn on the frame"
        )
        assert client.post("/api/pregrasp/demo/save", json={}).status_code == 200
        root = tmp_path / "demos" / "stacked"
        assert (root / manipulation.runtime.OBJECTS_FILE).exists()
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
        loaded = client.post("/api/pregrasp/demo/load", json={"name": "stacked"}).json()
        assert loaded["teach_pending"] is False
        again = client.get("/api/pregrasp/demo/objects").json()["objects"]
        assert [(o["name"], o["status"]) for o in again] == [("gamepad", "done")], (
            "the tracked object comes back with the demo"
        )
        with manipulation.runtime._state.lock:
            assert np.array_equal(manipulation.runtime._state.demo.objects["gamepad"]["seen"], seen)
        assert client.post("/api/pregrasp/demo/objects/remove", json={"name": "gamepad"}).status_code == 200
        assert not (root / manipulation.runtime.OBJECTS_FILE).exists()
        # A worker that finds nothing under the click says so.
        client.post("/api/pregrasp/demo/objects", json={"i": 0, "x": 5, "y": 5, "name": "nothing"})
        job = client.get("/api/pregrasp/worker/job", params={"wait": 0}).json()
        client.post(
            "/api/pregrasp/worker/result",
            params={"id": job["id"]},
            content=_npz(meta=json.dumps({"ok": False, "reason": "nothing segments at that pixel"})),
        )
        failed = client.get("/api/pregrasp/demo/objects").json()["objects"][0]
        assert failed["status"] == "failed" and "nothing segments" in failed["reason"]
    finally:
        manipulation.runtime._state.worker.proc = None
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()


def _demo_with_object(tmp_path, t0, n_samples=30, n_frames=15):
    """A demo recorded without a teach, with one object designated on its stream and marks bound to it."""
    from scipy.spatial.transform import Rotation

    rec = write_stream(tmp_path / "demos" / ".recordings" / "work", n_frames, t0=t0, hz=15.0)
    t = np.arange(n_samples) / 30.0
    demo = manipulation.runtime._Demo(
        name="bound",
        concept="demo",
        fps=30.0,
        t=t,
        tips=np.tile(np.eye(4), (n_samples, 1, 1)),
        grippers=np.zeros(n_samples),
        q_obs=np.zeros((n_samples, 7)),
        q_cmd=np.zeros((n_samples, 7)),
        deltas=np.tile(np.eye(4), (n_samples, 1, 1)),
        seen=np.zeros(n_samples, dtype=bool),
        delta0=np.eye(4),
        t0=t0,
        intr=dict(INTR),
        recording=str(rec),
    )
    deltas = np.tile(np.eye(4), (n_frames, 1, 1))
    for f in range(n_frames):  # the object slid 1 mm and turned 1 deg per frame after the click on frame 2
        deltas[f][:3, :3] = Rotation.from_euler("z", f - 2, degrees=True).as_matrix()
        deltas[f][:3, 3] = [(f - 2) / 1000.0, 0.0, 0.0]
    seen = np.ones(n_frames, dtype=bool)
    seen[5:7] = False  # hidden while the pre-grasp at frame 6 was shown
    demo.objects["gamepad"] = {
        "frame": 2,
        "click": [424, 240],
        "status": "done",
        "deltas": deltas,
        "seen": seen,
        "masks": np.zeros((n_frames, H // 4, W // 4), dtype=bool),
        "mask": box_mask(),
    }
    return demo


def test_marks_bound_to_a_designated_object_use_its_live_find_and_its_pose_in_the_demo(
    client, tmp_path, monkeypatch
):
    from scipy.spatial.transform import Rotation

    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path / "demos")
    t0 = time.time()
    demo = _demo_with_object(tmp_path, t0)
    manipulation.runtime._state.worker.proc = _FakeProc()
    try:
        with manipulation.runtime._state.lock:
            (
                manipulation.runtime._state.demo,
                manipulation.runtime._state.teach,
                manipulation.runtime._state.test,
            ) = demo, None, None
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()
        post = lambda kps: client.post("/api/pregrasp/demo/keypoints", json={"keypoints": kps})  # noqa: E731
        assert post([{"t": 0.4, "kind": "pregrasp", "object": "cube"}]).status_code == 422, (
            "only a tracked object"
        )
        mixed = [{"t": 0.4, "kind": "pregrasp", "object": "gamepad"}, {"t": 0.7, "kind": "grasp_end"}]
        assert post(mixed).status_code == 422, "the pre-grasp and the grasp are for one object"
        unnamed = [{"t": 0.4, "kind": "pregrasp"}, {"t": 0.7, "kind": "grasp_end"}]
        assert post(unnamed).status_code == 422, "nothing was taught before this demo"
        demo.keypoints = unnamed  # as saved before the editor named the object
        ref_motion, problem = manipulation.planning._reference_motion(demo, None)
        assert ref_motion is None and "name no object" in problem, "no taught object to fall back on"
        bound = [
            {"t": 0.4, "kind": "pregrasp", "object": "gamepad"},
            {"t": 0.7, "kind": "grasp_end", "object": "gamepad"},
        ]
        assert post(bound).status_code == 200
        assert manipulation.planning._session_object(demo) == "gamepad"
        ref_motion, problem = manipulation.planning._reference_motion(demo, None)
        assert ref_motion is None and "click gamepad in the camera view" in problem, (
            "the act needs the live find first"
        )
        assert client.post("/api/pregrasp/act", json={}).status_code == 409
        # The live click teaches with the demo's view of the object as its reference.
        monkeypatch.setattr(
            manipulation.camera,
            "_frame",
            lambda: _async((np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.45, np.float32), dict(INTR))),
        )
        r = client.post(
            "/api/pregrasp/teach/capture",
            json={"mode": "features", "click": [400, 250], "ref_object": "gamepad"},
        )
        assert r.status_code == 200, r.text
        job = client.get("/api/pregrasp/worker/job", params={"wait": 0}).json()
        assert (
            job["kind"] == "teach"
            and job["concept"] == "gamepad"
            and job["ref_frame"] == 2
            and job["ref_recording"] == demo.recording
        )
        frame = np.load(
            io.BytesIO(client.get("/api/pregrasp/worker/frame.npz", params={"id": job["id"]}).content)
        )
        assert np.array_equal(frame["ref_mask"], box_mask()), "the demo's mask travels with the job"
        ref_delta = np.eye(4)
        ref_delta[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
        ref_delta[:3, 3] = [0.03, 0.01, 0.0]
        result = _npz(
            meta=json.dumps(
                {
                    "ok": True,
                    "n_points": 50,
                    "radius_mm": 40.0,
                    "shape_class": "box",
                    "yaw_observable": True,
                    "face": None,
                    "ref_ok": True,
                    "ref_inliers": 80,
                    "ref_turn_deg": 20.0,
                }
            ),
            mask=box_mask(),
            uv=np.zeros((50, 2)),
            xyz=np.zeros((50, 3)),
            ref_delta=ref_delta,
        )
        assert (
            client.post("/api/pregrasp/worker/result", params={"id": job["id"]}, content=result).status_code
            == 200
        )
        st = client.get("/api/pregrasp/state").json()
        assert st["teach"]["ref"] == {
            "object": "gamepad",
            "ok": True,
            "inliers": 80,
            "turn_deg": 20.0,
            "reason": "",
            # This worker answer predates the card's size, so the find's strength is unknown, not weak.
            "card_points": None,
            "strong": None,
            "share": None,
        }
        assert st["demo"]["objects"] == ["gamepad"]
        # The motion the act uses: the live track, times the find, times the inverse of where the demo had the
        # object on its pose frame: the frame it was clicked on, 2, whose view the find matched, so the find alone.
        ref_motion, problem = manipulation.planning._reference_motion(demo, manipulation.runtime._state.teach)
        assert problem == "" and np.allclose(
            ref_motion, ref_delta @ np.linalg.inv(demo.objects["gamepad"]["deltas"][2])
        )
        assert np.allclose(ref_motion, ref_delta)
        failed = dict(
            manipulation.runtime._state.teach.keypoints["ref"],
            ok=False,
            delta=None,
            reason="the live view does not match",
        )
        manipulation.runtime._state.teach.keypoints["ref"] = failed
        assert manipulation.planning._reference_motion(demo, manipulation.runtime._state.teach) == (
            None,
            "the live view does not match",
        )
    finally:
        manipulation.runtime._state.worker.proc = None
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
                manipulation.runtime._state.test
            ) = None
            manipulation.runtime._state.teach_job = None
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()


def test_the_editor_shows_the_followed_objects_own_visibility(client, tmp_path, monkeypatch):
    """The editor's "object hidden" came from the live tracker's frames during the recording, a few a second, so a
    sample between two of them read hidden while the object's outline was drawn on every frame. With marks on a
    designated object it is that object's own track on the recording, the one the outline comes from."""
    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path / "demos")
    demo = _demo_with_object(tmp_path, time.time())  # its object is hidden on stream frames 5 and 6
    with manipulation.runtime._state.lock:
        manipulation.runtime._state.demo = demo
    try:
        demo.keypoints = [{"t": 0.4, "kind": "pregrasp"}, {"t": 0.7, "kind": "grasp_end"}]
        assert client.get("/api/pregrasp/demo/curve").json()["seen"] == [0] * 30, (
            "unnamed marks: the live tracker's"
        )
        demo.keypoints = [
            {"t": 0.4, "kind": "pregrasp", "object": "gamepad"},
            {"t": 0.7, "kind": "grasp_end", "object": "gamepad"},
        ]
        seen = client.get("/api/pregrasp/demo/curve").json()["seen"]
        hidden = {i for i, v in enumerate(seen) if not v}
        # Samples 10 and 12 fall on frames 5 and 6; the samples between frames may go either way.
        assert {10, 12} <= hidden <= set(range(9, 15)), sorted(hidden)
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None


def test_the_editor_draws_the_tracked_pose_and_reports_the_frame_the_act_reads_it_from(
    client, tmp_path, monkeypatch
):
    """The demo view drew the object's outline but not its tracked pose, so a pose the tracker had tipped could not
    be seen, and nothing showed which frame the act reads the object's demo pose from."""
    import cv2
    from scipy.spatial.transform import Rotation

    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path / "demos")
    demo = _demo_with_object(tmp_path, time.time())  # hidden on stream frames 5 and 6
    demo.keypoints = [
        {"t": 0.4, "kind": "pregrasp", "object": "gamepad"},
        {"t": 0.7, "kind": "grasp_end", "object": "gamepad"},
    ]
    t_bc = np.eye(4)  # a camera looking down at the table at an angle
    t_bc[:3, :3] = Rotation.from_euler("x", 150, degrees=True).as_matrix()
    t_bc[:3, 3] = [0.0, 0.3, 0.4]

    def red(i: int) -> int:  # the pose's x axis; the synthetic frames are grey and the outline is blue
        jpg = client.get("/api/pregrasp/demo/frame.jpg", params={"i": i}).content
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).astype(int)
        return int(((bgr[..., 2] > 180) & (bgr[..., 1] < 90) & (bgr[..., 0] < 90)).sum())

    with manipulation.runtime._state.lock:
        manipulation.runtime._state.demo = demo
    try:
        assert red(8) == 0, "without the arm's camera calibration the frames render, without a pose"
        monkeypatch.setattr(manipulation.camera, "_t_base_cam", lambda: t_bc)
        times = np.loadtxt(pathlib.Path(demo.recording) / "times.txt")
        pose_t = client.get("/api/pregrasp/demo/objects").json()["objects"][0]["pose_t"]
        assert pose_t == pytest.approx(times[2] - demo.t0), "the frame the object was clicked on"
        assert manipulation.planning._pose_frame(demo, "gamepad") == 2, (
            "the same frame the act's reference motion uses"
        )
        assert red(8) > 5 and red(2) > 5, "every frame carries the pose, the frame the act reads included"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None


@pytest.mark.usefixtures("act_simulation")
def test_every_act_finds_its_object_afresh_where_the_tracker_last_saw_it(tmp_path, monkeypatch):
    """A track kept since an earlier find goes on adding points and drifts, so an act finds its object again
    first: a click deep inside where the tracker last saw it, for the object the marks name, and no pose is
    used until the restarted tracker has certified a view. A failed find ends the act before anything moves."""
    import asyncio

    from scipy.spatial.transform import Rotation

    demo = _demo_with_object(tmp_path, time.time())
    demo.keypoints = [
        {"t": 0.4, "kind": "pregrasp", "object": "gamepad"},
        {"t": 0.7, "kind": "grasp_end", "object": "gamepad"},
    ]
    monkeypatch.setattr(
        manipulation.camera,
        "_frame",
        lambda: _async((np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.45, np.float32), dict(INTR))),
    )
    monkeypatch.setattr(manipulation.runtime, "ACT_STEP_TIMEOUT_S", 0.5)
    monkeypatch.setattr(
        manipulation.runtime._state, "groups_with_acts", False
    )  # held poses: no point groups view here
    taught = np.zeros((H, W), dtype=bool)
    taught[100:141, 100:141] = True  # where the last find had it
    seen = np.zeros((H, W), dtype=bool)
    seen[300:341, 500:541] = True  # where the tracker last saw it, since then moved by hand
    fresh = np.eye(4)
    fresh[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()

    def start_from_an_old_find() -> None:
        teach = manipulation.runtime._Teach(
            at="t",
            box=(0, 0, 0, 0),
            rgb=np.zeros((H, W, 3), np.uint8),
            depth_m=np.full((H, W), 0.45, np.float32),
            intr=dict(INTR),
            keypoints={
                "mode": "features",
                "concept": "gamepad",
                "mask": taught,
                "n_points": 50,
                "ref": {"object": "gamepad", "ok": True, "delta": np.eye(4), "inliers": 80, "reason": ""},
            },
        )
        live = manipulation.runtime._Test(
            at="t", rgb=None, result={"ok": True, "live_mask": seen, "delta_cam": np.eye(4)}
        )
        with manipulation.runtime._state.lock:
            (
                manipulation.runtime._state.demo,
                manipulation.runtime._state.teach,
                manipulation.runtime._state.test,
            ) = demo, teach, live
            manipulation.runtime._state.teach_job = None
            manipulation.runtime._state.track.history.clear()
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()

    async def play_the_worker(meta: dict, then_tracked: bool) -> manipulation.runtime._Job:
        for _ in range(200):
            await asyncio.sleep(0.01)
            with manipulation.runtime._state.lock:
                job_id = manipulation.runtime._state.teach_job
            if job_id is not None:
                break
        else:
            raise AssertionError("no find was asked of the worker")
        job = manipulation.runtime._state.worker.jobs[job_id]
        job.result = {
            "n_points": 50,
            "radius_mm": 40.0,
            "shape_class": "box",
            "yaw_observable": True,
            "mask": seen,
            "uv": np.zeros((50, 2)),
            "xyz": np.zeros((50, 3)),
            "ref_delta": fresh,
            **meta,
        }
        manipulation.acquisition._apply_teach_result(job)
        if then_tracked:
            await asyncio.sleep(0.05)
            with manipulation.runtime._state.lock:
                manipulation.runtime._state.track.history.append((time.time(), True, np.eye(4)))
        return job

    served = []

    async def serve(metas, then_tracked):
        for meta in metas:
            job = await play_the_worker(meta, then_tracked)
            served.append(job)
        return job

    async def find(meta: dict | list, then_tracked: bool) -> tuple[manipulation.runtime._Job, str]:
        worker = asyncio.create_task(serve(meta if isinstance(meta, list) else [meta], then_tracked))
        why = await manipulation.acquisition._find_afresh("gamepad", lambda: False)
        return await worker, why

    manipulation.runtime._state.worker.proc = _FakeProc()
    try:
        start_from_an_old_find()
        job, why = asyncio.run(find({"ok": True, "ref_ok": True, "ref_inliers": 90}, then_tracked=True))
        assert why == "", why
        assert job.kind == "teach" and job.concept == "gamepad" and job.extra["ref_object"] == "gamepad"
        assert abs(job.click[0] - 520) <= 1 and abs(job.click[1] - 320) <= 1, (
            f"the click lands inside where the tracker last saw it, not the old find: {job.click}"
        )
        assert np.allclose(manipulation.runtime._state.teach.keypoints["ref"]["delta"], fresh), (
            "the act uses the new find"
        )

        start_from_an_old_find()
        _, why = asyncio.run(find({"ok": True, "ref_ok": True, "ref_inliers": 90}, then_tracked=False))
        assert why == "the tracker has not seen gamepad since finding it", (
            "no pose from before the find is used while the restarted tracker has none"
        )

        start_from_an_old_find()
        weak = {"ok": True, "ref_ok": True, "ref_inliers": 32, "ref_card_points": 400}
        _, why = asyncio.run(find(weak, then_tracked=True))
        assert why.startswith("a weak find: gamepad matched 32 of the demo view's 400 points"), why

        start_from_an_old_find()
        strong = {"ok": True, "ref_ok": True, "ref_inliers": 116, "ref_card_points": 400}
        _, why = asyncio.run(find(strong, then_tracked=True))
        assert why == "", why

        start_from_an_old_find()
        mismatch = {"ok": True, "ref_ok": False, "ref_reason": "the live view does not match the demo's view"}
        _, why = asyncio.run(find([mismatch] * 3, then_tracked=True))
        assert why == "the live view does not match the demo's view"
        assert len({tuple(job.click) for job in served[-3:]}) == 3

        start_from_an_old_find()
        # A button-sized proposal fails registration. Retry in the original
        # observed outline, not the rejected subpart's replacement teach mask.
        tiny = np.zeros_like(seen)
        tiny[317:324, 517:524] = True
        _, why = asyncio.run(find([{**mismatch, "mask": tiny}, strong], then_tracked=True))
        assert why == "", why
        retry = served[-1].click
        assert seen[retry[1], retry[0]] and not tiny[retry[1], retry[0]]
        assert served[-1].log["attempt"] == 2

        start_from_an_old_find()
        _, why = asyncio.run(find([mismatch, weak], then_tracked=True))
        assert why.startswith("a weak find:"), "a retry must not relax pose acceptance"

        start_from_an_old_find()
        act = manipulation.runtime._state.act
        act.on, act.ok, act.reason, act.step, act.stop_requested = True, None, "", "starting", False

        async def act_with_a_failed_find() -> None:
            worker = asyncio.create_task(serve([mismatch] * 3, then_tracked=True))
            await manipulation.execution._act_task(0.5)
            await worker

        asyncio.run(act_with_a_failed_find())
        assert (act.ok, act.step, act.reason, act.on) == (
            False,
            "aborted",
            "the live view does not match the demo's view",
            False,
        ), "the act ends on its find, before planning or moving"
    finally:
        manipulation.runtime._state.worker.proc = None
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
                manipulation.runtime._state.test
            ) = None
            manipulation.runtime._state.teach_job = None
            manipulation.runtime._state.track.history.clear()
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()


async def _async(value):
    return value


def test_the_worker_locates_an_object_against_the_demo_view_and_teaches_nothing(worker, tmp_path):
    """The object a place goes onto, and the held object in the gripper, are found against the demo's view of them
    while the live track follows something else: a locate takes no card, track or Point2Pose session to touch."""
    import cv2

    from lerobot.showservo.pose import CameraIntrinsics

    rec = write_stream(tmp_path / "rec", 3, t0=time.time())
    intr = CameraIntrinsics(fx=INTR["fx"], fy=INTR["fy"], cx=INTR["cx"], cy=INTR["cy"])
    rgb = cv2.cvtColor(cv2.imread(str(rec / "rgb" / "000001.jpg")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(rec / "depth" / "000001.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    live = worker._Frame(rgb, depth, "live")

    class Sam:
        def mask_at(self, rgb, x, y):
            return box_mask() if box_mask()[y, x] else None

    ref = {"recording": str(rec), "frame": 1, "mask": box_mask()}
    out = np.load(io.BytesIO(worker._locate(live, Sam(), _GridTier(), intr, [360, 240], ref)))
    meta = json.loads(str(out["meta"]))
    assert (
        meta["ok"]
        and meta["ref_ok"]
        and meta["ref_card_points"] > 0
        and meta["ref_inliers"] <= meta["ref_card_points"]
    )
    assert np.allclose(out["ref_delta"], np.eye(4), atol=1e-3) and np.array_equal(out["mask"], box_mask())
    nothing = json.loads(
        str(np.load(io.BytesIO(worker._locate(live, Sam(), _GridTier(), intr, [10, 10], ref)))["meta"])
    )
    assert not nothing["ok"] and "under the click" in nothing["reason"]
    unclicked = json.loads(
        str(np.load(io.BytesIO(worker._locate(live, Sam(), _GridTier(), intr, None, ref)))["meta"])
    )
    assert not unclicked["ok"] and "click" in unclicked["reason"]

    refreshed = np.load(
        io.BytesIO(worker._locate(live, Sam(), _GridTier(), intr, [360, 240], {**ref, "refresh_mask": True}))
    )
    assert json.loads(str(refreshed["meta"]))["ref_ok"]
    np.testing.assert_array_equal(refreshed["ref_mask"], box_mask())


def _two_object_demo(tmp_path, t0):
    """The bound demo with a second object, "box", designated on its stream: what a place goes onto."""
    demo = _demo_with_object(tmp_path, t0)
    target = np.zeros((H, W), dtype=bool)
    target[300:360, 500:600] = True
    n = len(_stream_times_of(demo))
    deltas = np.tile(np.eye(4), (n, 1, 1))
    for f in range(n):  # the box crept 1 mm per frame along y in the demo; hidden on frame 9
        deltas[f][:3, 3] = [0.0, f / 1000.0, 0.0]
    seen = np.ones(n, dtype=bool)
    seen[9] = False
    demo.objects["box"] = {
        "frame": 1,
        "click": [550, 330],
        "status": "done",
        "deltas": deltas,
        "seen": seen,
        "masks": np.zeros((n, H // 4, W // 4), dtype=bool),
        "mask": target,
    }
    return demo, target


def _stream_times_of(demo):
    return manipulation.recording._stream_times(demo.recording)


@pytest.mark.usefixtures("act_simulation")
def test_a_place_follows_its_own_object_found_and_tracked_apart_from_the_picked_one(
    client, tmp_path, monkeypatch
):
    """The pre-grasps and the grasp follow the object picked; the pre-places and the place follow the one it goes
    onto. That object is found by a locate against the demo's view of it, which leaves the picked object's teach alone
    and has the box join the live session that follows the gamepad; the place moves by what that find says, against
    where the demo had the object at the first pre-place."""
    import asyncio

    from scipy.spatial.transform import Rotation

    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path / "demos")
    monkeypatch.setattr(
        manipulation.runtime._state, "groups_with_acts", False
    )  # held poses: no point groups view here
    demo, target = _two_object_demo(tmp_path, time.time())
    live_rgb = np.full((H, W, 3), 200, np.uint8)
    monkeypatch.setattr(
        manipulation.camera,
        "_frame",
        lambda: _async((live_rgb, np.full((H, W), 0.45, np.float32), dict(INTR))),
    )
    manipulation.runtime._state.worker.proc = _FakeProc()
    try:
        with manipulation.runtime._state.lock:
            (
                manipulation.runtime._state.demo,
                manipulation.runtime._state.teach,
                manipulation.runtime._state.test,
            ) = demo, None, None
            manipulation.runtime._state.located.clear()
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()
        post = lambda kps: client.post("/api/pregrasp/demo/keypoints", json={"keypoints": kps})  # noqa: E731
        grasp = [
            {"t": 0.2, "kind": "pregrasp", "object": "gamepad"},
            {"t": 0.4, "kind": "grasp_end", "object": "gamepad"},
        ]
        r = post([*grasp, {"t": 0.6, "kind": "preplace", "object": "gamepad"}])
        assert r.status_code == 422 and "another object" in r.text
        r = post([*grasp, {"t": 0.6, "kind": "preplace", "object": "mug"}])
        assert r.status_code == 422 and "not a tracked object" in r.text
        marks = [
            *grasp,
            {"t": 0.6, "kind": "preplace", "object": "box"},
            {"t": 0.8, "kind": "place_end", "object": "box"},
        ]
        r = post(marks)
        assert r.status_code == 200, r.text
        assert (
            r.json()["act_objects"] == ["gamepad", "box"]
            and manipulation.planning._session_object(demo) == "gamepad"
        )
        assert manipulation.planning._object_motion(demo, "box", np.eye(4)) == (
            None,
            "click box in the camera view to find it",
        )

        moved = np.eye(4)
        moved[:3, :3] = Rotation.from_euler("z", 15, degrees=True).as_matrix()
        moved[:3, 3] = [0.04, -0.02, 0.0]

        async def answer(inliers: int) -> manipulation.runtime._Job:
            for _ in range(500):
                await asyncio.sleep(0.002)
                with manipulation.runtime._state.lock:
                    pending = list(manipulation.runtime._state.worker.pending)
                    manipulation.runtime._state.worker.pending.clear()  # taken, as the worker's poll takes them
                if pending:
                    break
            job = manipulation.runtime._state.worker.jobs[pending[0]]
            job.result = {
                "ok": True,
                "ref_ok": True,
                "ref_inliers": inliers,
                "ref_card_points": 400,
                "ref_turn_deg": 15.0,
                "ref_delta": moved,
                "mask": target,
            }
            return job

        async def click(inliers: int):
            worker = asyncio.create_task(answer(inliers))
            info = await manipulation.acquisition.locate(
                manipulation.acquisition.LocateBody(click=[550, 330], object="box")
            )
            return await worker, info

        job, info = asyncio.run(click(160))
        assert job.kind == "locate" and job.concept == "box" and job.click == [550, 330]
        assert job.extra == {
            "ref_recording": demo.recording,
            "ref_frame": 1,
            "ref_object": "box",
            "ref_symmetry": {"kind": "none", "axis": "surface_normal", "order": 1},
            "track": True,
            "scene": ["gamepad"],
        }, "the box joins the live session beside the gamepad"
        assert np.array_equal(job.arrays["ref_mask"], target), "the box's own view in the demo"
        assert info["ok"] and info["strong"] and info["inliers"] == 160
        assert manipulation.runtime._state.teach is None and manipulation.runtime._state.teach_job is None, (
            "the picked object's teach and track untouched"
        )
        # Placed onto where the box is now: the find, against where the demo had it on its pose frame, the frame it
        # was clicked on (1), which comes before the place.
        motion, problem = manipulation.planning._object_motion(demo, "box", np.eye(4))
        assert problem == "" and np.allclose(motion, moved @ np.linalg.inv(demo.objects["box"]["deltas"][1]))
        assert client.get("/api/pregrasp/state").json()["located"]["box"]["strong"] is True
        # With the point groups, the find is moved on by how the box moved since the find's frame was read.
        pushed = np.eye(4)
        pushed[:3, 3] = [0.0, 0.03, 0.0]
        asked = []
        motion, problem = manipulation.planning._object_motion(
            demo, "box", np.eye(4), carry=lambda s: asked.append(s) or pushed
        )
        assert asked == [manipulation.runtime._state.located["box"]["stamp"]], "carried from the find's frame"
        assert np.allclose(motion, pushed @ moved @ np.linalg.inv(demo.objects["box"]["deltas"][1]))

        asyncio.run(click(32))
        motion, problem = manipulation.planning._object_motion(demo, "box", np.eye(4))
        assert motion is None and problem.startswith(
            "a weak find: box matched 32 of the demo view's 400 points"
        )
    finally:
        manipulation.runtime._state.worker.proc = None
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
                manipulation.runtime._state.test
            ) = None
            manipulation.runtime._state.located.clear()
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()


class _TipKinematics:
    """A fake arm: the tip is the first three joints in millimetres, never turned; the IK closes half the gap per call."""

    def forward_kinematics(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3], dtype=float) / 1000.0
        return pose

    def inverse_kinematics(self, seed, pose):
        q = np.asarray(seed, dtype=float).copy()
        q[:3] += 0.5 * (np.asarray(pose[:3, 3]) * 1000.0 - q[:3])
        return q


class _TipYawKinematics(_TipKinematics):
    """The fake arm with a wrist: joint 4 (wrist_flex) turns the tip about the vertical, in degrees; the IK closes half
    its gap per call, the shorter way round."""

    def forward_kinematics(self, q):
        pose = super().forward_kinematics(q)
        a = np.radians(float(q[4]))
        pose[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        return pose

    def inverse_kinematics(self, seed, pose):
        q = super().inverse_kinematics(seed, pose)
        want = np.degrees(np.arctan2(pose[1, 0], pose[0, 0]))
        q[4] += 0.5 * ((want - q[4] + 180.0) % 360.0 - 180.0)
        return q


# The place demo's samples: the last pre-grasp, the grip becoming firm, the lift, the grasp end, the last pre-place,
# the release and the place end.
PLACE_AT = {
    "pregrasp": 10,
    "grip": 28,
    "lift": 45,
    "grasp_end": 52,
    "preplace": 75,
    "release": 88,
    "place_end": 95,
}


def _place_demo(tmp_path, t0, wrist_deg=0.0):
    """A demo at 30 Hz with its stream at 15 Hz: the arm comes down by 0.83 s and stands there while the gripper
    closes on the gamepad, firm at 0.93 s; it lifts at 1.5 s, carries the gamepad above the box, holds it still there
    (1.93 to 2.6 s), sets it down and lets go at 2.93 s. The fake arm's tip is its first three joints in millimetres;
    the gamepad stops the fingers 3 units short of the closing command. With ``wrist_deg`` the arm has a wrist
    (:class:`_TipYawKinematics`), turned that far during the carry and kept through the place."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi = MOTOR_NAMES.index("gripper")
    n = 100
    s = np.arange(n)
    q = np.zeros((n, 7))
    q[:, 0] = np.interp(s, [0, 25, 52, 58, 99], [100, 120, 120, 180, 180])
    q[:, 2] = np.interp(s, [0, 25, 45, 52, 78, 86, 99], [60, 20, 20, 60, 60, 25, 25])
    q[:, 4] = np.interp(s, [0, PLACE_AT["grasp_end"], PLACE_AT["preplace"], 99], [0, 0, wrist_deg, wrist_deg])
    gripping = (s >= PLACE_AT["grip"]) & (s < PLACE_AT["release"])
    q_cmd = q.copy()
    q_cmd[:, gi] = np.where(gripping, 85.0, 60.0)
    q[:, gi] = np.where(gripping, 82.0, 60.0)
    kin = _TipYawKinematics() if wrist_deg else _TipKinematics()
    rec = write_stream(tmp_path / "demos" / ".recordings" / "place", 50, t0=t0, hz=15.0)
    demo = manipulation.runtime._Demo(
        name="place",
        concept="demo",
        fps=30.0,
        t=s / 30.0,
        tips=np.stack([kin.forward_kinematics(x) for x in q]),
        grippers=q[:, gi],
        q_obs=q,
        q_cmd=q_cmd,
        deltas=np.tile(np.eye(4), (n, 1, 1)),
        seen=np.zeros(n, dtype=bool),
        delta0=np.eye(4),
        t0=t0,
        intr=dict(INTR),
        recording=str(rec),
    )
    box = np.zeros((H, W), dtype=bool)
    box[330:380, 560:660] = True
    for name, mask in (("gamepad", box_mask()), ("box", box)):
        demo.objects[name] = {
            "frame": 2,
            "click": [360, 240] if name == "gamepad" else [600, 350],
            "status": "done",
            "deltas": np.tile(np.eye(4), (50, 1, 1)),
            "seen": np.ones(50, dtype=bool),
            "masks": np.tile(mask[::4, ::4], (50, 1, 1)),
            "mask": mask,
        }
    demo.keypoints = [
        {"t": float(demo.t[PLACE_AT["pregrasp"]]), "kind": "pregrasp", "object": "gamepad"},
        {"t": float(demo.t[PLACE_AT["grasp_end"]]), "kind": "grasp_end", "object": "gamepad"},
        {"t": float(demo.t[PLACE_AT["preplace"]]), "kind": "preplace", "object": "box"},
        {"t": float(demo.t[PLACE_AT["place_end"]]), "kind": "place_end", "object": "box"},
    ]
    return demo, kin, box


def test_the_live_path_preview_is_planned_again_only_when_a_key_frame_moves_visibly(tmp_path, monkeypatch):
    """The live view drew the act's path by planning it on every rendered frame (44-51 ms each through the acts of
    2026-10-16, beside the motor loop). It keeps drawing the last path until a key frame's target has moved further
    than the line shows."""
    demo, _, _ = _place_demo(tmp_path, time.time())
    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=None,
        depth_m=None,
        intr=INTR,
        keypoints={"ref": {"object": "gamepad", "ok": True, "delta": np.eye(4)}},
    )
    planned = []
    real_plan = core.plan_key_frames

    def counting(*args, **kwargs):
        planned.append(1)
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(core, "plan_key_frames", counting)
    manipulation.planning._preview_last.clear()

    def moved(x_m):
        delta = np.eye(4)
        delta[0, 3] = x_m
        return delta

    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.teach = teach
        first = manipulation.planning._act_preview(demo, np.eye(4), np.eye(4))
        assert first is not None and len(planned) == 1
        assert manipulation.planning._act_preview(demo, moved(0.001), np.eye(4)) is first, (
            "1 mm: the same line"
        )
        assert len(planned) == 1
        again = manipulation.planning._act_preview(demo, moved(0.005), np.eye(4))
        assert len(planned) == 2 and again is not first
        np.testing.assert_allclose(again[0, :3, 3] - first[0, :3, 3], [0.005, 0.0, 0.0], atol=1e-9)
        manipulation.planning._act_preview(demo, moved(0.005), np.eye(4), preview_pose=np.eye(4))
        assert len(planned) == 3, "a review's candidate is planned as asked"
    finally:
        manipulation.planning._preview_last.clear()
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = None


def test_a_world_waypoint_is_saved_with_its_frame(tmp_path, client):
    """The marks are rebuilt from what each one says; a waypoint's frame is part of it, kept, or saving it is refused."""
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        lift = {"t": float(demo.t[-1]), "kind": "waypoint", "frame": "world"}
        r = client.post("/api/pregrasp/demo/keypoints", json={"keypoints": [*demo.keypoints, lift]})
        assert r.status_code == 200, r.text
        assert r.json()["keypoints"][-1] == lift and demo.keypoints[-1] == lift
        r = client.post(
            "/api/pregrasp/demo/keypoints",
            json={"keypoints": [*demo.keypoints[:-1], {**lift, "frame": None}]},
        )
        assert r.status_code == 422 and "world frame" in r.json()["detail"]
    finally:
        _end_place_state()


def _end_place_state():
    with manipulation.runtime._state.lock:
        manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
            manipulation.runtime._state.test
        ) = None
        manipulation.runtime._state.located.clear()
        manipulation.runtime._state.targets.clear()
        manipulation.runtime._state.track.on = False
        manipulation.runtime._state.track.history = []
        manipulation.runtime._state.track.last = {}
        manipulation.runtime._state.worker.pending.clear()
        manipulation.runtime._state.worker.jobs.clear()
        manipulation.runtime._state.act = manipulation.runtime._Act()


def test_the_editors_check_says_what_the_demo_made_holding_and_refuses_a_key_frame_on_the_held_object(
    client, tmp_path
):
    """The editor shows unsaved key frames as the server reads them: which the demo made holding something (from the
    firm grip in the grasp's follow to where the gripper let go), the object a landing turns about, and why a list
    cannot be saved. A key frame held in the frame of the object the gripper holds then is refused, there and on Save:
    the gripper carries that object."""
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    t = demo.t
    keys = [
        {
            "t": float(t[PLACE_AT["pregrasp"]]),
            "kind": "key",
            "frame": "gamepad",
            "move": "line",
            "grip": "first",
        },
        {
            "t": float(t[PLACE_AT["grasp_end"]]),
            "kind": "key",
            "frame": "gamepad",
            "move": "follow",
            "grip": "first",
        },
        {"t": float(t[PLACE_AT["preplace"]]), "kind": "key", "frame": "box", "move": "line", "grip": "first"},
        {
            "t": float(t[PLACE_AT["place_end"]]),
            "kind": "key",
            "frame": "box",
            "move": "follow",
            "grip": "first",
        },
    ]
    check = lambda kps: client.post("/api/pregrasp/demo/key_frames/check", json={"keypoints": kps}).json()  # noqa: E731
    post = lambda kps: client.post("/api/pregrasp/demo/keypoints", json={"keypoints": kps})  # noqa: E731
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        r = check(keys)
        assert r["problem"] == "" and [k["held"] for k in r["key_frames"]] == [False, False, True, True]
        assert r["hold"] == {
            "object": "gamepad",
            "grip_t": pytest.approx(float(t[PLACE_AT["grip"]])),
            "release_t": pytest.approx(float(t[PLACE_AT["release"]])),
        }
        assert r["landing_object"] == "box"
        old = check(demo.keypoints)  # the marks saved before the key frames make the same ones
        assert [(k["t"], k["frame"], k["move"], k["grip"]) for k in old["key_frames"]] == [
            (k["t"], k["frame"], k["move"], k["grip"]) for k in keys
        ]
        on_held = [*keys[:2], *({**k, "frame": "gamepad"} for k in keys[2:])]
        r = check(on_held)
        assert "while the gripper holds gamepad" in r["problem"], r["problem"]
        assert post(on_held).status_code == 422 and "while the gripper holds gamepad" in post(on_held).text
        assert "frame of the key frame before" in check([keys[0], {**keys[1], "frame": "box"}])["problem"]
        r = post(keys)
        assert r.status_code == 200 and r.json()["act_objects"] == ["gamepad", "box"]
        assert demo.keypoints == keys, "kept as the key frames they are"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None


def test_a_lift_is_checked_and_saved_and_the_removed_runner_is_rejected(client, tmp_path):
    """A lift after the grasp: made holding, labelled, its height kept through the check and the Save; a follow right
    after it is refused, as it would replay the demo from where its arm was. The removed runner cannot be selected through the API."""
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    t = demo.t
    keys = [
        {
            "t": float(t[PLACE_AT["pregrasp"]]),
            "kind": "key",
            "frame": "gamepad",
            "move": "line",
            "grip": "first",
        },
        {
            "t": float(t[PLACE_AT["grasp_end"]]),
            "kind": "key",
            "frame": "gamepad",
            "move": "follow",
            "grip": "first",
        },
        {"t": float(t[60]), "kind": "key", "frame": "world", "move": "lift", "grip": "first", "up_mm": 45.0},
        {"t": float(t[PLACE_AT["preplace"]]), "kind": "key", "frame": "box", "move": "line", "grip": "first"},
        {
            "t": float(t[PLACE_AT["place_end"]]),
            "kind": "key",
            "frame": "box",
            "move": "follow",
            "grip": "first",
        },
    ]
    check = lambda kps: client.post("/api/pregrasp/demo/key_frames/check", json={"keypoints": kps}).json()  # noqa: E731
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        r = check(keys)
        assert r["problem"] == ""
        lift = r["key_frames"][2]
        assert (lift["label"], lift["move"], lift["up_mm"], lift["held"]) == ("lift 1", "lift", 45.0, True)
        assert "straight line between them" in check([*keys[:3], {**keys[4], "frame": "world"}])["problem"]
        assert client.post("/api/pregrasp/demo/keypoints", json={"keypoints": keys}).status_code == 200
        assert demo.keypoints[2]["up_mm"] == 45.0, "the height is kept with the demo"
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach = manipulation.runtime._Teach(
                at="t", box=(0, 0, 0, 0), rgb=None, depth_m=None, intr={}, keypoints={}
            )
        r = client.post("/api/pregrasp/act", json={"runner": "marks"})
        assert r.status_code == 422 and "runner" in r.text
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = None


def test_a_locate_against_another_demo_or_view_does_not_move_the_place(client, tmp_path, monkeypatch):
    """A locate's motion is from the view it matched; kept by object name alone, one made against the previous demo
    moved this demo's place."""
    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path / "demos")
    demo, target = _two_object_demo(tmp_path, time.time())
    demo.keypoints = [
        {"t": 0.2, "kind": "pregrasp", "object": "gamepad"},
        {"t": 0.4, "kind": "grasp_end", "object": "gamepad"},
        {"t": 0.6, "kind": "preplace", "object": "box"},
    ]
    found = {
        "object": "box",
        "ok": True,
        "delta": np.eye(4),
        "inliers": 160,
        "card_points": 400,
        "mask": target,
    }
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.located["box"] = {**found, "view": ["an earlier demo", 1]}
        assert manipulation.planning._object_motion(demo, "box", np.eye(4)) == (
            None,
            "click box in the camera view to find it",
        )
        assert client.get("/api/pregrasp/state").json()["located"] == {}
        manipulation.runtime._state.located["box"] = {
            **found,
            "view": [demo.name, 4],
        }  # the box designated again, elsewhere
        assert manipulation.planning._object_motion(demo, "box", np.eye(4))[0] is None
        manipulation.runtime._state.located["box"] = {**found, "view": [demo.name, 1]}
        assert manipulation.planning._object_motion(demo, "box", np.eye(4))[1] == ""
        assert set(client.get("/api/pregrasp/state").json()["located"]) == {"box"}
        r = client.post("/api/pregrasp/locate", json={"click": [10, 10], "object": "mug"})
        assert r.status_code == 409 and "mug" not in manipulation.runtime._state.located, (
            "nothing kept for an unknown object"
        )
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.located.clear()


def test_an_objects_pose_is_read_where_it_was_set_else_where_it_was_clicked_else_where_last_seen(
    client, tmp_path
):
    """The pose frame was the last frame seen by the first pre-grasp, so a clear view of the object meant a waypoint
    the arm had to visit: the cube a place goes onto is in full view only while the arm is far from it. Each object's
    pose is now read on a frame of its own: one the operator sets, else the frame it was clicked on, where its view
    is the find's own, as long as that comes before its stage replays; else the old rule."""
    demo = _demo_with_object(
        tmp_path, time.time()
    )  # the gamepad clicked on stream frame 2; hidden on 5 and 6
    times = manipulation.recording._stream_times(demo.recording) - demo.t0
    grasp = [
        {"t": 0.4, "kind": "pregrasp", "object": "gamepad"},
        {"t": 0.7, "kind": "grasp_end", "object": "gamepad"},
    ]
    demo.keypoints = list(grasp)
    assert manipulation.planning._pose_choice(demo, "gamepad") == (2, "clicked")
    demo.keypoints = [*grasp, {"t": float(times[8]), "kind": "pose", "object": "gamepad"}]
    assert manipulation.planning._pose_choice(demo, "gamepad") == (8, "set")
    demo.objects["gamepad"]["frame"] = (
        9  # clicked after the last pre-grasp, at 0.6 s: the arm may have moved it
    )
    demo.keypoints = [{"t": 0.3, "kind": "pregrasp", "object": "gamepad"}, grasp[1]]
    assert manipulation.planning._pose_choice(demo, "gamepad") == (4, "last seen by its first key frame")
    demo.objects["gamepad"]["frame"] = 2
    demo.keypoints = []
    assert manipulation.planning._pose_choice(demo, "gamepad") is None, "no stage yet"
    with manipulation.runtime._state.lock:
        manipulation.runtime._state.demo = demo
    try:
        listed = client.get("/api/pregrasp/demo/objects").json()["objects"][0]
        assert listed["pose_t"] == pytest.approx(times[2]) and listed["pose_from"] == "clicked"
        post = lambda kps: client.post("/api/pregrasp/demo/keypoints", json={"keypoints": kps})  # noqa: E731
        r = post([*grasp, {"t": float(times[5]), "kind": "pose", "object": "gamepad"}])
        assert r.status_code == 422 and "hidden" in r.text, "a pose is read where the object can be seen"
        r = post([*grasp, {"t": float(times[7]), "kind": "pose", "object": "gamepad"}])
        assert r.status_code == 422 and "no later than its last pre-grasp" in r.text
        r = post([*grasp, {"t": float(times[3]), "kind": "pose", "object": "gamepad"}])
        assert r.status_code == 200, r.text
        listed = client.get("/api/pregrasp/demo/objects").json()["objects"][0]
        assert listed["pose_t"] == pytest.approx(times[3]) and listed["pose_from"] == "set"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None


def test_the_place_object_is_drawn_where_its_find_put_it(tmp_path):
    """The box a place goes onto is not tracked: the live view draws it where its last find put it, its surface from
    the demo's view carried by the find's motion, so a find that missed shows against the real box."""
    import cv2

    demo, _kin, box = _place_demo(tmp_path, time.time())
    points = manipulation.tracking._view_points(demo, "box")
    assert manipulation.tracking._view_points(demo, "box") is points, "read once"
    spans = []
    for shift in ([0.0, 0.0, 0.0], [0.03, 0.0, 0.0]):
        delta = np.eye(4)
        delta[:3, 3] = shift
        mask = manipulation.rendering._found_mask(dict(INTR), points, delta, (H, W))
        uv = manipulation.rendering._project_cam(dict(INTR), points + shift)
        ys, xs = np.nonzero(mask)
        assert abs(xs.min() - uv[:, 0].min()) <= 4 and abs(xs.max() - uv[:, 0].max()) <= 4
        assert abs(ys.min() - uv[:, 1].min()) <= 4 and abs(ys.max() - uv[:, 1].max()) <= 4
        spans.append(xs.mean())
    z = float(points[:, 2].mean())
    assert spans[1] - spans[0] == pytest.approx(INTR["fx"] * 0.03 / z, abs=2.0), "moved with the find"
    if np.allclose(points[:, 2], z):  # a flat view at one depth: the outline is the box's own mask
        assert np.mean(manipulation.rendering._found_mask(dict(INTR), points, np.eye(4), (H, W))[box]) > 0.95

    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=np.zeros((H, W, 3), np.uint8),
        depth_m=np.full((H, W), 0.45, np.float32),
        intr=dict(INTR),
        keypoints={"mode": "features", "concept": "gamepad"},
    )
    found = {"object": "box", "ok": True, "delta": np.eye(4), "inliers": 150, "card_points": 400}
    found["view"] = [demo.name, int(demo.objects["box"]["frame"])]

    def orange(jpeg: bytes) -> np.ndarray:
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        b, g, r = (bgr[..., k].astype(int) for k in range(3))
        return (r > 180) & (g > 90) & (g < 190) & (b < 90)

    status = {"state": "tracking", "algo": "p2p"}
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo, manipulation.runtime._state.located = demo, {"box": found}
        drawn = orange(
            manipulation.rendering._render_live(np.zeros((H, W, 3), np.uint8), {}, None, None, teach, status)
        )
        edge = cv2.dilate(box.astype(np.uint8), np.ones((5, 5), np.uint8)) & ~cv2.erode(
            box.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
        assert drawn[edge.astype(bool)].mean() > 0.3, "the box outlined where it was found"
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.located = {}
        assert not orange(
            manipulation.rendering._render_live(np.zeros((H, W, 3), np.uint8), {}, None, None, teach, status)
        )[edge.astype(bool)].any(), "nothing drawn without a find"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.located = {}


def test_the_place_object_moves_with_its_trusted_share_of_the_session_and_only_then(tmp_path):
    """The held object's finds in the gripper join no session; the place object's find that joined it moves with its
    share of each tracked frame, but not while Point2Pose has it lost or too few of the tracks it began with are seen;
    a newer find replaces the one followed."""
    demo, _kin, box = _place_demo(tmp_path, time.time())
    anchor = np.eye(4)
    anchor[:3, 3] = [0.03, 0.01, 0.0]
    found = {"object": "box", "ok": True, "delta": anchor.copy(), "mask": box, "tracking": True}
    found["view"] = [demo.name, int(demo.objects["box"]["frame"])]
    moved = np.eye(4)
    moved[:3, 3] = [0.0, 0.02, 0.0]

    def share(**kw):
        return {"others": [{"name": "box", "ok": True, "lost": False, "n_visible": 50, "n_tracks": 50, **kw}]}

    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        manipulation.acquisition._store_located("box", {**found, "tracking": False})
        assert "box" not in manipulation.runtime._state.targets, "not followed without joining the session"
        manipulation.acquisition._store_located("box", dict(found))
        assert "box" in manipulation.runtime._state.targets
        manipulation.tracking._apply_others({**share(lost=True), "other_delta_0": moved}, (H, W))
        assert np.allclose(manipulation.runtime._state.located["box"]["delta"], anchor), "lost: not moved"
        manipulation.tracking._apply_others({**share(n_visible=3), "other_delta_0": moved}, (H, W))
        assert np.allclose(manipulation.runtime._state.located["box"]["delta"], anchor), (
            "too few tracks seen: not moved"
        )
        assert manipulation.runtime._state.targets["box"].last["state"] == "untrusted"
        manipulation.tracking._apply_others({**share(), "other_delta_0": moved}, (H, W))
        assert np.allclose(manipulation.runtime._state.located["box"]["delta"], moved @ anchor), (
            "moved with its share"
        )
        assert (
            manipulation.acquisition._located_info(manipulation.runtime._state.located["box"])["track"]
            == "tracking"
        )
        manipulation.acquisition._store_located("box", {**found, "delta": np.eye(4)})
        manipulation.tracking._apply_others({"others": [], "other_delta_0": moved}, (H, W))
        assert np.allclose(manipulation.runtime._state.located["box"]["delta"], np.eye(4)), (
            "the newer find, not in this frame"
        )
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.located = {}
            manipulation.runtime._state.targets = {}


def test_every_object_a_key_frame_is_held_in_is_followed_by_its_own_share_and_drawn(tmp_path):
    """Key frames in three objects' frames and the world's: the first object's find starts the live session and the
    other two join it. Each is moved by its own share of a tracked frame and by nothing else's, each has its own
    motion since the demo, and the live view draws both where the act would aim."""
    import cv2

    demo, box = _two_object_demo(tmp_path, time.time())
    mug = np.zeros((H, W), dtype=bool)
    mug[100:150, 150:250] = True
    n = len(_stream_times_of(demo))
    demo.objects["mug"] = {
        **demo.objects["box"],
        "frame": 1,
        "mask": mug,
        "deltas": np.tile(np.eye(4), (n, 1, 1)),
    }
    demo.keypoints = [
        {"t": 0.10, "kind": "key", "frame": "gamepad", "move": "line", "grip": "first"},
        {"t": 0.20, "kind": "key", "frame": "box", "move": "line", "grip": "first"},
        {"t": 0.30, "kind": "key", "frame": "box", "move": "follow", "grip": "first"},
        {"t": 0.40, "kind": "key", "frame": "mug", "move": "line", "grip": "first"},
        {"t": 0.50, "kind": "key", "frame": "world", "move": "line", "grip": "along"},
    ]
    assert core.keypoints_problem(demo.keypoints, 0.0, float(demo.t[-1])) == ""
    assert manipulation.planning._act_objects(demo) == ["gamepad", "box", "mug"]
    assert manipulation.planning._session_object(demo) == "gamepad" and manipulation.planning._other_objects(
        demo
    ) == ["box", "mug"]
    assert manipulation.acquisition._session_names(demo, but="gamepad") == ["box", "mug"], (
        "both join the session's start"
    )
    anchors = {name: np.eye(4) for name in ("box", "mug")}
    anchors["box"][:3, 3], anchors["mug"][:3, 3] = [0.03, 0.01, 0.0], [-0.02, 0.0, 0.0]
    moved = {name: np.eye(4) for name in ("box", "mug")}
    moved["box"][:3, 3], moved["mug"][:3, 3] = [0.0, 0.02, 0.0], [0.01, 0.0, 0.0]

    def share(name, **kw):
        return {"name": name, "ok": True, "lost": False, "n_visible": 50, "n_tracks": 50, **kw}

    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=np.zeros((H, W, 3), np.uint8),
        depth_m=np.full((H, W), 0.45, np.float32),
        intr=dict(INTR),
        keypoints={"mode": "features", "concept": "gamepad"},
    )
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        for name, mask in (("box", box), ("mug", mug)):
            view = [demo.name, int(demo.objects[name]["frame"])]
            found = {"object": name, "ok": True, "delta": anchors[name].copy(), "mask": mask, "view": view}
            manipulation.acquisition._store_located(name, {**found, "tracking": True})
        assert set(manipulation.runtime._state.targets) == {"box", "mug"}
        both = {
            "others": [share("box"), share("mug")],
            "other_delta_0": moved["box"],
            "other_delta_1": moved["mug"],
        }
        manipulation.tracking._apply_others(both, (H, W))
        for name in ("box", "mug"):
            assert np.allclose(
                manipulation.runtime._state.located[name]["delta"], moved[name] @ anchors[name]
            ), name
        mug_lost = {"others": [share("mug", lost=True), share("box")], "other_delta_0": np.eye(4)}
        mug_lost["other_delta_1"] = moved["box"] @ moved["box"]
        manipulation.tracking._apply_others(mug_lost, (H, W))
        assert np.allclose(
            manipulation.runtime._state.located["box"]["delta"], moved["box"] @ moved["box"] @ anchors["box"]
        )
        assert np.allclose(
            manipulation.runtime._state.located["mug"]["delta"], moved["mug"] @ anchors["mug"]
        ), "lost: held"
        assert manipulation.runtime._state.targets["mug"].last == {"state": "untrusted", "reason": "lost"}
        motions = manipulation.planning._found_motions(demo, np.eye(4))
        for name in ("box", "mug"):
            f = manipulation.planning._pose_frame(demo, name)
            expect = manipulation.runtime._state.located[name]["delta"] @ np.linalg.inv(
                demo.objects[name]["deltas"][f]
            )
            assert np.allclose(motions[name], expect), name

        def orange(jpeg: bytes) -> np.ndarray:
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            b, g, r = (bgr[..., k].astype(int) for k in range(3))
            return (r > 180) & (g > 90) & (g < 190) & (b < 90)

        for name in ("box", "mug"):  # each drawn where its find put it
            manipulation.runtime._state.located[name]["delta"] = np.eye(4)
        drawn = orange(
            manipulation.rendering._render_live(
                np.zeros((H, W, 3), np.uint8), {}, None, None, teach, {"state": "tracking"}
            )
        )
        for name, mask in (("box", box), ("mug", mug)):
            m = mask.astype(np.uint8)
            edge = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool) & ~cv2.erode(
                m, np.ones((5, 5), np.uint8)
            ).astype(bool)
            assert drawn[edge].mean() > 0.3, f"{name} outlined where it was found"
        manipulation.acquisition._store_located(
            "mug", {"object": "mug", "ok": False, "delta": None, "mask": None}
        )
        assert set(manipulation.runtime._state.targets) == {"box"}, (
            "a failed find of one stops its own track only"
        )
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.located = {}
            manipulation.runtime._state.targets = {}


def test_key_frame_marks_are_checked_as_key_frames():
    p = core.keypoints_problem
    line = {"t": 1.0, "kind": "key", "frame": "box", "move": "line", "grip": "first"}
    follow = {"t": 2.0, "kind": "key", "frame": "box", "move": "follow", "grip": "first"}
    assert p([line, follow], 0.0, 10.0) == ""
    assert "one way" in p([line, {"t": 3.0, "kind": "pregrasp", "object": "box"}], 0.0, 10.0)
    assert "names its frame" in p([{**line, "frame": None}], 0.0, 10.0)
    assert "move is one of" in p([{**line, "move": "jump"}], 0.0, 10.0)
    assert "frame of the key frame before" in p([line, {**follow, "frame": "mug"}], 0.0, 10.0)
    assert p([{"t": 0.5, "kind": "pose", "object": "box"}, line, follow], 0.0, 10.0) == ""
    assert "before the arm can have moved it" in p(
        [{"t": 1.5, "kind": "pose", "object": "box"}, line, follow], 0.0, 10.0
    )


def test_the_place_object_is_followed_only_while_enough_of_it_is_seen(client, tmp_path):
    """Act 2026-10-07 22:21: the gripper came over the cube and its track, still trusted with 49 of its 100 points
    seen, jumped 163 mm. A frame moves the place object only while at least the trust share of its tracks is seen;
    the share is a run-time option the page's slider sets."""
    demo, _kin, box = _place_demo(tmp_path, time.time())
    anchor = np.eye(4)
    anchor[:3, 3] = [0.03, 0.01, 0.0]
    found = {"object": "box", "ok": True, "delta": anchor.copy(), "mask": box, "tracking": True}
    found["view"] = [demo.name, int(demo.objects["box"]["frame"])]
    moved = np.eye(4)
    moved[:3, 3] = [0.0, 0.02, 0.0]
    covered = {"others": [{"name": "box", "ok": True, "lost": False, "n_visible": 40, "n_tracks": 50}]}
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        manipulation.acquisition._store_located("box", dict(found))
        assert (
            client.get("/api/pregrasp/state").json()["trust_share"]
            == manipulation.runtime.TRUST_SHARE_DEFAULT
        )
        manipulation.tracking._apply_others({**covered, "other_delta_0": moved}, (H, W))
        assert np.allclose(manipulation.runtime._state.located["box"]["delta"], anchor), (
            "80% seen: stays where it was"
        )
        assert manipulation.runtime._state.targets["box"].last["state"] == "untrusted"
        assert "80%" in manipulation.runtime._state.targets["box"].last["reason"]
        r = client.post("/api/pregrasp/options", json={"trust_share": 0.75})
        assert r.status_code == 200 and r.json()["trust_share"] == 0.75
        assert client.get("/api/pregrasp/state").json()["flat"] is False, "an option not given is left alone"
        manipulation.tracking._apply_others({**covered, "other_delta_0": moved}, (H, W))
        assert np.allclose(manipulation.runtime._state.located["box"]["delta"], moved @ anchor), (
            "75% asked: followed"
        )
        assert client.post("/api/pregrasp/options", json={"trust_share": 1.5}).status_code == 422
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.located = {}
            manipulation.runtime._state.targets = {}
            manipulation.runtime._state.trust_share = manipulation.runtime.TRUST_SHARE_DEFAULT


def test_an_objects_symmetry_is_kept_with_the_demo_and_sent_with_its_finds(client, tmp_path, monkeypatch):
    """The gamepad reads the same turned end to end (order 2), the cube turned a quarter (order 4). Declared on the
    demo's objects, the order is written beside the demo, read back with it, and sent with each find against the
    demo's view, where the worker folds the turn by it; a demo saved before has none."""
    import asyncio

    demo, _kin, box = _place_demo(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "place")
    pathlib.Path(demo.root).mkdir(parents=True)
    queued = []

    def fake_queue(kind, concept, rgb, depth_m, intr, **kw):
        job = manipulation.runtime._Job(
            id="j", kind=kind, concept=concept, rgb=rgb, depth_m=depth_m, intr=intr, created=0.0
        )
        job.result = {"ok": False, "reason": "not looked for"}
        queued.append(job)
        return job

    monkeypatch.setattr(manipulation.worker, "_queue_job", fake_queue)
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.worker.proc = _FakeProc()
        assert next(
            o for o in client.get("/api/pregrasp/demo/objects").json()["objects"] if o["name"] == "box"
        )["symmetry"] == {"kind": "none", "axis": "surface_normal", "order": 1}, "none until declared"
        r = client.post("/api/pregrasp/demo/objects/symmetry", json={"name": "box", "order": 4})
        assert r.status_code == 200
        face_four = {"kind": "discrete", "axis": "surface_normal", "order": 4}
        assert next(o for o in r.json()["objects"] if o["name"] == "box")["symmetry"] == face_four
        assert manipulation.demos._read_objects(pathlib.Path(demo.root))["box"]["symmetry"] == face_four, (
            "kept beside the demo"
        )
        assert (
            manipulation.demos._read_objects(pathlib.Path(demo.root))["gamepad"]["symmetry"]["kind"] == "none"
        )
        r = client.post(
            "/api/pregrasp/demo/objects/symmetry",
            json={"name": "box", "kind": "continuous", "axis": "long_axis"},
        )
        assert r.status_code == 200
        assert manipulation.demos._read_objects(pathlib.Path(demo.root))["box"]["symmetry"] == {
            "kind": "continuous",
            "axis": "long_axis",
            "order": 1,
        }
        r = client.post(
            "/api/pregrasp/demo/objects/symmetry",
            json={"name": "box", "kind": "continuous", "axis": "long_axis", "end_swap": True},
        )
        assert r.status_code == 200
        assert (
            manipulation.demos._read_objects(pathlib.Path(demo.root))["box"]["symmetry"]["end_swap"] is True
        )
        assert (
            client.post(
                "/api/pregrasp/demo/objects/symmetry",
                json={"name": "box", "kind": "continuous", "axis": "surface_normal", "end_swap": True},
            ).status_code
            == 422
        )
        assert (
            client.post("/api/pregrasp/demo/objects/symmetry", json={"name": "nope", "order": 2}).status_code
            == 404
        )
        assert (
            client.post("/api/pregrasp/demo/objects/symmetry", json={"name": "box", "order": 0}).status_code
            == 422
        )
        rgb, depth = np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.45, np.float32)
        asyncio.run(
            manipulation.acquisition._locate(
                "box", rgb, depth, dict(INTR), [600, 350], lambda: False, track=False
            )
        )
        assert queued[-1].extra["ref_symmetry"] == {
            "kind": "continuous",
            "axis": "long_axis",
            "order": 1,
            "end_swap": True,
        }, "the axis, turn family and end equivalence are sent with the find"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.worker.proc = None
            manipulation.runtime._state.located = {}


def test_a_places_landing_rule_is_kept_with_the_demo_and_read_back(client, tmp_path):
    """The landing rule is the operator's: exact unless asked, set on the demo, written beside it with the marks and
    kept when they are cleared; a demo saved before rules existed, or with one this version does not read, lands as
    shown."""
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "place")
    root = pathlib.Path(demo.root)
    root.mkdir(parents=True)
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        assert manipulation.demos._demo_info(demo)["landing"] == "exact", "no symmetry unless asked"
        assert manipulation.demos._demo_info(demo)["landing_axis"] == "surface_normal"
        r = client.post("/api/pregrasp/demo/landing", json={"landing": "turn"})
        assert r.status_code == 200 and r.json()["landing"] == "turn"
        assert (
            manipulation.demos._read_landing(root) == "turn"
            and manipulation.demos._read_keypoints(root) == demo.keypoints
        )
        assert client.post("/api/pregrasp/demo/keypoints", json={"keypoints": []}).status_code == 200
        assert manipulation.demos._read_landing(root) == "turn", "kept when the marks are cleared"
        r = client.post("/api/pregrasp/demo/landing", json={"landing": "turn", "axis": "long_axis"})
        assert r.status_code == 200 and r.json()["landing_axis"] == "long_axis"
        assert manipulation.demos._read_landing_axis(root) == "long_axis"
        assert (
            client.post(
                "/api/pregrasp/demo/landing", json={"landing": "turn", "axis": "sideways"}
            ).status_code
            == 422
        )
        assert client.post("/api/pregrasp/demo/landing", json={"landing": "anyhow"}).status_code == 422
        (root / manipulation.runtime.KEYPOINTS_FILE).write_text(
            json.dumps({"keypoints": [], "landing": "sideways"})
        )
        assert manipulation.demos._read_landing(root) == "exact"
        assert manipulation.demos._read_landing_axis(root) == "surface_normal"
        (root / manipulation.runtime.KEYPOINTS_FILE).write_text(json.dumps({"keypoints": []}))
        assert manipulation.demos._read_landing(root) == "exact"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None


def test_a_demo_saved_before_symmetry_reads_as_none(tmp_path):
    """Objects written as (name, frame, click), before symmetry was kept, read back with none."""
    root = tmp_path / "old"
    root.mkdir()
    m = np.zeros((H, W), bool)
    np.savez_compressed(
        root / manipulation.runtime.OBJECTS_FILE,
        names=json.dumps([["box", 2, [600, 350]]]),
        o0_deltas=np.tile(np.eye(4), (3, 1, 1)),
        o0_seen=np.ones(3, bool),
        o0_masks=np.zeros((3, H // 4, W // 4), bool),
        o0_mask=m,
    )
    assert manipulation.demos._read_objects(root)["box"]["symmetry"] == {
        "kind": "none",
        "axis": "surface_normal",
        "order": 1,
    }


def test_a_tracked_frame_moves_the_place_object_by_its_share(client, tmp_path, monkeypatch):
    """The worker answers each tracked frame of the picked object with every other object of its Point2Pose session:
    the share of the object a place goes onto moves that object's find, through the same result as the picked one."""
    demo, _kin, box = _place_demo(tmp_path, time.time())
    # Dispatch/composition test: this synthetic depth does not depict the moved
    # objects. Geometric rejection and correction are exercised separately.
    monkeypatch.setattr(
        manipulation.tracking,
        "_surface_decision",
        lambda pose, *args: {
            "state": "unchanged",
            "pose": None if pose is None else np.asarray(pose).tolist(),
            "observed_dof": 0,
        },
    )
    anchor = np.eye(4)
    anchor[:3, 3] = [0.03, 0.01, 0.0]
    moved = np.eye(4)
    moved[:3, 3] = [0.0, 0.02, 0.0]
    rgb, depth = np.full((H, W, 3), 200, np.uint8), np.full((H, W), 0.45, np.float32)
    ref = {"object": "gamepad", "ok": True, "delta": np.eye(4), "inliers": 150, "card_points": 400}
    keypoints = {"mode": "features", "concept": "gamepad", "mask": box_mask(), "n_points": 50, "ref": ref}
    keypoints.update(xyz=np.zeros((50, 3)), radius_mm=40.0, shape_class="box", yaw_observable=True, face=None)
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.teach = manipulation.runtime._Teach(
                at="t", box=(0, 0, 0, 0), rgb=rgb, depth_m=depth, intr=dict(INTR), keypoints=keypoints
            )
            manipulation.runtime._state.track.on, manipulation.runtime._state.track.algo = True, "p2p"
        view = [demo.name, int(demo.objects["box"]["frame"])]
        manipulation.acquisition._store_located(
            "box", {"object": "box", "ok": True, "delta": anchor, "mask": box, "tracking": True, "view": view}
        )
        job = manipulation.worker._queue_job(
            "track", "gamepad", rgb, depth, dict(INTR), algo="p2p", compress=False
        )
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.track.job = job.id
        others = [{"name": "box", "ok": True, "lost": False, "n_visible": 50, "n_tracks": 50}]
        meta = {
            "ok": True,
            "state": "tracking",
            "algo": "p2p",
            "n_inliers": 50,
            "n_matches": 50,
            "others": others,
        }
        result = _npz(
            meta=json.dumps(meta),
            delta=np.eye(4),
            live_uv=np.zeros((3, 2)),
            other_delta_0=moved,
            other_mask_0=box,
        )
        assert (
            client.post("/api/pregrasp/worker/result", params={"id": job.id}, content=result).status_code
            == 200
        )
        assert np.allclose(manipulation.runtime._state.located["box"]["delta"], moved @ anchor)
        assert client.get("/api/pregrasp/state").json()["located"]["box"]["track"] == "tracking"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
                manipulation.runtime._state.test
            ) = None
            manipulation.runtime._state.located = {}
            manipulation.runtime._state.targets = {}
            manipulation.runtime._state.track.on, manipulation.runtime._state.track.job = False, None
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()


def test_where_an_object_was_last_seen_is_kept_beside_the_demo(tmp_path, monkeypatch):
    """A find or a tracked frame of an object keeps where it was seen, as the click a find there would use, beside the
    saved demo; a tracked object's point is written at most every LAST_SEEN_EVERY_S. Each write is logged with what
    its outline came from and the outline, so a click that walked off the object can be traced back."""
    demo, _kin, box = _place_demo(tmp_path, time.time())
    demo.root = str(tmp_path / "saved")
    pathlib.Path(demo.root).mkdir()
    path = pathlib.Path(demo.root) / manipulation.runtime.LAST_SEEN_FILE
    monkeypatch.setattr(manipulation.runtime, "_last_seen_written", {})
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        manipulation.acquisition._remember_seen("box", box, box.shape, "find 1")
        manipulation.runtime._SEEN_EXECUTOR.submit(lambda: None).result()  # the write is done
        click = json.loads(path.read_text())["box"]["click"]
        assert box[click[1], click[0]], "a click on the box"
        moved = np.roll(box, 80, axis=1)
        manipulation.acquisition._remember_seen("box", moved, box.shape, "tracked by Point2Pose")
        manipulation.runtime._SEEN_EXECUTOR.submit(lambda: None).result()
        assert json.loads(path.read_text())["box"]["click"] == click, "not again so soon"
        monkeypatch.setattr(manipulation.runtime, "LAST_SEEN_EVERY_S", 0.0)
        manipulation.acquisition._remember_seen("box", moved, box.shape, "tracked by Point2Pose")
        manipulation.runtime._SEEN_EXECUTOR.submit(lambda: None).result()
        new = json.loads(path.read_text())["box"]["click"]
        assert moved[new[1], new[0]] and not box[new[1], new[0]], "where it was seen last"
        trail = pathlib.Path(demo.root) / manipulation.runtime.FINDS_DIR / manipulation.runtime.SEEN_LOG
        rows = [json.loads(line) for line in trail.read_text().splitlines()]
        assert [(r["click"], r["source"]) for r in rows] == [
            (click, "find 1"),
            (new, "tracked by Point2Pose"),
        ]
        ys, xs = np.nonzero(moved)
        assert rows[1]["outline"]["px"] == int(moved.sum()) and rows[1]["outline"]["middle"] == new
        assert rows[1]["outline"]["box"] == [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None


def test_each_find_is_logged_beside_the_demo_with_its_click_where_it_came_from_and_what_it_cut(
    tmp_path, monkeypatch
):
    """An act's find of the box clicks the middle of the outline kept for it. When that outline spills off the box the
    find can fail with nothing on record to say why: each find is logged beside the demo with why it was asked, the
    click, the outline the click came from, what the find cut out and matched, and the frame it ran on (not for one
    of the demo's own frames); the act's record lists its finds."""
    import asyncio

    import cv2

    demo, _kin, box = _place_demo(tmp_path, time.time())
    demo.root = str(tmp_path / "saved")
    pathlib.Path(demo.root).mkdir()
    spill = box.copy()
    spill[300:400, 420:560] = True  # the tray beside the box, cut out with it
    rgb = np.full((H, W, 3), 40, np.uint8)
    rgb[box] = (60, 200, 60)
    depth = np.full((H, W), 0.45, np.float32)
    answers = iter(
        [{"ok": True, "ref_ok": False, "ref_reason": "the live view does not match", "mask": spill}]
    )

    def fake_queue(kind, concept, rgb, depth_m, intr, **kw):
        job = manipulation.runtime._Job(
            id="j", kind=kind, concept=concept, rgb=rgb, depth_m=depth_m, intr=intr, created=0.0
        )
        job.result = next(
            answers, {"ok": True, "ref_ok": True, "ref_delta": np.eye(4), "ref_inliers": 150, "mask": box}
        )
        return job

    monkeypatch.setattr(manipulation.worker, "_queue_job", fake_queue)
    monkeypatch.setattr(manipulation.camera, "_frame", lambda: _async((rgb, depth, dict(INTR))))
    monkeypatch.setattr(manipulation.runtime, "_last_seen_written", {})
    kept = {
        "object": "box",
        "ok": True,
        "mask": spill,
        "at": 1.0,
        "tracked_at": 2.0,
        "find_id": "an earlier find",
    }
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.worker.proc = _FakeProc()
            manipulation.runtime._state.located["box"] = kept
            manipulation.runtime._state.act = manipulation.runtime._Act(on=True)
        why = asyncio.run(manipulation.acquisition._locate_afresh("box", lambda: False, track=False))
        assert why == "the live view does not match"
        assert manipulation.runtime._state.located["box"]["mask"] is None
        assert "click box" in asyncio.run(
            manipulation.acquisition._locate_afresh("box", lambda: False, track=False)
        ), "a failed background mask must not seed another automatic find"
        log = {"why": "measuring how the demo held it", "came_from": {"demo_frame": 7}}
        asyncio.run(
            manipulation.acquisition._locate(
                "box", rgb, depth, dict(INTR), [600, 350], lambda: False, log=log
            )
        )
        teach = manipulation.runtime._Job(
            id="t", kind="teach", concept="gamepad", rgb=rgb, depth_m=depth, intr=INTR, created=0.0
        )
        teach.click, teach.extra, teach.log = (
            [360, 240],
            {"ref_object": "gamepad"},
            {"why": "clicked on the page"},
        )
        teach.result = {
            "ok": True,
            "mask": box_mask(),
            "uv": np.zeros((50, 2)),
            "xyz": np.zeros((50, 3)),
            "n_points": 50,
            "radius_mm": 40.0,
            "shape_class": "box",
            "yaw_observable": True,
            "ref_ok": True,
            "ref_delta": np.eye(4),
            "ref_inliers": 236,
            "ref_card_points": 373,
        }
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach_job = "t"
        manipulation.acquisition._apply_teach_result(teach)
        manipulation.runtime._FIND_LOG_EXECUTOR.submit(lambda: None).result()  # the writes are done
        d = pathlib.Path(demo.root) / manipulation.runtime.FINDS_DIR
        rows = [json.loads(line) for line in (d / manipulation.runtime.FINDS_LOG).read_text().splitlines()]
        act_find, demo_find, taught = rows
        assert (taught["kind"], taught["object"], taught["why"], taught["click"]) == (
            "teach",
            "gamepad",
            "clicked on the page",
            [360, 240],
        )
        assert taught["used"] and taught["ok"] and taught["inliers"] == 236 and taught["strong"] is True
        assert (d / f"{taught['id']}_cut.png").exists()
        manipulation.runtime._SEEN_EXECUTOR.submit(lambda: None).result()
        seen = [json.loads(line) for line in (d / manipulation.runtime.SEEN_LOG).read_text().splitlines()]
        assert seen[-1]["source"] == f"find {taught['id']}", "a last-seen click names the find it came from"
        middle = manipulation.acquisition._deepest_pixel(spill, spill.shape)
        assert act_find["why"] == "act: found again where it was kept" and act_find["click"] == middle
        came = act_find["came_from"]
        assert came["kept_outline"]["px"] == int(spill.sum()) and came["kept_outline"]["middle"] == middle
        assert (came["kept_find"], came["found_at"], came["tracked_at"]) == ("an earlier find", 1.0, 2.0)
        assert act_find["ok"] is False and act_find["reason"] == "the live view does not match"
        assert act_find["cut"]["px"] == int(spill.sum()) and act_find["during_act"] is True
        fid = act_find["id"]
        assert np.array_equal(cv2.imread(str(d / f"{fid}_cut.png"), cv2.IMREAD_UNCHANGED) > 0, spill)
        assert np.array_equal(
            cv2.imread(str(d / f"{fid}_depth.png"), cv2.IMREAD_UNCHANGED), np.full((H, W), 450)
        )
        assert cv2.imread(str(d / f"{fid}.jpg")).shape == (H, W, 3), "the frame it ran on, to replay it"
        assert (
            demo_find["ok"] is True
            and demo_find["inliers"] == 150
            and demo_find["came_from"]["demo_frame"] == 7
        )
        assert not (d / f"{demo_find['id']}.jpg").exists(), "a demo frame is in the demo already"
        assert manipulation.runtime._state.act.finds == [fid, demo_find["id"], taught["id"]], (
            "the act's record lists them"
        )
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
                manipulation.runtime._state.test
            ) = None
            manipulation.runtime._state.teach_job = None
            manipulation.runtime._state.worker.proc = None
            manipulation.runtime._state.located = {}
            manipulation.runtime._state.targets = {}
            manipulation.runtime._state.act = manipulation.runtime._Act()


def test_an_acts_record_keeps_the_place_objects_mask_and_where_the_act_holds_it(tmp_path, monkeypatch):
    """A replay of an act could not draw the object placed onto's mask, and drew where the act held it only by
    rebuilding that from the record's parts: each recorded frame keeps both, the pose as the act held it then, with
    the find it came from; and the point groups' newest picture, which says what carried each object, with its
    wireframe and tinted surfaces as the camera view painted them."""
    demo, _kin, box = _place_demo(tmp_path, time.time())
    held = np.eye(4)
    held[:3, 3] = [0.01, 0.02, 0.0]
    run = manipulation.recording._Run(root=tmp_path / "act", meta={})
    rgb, depth = np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.45, np.float32)
    job = manipulation.runtime._Job(
        id="j", kind="track", concept="gamepad", rgb=rgb, depth_m=depth, intr=INTR, created=1.0
    )
    r = {"ok": True, "state": "tracking", "other_mask_0": box, "other_track_uv_0": np.zeros((3, 2))}
    r["others"] = [{"name": "box", "ok": True}]
    found = {
        "object": "box",
        "ok": True,
        "delta": held,
        "stamp": None,
        "find_id": "f1",
        "view": [demo.name, 2],
    }
    drawing = {
        "points": [[600, 350, 3, 1], [610, 352, 3, 0], [100, 100, 0, 1]],
        "base": 0,
        "quiet": [],
        "stamp": 0.9,
        "objects": {
            "box": {"group": 3, "own": False, "why": "", "n_seen": 0, "n_grouped": 2, "own_points": [0, 1]}
        },
        "groups": {"3": {"members": 2, "n_fit": 1, "rms_mm": 1.5, "supported": True}},
        "links": [[0, 1]],
        "joins": [],
        "depth": [450, 452, 0],
        "depth_range": [400.0, 620.0],
        "surfaces": "iVBORw0KGgo=",  # the label map's PNG, as the view sends it
        "scale": 4,
    }
    monkeypatch.setattr(manipulation.runtime._state, "groups_with_acts", True)
    monkeypatch.setattr(
        manipulation.runtime._state, "groups_feed", manipulation.runtime._GroupsFeed(drawing=drawing)
    )
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.located["box"] = found
        run.frame(job, r, {"ok": True}, "place", None, None)
        manipulation.runtime._RUN_EXECUTOR.submit(lambda: None).result()  # the frame is on disk
        f0 = np.load(tmp_path / "act" / "frames" / "000000.npz")
        assert np.array_equal(f0["other_mask_0"], box) and np.allclose(f0["held_0"], held)
        assert run.frames[0]["held"] == [
            {"object": "box", "find_id": "f1", "stamp": None, "tracked_at": None}
        ]
        assert run.frames[0]["others"] == ["box"], "the other arrays' objects, in order"
        assert f0["groups_points"].tolist() == drawing["points"], (
            "every track's pixel, group and whether seen"
        )
        kept = run.frames[0]["groups"]
        assert (
            kept["objects"]["box"]["group"] == 3
            and kept["groups"]["3"]["n_fit"] == 1
            and kept["stamp"] == 0.9
        )
        assert f0["groups_links"].tolist() == [[0, 1]] and f0["groups_depth"].tolist() == [450, 452, 0], (
            "the wireframe: its links and each track's depth"
        )
        assert f0["groups_joins"].shape == (0, 2) and kept["depth_range"] == [400.0, 620.0]
        assert kept["surfaces"] == "iVBORw0KGgo=" and kept["scale"] == 4, (
            "the tinted surfaces, for the replay"
        )
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.located = {}


def test_a_loaded_demo_finds_its_objects_where_they_were_last_seen_without_a_click(tmp_path, monkeypatch):
    """After a restart nothing in memory says where the objects are: the load finds them again where they were last
    seen, once the worker and the camera are up: the one a place goes onto first, then the picked one by a teach
    against the demo's view whose Point2Pose session starts with both, one start for the two."""
    import asyncio

    from lerobot.gui.api import showservo

    demo, _kin, box = _place_demo(tmp_path, time.time())
    demo.root = str(tmp_path / "saved")
    pathlib.Path(demo.root).mkdir()
    (pathlib.Path(demo.root) / manipulation.runtime.LAST_SEEN_FILE).write_text(
        json.dumps({"gamepad": {"click": [300, 250]}, "box": {"click": [560, 300]}})
    )
    live_rgb = np.full((H, W, 3), 200, np.uint8)
    monkeypatch.setattr(
        manipulation.camera,
        "_frame",
        lambda: _async((live_rgb, np.full((H, W), 0.45, np.float32), dict(INTR))),
    )
    camera = {"on": False}
    monkeypatch.setattr(showservo, "live_camera", lambda: object() if camera["on"] else None)
    asked: list[manipulation.runtime._Job] = []

    async def run():
        async def worker():
            while True:
                await asyncio.sleep(0.002)
                with manipulation.runtime._state.lock:
                    pending = list(manipulation.runtime._state.worker.pending)
                    manipulation.runtime._state.worker.pending.clear()
                for job_id in pending:
                    job = manipulation.runtime._state.worker.jobs[job_id]
                    asked.append(job)
                    if job.kind == "teach":
                        job.result = {
                            "ok": True,
                            "ref_ok": True,
                            "ref_inliers": 150,
                            "ref_card_points": 400,
                            "ref_delta": np.eye(4),
                            "n_points": 50,
                            "radius_mm": 40.0,
                            "shape_class": "box",
                            "yaw_observable": True,
                            "mask": box_mask(),
                            "uv": np.zeros((50, 2)),
                            "xyz": np.zeros((50, 3)),
                        }
                        manipulation.acquisition._apply_teach_result(job)
                    elif job.kind == "locate":
                        job.result = {
                            "ok": True,
                            "ref_ok": True,
                            "ref_inliers": 150,
                            "ref_card_points": 400,
                            "ref_delta": np.eye(4),
                            "mask": box,
                            "tracking": bool(job.extra.get("track")),
                        }

        feed = asyncio.create_task(worker())
        refind = asyncio.create_task(manipulation.acquisition._refind_last_seen(demo))
        await asyncio.sleep(0.6)
        assert not asked, "nothing before the worker and the camera are up"
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.worker.log.append("worker ready")
        camera["on"] = True
        await asyncio.sleep(0.6)
        assert not asked, "nor while the load's own teach of the demo's object is in flight"
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach_job = None
        await asyncio.wait_for(refind, timeout=5.0)
        await asyncio.sleep(0.05)
        feed.cancel()

    try:
        with manipulation.runtime._state.lock:
            (
                manipulation.runtime._state.demo,
                manipulation.runtime._state.teach,
                manipulation.runtime._state.test,
            ) = demo, None, None
            manipulation.runtime._state.located = {}
            manipulation.runtime._state.worker.log = []
            manipulation.runtime._state.teach_job = "the load's own teach"
        manipulation.runtime._state.worker.proc = _FakeProc()
        asyncio.run(run())
        locate = next(j for j in asked if j.kind == "locate")
        assert locate.click == [560, 300] and not locate.extra.get("track"), (
            "the box first, joining nothing alone"
        )
        teach = next(j for j in asked if j.kind == "teach")
        assert asked.index(locate) < asked.index(teach)
        assert teach.click == [300, 250] and teach.extra["ref_object"] == "gamepad"
        assert teach.extra["more"] == ["box"] and np.array_equal(teach.arrays["more_mask_0"], box), (
            "the session starts with both"
        )
        assert (
            manipulation.runtime._state.teach is not None
            and manipulation.acquisition._located(demo, "box")["ok"]
        )
        assert "box" in manipulation.runtime._state.targets, "the box followed in that session"
    finally:
        manipulation.runtime._state.worker.proc = None
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
                manipulation.runtime._state.test
            ) = None
            manipulation.runtime._state.located = {}
            manipulation.runtime._state.targets = {}
            manipulation.runtime._state.worker.log = []
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()
            manipulation.runtime._state.track.on = False


class _FakeSession:
    """A Point2Pose session as the bridge answers it, for any number of objects: object i moves i + 1 mm along x per
    step since the session's start."""

    def __init__(self):
        self.n, self.inits, self.steps = 0, 0, 0

    def init(self, rgb, depth_m, masks, intr):
        masks = np.asarray(masks, dtype=bool)
        self.n, self.inits, self.steps = len(masks[None] if masks.ndim == 2 else masks), self.inits + 1, 0
        return self._reply()

    def step(self, rgb, depth_m):
        self.steps += 1
        return self._reply()

    def _reply(self):
        each = {"lost": False, "n_visible": 40, "n_tracks": 50, "mean_residual_m": 0.001}
        r = {"ok": True, "objects": [dict(each) for _ in range(self.n)]}
        for i in range(self.n):
            d = np.eye(4)
            d[0, 3] = 0.001 * (i + 1) * self.steps
            r.update({f"delta_{i}": d, f"mask_{i}": np.ones((4, 4), bool), f"model_{i}": np.zeros((0, 3))})
            r.update({f"live_uv_{i}": np.zeros((0, 2)), "mean_residual_m": 0.001})
            r.update({f"track_idx_{i}": np.arange(3) + 3 * i, f"track_uv_{i}": np.full((3, 2), float(i)),
                      f"track_vis_{i}": np.ones(3, bool)})  # fmt: skip
        return r


def _frame_of(worker, k: int):
    return worker._Frame(np.full((4, 4, 3), k, np.uint8), np.full((4, 4), 0.45, np.float32), f"f{k}")


def test_the_live_session_carries_each_objects_motion_across_a_restart(worker):
    """Point2Pose fixes its objects when a session starts: an object found anew starts it over with every object's
    newest mask, and each one's motion goes on from where it was (its pose in the new session times its motion when
    that began), so nothing jumps."""
    bridge = _FakeSession()
    scene = worker.Scene(lambda: bridge)
    intr = worker.CameraIntrinsics(fx=600.0, fy=600.0, cx=2.0, cy=2.0)
    assert scene.start(_frame_of(worker, 0), intr, {"gamepad": np.ones((4, 4), bool)})
    for k in range(1, 4):
        scene.step(_frame_of(worker, k).rgb, None)
    assert scene.share("gamepad")["delta"][0, 3] == pytest.approx(0.003)
    assert scene.start(_frame_of(worker, 4), intr, {"cube": np.ones((4, 4), bool)}, keep=["gamepad"])
    assert scene.order == ["gamepad", "cube"] and bridge.n == 2
    assert scene.share("gamepad")["delta"][0, 3] == pytest.approx(0.003), "no jump at the restart"
    scene.step(_frame_of(worker, 5).rgb, None)
    assert scene.share("gamepad")["delta"][0, 3] == pytest.approx(0.004), "its motion goes on from there"
    assert scene.share("cube")["delta"][0, 3] == pytest.approx(0.002), "the cube's from its own find"
    rgb = _frame_of(worker, 6).rgb
    scene.step(rgb, None)
    scene.step(rgb, None)
    assert bridge.steps == 2, "one step per frame, however many objects read it"
    empty = {"cube": np.zeros((4, 4), bool)}
    assert not scene.start(_frame_of(worker, 7), intr, empty), "an empty mask starts nothing"
    assert scene.order == ["gamepad", "cube"], "the session as it was"


def test_each_object_s_share_carries_its_tracks_and_which_start_numbered_them(worker):
    """For the server's depth check: every object's share carries its tracks (number, pixel, seen by the colour
    tracker) and how many times the session has started, since each start numbers the tracks afresh; the frame's reply
    to the server passes the others' tracks on beside their motion."""
    bridge = _FakeSession()
    scene = worker.Scene(lambda: bridge)
    intr = worker.CameraIntrinsics(fx=600.0, fy=600.0, cx=2.0, cy=2.0)
    assert scene.start(_frame_of(worker, 0), intr, {"gamepad": np.ones((4, 4), bool)})
    scene.step(_frame_of(worker, 1).rgb, None)
    share = scene.share("gamepad")
    assert share["session"] == 1 and list(share["track_idx"]) == [0, 1, 2] and share["track_vis"].all()
    assert scene.start(_frame_of(worker, 2), intr, {"cube": np.ones((4, 4), bool)}, keep=["gamepad"])
    assert scene.share("gamepad")["session"] == scene.share("cube")["session"] == 2
    arrays = worker._others_arrays(scene.others("gamepad"))
    assert list(arrays["other_track_idx_0"]) == [3, 4, 5] and arrays["other_track_uv_0"].shape == (3, 2)
    assert arrays["other_track_vis_0"].all()


def test_a_frame_for_an_object_the_session_no_longer_has_leaves_the_session_alone(worker):
    """After the load's own teach of the demo's object, the gamepad's teach started the session with the gamepad and
    the cube; a frame asked earlier for the old object then restarted the session with it alone, and the next
    gamepad frame did the same with the gamepad alone: the cube was gone. Such a frame now changes nothing."""
    import types

    bridge = _FakeSession()
    scene = worker.Scene(lambda: bridge)
    intr = worker.CameraIntrinsics(fx=600.0, fy=600.0, cx=2.0, cy=2.0)
    assert scene.start(
        _frame_of(worker, 0), intr, {"gamepad": np.ones((4, 4), bool), "cube": np.ones((4, 4), bool)}
    )
    models = types.SimpleNamespace(scene=lambda mode="p2p": scene)
    old = types.SimpleNamespace(  # what a card the tracker reads carries
        scene=_frame_of(worker, 0),
        mask=np.ones((4, 4), bool),
        shape_class="box",
        yaw_observable=True,
        face=None,
        table_normal=None,
        xyz=np.zeros((0, 3)),
    )
    try:
        reply = worker._track(
            {"concept": "object_1", "algo": "p2p"},
            _frame_of(worker, 1),
            {"object_1": old},
            {},
            None,
            None,
            intr,
            models,
        )
    except Exception as e:  # what the session looks like afterwards is the point; the call is checked next
        reply = e
    assert scene.order == ["gamepad", "cube"] and bridge.inits == 1, "the session as it was"
    assert not isinstance(reply, Exception), reply
    meta = json.loads(str(np.load(io.BytesIO(reply))["meta"]))
    assert meta["state"] == "not tracked" and not meta["ok"]


def test_the_live_track_waits_out_a_camera_it_cannot_read_and_carries_on(monkeypatch):
    """A frame the camera cannot give for a moment (it was the Servo session restarting) left the track stopped for
    good, while the live view went on showing its last frame as tracking. It waits, says why, and asks again."""
    import asyncio

    from fastapi import HTTPException

    monkeypatch.setattr(manipulation.runtime, "TRACK_FRAME_RETRY_S", 0.01)
    camera = {"on": False}
    live_rgb = np.full((H, W, 3), 200, np.uint8)

    async def frame():
        if not camera["on"]:
            raise HTTPException(409, "start a live camera session in the Servo tab first")
        return live_rgb, np.full((H, W), 0.45, np.float32), dict(INTR)

    monkeypatch.setattr(manipulation.camera, "_frame", frame)
    keypoints = {"mode": "features", "concept": "box", "n_points": 50}
    states: list[str] = []

    async def run():
        manipulation.tracking._begin_track()
        tr = manipulation.runtime._state.track
        for _ in range(20):
            await asyncio.sleep(0.01)
            states.append((tr.last or {}).get("state"))
        assert tr.on, "still on while the camera cannot be read"
        camera["on"] = True
        for _ in range(200):
            await asyncio.sleep(0.005)
            with manipulation.runtime._state.lock:
                pending = list(manipulation.runtime._state.worker.pending)
            if pending:
                break
        assert pending and manipulation.runtime._state.worker.jobs[pending[0]].kind == "track", (
            "asked for a frame again"
        )
        with manipulation.runtime._state.lock:
            tr.on = False
        tr.done.set()
        await asyncio.wait_for(tr.task, timeout=2.0)

    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach = manipulation.runtime._Teach(
                at="t",
                box=(0, 0, 0, 0),
                rgb=live_rgb,
                depth_m=np.zeros((H, W)),
                intr=dict(INTR),
                keypoints=keypoints,
            )
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()
        manipulation.runtime._state.worker.proc = _FakeProc()
        asyncio.run(run())
        assert "waiting" in states and "stopped" not in states
    finally:
        manipulation.runtime._state.worker.proc = None
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach = None
            manipulation.runtime._state.track.on = False
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()


@pytest.mark.parametrize("trusted", [False, True])
def test_the_live_sessions_model_is_served_with_the_frame_and_motions_that_place_it(client, trusted):
    """The worker's model survives the HTTP boundary with its frame and raw, trusted and held motions distinct.
    A rejected view stays inspectable without claiming its fit was trusted; no data before the first frame."""
    with manipulation.runtime._state.lock:
        manipulation.runtime._state.track.model = None
    assert client.get("/api/pregrasp/track/model.npz").status_code == 404
    depth = np.full((H, W), 0.45, np.float32)
    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=np.zeros((H, W, 3), np.uint8),
        depth_m=depth,
        intr=dict(INTR),
        keypoints={
            "mode": "features",
            "concept": "gamepad",
            "xyz": np.full((50, 3), 0.45),
            "n_points": 50,
            "radius_mm": 40.0,
            "shape_class": "box",
            "yaw_observable": True,
        },
    )
    model = np.arange(12.0).reshape(4, 3)
    held = np.eye(4)
    held[:3, 3] = [0.01, 0.0, 0.0]
    job = manipulation.runtime._Job(
        id="j",
        kind="track",
        concept="gamepad",
        rgb=np.zeros((H, W, 3), np.uint8),
        depth_m=depth,
        intr=dict(INTR),
        created=time.time(),
    )
    payload = _npz(
        meta=json.dumps(
            {
                "ok": True,
                "state": "tracking",
                "n_inliers": 50 if trusted else 0,
                "n_matches": 50,
                "n_tracks": 50,
                "n_model_points": 4,
            }
        ),
        model_xyz=model,
        mask=np.zeros((H, W), bool),
        delta=held,
    )
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach, manipulation.runtime._state.track.job = teach, "j"
            manipulation.runtime._state.test = manipulation.runtime._Test(
                at="t", rgb=None, result={"ok": True, "delta_cam": held}
            )
            manipulation.runtime._state.worker.jobs[job.id] = job
        assert (
            client.post("/api/pregrasp/worker/result", params={"id": job.id}, content=payload).status_code
            == 200
        )
        assert job.result_received_at <= job.result_decoded_at <= job.verified_at <= job.rendered_at
        r = client.get("/api/pregrasp/track/model.npz")
        assert r.status_code == 200
        z = np.load(io.BytesIO(r.content))
        assert np.allclose(z["model_xyz"], model) and np.allclose(z["teach_xyz"], 0.45)
        assert ("delta_cam" in z.files) == trusted
        assert np.allclose(z["raw_delta_cam"], held), (
            "a rejected fit remains inspectable, without being trusted"
        )
        if trusted:
            assert np.allclose(z["delta_cam"], held)
        assert np.allclose(z["held_delta_cam"], held)
        assert z["depth_m"].shape == (H, W) and z["mask"].shape == (H, W)
        assert np.allclose(z["intr"], [INTR["fx"], INTR["fy"], INTR["cx"], INTR["cy"]])
        state = client.get("/api/pregrasp/state")
        assert state.status_code == 200, "large model arrays stay out of the JSON readout"
        assert "model_xyz" not in state.json()["test"]
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach = manipulation.runtime._state.test = (
                manipulation.runtime._state.track.model
            ) = None
            manipulation.runtime._state.track.job = None
            manipulation.runtime._state.track.last = {}
            manipulation.runtime._state.track.history = []
            manipulation.runtime._state.track.hold_views = []
            manipulation.runtime._state.worker.jobs.pop(job.id, None)
