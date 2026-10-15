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

    monkeypatch.setattr(jog, "update_playback_next", update_next)
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


def test_act_refuses_recovered_axis_pose_without_a_checked_grasp_choice(client, tmp_path):
    demo, _, _ = _place_demo(tmp_path, time.time())
    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=None,
        depth_m=None,
        intr={},
        keypoints={
            "ref": {
                "object": "gamepad",
                "ok": True,
                "delta": np.eye(4),
                "surface_constraint": {"state": "recovered"},
            }
        },
    )
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.teach = teach
            manipulation.runtime._state.teach_job = None
        response = client.post("/api/pregrasp/act", json={})
        assert response.status_code == 409
        assert "collision-checked grasp selection" in response.text
        assert manipulation.runtime._state.act.on is False
        _, problem = manipulation.planning._reference_motion(demo, teach)
        assert "collision-checked grasp selection" in problem
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = None


def _run_place_act(
    tmp_path,
    monkeypatch,
    empty_grip=False,
    slow_grip=False,
    slip=False,
    weak_carry=False,
    blind=False,
    blind_carry=False,
    inject=None,
    correct_hold=True,
    allow_grasp_prediction=False,
    finds_wait_for_the_arm=False,
    box_moves_in_the_carry=None,
    covered_after_find=False,
    sag_deg=0.0,
    wrist_deg=0.0,
    box_turn_deg=0.0,
    landing="exact",
    ranges=None,
    world=False,
    as_key_frames=False,
    two_preplaces=False,
    low_preplace=False,
    drop_in_lift=False,
    lift_mm=0.0,
    lift_before_mm=0.0,
    initial_find_inliers=150,
    fresh_find_inliers=150,
    held_turn_deg=0.0,
    correction_sag_deg=0.0,
):
    """Run the act on the place demo against a fake arm and a fake worker. The gamepad lies 10 mm from where the demo
    had it, the box 30 mm and 10 mm; the demo held the gamepad 20 mm below the fingertip and the act holds it 6 mm
    further along x; with ``slip`` the lift moves it another 6 mm. With ``weak_carry`` the demo's frames after the
    lift show too little of the gamepad to find it; with ``blind`` no view of it in the gripper matches at all.
    ``inject`` and ``correct_hold`` are what the operator asked the act for (:class:`manipulation.execution.InjectBody`). With
    ``finds_wait_for_the_arm`` the worker answers nothing until the arm has been given a target. With
    ``box_moves_in_the_carry``, a motion (camera frame), the session that follows the gamepad, which the box joins at
    the act's start, sees the box moved by it once the gamepad is carried. With ``covered_after_find`` the arm hides
    the gamepad from the camera once it is found again: the restarted track never sees it. With ``sag_deg`` the
    shoulder reads that much short of its command while the gripper holds the gamepad, as the load holds it down. With
    ``wrist_deg`` the demo turns the arm's wrist that far for the place (:func:`_place_demo`); ``box_turn_deg`` turns
    the box about its middle as well as moving it; ``landing`` is the demo's landing rule and ``ranges`` what the arm
    reports as its servos' ranges. With ``world`` the demo's
    arm rises 20 mm and moves 10 mm along x after the place end, its gripper opening on to 75, and its last sample is a
    world waypoint. With ``as_key_frames`` the marks are stored as the key frames they make ("key" marks). With
    ``two_preplaces`` a second pre-place comes before the one marked, in the carry; with ``drop_in_lift`` the gamepad
    falls out of the fingers once the lift has raised the arm above 40 mm, the gripper then reaching its command.
    With ``held_turn_deg`` the object is held turned that far about the tip's z axis. With ``lift_mm`` the key frames
    (as "key" marks) rise that far straight up after the grasp; with
    ``lift_before_mm`` that far after the pre-grasp, a second pre-grasp then coming down to the grasp. Post: (demo, sim, views, the motions and holds); ``sim["sent_at"]`` has each streamed sample's time and how many
    streams had ended by then, ``sim["first_answer"]`` when the worker first answered."""
    import asyncio
    import time as _time

    from lerobot.gui.api import jog
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    monkeypatch.setattr(
        manipulation.runtime._state, "groups_with_acts", False
    )  # held poses: no point groups view here
    demo, kin, box = _place_demo(tmp_path, _time.time(), wrist_deg=wrist_deg)
    demo.landing = landing
    if world:
        end_at, last = PLACE_AT["place_end"], len(demo.t) - 1
        for i in range(end_at + 1, last + 1):
            a = (i - end_at) / (last - end_at)
            demo.q_obs[i, 0] = demo.q_obs[end_at, 0] + 10.0 * a
            demo.q_obs[i, 2] = demo.q_obs[end_at, 2] + 20.0 * a
            demo.q_cmd[i, :6] = demo.q_obs[i, :6]
            demo.q_cmd[i, MOTOR_NAMES.index("gripper")] = 60.0 + 15.0 * a
            demo.tips[i] = kin.forward_kinematics(demo.q_obs[i])
        demo.keypoints.append({"t": float(demo.t[last]), "kind": "waypoint", "frame": "world"})
    if two_preplaces:
        demo.keypoints.append({"t": float(demo.t[66]), "kind": "preplace", "object": "box"})
    if low_preplace:
        assert two_preplaces
        for mark in demo.keypoints:
            if mark["kind"] == "preplace" and mark["t"] == float(demo.t[PLACE_AT["preplace"]]):
                mark["t"] = float(demo.t[85])  # near contact, after the earlier high observation line
    if as_key_frames:
        demo.keypoints = [m for m in demo.keypoints if m["kind"] == "pose"] + [
            {"t": k["t"], "kind": "key", "frame": k["frame"], "move": k["move"], "grip": k["grip"]}
            for k in core.key_frames(demo.keypoints)
        ]
    if lift_mm:
        lift = {"t": float(demo.t[60]), "kind": "key", "frame": "world", "move": "lift", "grip": "first"}
        demo.keypoints = [m for m in demo.keypoints if m["kind"] == "pose"] + sorted(
            [
                *(
                    {"t": k["t"], "kind": "key", "frame": k["frame"], "move": k["move"], "grip": k["grip"]}
                    for k in core.key_frames(demo.keypoints)
                ),
                {**lift, "up_mm": lift_mm},
            ],
            key=lambda k: k["t"],
        )
    if lift_before_mm:
        keys = [
            {"t": k["t"], "kind": "key", "frame": k["frame"], "move": k["move"], "grip": k["grip"]}
            for k in core.key_frames(demo.keypoints)
        ]
        extra = [
            {
                "t": float(demo.t[14]),
                "kind": "key",
                "frame": "world",
                "move": "lift",
                "grip": "first",
                "up_mm": lift_before_mm,
            },
            {"t": float(demo.t[20]), "kind": "key", "frame": "gamepad", "move": "line", "grip": "first"},
        ]
        demo.keypoints = [m for m in demo.keypoints if m["kind"] == "pose"] + sorted(
            [*keys, *extra], key=lambda k: k["t"]
        )
    pick_moved, box_moved, hold_demo, hold_now = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
    pick_moved[:3, 3] = [0.010, 0.0, 0.0]
    box_moved[:3, 3] = [0.030, 0.010, 0.0]
    if box_turn_deg:
        box_moved = box_moved @ core.turn_about(
            manipulation.planning._landing_centre(demo, "box", np.eye(4)), box_turn_deg
        )
    hold_demo[:3, 3] = [0.0, 0.0, -0.020]
    hold_now[:3, 3] = [0.006, 0.0, -0.020]
    hold_now[:3, :3] = core.turn_about(np.zeros(3), held_turn_deg)[:3, :3]
    hold_carried = hold_now.copy()
    if slip:
        hold_carried[:3, 3] = [0.012, 0.0, -0.020]
    live_rgb = np.full((H, W, 3), 200, np.uint8)
    captured: list[np.ndarray] = []  # the fake arm's tip at each live frame, by the frame's first pixel

    def frame():
        """The live camera: each frame says which it is, so a find read after the arm moved on sees the arm as it
        was when the frame was taken."""
        assert len(captured) < 256 * 256, "the first two values number the frames"
        rgb = live_rgb.copy()
        rgb[0, 0, 0], rgb[0, 0, 1] = len(captured) % 256, len(captured) // 256
        captured.append(kin.forward_kinematics(sim["q"]))
        # The synthetic reference is a front-facing plane at 0.45 m. Carry its depth with the same hold the fake
        # matcher observes; a stationary depth image would contradict the visible object after the lift.
        z = 0.45
        if sim["grip"] >= 82.0:
            held = hold_now if captured[-1][2, 3] < 0.04 else hold_carried
            z += (captured[-1] @ held)[2, 3]
        return _async((rgb, np.full((H, W), z, np.float32), dict(INTR)))

    monkeypatch.setattr(manipulation.camera, "_frame", frame)
    sim = {
        "q": np.array([80.0, -20.0, 90.0, 0, 0, 0, 60.0]),
        "grip": 60.0,
        "closed_at": None,
        "targets": [],
        "target_at": [],
        "first_move_at": None,
        "streamed": [],
        "sent_at": [],
        "stops": 0,
        "first_answer": None,
        "first_answer_wall": None,
        "find_answers": [],
        "grips_set": [],
    }

    def grip_obs():
        dropped = drop_in_lift and kin.forward_kinematics(sim["q"])[2, 3] > 0.040
        final = (
            sim["grip"] if empty_grip or dropped or sim["grip"] < 82.0 else 82.0
        )  # the gamepad stops the fingers
        if (
            slow_grip and sim["closed_at"] is not None
        ):  # still closing, 5 units behind and catching up over 0.3 s
            final -= 5.0 * max(0.0, 1.0 - (_time.monotonic() - sim["closed_at"]) / 0.3)
        return final

    def set_target_pose(pose):
        pytest.fail("Act must use the checked joint stream, not a separate Cartesian walk")

    def status():
        return {"connected": True, "halted": False, "holding": False}

    async def joints_start(q_first):
        sim["q"] = np.array([q_first[m] for m in MOTOR_NAMES])

    async def joints_stop():
        sim["stops"] += 1

    def set_target_joints(q):
        sim["q"] = np.array([q[m] for m in MOTOR_NAMES])
        if sim["first_move_at"] is None and np.max(np.abs(sim["q"][:6] - [80, -20, 90, 0, 0, 0])) > 0.1:
            sim["first_move_at"] = _time.monotonic()
        if float(q["gripper"]) >= 84.0 > sim["grip"]:
            sim["closed_at"] = _time.monotonic()
        sim["grip"] = float(q["gripper"])
        sim["streamed"].append(sim["q"].copy())
        sim["sent_at"].append((_time.monotonic(), sim["stops"]))

    def tip_and_anchor():
        q = {m: float(sim["q"][k]) for k, m in enumerate(MOTOR_NAMES)}
        q["gripper"] = grip_obs()
        if sim["grip"] >= 82.0:  # holding: the load keeps the shoulder short of its command
            q["shoulder_lift"] += sag_deg
            if sim["stops"] >= 2:
                q["shoulder_lift"] += correction_sag_deg
        return kin.forward_kinematics(sim["q"]), np.eye(4), q

    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    monkeypatch.setattr(jog, "servo_ranges", lambda: ranges)
    monkeypatch.setattr(jog, "current_tip_and_anchor", tip_and_anchor)
    monkeypatch.setattr(jog, "set_target_pose", set_target_pose)
    monkeypatch.setattr(jog, "current_status", status)
    monkeypatch.setattr(jog, "current_gripper", grip_obs)
    monkeypatch.setattr(
        jog, "set_gripper", lambda g: sim["grips_set"].append(g) or sim.__setitem__("grip", g)
    )
    monkeypatch.setattr(jog, "walk_limits", lambda: (0.04, np.radians(30)))
    monkeypatch.setattr(jog, "set_walk_limits", lambda lin, ang: None)
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    monkeypatch.setattr(jog, "joints_start", joints_start)
    monkeypatch.setattr(jog, "joints_stop", joints_stop)
    monkeypatch.setattr(jog, "set_target_joints", set_target_joints)
    fake_playback(monkeypatch, set_target_joints)
    monkeypatch.setattr(jog, "start_record", lambda: _time.time())
    monkeypatch.setattr(jog, "stop_record", lambda: [])
    monkeypatch.setattr(jog, "fk_tip", lambda q: np.eye(4))
    monkeypatch.setattr(manipulation.camera, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(manipulation.runtime, "ACT_TICK_S", 0.002)
    monkeypatch.setattr(manipulation.runtime, "HOLD_VIEW_GAP_S", 0.002)
    monkeypatch.setattr(manipulation.runtime, "ACT_STEP_TIMEOUT_S", 5.0)
    monkeypatch.setattr(manipulation.runtime, "TRIALS_PATH", tmp_path / "trials.jsonl")
    monkeypatch.setattr(manipulation.runtime, "_trials", None, raising=False)
    monkeypatch.setattr(manipulation.demos, "_demos_root", lambda: tmp_path / "demos")
    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=live_rgb,
        depth_m=np.full((H, W), 0.45, np.float32),
        intr=dict(INTR),
        keypoints={
            "mode": "features",
            "concept": "gamepad",
            "mask": box_mask(),
            "n_points": 50,
            "ref": {
                "object": "gamepad",
                "ok": True,
                "delta": pick_moved,
                "inliers": initial_find_inliers,
                "card_points": 400,
            },
        },
    )  # a live track since an earlier find: the act starts on it while the find is started over beside the arm
    with manipulation.runtime._state.lock:
        (
            manipulation.runtime._state.demo,
            manipulation.runtime._state.teach,
            manipulation.runtime._state.teach_job,
        ) = demo, teach, None
        manipulation.runtime._state.test = manipulation.runtime._Test(
            at="t", rgb=None, result={"ok": True, "live_mask": box_mask(), "delta_cam": np.eye(4)}
        )
        manipulation.runtime._state.located = {  # where an earlier find left the box, against the demo's view of it
            "box": {
                "object": "box",
                "ok": True,
                "delta": box_moved
                @ demo.objects["box"]["deltas"][manipulation.planning._pose_frame(demo, "box")],
                "mask": box,
                "inliers": 150,
                "card_points": 400,
                "view": [demo.name, int(demo.objects["box"]["frame"])],
            }
        }
        manipulation.runtime._state.track.on = True
        manipulation.runtime._state.track.last = {"state": "tracking"}
        manipulation.runtime._state.track.history = []
        manipulation.runtime._state.worker.pending.clear()
        manipulation.runtime._state.worker.jobs.clear()
        manipulation.runtime._state.act = manipulation.runtime._Act(
            on=True,
            speed=4.0,
            inject=inject,
            correct_hold=correct_hold,
            allow_grasp_prediction=allow_grasp_prediction,
        )
    manipulation.runtime._state.worker.proc = _FakeProc()
    views = {"demo": 0, "live": 0, "live_z": []}

    async def worker():
        """Answers the act's jobs as the real worker would, and keeps the live track certifying the gamepad."""
        taught = False
        while True:
            await asyncio.sleep(0.002)
            if finds_wait_for_the_arm and sim["first_move_at"] is None:
                continue
            with manipulation.runtime._state.lock:
                pending = list(manipulation.runtime._state.worker.pending)
                manipulation.runtime._state.worker.pending.clear()
                if taught and not covered_after_find:
                    manipulation.runtime._state.track.history.append((_time.time(), True, np.eye(4)))
                elif taught:
                    manipulation.runtime._state.track.last = {"state": "lost"}
            if box_moves_in_the_carry is not None and taught:  # the box's share of each tracked frame
                carried = (
                    sim["grip"] >= 82.0 and kin.forward_kinematics(sim["q"])[2, 3] > 0.05
                )  # closed, lifted
                sim["box_moved"] = sim.get("box_moved") or carried
                share = {"name": "box", "ok": True, "lost": False, "n_visible": 50, "n_tracks": 50}
                moved = box_moves_in_the_carry if sim["box_moved"] else np.eye(4)
                manipulation.tracking._apply_others({"others": [share], "other_delta_0": moved}, (H, W))
            if pending and sim["first_answer"] is None:
                sim["first_answer"] = _time.monotonic()
                sim["first_answer_wall"] = _time.time()
            for job_id in pending:
                job = manipulation.runtime._state.worker.jobs[job_id]
                found = {
                    "ok": True,
                    "ref_ok": True,
                    "ref_inliers": 150,
                    "ref_card_points": 400,
                    "ref_turn_deg": 0.0,
                }
                if job.kind == "teach":
                    sim["find_answers"].append(_time.monotonic())
                    job.result = {
                        **found,
                        "ref_inliers": fresh_find_inliers,
                        "n_points": 50,
                        "radius_mm": 40.0,
                        "shape_class": "box",
                        "yaw_observable": True,
                        "mask": box_mask(),
                        "uv": np.zeros((50, 2)),
                        "xyz": np.zeros((50, 3)),
                        "ref_delta": pick_moved,
                    }
                    manipulation.acquisition._apply_teach_result(job)
                    with manipulation.runtime._state.lock:
                        manipulation.runtime._state.test = manipulation.runtime._Test(
                            at="t",
                            rgb=None,
                            result={"ok": True, "live_mask": box_mask(), "delta_cam": np.eye(4)},
                        )
                    taught = True
                elif job.concept == "box":
                    job.result = {**found, "ref_delta": box_moved, "mask": box}
                    job.result["tracking"] = bool(job.extra.get("track"))
                elif (
                    int(round(float(job.rgb.mean()))) == 200
                ):  # the live camera: the gamepad in the gripper as the frame was taken, at the bottom or carried
                    tip = captured[int(job.rgb[0, 0, 0]) + 256 * int(job.rgb[0, 0, 1])]
                    views["live"] += 1
                    views["live_z"].append(float(tip[2, 3]))
                    hold = hold_now if tip[2, 3] < 0.04 else hold_carried
                    job.result = {**found, "ref_delta": tip @ hold, "mask": box_mask()}
                    if blind or (blind_carry and tip[2, 3] > 0.04):
                        job.result = {
                            "ok": True,
                            "ref_ok": False,
                            "ref_reason": "the live view does not match",
                        }
                else:  # a frame of the demo's own recording, which reads its frame number
                    views["demo"] += 1
                    k = round(float(job.rgb.mean()))
                    i = int(np.argmin(np.abs(demo.t - k / 15.0)))
                    weak = weak_carry and demo.t[i] > demo.t[PLACE_AT["lift"]]
                    job.result = {
                        **found,
                        "ref_inliers": 30 if weak else 150,
                        "ref_delta": demo.tips[i] @ hold_demo,
                        "mask": box_mask(),
                    }
                    if blind:
                        job.result = {
                            "ok": True,
                            "ref_ok": False,
                            "ref_reason": "the live view does not match",
                        }

    async def run():
        feed = asyncio.create_task(worker())
        try:
            await asyncio.wait_for(manipulation.execution._act_task(4.0), timeout=30.0)
        finally:
            feed.cancel()

    try:
        asyncio.run(run())
    finally:
        manipulation.runtime._state.worker.proc = None
    return (
        demo,
        sim,
        views,
        {
            "pick": pick_moved,
            "box": box_moved,
            "demo": hold_demo,
            "now": hold_now,
            "carried": hold_carried,
        },
    )


def _act_targets(tmp_path):
    row = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
    return json.loads((pathlib.Path(row["run"]) / "act.json").read_text())["targets"]


@pytest.mark.usefixtures("act_simulation")
def test_after_letting_go_the_act_goes_on_to_the_world_waypoints_where_the_demos_arm_was(
    tmp_path, monkeypatch
):
    """The place ends at the release; the lift away after it is a world waypoint: the demo's own fingertip pose,
    reached in a straight line from where the place left the arm, not moved with the box (which lies 30 mm off)."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi = MOTOR_NAMES.index("gripper")
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, world=True)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        kin = _TipKinematics()
        tips = np.stack([kin.forward_kinematics(q)[:3, 3] for q in sim["streamed"]])
        fix = m["demo"] @ np.linalg.inv(m["now"])
        placed = (m["box"] @ demo.tips[PLACE_AT["place_end"]] @ fix)[:3, 3]
        k = int(np.flatnonzero(np.linalg.norm(tips - placed, axis=1) <= 0.0005)[-1])
        assert sim["streamed"][k][gi] == 60.0, "the place ends let go"
        assert np.linalg.norm(tips[-1] - demo.tips[-1][:3, 3]) <= 0.0005, "where the demo's arm was"
        assert np.linalg.norm(tips[-1] - (m["box"] @ demo.tips[-1])[:3, 3]) > 0.02, "not moved with the box"
        line = tips[k:] - placed
        d = (tips[-1] - placed) / np.linalg.norm(tips[-1] - placed)
        assert np.abs(line - np.outer(line @ d, d)).max() <= 0.0005, "a straight line"
        assert sim["streamed"][-1][gi] == pytest.approx(60.0), (
            "withdrawal keeps the released opening instead of regrasping the object"
        )
        row = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
        assert row["result"] == "done"
    finally:
        _end_place_state()


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


@pytest.mark.parametrize(
    "scenario",
    [
        {},
        {"world": True},
        {"slip": True},
        {"weak_carry": True},
        {"wrist_deg": 80.0, "box_turn_deg": 14.5, "landing": "turn", "ranges": "wrist"},
        {"inject": {"at": "aim", "dx_mm": 8.0, "rz_deg": 5.0}},
        {"correct_hold": False},
        {"empty_grip": True},
        {"drop_in_lift": True},
    ],
    ids=[
        "place",
        "world-waypoint",
        "slip",
        "weak-carry",
        "landing-turn",
        "inject-aim",
        "uncorrected",
        "missed",
        "dropped-in-lift",
    ],
)
@pytest.mark.usefixtures("act_simulation")
def test_key_frame_execution_outcomes_across_grasp_and_placement_scenarios(tmp_path, monkeypatch, scenario):
    """Exercise supported scenarios against explicit outcomes, with no second runner as oracle."""
    if scenario.get("ranges") == "wrist":
        scenario = {**scenario, "ranges": (-WRIST_HI, WRIST_HI)}
    try:
        _demo, sim, _views, _m = _run_place_act(tmp_path, monkeypatch, **scenario)
        act = manipulation.runtime._state.act
        failed = any(scenario.get(k) for k in ("empty_grip", "drop_in_lift", "weak_carry", "inject"))
        assert act.ok is (not bool(failed)), act.reason
        if scenario.get("inject"):
            assert "no solved joint path" in act.reason and not sim["streamed"]
        else:
            assert sim["streamed"], "the simulated grasp must actually execute"
        if scenario.get("empty_grip") or scenario.get("drop_in_lift"):
            assert "grasp missed" in act.reason
        elif scenario.get("weak_carry"):
            assert "cannot verify" in act.reason and "gripper closed" in act.reason
            assert act.place["held_view_attempts"], "try the alternate held references after lifting"
        elif scenario.get("inject"):
            assert not sim["targets"], "failed preflight must not command the arm"
        else:
            assert act.place and act.place["grasp_held"]
            assert np.isfinite(np.asarray(sim["streamed"])).all()
    finally:
        _end_place_state()


@pytest.mark.parametrize("world", [False, True], ids=["place", "world-waypoint"])
def test_key_frames_marked_as_key_frames_act_as_the_marks_they_translate(tmp_path, monkeypatch, world):
    """The place demo's marks stored as the key frames they make: the key-frame act sends the arm the same walk targets
    and streamed samples, as nothing it does reads a mark's kind."""
    runs = {}
    for as_key in (False, True):
        try:
            demo, sim, _views, _m = _run_place_act(
                tmp_path / str(as_key), monkeypatch, world=world, as_key_frames=as_key
            )
            assert all(k["kind"] in ("key", "pose") for k in demo.keypoints) == as_key
            act = manipulation.runtime._state.act
            assert act.ok, act.reason
            runs[as_key] = (
                [
                    p
                    for k, p in enumerate(sim["targets"])
                    if k == 0 or not np.allclose(p, sim["targets"][k - 1])
                ],
                [q.copy() for q in sim["streamed"]],
                dict(act.place or {}),
            )
        finally:
            _end_place_state()
    (walks_m, streamed_m, place_m), (walks_k, streamed_k, place_k) = runs[False], runs[True]
    assert len(walks_k) == len(walks_m) and all(
        np.allclose(a, b) for a, b in zip(walks_k, walks_m, strict=True)
    )
    assert len(streamed_k) == len(streamed_m) and all(
        np.allclose(a, b) for a, b in zip(streamed_k, streamed_m, strict=True)
    )
    assert place_k["hold_used"] == place_m["hold_used"] and np.allclose(place_k["fix"], place_m["fix"])


@pytest.mark.usefixtures("act_simulation")
def test_the_act_sets_the_held_object_down_where_the_demo_did_on_the_target_however_it_is_gripped(
    tmp_path, monkeypatch
):
    """Pick, carry, place: the box is found where it is now; the demo's hold is measured on its still frames at the
    grip and while carried; the grasp streams through, its grip checked and the hold seen in the gripper on the fly
    while the arm stands still as the demo's did, and again at the pre-place; the place is corrected by the change,
    so the gamepad lands where the demo set it on the box although the gripper holds it 6 mm off."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi = MOTOR_NAMES.index("gripper")
    pre, end_at = PLACE_AT["preplace"], PLACE_AT["place_end"]
    try:
        demo, sim, views, m = _run_place_act(tmp_path, monkeypatch)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert (views["demo"], views["live"]) == (6, 10), (
            "three demo frames at the grip and three carried; 5 + 5 live"
        )
        assert all(z < 0.04 for z in views["live_z"][:5]) and all(z > 0.05 for z in views["live_z"][5:]), (
            "the first live views at the grip, before the lift; the rest at the pre-place"
        )
        grasp_sent = [w for w, stops in sim["sent_at"] if stops == 0]
        assert max(np.diff(grasp_sent)) < 0.1, "no pause in the grasp: the grip is read beside it"
        assert act.place["grasp_held"] is True and act.place["grasp_short"] == pytest.approx(3.0)
        assert [act.place[k]["n"] for k in ("demo_grip", "demo_carry", "live_grip", "live_place")] == [
            3,
            3,
            5,
            5,
        ]
        assert act.place["hold_used"] == "pre-place" and act.place["grip_vs_place_mm"] == pytest.approx(
            0.0, abs=0.01
        )
        assert act.place["shift_mm"] == pytest.approx(6.0, abs=0.01), (
            "the correction is the change in the hold"
        )
        fix = m["demo"] @ np.linalg.inv(m["now"])
        carried = [q for q, (_w, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops == 1]
        assert (
            np.linalg.norm(
                _TipKinematics().forward_kinematics(carried[-1])[:3, 3] - (m["box"] @ demo.tips[pre])[:3, 3]
            )
            <= 0.0005
        ), "carried onto the moved box, on the joints planned for it"
        assert {q[gi] for q in carried} == {85.0}, (
            "the carry keeps the grasp's closing, not the fingers' reading"
        )
        placing = [q for q, (_w, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops == 2]
        assert (
            min(
                core.pose_residual(_TipKinematics().forward_kinematics(q), m["box"] @ demo.tips[pre] @ fix)[0]
                for q in placing
            )
            < 0.0005
        ), "then through the corrected pre-place on the checked joint path"
        assert placing[0][gi] == 85.0, "the correction keeps the closing, not the fingers' reading"
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (m["box"] @ demo.tips[end_at] @ fix)[:3, 3]) <= 0.0005
        # The point: held as it is now, the gamepad ends where the demo set it down on the box.
        assert (
            np.linalg.norm((end @ m["now"])[:3, 3] - (m["box"] @ demo.tips[end_at] @ m["demo"])[:3, 3])
            <= 0.0005
        )
        assert sim["streamed"][-1][gi] == 60.0, "let go, as the demo did"
        row = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
        assert row["place"]["hold_used"] == "pre-place" and row["place"]["shift_mm"] == pytest.approx(
            6.0, abs=0.01
        )
        # The act's recording says what the place was aimed by: the target's find and the motion it gave.
        recording = json.loads((pathlib.Path(row["run"]) / "act.json").read_text())
        target = recording["target"]
        assert target["object"] == "box" and target["inliers"] == 150 and target["strong"] is True
        assert np.allclose(target["motion_base"], m["box"]) and target["pose_frame"] == 2
        place = recording["place"]
        assert place["hold_used"] == "pre-place" and np.allclose(place["fix"], fix)
        assert np.allclose(place["demo_carry"]["hold"], m["demo"])
        assert np.allclose(place["live_place"]["hold"], m["now"]), (
            "the act record retains both measured holds, so object placement can be reconstructed "
            "without the trials index or rerunning perception"
        )
    finally:
        _end_place_state()


def test_a_lift_after_the_grasp_raises_the_held_object_straight_up_before_the_carry_crosses_over(
    tmp_path, monkeypatch
):
    """The carry begins straight up from where the grasp left the arm, by the lift's height, the grasp's closing kept,
    before its line crosses to the box: the line from the pick to the pre-place knocked the target aside on the rig.
    The place is corrected for the hold as before, so the gamepad still lands where the demo set it."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi, kin = MOTOR_NAMES.index("gripper"), _TipKinematics()
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, lift_mm=30.0)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        sent = list(zip(sim["streamed"], sim["sent_at"], strict=True))
        grasp_end = kin.forward_kinematics([q for q, (_w, s) in sent if s == 0][-1])[:3, 3]
        carried = [q for q, (_w, s) in sent if s == 1]
        tips = np.stack([kin.forward_kinematics(q)[:3, 3] for q in carried])
        off = np.any(
            np.abs(tips[:, :2] - grasp_end[:2]) > 0.0002, axis=1
        )  # the joints solve to a fraction of a mm
        n_up = int(np.argmax(off))
        assert n_up > 1 and np.all(np.diff(tips[:n_up, 2]) > 0), (
            "first straight up, above where the grasp ended"
        )
        assert tips[n_up - 1, 2] - grasp_end[2] == pytest.approx(0.030, abs=0.0005), "by the lift's height"
        assert {q[gi] for q in carried[:n_up]} == {85.0}, "the grasp's closing kept"
        assert tips[n_up:, 2].max() <= tips[n_up - 1, 2] + 1e-9, (
            "the line to the pre-place goes on from up there"
        )
        end = kin.forward_kinematics(sim["streamed"][-1])
        assert (
            np.linalg.norm(
                (end @ m["now"])[:3, 3] - (m["box"] @ demo.tips[PLACE_AT["place_end"]] @ m["demo"])[:3, 3]
            )
            <= 0.0005
        ), "the gamepad ends where the demo set it down on the box"
    finally:
        _end_place_state()


def test_a_lift_before_the_grasp_is_walked_straight_up_from_where_the_arm_stands(tmp_path, monkeypatch):
    """A checked pre-grasp lift rises from the arm's arrival pose before the next pre-grasp descends."""
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, lift_before_mm=25.0)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        pre = m["pick"] @ demo.tips[PLACE_AT["pregrasp"]]
        targets = _act_targets(tmp_path)
        first = [np.asarray(x["pose"]) for x in targets if x["step"] == "key frame 1" and x.get("pose")]
        lift = [np.asarray(x["pose"]) for x in targets if x["step"] == "lift 1" and x.get("pose")]
        second = [np.asarray(x["pose"]) for x in targets if x["step"] == "key frame 2" and x.get("pose")]
        assert first and lift and second and sim["streamed"]
        assert np.allclose(first[-1], pre)
        up = lift[-1]
        near = manipulation.runtime.ACT_ARRIVE_M
        assert np.allclose(up[:3, :3], pre[:3, :3]) and np.all(np.abs(up[:2, 3] - pre[:2, 3]) <= near), (
            "straight up"
        )
        assert 0.025 - near <= up[2, 3] - pre[2, 3] <= 0.025 + near, "by the lift's height"
        assert np.allclose(second[-1], m["pick"] @ demo.tips[20]), "then the second pre-grasp"
    finally:
        _end_place_state()


WRIST_HI = np.array(
    [np.nan, np.nan, np.nan, np.nan, 93.3, np.nan, np.nan]
)  # the servos' ranges: only the wrist's bites


@pytest.mark.usefixtures("act_simulation")
def test_a_place_landed_as_shown_past_a_servos_range_is_refused_before_the_arm_moves(tmp_path, monkeypatch):
    """The demo set the gamepad down with the wrist at 80 deg; the box has since turned 40 deg about its middle. Landed
    exactly as shown, the wrist would need 120, past its servo's 93: the act says so before anything moves."""
    try:
        _demo, sim, _views, _m = _run_place_act(
            tmp_path, monkeypatch, wrist_deg=80.0, box_turn_deg=40.0, ranges=(-WRIST_HI, WRIST_HI)
        )
        act = manipulation.runtime._state.act
        assert not act.ok and "drives wrist_flex toward 120 deg, past its servo's range" in act.reason, (
            act.reason
        )
        assert sim["streamed"] == [] and sim["targets"] == [], "refused before the arm moved"
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_a_place_needing_a_joint_just_past_its_servos_range_is_made_with_the_joint_held_there(
    tmp_path, monkeypatch
):
    """An act of 2026-10-08 stopped after its grasp: its place needed wrist_flex at -94 deg against its servo's -93.
    Held at its range, the wrist still sets the gamepad down within the reach tolerance: the act goes on, and never
    asks the wrist past its servo."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    wf = MOTOR_NAMES.index("wrist_flex")
    try:
        _demo, sim, _views, _m = _run_place_act(
            tmp_path, monkeypatch, wrist_deg=80.0, box_turn_deg=14.5, ranges=(-WRIST_HI, WRIST_HI)
        )
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert max(abs(q[wf]) for q in sim["streamed"]) == pytest.approx(93.3), "held at its servo's range"
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_a_place_free_to_turn_about_the_box_lands_where_the_arm_stays_nearest_the_demo(tmp_path, monkeypatch):
    """The same, the place free to land turned any way about the box's middle: the act takes a turn the wrist reaches,
    nearer the demo's wrist than landing as shown, and the carry, the hold's correction and the place all aim by it.
    (Which turn is nearest the demo's joints is :func:`core.rank_landings`' to say, tested on its own.)"""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    wf = MOTOR_NAMES.index("wrist_flex")
    try:
        demo, sim, _views, m = _run_place_act(
            tmp_path,
            monkeypatch,
            wrist_deg=80.0,
            box_turn_deg=40.0,
            landing="turn",
            ranges=(-WRIST_HI, WRIST_HI),
        )
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        row = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
        target = json.loads((pathlib.Path(row["run"]) / "act.json").read_text())["target"]
        turn = target["landing"]["turn_deg"]
        assert target["landing"]["rule"] == "turn" and turn is not None and target["landing"]["reachable"] > 1
        assert all(abs(q[wf]) <= 93.3 for q in sim["streamed"]), "no joint asked past its servo"
        carried = [q for q, (_w, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops == 1]
        assert abs(carried[-1][wf] - 80.0) < 40.0, (
            "at the pre-place, nearer the demo's wrist than as shown (120)"
        )
        landed = core.landed(
            m["box"], np.asarray(target["landing_centre"]), turn, np.asarray(target["landing_axis"])
        )
        fix = m["demo"] @ np.linalg.inv(m["now"])
        placing = [q for q, (_w, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops == 2]
        assert (
            min(
                core.pose_residual(
                    _TipYawKinematics().forward_kinematics(q), landed @ demo.tips[PLACE_AT["preplace"]] @ fix
                )[0]
                for q in placing
            )
            < 0.0005
        ), "the hold's correction streams through the checked landing"
        end = _TipYawKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (landed @ demo.tips[PLACE_AT["place_end"]] @ fix)[:3, 3]) <= 0.0005
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize("landing", ["exact", "turn"])
def test_a_changed_hold_is_planned_before_moving_and_can_choose_an_allowed_landing_again(
    tmp_path, monkeypatch, landing
):
    """The carry fits the wrist's range, but the changed hold needs another 40 degrees. Refuse before moving the
    correction when that orientation is required; otherwise find a permitted landing and stream its checked path.
    The rig used to walk blindly into its wrist limit here, timing out before it planned the corrected place."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi, wf = MOTOR_NAMES.index("gripper"), MOTOR_NAMES.index("wrist_flex")
    try:
        demo, sim, _views, m = _run_place_act(
            tmp_path,
            monkeypatch,
            wrist_deg=80.0,
            held_turn_deg=-40.0,
            landing=landing,
            ranges=(-WRIST_HI, WRIST_HI),
        )
        act = manipulation.runtime._state.act
        row = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
        record = json.loads((pathlib.Path(row["run"]) / "act.json").read_text())
        plans = [p for p in record["plans"] if p["stage"].startswith("hold correction")]
        assert plans and not plans[0]["ok"], "reject the now unreachable correction before commanding it"
        assert all("corrected for the hold" not in t["step"] for t in record["targets"]), (
            "no unplanned Cartesian correction target was sent"
        )
        placed = [q for q, (_w, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops >= 2]
        if landing == "exact":
            assert not act.ok and "wrist_flex" in act.reason
            assert not placed, "the arm must hold still when no corrected path is feasible"
            assert sim["streamed"][-1][gi] == 85.0, "a refusal must not open the grasp"
        else:
            assert act.ok, act.reason
            assert len(plans) == 2 and plans[1]["ok"]
            turn = plans[1]["landing"]["turn_deg"]
            assert turn != record["target"]["landing"]["turn_deg"]
            assert all(abs(q[wf]) <= WRIST_HI[wf] for q in sim["streamed"])
            assert placed[0][gi] == 85.0 and placed[-1][gi] == 60.0
            landed = core.landed(
                m["box"],
                np.asarray(record["target"]["landing_centre"]),
                turn,
                np.asarray(record["target"]["landing_axis"]),
            )
            actual = _TipYawKinematics().forward_kinematics(placed[-1]) @ m["now"]
            intended = landed @ demo.tips[PLACE_AT["place_end"]] @ m["demo"]
            metres, deg = core.pose_residual(actual, intended)
            assert metres < core.ACT_REACH_TOL_M and deg < core.ACT_REACH_TOL_DEG, (
                "the measured held object still satisfies the demo's relation at the permitted landing"
            )
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_the_arm_must_settle_on_the_checked_hold_correction_before_it_descends_or_releases(
    tmp_path, monkeypatch
):
    """A reachable plan is not an observed arrival: the shoulder jams during the correction, so the grasp must stay
    closed at the hover. Streaming straight through the checked plan would release above the wrong location."""
    try:
        _demo, sim, _views, _m = _run_place_act(tmp_path, monkeypatch, correction_sag_deg=10.0)
        act = manipulation.runtime._state.act
        assert not act.ok and "stopped 10 deg short" in act.reason
        placed = [q for q, (_w, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops >= 2]
        assert placed and all(q[-1] == 85.0 for q in placed), "the taught release was never commanded"
        assert all(_TipKinematics().forward_kinematics(q)[2, 3] > 0.05 for q in placed), (
            "the descent must wait for the loaded arm to reach the corrected hover"
        )
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_a_landing_that_no_longer_plans_after_the_grasp_is_chosen_again(tmp_path, monkeypatch):
    """The act of 2026-10-10 11:15 stopped after its grasp, its place 5 mm out of reach at the turn its start had taken,
    though other turns planned. Here the box turns 40 deg about its middle while the gamepad is lifted: landed at the
    start's turn the wrist would need 40 deg more, past its servo's 93; the carry takes the turn that lands as the
    start's did on the box as it lay then, the place aims by it, and the act's record keeps both carry plans."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    wf = MOTOR_NAMES.index("wrist_flex")
    probe, _kin, _box = _place_demo(tmp_path / "probe", time.time(), wrist_deg=80.0)
    found_at = np.array(
        [0.030, 0.010, 0.0]
    )  # where the harness's earlier find has the box: moved, not turned
    axis, centre = manipulation.planning._landing_axis(probe, "box", np.eye(4))
    turned = core.turn_about(centre + found_at, 40.0, axis)
    try:
        demo, sim, _views, m = _run_place_act(
            tmp_path,
            monkeypatch,
            wrist_deg=80.0,
            landing="turn",
            ranges=(-WRIST_HI, WRIST_HI),
            box_moves_in_the_carry=turned,
        )
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert np.allclose(m["box"][:3, 3], found_at)
        row = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
        record = json.loads((pathlib.Path(row["run"]) / "act.json").read_text())
        kept = record["target"]["landing"]["turn_deg"]
        carries = [p for p in record["plans"] if p["stage"].startswith("carry")]
        assert [p["ok"] for p in carries] == [False, True], carries
        assert (
            "drives wrist_flex toward" in carries[0]["reason"]
            and "past its servo's range" in carries[0]["reason"]
        )
        again = carries[1]["landing"]["turn_deg"]
        assert again == (kept - 40.0) % 360.0, "turned back by the box's turn"
        assert all(abs(q[wf]) <= 93.3 for q in sim["streamed"]), "no joint asked past its servo"
        landed = core.landed(
            turned @ m["box"],
            np.asarray(record["target"]["landing_centre"]),
            again,
            np.asarray(record["target"]["landing_axis"]),
        )
        fix = m["demo"] @ np.linalg.inv(m["now"])
        end = _TipYawKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (landed @ demo.tips[PLACE_AT["place_end"]] @ fix)[:3, 3]) <= 0.0005
    finally:
        _end_place_state()


def test_the_reach_preview_searches_a_landing_once_per_find_and_never_during_an_act(
    client, tmp_path, monkeypatch
):
    """The editor polls the reach every 3 s. Searching the landing's turns each time held the arm's loop up for a
    second, every time: the preview searches once per find of the target and plans that turn after; during an act it
    does not plan at all."""
    from lerobot.gui.api import jog
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    demo, kin, box = _place_demo(tmp_path, time.time(), wrist_deg=80.0)
    demo.landing = "turn"
    centre = manipulation.planning._landing_centre(demo, "box", np.eye(4))
    q0 = np.array([80.0, -20.0, 90.0, 0.0, 0.0, 0.0, 60.0])
    searches = []
    rank = core.rank_landings
    monkeypatch.setattr(core, "rank_landings", lambda *a, **k: searches.append(1) or rank(*a, **k))
    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    monkeypatch.setattr(
        jog,
        "current_tip_and_anchor",
        lambda: (kin.forward_kinematics(q0), np.eye(4), dict(zip(MOTOR_NAMES, q0, strict=True))),
    )
    monkeypatch.setattr(jog, "servo_ranges", lambda: (-WRIST_HI, WRIST_HI))
    monkeypatch.setattr(jog, "walk_limits", lambda: (0.04, np.radians(30)))
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    monkeypatch.setattr(manipulation.camera, "_t_base_cam", lambda: np.eye(4))
    moved = np.eye(4)
    moved[:3, 3] = [0.030, 0.010, 0.0]
    moved = moved @ core.turn_about(centre, 40.0)
    f = manipulation.planning._pose_frame(demo, "box")

    def found(at):
        return {
            "object": "box",
            "ok": True,
            "delta": moved @ demo.objects["box"]["deltas"][f],
            "mask": box,
            "inliers": 150,
            "card_points": 400,
            "at": at,
            "view": [demo.name, int(demo.objects["box"]["frame"])],
        }

    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=np.zeros((H, W, 3), np.uint8),
        depth_m=np.full((H, W), 0.45, np.float32),
        intr=dict(INTR),
        keypoints={"mode": "features", "ref": {"object": "gamepad", "ok": True, "delta": np.eye(4)}},
    )
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo, manipulation.runtime._state.teach = demo, teach
            manipulation.runtime._state.test = manipulation.runtime._Test(
                at="t", rgb=None, result={"ok": True, "delta_cam": np.eye(4)}
            )
            manipulation.runtime._state.located = {"box": found(1.0)}
            manipulation.runtime._state.reach_landing = {}
        first = client.get("/api/pregrasp/demo/reach").json()
        second = client.get("/api/pregrasp/demo/reach").json()
        assert len(searches) == 1, "the second poll plans the turn the first one found"
        assert first["landing"]["turn_deg"] is not None and second["landing"] == first["landing"]
        assert second["ok"] == first["ok"] and second["marks"] == first["marks"]
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.located = {"box": found(2.0)}  # found again: searched again
        client.get("/api/pregrasp/demo/reach")
        assert len(searches) == 2
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.act = manipulation.runtime._Act(on=True)
        r = client.get("/api/pregrasp/demo/reach")
        assert r.status_code == 409 and "act" in r.json()["detail"] and len(searches) == 2
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = (
                manipulation.runtime._state.test
            ) = None
            manipulation.runtime._state.located = {}
            manipulation.runtime._state.reach_landing = {}
            manipulation.runtime._state.act = manipulation.runtime._Act()


@pytest.mark.usefixtures("act_simulation")
def test_a_shift_during_the_lift_is_caught_at_the_pre_place(tmp_path, monkeypatch):
    """The grip measured before the lift is not the grip after it when the lift moves the object in the fingers: the
    pre-place measurement, seen well on both sides, is the one the place uses, and the shift is reported."""
    end_at = PLACE_AT["place_end"]
    try:
        demo, sim, views, m = _run_place_act(tmp_path, monkeypatch, slip=True)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert act.place["hold_used"] == "pre-place"
        assert act.place["grip_vs_place_mm"] == pytest.approx(6.0, abs=0.01), "the lift moved it 6 mm"
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert (
            np.linalg.norm((end @ m["carried"])[:3, 3] - (m["box"] @ demo.tips[end_at] @ m["demo"])[:3, 3])
            <= 0.0005
        ), "placed for how it sits after the lift"
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize("blind", [False, True])
def test_missing_demo_carry_reference_stops_before_the_grasp(tmp_path, monkeypatch, blind):
    """An unusable saved reference is known before picking; do not strand a payload at the carry check."""
    from lerobot.gui.api.manipulation import held_views

    monkeypatch.setattr(held_views, "references", lambda *a: [])
    try:
        _demo, sim, _views, _m = _run_place_act(tmp_path, monkeypatch, weak_carry=not blind, blind=blind)
        act = manipulation.runtime._state.act
        assert not act.ok and "demo carry reference before grasp" in act.reason
        assert act.place["reference_preflight"] == "failed before grasp"
        assert not [q for q in sim["streamed"] if q[-1] >= 82.0], (
            "empty-arm approach may stream, but grasp and loaded motion wait for reference validation"
        )
        assert sim["grip"] < 82.0, "leave the empty gripper open"
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_missing_carried_hold_stops_closed_even_when_a_grasp_estimate_exists(tmp_path, monkeypatch):
    """Neither an earlier visual grip nor an encoder grasp estimate can certify the pose after a lift."""
    try:
        _demo, sim, _views, _m = _run_place_act(
            tmp_path,
            monkeypatch,
            blind_carry=True,
        )
        act = manipulation.runtime._state.act
        assert not act.ok and "cannot verify" in act.reason and "gripper closed" in act.reason
        assert act.place["hold_used"] == "unverified"
        assert act.place["grasp_estimate_available"]
        assert act.place["grip_measurement_available"], "even a visual grip is stale after carrying"
        assert not [q for q, (_, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops >= 2], (
            "no correction, descent or release follows a failed verification"
        )
        assert sim["streamed"][-1][-1] == 85.0
        assert _TipKinematics().forward_kinematics(sim["streamed"][-1])[2, 3] > 0.05
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize("missing", ["demo", "live", "both"])
def test_explicit_grasp_prediction_places_without_claiming_a_carried_measurement(
    tmp_path, monkeypatch, missing
):
    """The opted-in prediction assumes the grasp persisted, and remains identified as such in saved evidence."""
    try:
        demo, sim, _views, m = _run_place_act(
            tmp_path,
            monkeypatch,
            weak_carry=missing == "demo",
            blind_carry=missing == "live",
            blind=missing == "both",
            allow_grasp_prediction=True,
        )
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert act.place["hold_evidence"] == "predicted" and act.place["fallback_reason"]
        assert act.place["hold_used"] == ("grasp prediction" if missing == "both" else "grip prediction")
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        expected = m["box"] @ demo.tips[PLACE_AT["place_end"]] @ np.asarray(act.place["fix"])
        assert core.pose_residual(end, expected)[0] <= 0.0005
        row = manipulation.recording._load_trials()[-1]
        assert row["allow_grasp_prediction"] and row["place"]["hold_evidence"] == "predicted"
        meta = json.loads((pathlib.Path(row["run"]) / "act.json").read_text())
        assert meta["allow_grasp_prediction"] and meta["hold_selection"]["evidence"] == "predicted"
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_strict_visual_hold_measures_after_slip(tmp_path, monkeypatch):
    try:
        _demo, _sim, _views, _m = _run_place_act(
            tmp_path, monkeypatch, slip=True, allow_grasp_prediction=False
        )
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert act.place["hold_used"] == "pre-place" and act.place["hold_evidence"] == "measured"
        assert act.place["grip_vs_place_mm"] == pytest.approx(6.0, abs=0.01)
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_prediction_permission_does_not_invent_a_missing_grasp_attachment(tmp_path, monkeypatch):
    monkeypatch.setattr(manipulation.planning, "_grasp_pose_fix", lambda *a: None)
    try:
        _demo, sim, _views, _m = _run_place_act(
            tmp_path, monkeypatch, blind=True, allow_grasp_prediction=True
        )
        act = manipulation.runtime._state.act
        assert not act.ok and "cannot verify" in act.reason
        assert act.place["hold_used"] == "unverified" and sim["streamed"][-1][-1] == 85.0
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize("slow_grip", [False, True])
def test_the_act_stops_before_carrying_when_the_gripper_closed_on_nothing(tmp_path, monkeypatch, slow_grip):
    """The grasp check runs beside the stream, from the demo's firm grip: once the gripper's reading stops, an
    empty one is at its command, and the stream halts there. A gripper still closing reads short of its command with
    nothing in it, so the check waits for it to stop: 0.3 s here, longer than this test's grasp takes to stream at
    4x, so that act lifts and the check after the lift stops it (at 1x the demo stood still 0.57 s after its grip)."""
    try:
        demo, sim, views, m = _run_place_act(tmp_path, monkeypatch, empty_grip=True, slow_grip=slow_grip)
        act = manipulation.runtime._state.act
        assert not act.ok and act.reason.startswith(
            "the grasp missed: the gripper closed to 85.0, 0.0 short"
        ), act.reason
        assert not any(np.allclose(t, m["box"] @ demo.tips[PLACE_AT["preplace"]]) for t in sim["targets"]), (
            "nothing carried"
        )
        assert views["live"] == 0
        if not slow_grip:
            assert _TipKinematics().forward_kinematics(sim["streamed"][-1])[2, 3] < 0.025, (
                "halted at the grip, before the lift"
            )
    finally:
        _end_place_state()


def test_the_hold_is_measured_at_the_firm_grip_and_while_carried_on_still_frames(tmp_path):
    """Two windows of the demo show the hold. At the grip: from the moment the gripper stopped closing short of its
    command until the arm moves again, the object still where it lay; this is not the grasp's end mark, which follows
    the lift. While carried: from the grasp end to the last pre-place, and on through a pause marked at its start,
    until the arm moves again or the gripper starts to open."""
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    assert manipulation.holds._firm_grip(demo) == PLACE_AT["grip"], (
        "the reading stops 3 short of the command at 0.93 s"
    )
    assert [f for f, _ in manipulation.holds._still_held_frames(demo, "gamepad", "grip")] == [22, 19, 16], (
        "from 0.93 s until the arm moves again at 1.47 s, before the lift"
    )
    assert [f for f, _ in manipulation.holds._still_held_frames(demo, "gamepad", "carry")] == [38, 35, 32]
    demo.keypoints[2] = {
        "t": float(demo.t[59]),
        "kind": "preplace",
        "object": "box",
    }  # the pause's first sample
    assert [f for f, _ in manipulation.holds._still_held_frames(demo, "gamepad", "carry")] == [38, 35, 32], (
        "the pause runs on after the mark until the arm moves again"
    )
    demo.q_cmd[65:, -1] = (
        60.0  # let go in the middle of the pause, at 2.17 s: no longer the same hold from there
    )
    assert [f for f, _ in manipulation.holds._still_held_frames(demo, "gamepad", "carry")] == [32], (
        "frame 32 (2.13 s) is the last before letting go, and 30 and 31 too close to it to be other views"
    )


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


@pytest.mark.parametrize("occluder", [0.0, 0.4, 0.65, np.nan])
def test_a_partial_held_view_clicks_visible_surface_instead_of_the_covered_middle(occluder):
    """A large foreground wrist hid the predicted centre while a narrow end remained usable. Padding the projected
    surface must not manufacture an eligible pixel on that wrist or on invalid/background depth."""
    intr = {"fx": 200.0, "fy": 200.0, "cx": 40.0, "cy": 30.0}
    yy, xx = np.mgrid[12:49:2, 12:69:2]
    points = np.c_[((xx.ravel() - 40) * 0.5 / 200), ((yy.ravel() - 30) * 0.5 / 200), np.full(xx.size, 0.5)]
    depth = np.full((60, 80), occluder, dtype=np.float32)
    depth[18:43, 58:67] = 0.505  # a visible end, including a small error in the hold prediction
    click = manipulation.holds._held_click(points, np.eye(4), np.eye(4), intr, depth, 0.02)
    assert click is not None and 58 <= click[0] < 67 and 18 <= click[1] < 43
    depth[18:43, 58:67] = occluder
    assert manipulation.holds._held_click(points, np.eye(4), np.eye(4), intr, depth, 0.02) is None
    # Points behind the camera and outside the image never supply support, even with matching depth.
    depth[:] = 0.5
    absent = np.array([[0.0, 0.0, -0.5], [1.0, 0.0, 0.5], [np.nan, 0.0, 0.5]])
    assert manipulation.holds._held_click(absent, np.eye(4), np.eye(4), intr, depth, 0.02) is None


def test_the_held_object_is_clicked_where_the_live_track_has_it_when_the_predicted_middle_fails(
    tmp_path, monkeypatch
):
    """The click on the held object goes where the demo's hold puts its middle; a finger there makes the find fail, so
    the next views alternate with where the live track still has the object."""
    import asyncio

    from lerobot.gui.api import jog

    demo, kin, _box = _place_demo(tmp_path, time.time())
    hold = np.eye(4)
    hold[:3, 3] = [0.0, 0.0, -0.02]
    tracked = np.zeros((H, W), dtype=bool)
    tracked[100:140, 700:780] = True  # where the live track has the gamepad, away from the predicted middle
    tip = kin.forward_kinematics(np.array([210.0, 10.0, 60.0, 0, 0, 0, 82.0]))
    clicks = []

    async def locate(obj, rgb, depth, intr, click, stopped, **_):
        clicks.append(tuple(click))
        if not tracked[click[1], click[0]]:
            return {"object": obj, "ok": False, "delta": None, "reason": "the live view does not match"}
        return {
            "object": obj,
            "ok": True,
            "delta": tip @ hold,
            "inliers": 150,
            "card_points": 400,
            "reason": "",
        }

    # The reference surface is at 0.45 m, carried 0.06 - 0.02 m further away. Depth supports the prediction,
    # but appearance rejects it: the separately tracked region is the successful alternative.
    blank = (np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.49, np.float32), dict(INTR))
    monkeypatch.setattr(manipulation.acquisition, "_locate", locate)
    monkeypatch.setattr(manipulation.camera, "_frame", lambda: _async(blank))
    monkeypatch.setattr(jog, "current_tip_and_anchor", lambda: (tip, np.eye(4), {}))
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
            manipulation.runtime._state.test = manipulation.runtime._Test(
                at="t", rgb=None, result={"ok": True, "live_mask": tracked}
            )
            manipulation.runtime._state.track.last = {"state": "tracking"}
        avg, problem = asyncio.run(
            manipulation.holds._live_hold(demo, "gamepad", hold, np.eye(4), lambda: False)
        )
        assert problem == "" and avg["n"] == 5 and np.allclose(avg["hold"], hold), (problem, clicks)
        assert clicks[1:] == [clicks[1]] * 5 and clicks[0] != clicks[1], (
            "the predicted middle, then the track's"
        )
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.track.last = {
                "state": "lost"
            }  # a lost track's mask is where it last was, not now
        clicks.clear()
        avg, problem = asyncio.run(
            manipulation.holds._live_hold(demo, "gamepad", hold, np.eye(4), lambda: False)
        )
        assert avg is None and "0 of 7 views" in problem and len(set(clicks)) == 1, "nowhere else to click"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.test = None
            manipulation.runtime._state.track.last = {}


@pytest.mark.parametrize("quality", ["strong", "weak", "disagree", "slow"])
def test_a_lost_held_track_tries_fresh_visual_matches_within_a_bounded_wait(monkeypatch, quality):
    """An empty segmentation mask can latch a track lost despite visible surface. Reference matching may recover
    the hold in the same coordinates, but must still reject weak, inconsistent or insufficient fresh views."""
    import asyncio

    from scipy.spatial.transform import Rotation

    points = np.array([[0.0, 0.0, 0.5], [0.01, 0.0, 0.5], [0.0, 0.01, 0.5]])
    tip = np.eye(4)
    tip[:3, 3] = [0.1, -0.05, 0.03]
    t_bc = np.eye(4)
    t_bc[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    predicted = np.linalg.inv(tip)
    measured = predicted.copy()
    measured[0, 3] += 0.003  # visual correction, not the predicted hold echoed back
    monkeypatch.setattr(manipulation.planning, "_object_points", lambda *a: points @ t_bc[:3, :3].T)

    async def lost(*args):
        return None, "empty segmentation supplied no fit"

    async def view(_obj=None):
        return np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.5), INTR, tip, None

    calls, clock = [], [0.0]

    async def locate(obj, rgb, depth, intr, click, stopped, **kw):
        calls.append(click)
        clock[0] += 0.6 if quality == "slow" else 0.01
        result = measured.copy()
        if quality == "disagree":
            result[0, 3] += 0.02 * len(calls)
        return {
            "object": obj,
            "ok": not stopped(),
            "delta": np.linalg.inv(t_bc) @ tip @ result @ t_bc,
            "inliers": 10 if quality == "weak" else 150,
            "card_points": 400,
        }

    monkeypatch.setattr(manipulation.holds, "_live_hold_by_track", lost)
    monkeypatch.setattr(manipulation.holds, "_held_view", view)
    monkeypatch.setattr(manipulation.acquisition, "_locate", locate)
    monkeypatch.setattr(manipulation.runtime, "HOLD_TRACK_WAIT_S", 1.0)
    monkeypatch.setattr(manipulation.holds.time, "monotonic", lambda: clock[0])
    # The prediction and result both transform the BASE-frame reference, as in the real caller.
    hold, problem = asyncio.run(
        manipulation.holds._live_hold(None, "part", predicted, t_bc, lambda: False, method="track")
    )
    assert calls, "a lost track must not bypass the independent current-frame measurement"
    if quality == "strong":
        assert not problem and hold["method"] == "find" and hold["views"] == core.HOLD_VIEWS
        np.testing.assert_allclose(hold["hold"], measured, atol=1e-9)
    else:
        assert hold is None and "empty segmentation" in problem
    if quality == "slow":
        assert len(calls) == 2, "the shared wait budget must stop retries, including an in-flight find"


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


def test_the_grasp_pose_corrects_for_where_the_arm_landed():
    """The arm landed 3 mm and 4 deg off its planned grasp: the object then sits that much differently in the
    gripper, and the place corrected by the grasp pose puts it where the demo did."""
    from scipy.spatial.transform import Rotation

    demo_tip, motion, miss = np.eye(4), np.eye(4), np.eye(4)
    demo_tip[:3, 3] = [0.2, 0.05, 0.02]
    motion[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    motion[:3, 3] = [0.01, -0.03, 0.0]
    miss[:3, :3] = Rotation.from_euler("y", 4, degrees=True).as_matrix()
    miss[:3, 3] = [0.003, 0.0, 0.0]
    live_tip = motion @ demo_tip @ miss
    fix = manipulation.planning._grasp_pose_fix(demo_tip, motion, live_tip)
    assert np.allclose(fix, miss)
    hold_demo = np.eye(4)
    hold_demo[:3, 3] = [0.0, 0.0, -0.02]
    object_now = motion @ demo_tip @ hold_demo  # where it lay, unmoved by the closing
    hold_now = np.linalg.inv(live_tip) @ object_now
    target, place_tip = np.eye(4), np.eye(4)
    target[:3, 3] = [0.05, 0.02, 0.0]
    place_tip[:3, 3] = [0.3, 0.1, 0.06]
    assert np.allclose(target @ place_tip @ fix @ hold_now, target @ place_tip @ hold_demo)


def _spy_injection(monkeypatch) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Record what the act gives :func:`manipulation.planning._grasp_pose_fix` (demo tip, object motion, live tip) and the pivots
    it turns an injected error about."""
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pivots: list[np.ndarray] = []
    fix, transform = manipulation.planning._grasp_pose_fix, manipulation.execution._inject_transform

    def spy_fix(demo_tip, motion, live_tip):
        calls.append((np.array(demo_tip), np.array(motion), np.array(live_tip)))
        return fix(demo_tip, motion, live_tip)

    def spy_transform(inject, pivot):
        pivots.append(np.array(pivot)[:3])
        return transform(inject, pivot)

    monkeypatch.setattr(manipulation.planning, "_grasp_pose_fix", spy_fix)
    monkeypatch.setattr(manipulation.execution, "_inject_transform", spy_transform)
    return calls, pivots


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize("sag", [3.4, 8.0])
def test_an_arm_held_short_under_load_settles_once_it_stops(tmp_path, monkeypatch, sag):
    """The act of 2026-10-07 16:33: after the lift the shoulder sat 3.4 deg short of its last target with the gamepad
    in the gripper, and the settle waited 20 s for the 2 deg it wanted. An arm that has stopped within ACT_STALL_DEG
    has arrived; one stopped further off says so at once instead of waiting."""
    import time as _time

    t0 = _time.monotonic()
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, sag_deg=sag)
        act = manipulation.runtime._state.act
        if sag <= manipulation.runtime.ACT_STALL_DEG:
            assert act.ok, act.reason
            assert act.place["settled_short_deg"] == pytest.approx(sag, abs=0.01)
        else:
            assert not act.ok and act.reason == f"the end: the arm stopped {sag:.0f} deg short of it", (
                act.reason
            )
            assert _time.monotonic() - t0 < manipulation.runtime.ACT_STEP_TIMEOUT_S, "not after the timeout"
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_an_object_the_arm_covers_after_its_find_is_grasped_where_the_find_put_it(tmp_path, monkeypatch):
    """The act of 2026-10-07 16:12: the gamepad's fresh find came in while the arm hovered over it, the restarted track
    never saw it under the gripper, and the act gave up after 20 s waiting for it. The find's own view is where the
    object is until the track sees it again: the act grasps and places as usual."""
    end_at = PLACE_AT["place_end"]
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, covered_after_find=True)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        fix = np.asarray(act.place["fix"])
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (m["box"] @ demo.tips[end_at] @ fix)[:3, 3]) <= 0.0005
    finally:
        _end_place_state()


def test_a_kept_failure_of_the_demos_hold_does_not_break_the_readout(client, tmp_path):
    """The demo's hold that failed is kept beside the measured ones so an act does not ask again; the page's readout
    of the demo showed every kept entry as a measured hold, raised on the failure, and the whole state failed with it:
    the page froze with the act it was following."""
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    o = demo.objects["gamepad"]
    for window in ("grip", "carry"):
        span = manipulation.holds._hold_window(demo, window)
        key = json.dumps(["gamepad", int(o["frame"]), window, span, np.round(np.eye(4), 5).tolist()])
        demo.holds[key] = {"problem": "gamepad is not visible enough in the gripper in the demo"}
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = demo
        assert manipulation.demos._demo_hold_info(demo) is None
        assert client.get("/api/pregrasp/state").status_code == 200
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize("fresh_inliers", [33, 150])
def test_a_tracking_session_with_a_weak_initial_find_must_refind_before_moving(
    tmp_path, monkeypatch, fresh_inliers
):
    """Good frame-to-frame tracking cannot certify its weak registration to the demo.

    On the rig, 33/373 agreeing reference points still started planning because
    an already-live session bypassed the strength check performed by a new find.
    Both runners must first obtain a strong reference, or stop without a target.
    """
    try:
        _demo, sim, _views, _m = _run_place_act(
            tmp_path, monkeypatch, initial_find_inliers=33, fresh_find_inliers=fresh_inliers
        )
        act = manipulation.runtime._state.act
        assert sim["find_answers"], "the weak reference must be replaced"
        if fresh_inliers == 33:
            assert not act.ok and act.reason.startswith("a weak find:"), act.reason
            assert not sim["targets"] and not sim["streamed"], "no motion on either weak reference"
        else:
            assert act.ok, act.reason
            assert sim["first_move_at"] > sim["find_answers"][0], "wait for a strong reference before moving"
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_the_arm_moves_while_the_act_finds_its_objects(tmp_path, monkeypatch):
    """Finding the box again, starting the gamepad's track over from a fresh find and measuring the demo's holds all
    take the worker's time; none of it holds the arm back. With a worker that answers nothing until the arm moves,
    the act walks toward the first pre-grasp on the track it began with and places as usual."""
    try:
        _demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, finds_wait_for_the_arm=True)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert sim["first_move_at"] < sim["first_answer"], "the arm moved before any find was answered"
        early = [
            x
            for x in _act_targets(tmp_path)
            if x["step"] == "pre-grasp 1" and x.get("pose") and x["t"] < sim["first_answer_wall"]
        ]
        assert early and np.allclose(np.asarray(early[0]["composition"]["motion"]), m["pick"]), (
            "the checked approach begins on the track it began with"
        )
    finally:
        _end_place_state()


def test_the_demos_hold_is_read_from_its_track_when_the_track_moves_with_the_fingertip(tmp_path, monkeypatch):
    """A find of the object's view at rest matched none of the pick-and-place demo's views of the gamepad in the
    gripper; its own track followed it there. When an object's track moves with the fingertip over the hold, the
    demo's hold is that track seen from the gripper at its still frames; an object whose track stays put gives none."""
    import asyncio

    from scipy.spatial.transform import Rotation

    # This test isolates gripper-frame composition; surface acceptance has its
    # own recorded and synthetic geometry tests.
    monkeypatch.setattr(
        manipulation.holds,
        "_demo_surface_views",
        lambda demo, obj, frames: [(i, {"pose": demo.objects[obj]["deltas"][f]}) for f, i in frames],
    )
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    times = manipulation.recording._stream_times(demo.recording) - demo.t0
    held = np.eye(4)
    held[:3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    held[:3, 3] = [0.004, -0.012, -0.018]  # how the gamepad sits in the gripper, the camera at the base
    deltas = np.tile(np.eye(4), (len(times), 1, 1))
    for f, tf in enumerate(times):
        if PLACE_AT["grip"] / 30.0 <= tf < PLACE_AT["release"] / 30.0:
            deltas[f] = demo.tips[int(np.argmin(np.abs(demo.t - tf)))] @ held
    with manipulation.runtime._state.lock:
        manipulation.runtime._state.demo = demo
    try:
        assert manipulation.holds._demo_hold_by_track(demo, "gamepad", np.eye(4), "carry") is None, (
            "a track that stays put"
        )
        demo.objects["gamepad"]["deltas"] = deltas
        hold = manipulation.holds._demo_hold_by_track(demo, "gamepad", np.eye(4), "carry")
        assert hold is not None and hold["n"] >= core.HOLD_MIN_VIEWS
        assert np.allclose(hold["hold"], held, atol=1e-6), "the track seen from the gripper"
        kept, problem = asyncio.run(
            manipulation.holds._demo_hold(demo, "gamepad", np.eye(4), lambda: False, "carry")
        )
        assert problem == "" and kept["method"] == "track", "no worker asked: the track answered"
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None


def test_the_live_hold_is_read_from_accepted_poses_while_the_arm_stands_still(tmp_path, monkeypatch):
    """With the demo's hold read from its track, the live hold is read the same way: the session's fits of the held
    object, trusted or not, over the frames read while the arm stands still, each seen from the gripper."""
    import asyncio

    from scipy.spatial.transform import Rotation

    from lerobot.gui.api import jog

    demo, _kin, _box = _place_demo(tmp_path, time.time())
    tip = np.eye(4)
    tip[:3, 3] = [0.15, -0.05, 0.08]
    held = np.eye(4)
    held[:3, :3] = Rotation.from_euler("x", 6, degrees=True).as_matrix()
    held[:3, 3] = [0.01, 0.0, -0.02]
    ref = np.eye(4)
    ref[:3, 3] = [0.02, 0.0, 0.0]  # the find of the demo's view that started the session
    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=np.zeros((H, W, 3), np.uint8),
        depth_m=np.zeros((H, W), np.float32),
        intr=dict(INTR),
        keypoints={"ref": {"object": "gamepad", "ok": True, "delta": ref}},
        stamp=1.0,
    )
    monkeypatch.setattr(jog, "current_tip_and_anchor", lambda: (tip, np.eye(4), {}))
    monkeypatch.setattr(manipulation.runtime, "ACT_TICK_S", 0.002)
    rng = np.random.default_rng(0)

    async def run():
        async def frames():  # the tracker's raw fits of the gamepad: motion since the find, a little noise
            while True:
                await asyncio.sleep(0.005)
                noisy = tip @ held @ np.linalg.inv(ref)
                noisy[:3, 3] += rng.normal(scale=0.0003, size=3)
                with manipulation.runtime._state.lock:
                    manipulation.runtime._state.track.hold_views.append((time.time(), noisy))

        feed = asyncio.create_task(frames())
        try:
            return await manipulation.holds._live_hold(
                demo, "gamepad", held, np.eye(4), lambda: False, method="track"
            )
        finally:
            feed.cancel()

    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo, manipulation.runtime._state.teach = demo, teach
            manipulation.runtime._state.track.hold_views = [
                (0.5, np.eye(4))
            ]  # read before the session's find: not a view of it
        hold, problem = asyncio.run(run())
        assert problem == "" and hold["method"] == "track" and hold["n"] >= core.HOLD_MIN_VIEWS
        assert np.allclose(hold["hold"][:3, 3], held[:3, 3], atol=0.001) and np.allclose(
            hold["hold"][:3, :3], held[:3, :3], atol=1e-6
        )
        assert (
            asyncio.run(
                manipulation.holds._live_hold(
                    demo, "gamepad", held, np.eye(4), lambda: False, views=[], method="track"
                )
            )[0]
            is None
        )
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = manipulation.runtime._state.teach = None
            manipulation.runtime._state.track.hold_views = []


def test_a_demo_hold_the_worker_measured_is_kept_and_one_it_did_not_answer_is_asked_again(tmp_path):
    """The pick-and-place demo's carry never shows the gamepad well enough: every act asked the worker for the same
    7 failing finds before it moved. A failure the worker measured is kept with the demo like a hold; a worker that
    was off measured nothing, and is asked again."""
    import asyncio

    demo, _kin, _box = _place_demo(tmp_path, time.time())
    asked = {"n": 0}
    with manipulation.runtime._state.lock:
        manipulation.runtime._state.demo = demo

    async def measure():
        async def worker():
            while True:
                await asyncio.sleep(0.001)
                with manipulation.runtime._state.lock:
                    pending = list(manipulation.runtime._state.worker.pending)
                    manipulation.runtime._state.worker.pending.clear()
                for job_id in pending:
                    asked["n"] += 1
                    manipulation.runtime._state.worker.jobs[job_id].result = {
                        "ok": True,
                        "ref_ok": False,
                        "ref_reason": "the live view does not match",
                    }

        feed = asyncio.create_task(worker())
        try:
            return await manipulation.holds._demo_hold(demo, "gamepad", np.eye(4), lambda: False, "carry")
        finally:
            feed.cancel()

    try:
        hold, off = asyncio.run(measure())
        assert hold is None and "start the worker first" in off and asked["n"] == 0
        assert not demo.holds, "nothing measured, nothing kept"
        manipulation.runtime._state.worker.proc = _FakeProc()
        hold, problem = asyncio.run(measure())
        assert hold is None and "not visible enough" in problem and asked["n"] > 0, (
            "asked again once it was on"
        )
        n = asked["n"]
        assert asyncio.run(measure()) == (None, problem) and asked["n"] == n, "kept: not asked a third time"
    finally:
        manipulation.runtime._state.worker.proc = None
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.demo = None
            manipulation.runtime._state.worker.pending.clear()
            manipulation.runtime._state.worker.jobs.clear()


def test_demo_hold_reports_weak_matches_and_surface_rejections_together(tmp_path, monkeypatch):
    """The last rejected surface must not hide earlier views that passed geometry but failed matching."""
    import asyncio

    demo, _kin, _box = _place_demo(tmp_path, time.time())
    calls = []

    async def locate(*args, **kwargs):
        calls.append(kwargs)
        return {
            "answered": True,
            "object": "gamepad",
            "ok": len(calls) == 1,
            "reason": "" if len(calls) == 1 else "surface overlap rejected",
            "inliers": 14,
            "card_points": 373,
        }

    monkeypatch.setattr(manipulation.holds, "_demo_hold_by_track", lambda *a: None)
    monkeypatch.setattr(manipulation.acquisition, "_locate", locate)
    hold, problem = asyncio.run(
        manipulation.holds._demo_hold(demo, "gamepad", np.eye(4), lambda: False, "carry")
    )
    assert hold is None and len(calls) > 1
    assert "1 view(s): a weak find" in problem and "14" in problem
    assert f"{len(calls) - 1} view(s): surface overlap rejected" in problem


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


@pytest.mark.usefixtures("act_simulation")
def test_the_place_object_is_followed_from_its_find_and_the_place_aims_where_it_is_now(tmp_path, monkeypatch):
    """The box is moved 20 mm while the gamepad is carried: the find at the act's start began a track of the box,
    the track moves the find, and the carry and the place aim at the box where it is now, not where it was found."""
    end_at, pre = PLACE_AT["place_end"], PLACE_AT["preplace"]
    later = np.eye(4)
    later[:3, 3] = [0.0, 0.020, 0.0]
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, box_moves_in_the_carry=later)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        fix = np.asarray(act.place["fix"])
        moved_box = later @ m["box"]
        assert act.place["target_moved_mm"] == pytest.approx(20.0, abs=0.5)
        assert any(
            core.pose_residual(_TipKinematics().forward_kinematics(q), moved_box @ demo.tips[pre] @ fix)[0]
            < 0.0005
            for q in sim["streamed"]
        ), "the checked correction passes through the pre-place on the moved box"
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (moved_box @ demo.tips[end_at] @ fix)[:3, 3]) <= 0.0005
        found = manipulation.acquisition._located(demo, "box")
        assert np.allclose(found["delta"], later @ m["box"]), "the find moved by the track"
        assert manipulation.acquisition._located_info(found)["track"] == "tracking"
        record = json.loads(
            (pathlib.Path(manipulation.recording._load_trials()[-1]["run"]) / "act.json").read_text()
        )
        steps = record["target_track"]
        assert steps and all(s["trusted"] for s in steps), "the act's record keeps what the box's track said"
        assert np.allclose(steps[-1]["delta"], later @ m["box"], atol=1e-4)
    finally:
        _end_place_state()


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


def test_task_turn_axis_is_independent_of_target_pose_symmetry(tmp_path, monkeypatch):
    """A visual long-axis symmetry cannot silently change a face-normal placement turn."""
    from lerobot.showservo import symmetry as pose_symmetry

    demo, _kin, _box = _place_demo(tmp_path, time.time())
    demo.objects["box"]["symmetry"] = {"kind": "discrete", "axis": "long_axis", "order": 2}
    demo.landing, demo.landing_axis = "turn", "surface_normal"
    seen = []
    original = pose_symmetry.observed_axis

    def axis(points, selector):
        seen.append(selector)
        return original(points, selector)

    monkeypatch.setattr(pose_symmetry, "observed_axis", axis)
    manipulation.planning._landing_axis(demo, "box", np.eye(4))
    assert seen[-1] == "surface_normal"
    assert len(core.landing_turns(demo.landing, demo.objects["box"]["symmetry"])) == 72
    demo.landing = "symmetry"
    manipulation.planning._landing_axis(demo, "box", np.eye(4))
    assert seen[-1] == "long_axis"
    assert len(core.landing_turns(demo.landing, demo.objects["box"]["symmetry"])) == 2


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


class _ReachKinematics(_TipKinematics):
    """The fake arm with a reach: its tip goes no further than ``radius`` metres from the base."""

    def __init__(self, radius: float):
        self.radius = radius

    def inverse_kinematics(self, seed, pose):
        p = np.asarray(pose[:3, 3], dtype=float)
        target = pose.copy()
        target[:3, 3] = p * min(1.0, self.radius / max(np.linalg.norm(p), 1e-9))
        return super().inverse_kinematics(seed, target)


def test_a_failed_approach_reports_residual_and_object_motion_without_claiming_physical_unreachability(
    tmp_path,
):
    """An IK path failure may be a configuration failure, even when an alternate grasp pose is reachable."""
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    teach = manipulation.runtime._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=np.zeros((H, W, 3), np.uint8),
        depth_m=np.full((H, W), 0.45, np.float32),
        intr=dict(INTR),
        keypoints={
            "mode": "features",
            "concept": "gamepad",
            "ref": {"object": "gamepad", "ok": True, "delta": np.eye(4)},
        },
    )
    farther = np.eye(4)
    farther[:3, 3] = [0.050, 0.0, 0.0]  # the box 50 mm farther out along the arm's x
    try:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach = teach
        plan = manipulation.planning._plan_act(
            demo,
            np.eye(4),
            np.eye(4),
            _ReachKinematics(0.20),
            np.array([100.0, 0.0, 60.0, 0, 0, 0, 60.0]),
            (0.04, np.radians(30)),
            ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            1.0,
            0,
            {"box": farther},
        )
        assert not plan["ok"]
        assert plan["reason"].startswith("the planned approach to pre-place 1 has no solved joint path"), (
            plan["reason"]
        )
        assert plan["reason"].endswith(
            "box's motion shifts that target 50 mm from the demo, including 50 mm radially away from the arm base"
        ), plan["reason"]
        mark = next(m for m in plan["marks"] if m["label"] == "pre-place 1")
        assert mark["object"] == "box" and mark["moved_mm"] == pytest.approx(50.0, abs=0.5)
        assert mark["residual_mm"] > 0 and mark["residual_deg"] >= 0
    finally:
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach = None


def test_an_injected_error_turns_about_its_pivot_and_moves():
    pivot = np.array([0.20, -0.05, 0.03])
    e = manipulation.execution._inject_transform({"dx_mm": 5.0, "rz_deg": 10.0}, pivot)
    assert np.allclose(e[:3, :3] @ pivot + e[:3, 3], pivot + [0.005, 0.0, 0.0]), "the pivot only moves"
    p = pivot + [0.01, 0.0, 0.0]
    turned = pivot + [0.01 * np.cos(np.radians(10)), 0.01 * np.sin(np.radians(10)), 0.0]
    assert np.allclose(e[:3, :3] @ p + e[:3, 3], turned + [0.005, 0.0, 0.0])
    assert np.allclose(manipulation.execution._inject_transform({}, pivot), np.eye(4))


def test_an_injected_error_is_capped_and_named():
    from pydantic import ValidationError

    body = manipulation.execution.ActBody(
        inject={"at": "find", "dz_mm": -30, "rx_deg": 20}, correct_hold=False
    )
    assert body.inject.at == "find" and not body.correct_hold
    for bad in ({"at": "aim", "dx_mm": 31}, {"at": "aim", "ry_deg": -21}, {"at": "sideways"}):
        with pytest.raises(ValidationError):
            manipulation.execution.ActBody(inject=bad)
    assert not manipulation.execution._injects(manipulation.execution.InjectBody()), (
        "all zeros injects nothing"
    )


@pytest.mark.usefixtures("act_simulation")
def test_an_injected_aim_is_measured_but_does_not_authorize_a_blind_place(tmp_path, monkeypatch):
    """An encoder grasp estimate can explain an injected aim, but does not observe the object after carrying."""
    inject = {
        "at": "aim",
        "dx_mm": 5.0,
        "dy_mm": 0.0,
        "dz_mm": -3.0,
        "rx_deg": 0.0,
        "ry_deg": 0.0,
        "rz_deg": 0.0,
    }
    calls, pivots = _spy_injection(monkeypatch)
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, blind_carry=True, inject=inject)
        act = manipulation.runtime._state.act
        assert not act.ok and "cannot verify" in act.reason
        g = demo.tips[manipulation.execution._grasp_index(demo)]
        assert pivots and np.allclose(pivots[-1], (m["pick"] @ g)[:3, 3]), "turned about the grip"
        miss = manipulation.execution._inject_transform(inject, (m["pick"] @ g)[:3, 3])
        demo_tip, motion, live_tip = calls[-1]
        assert np.allclose(motion, m["pick"]), "the act's own belief, unbent"
        assert core.pose_residual(live_tip, miss @ m["pick"] @ g)[0] <= 0.0005, "the arm went off its aim"
        expected = np.linalg.inv(g) @ np.linalg.inv(m["pick"]) @ miss @ m["pick"] @ g
        estimate = manipulation.planning._grasp_pose_fix(demo_tip, motion, live_tip)
        assert core.pose_residual(estimate, expected)[0] <= 0.0005
        assert act.place["hold_used"] == "unverified"
        assert sim["streamed"][-1][-1] == 85.0
        row = manipulation.recording._load_trials()[-1]
        assert row["inject"] == inject and row["correct_hold"] is True
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_a_wrong_find_does_not_authorize_placement_without_a_carried_pose(tmp_path, monkeypatch):
    """A wrong find looks correct to the grasp estimate; missing post-carry vision must prevent release."""
    inject = {
        "at": "find",
        "dx_mm": 0.0,
        "dy_mm": 6.0,
        "dz_mm": 0.0,
        "rx_deg": 0.0,
        "ry_deg": 0.0,
        "rz_deg": 0.0,
    }
    calls, pivots = _spy_injection(monkeypatch)
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, blind_carry=True, inject=inject)
        act = manipulation.runtime._state.act
        assert not act.ok and "cannot verify" in act.reason
        centre = manipulation.planning._object_points(demo, "gamepad", np.eye(4)).mean(axis=0)
        assert pivots and np.allclose(pivots[0], m["pick"][:3, :3] @ centre + m["pick"][:3, 3]), (
            "about its centre"
        )
        wrong = manipulation.execution._inject_transform(
            inject, m["pick"][:3, :3] @ centre + m["pick"][:3, 3]
        )
        g = demo.tips[manipulation.execution._grasp_index(demo)]
        _demo_tip, motion, live_tip = calls[-1]
        assert np.allclose(motion, wrong @ m["pick"]), "the act believes the wrong find"
        assert core.pose_residual(live_tip, wrong @ m["pick"] @ g)[0] <= 0.0005
        estimate = manipulation.planning._grasp_pose_fix(_demo_tip, motion, live_tip)
        assert core.pose_residual(estimate, np.eye(4))[0] <= 0.0005
        assert act.place["hold_used"] == "unverified" and sim["streamed"][-1][-1] == 85.0
        assert act.find_error is None
        assert np.allclose(manipulation.planning._delta_base(demo, np.eye(4), np.eye(4)), m["pick"]), (
            "the act's belief ended"
        )
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_the_hold_correction_switched_off_replays_the_place_against_the_target(tmp_path, monkeypatch):
    """A baseline for the injections: the place as the demo did it against where the target is now, the hold the
    act measured left out."""
    end_at = PLACE_AT["place_end"]
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, correct_hold=False)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert act.place["hold_used"] == "off" and np.allclose(act.place["fix"], np.eye(4))
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (m["box"] @ demo.tips[end_at])[:3, 3]) <= 0.0005
        assert manipulation.recording._load_trials()[-1]["correct_hold"] is False
    finally:
        _end_place_state()


def test_an_object_the_gripper_hides_is_reported_hidden_not_unheld(tmp_path):
    """The pick-and-place demo held still at its grip for 0.7 s, but the gamepad's own track saw nothing there under
    the gripper; the act said the demo never held it still."""
    import asyncio

    demo, _kin, _box = _place_demo(tmp_path, time.time())
    demo.objects["gamepad"]["seen"][10:25] = False  # under the gripper from 0.67 s to past the lift
    hold, problem = asyncio.run(
        manipulation.holds._demo_hold(demo, "gamepad", np.eye(4), lambda: False, "grip")
    )
    assert hold is None and problem.startswith(
        "gamepad is hidden in the gripper after the gripper closed on it"
    ), problem


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


@pytest.mark.parametrize("slip", [False, True])
def test_hold_correction_precedes_a_low_held_line(tmp_path, monkeypatch, slip):
    """A late pre-place may already touch the support. Observe and correct at the earlier high line, then
    approach the low line with the measured hold; do not slide the support with a correction made at contact."""
    try:
        demo, sim, views, motions = _run_place_act(
            tmp_path, monkeypatch, two_preplaces=True, low_preplace=True, slip=slip
        )
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert all(z > 0.05 for z in views["live_z"][5:]), "measure the carried hold above the support"
        kin = _TipKinematics()
        carried = [q for q, (_, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops == 1]
        assert kin.forward_kinematics(carried[-1])[2, 3] > 0.05, (
            "the uncorrected carry stops before descending"
        )
        assert act.place["shift_mm"] == pytest.approx(12.0 if slip else 6.0, abs=0.01)
        placing = [q for q, (_, stops) in zip(sim["streamed"], sim["sent_at"], strict=True) if stops >= 2]
        first_low = next(q for q in placing if kin.forward_kinematics(q)[2, 3] < 0.035)
        held = motions["carried"]
        intended = motions["box"] @ demo.tips[85] @ motions["demo"]
        actual = kin.forward_kinematics(first_low) @ held
        assert np.linalg.norm(actual[:2, 3] - intended[:2, 3]) < 0.0005, (
            "align before reaching contact height"
        )
        end = kin.forward_kinematics(placing[-1]) @ held
        goal = motions["box"] @ demo.tips[PLACE_AT["place_end"]] @ motions["demo"]
        assert np.linalg.norm(end[:3, 3] - goal[:3, 3]) < 0.0005
        assert placing[0][-1] == 85.0 and placing[-1][-1] == 60.0, (
            "keep the grip through correction, then release"
        )
    finally:
        _end_place_state()
