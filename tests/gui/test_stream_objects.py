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

from lerobot.gui.api import pregrasp

INTR = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0, "width": 848, "height": 480}
H, W = 480, 848  # the rig's camera


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
        import pregrasp_worker

        yield pregrasp_worker
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
    app.include_router(pregrasp.router)
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
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    t0 = time.time()
    n_samples, n_frames = 30, 15
    # The stream runs at half the arm's 30 Hz, as it does when the live tracker shares the camera.
    rec = write_stream(tmp_path / "demos" / ".recordings" / "work", n_frames, t0=t0, hz=15.0)
    t = np.arange(n_samples) / 30.0
    demo = pregrasp._Demo(
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
    pregrasp._state.worker.proc = _FakeProc()
    try:
        with pregrasp._state.lock:
            pregrasp._state.demo = demo
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
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
        assert (root / pregrasp.OBJECTS_FILE).exists()
        with pregrasp._state.lock:
            pregrasp._state.demo = None
        loaded = client.post("/api/pregrasp/demo/load", json={"name": "stacked"}).json()
        assert loaded["teach_pending"] is False
        again = client.get("/api/pregrasp/demo/objects").json()["objects"]
        assert [(o["name"], o["status"]) for o in again] == [("gamepad", "done")], (
            "the tracked object comes back with the demo"
        )
        with pregrasp._state.lock:
            assert np.array_equal(pregrasp._state.demo.objects["gamepad"]["seen"], seen)
        assert client.post("/api/pregrasp/demo/objects/remove", json={"name": "gamepad"}).status_code == 200
        assert not (root / pregrasp.OBJECTS_FILE).exists()
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
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.demo = None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


def _demo_with_object(tmp_path, t0, n_samples=30, n_frames=15):
    """A demo recorded without a teach, with one object designated on its stream and marks bound to it."""
    from scipy.spatial.transform import Rotation

    rec = write_stream(tmp_path / "demos" / ".recordings" / "work", n_frames, t0=t0, hz=15.0)
    t = np.arange(n_samples) / 30.0
    demo = pregrasp._Demo(
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

    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    t0 = time.time()
    demo = _demo_with_object(tmp_path, t0)
    pregrasp._state.worker.proc = _FakeProc()
    try:
        with pregrasp._state.lock:
            pregrasp._state.demo, pregrasp._state.teach, pregrasp._state.test = demo, None, None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
        post = lambda kps: client.post("/api/pregrasp/demo/keypoints", json={"keypoints": kps})  # noqa: E731
        assert post([{"t": 0.4, "kind": "pregrasp", "object": "cube"}]).status_code == 422, (
            "only a tracked object"
        )
        mixed = [{"t": 0.4, "kind": "pregrasp", "object": "gamepad"}, {"t": 0.7, "kind": "grasp_end"}]
        assert post(mixed).status_code == 422, "the pre-grasp and the grasp are for one object"
        unnamed = [{"t": 0.4, "kind": "pregrasp"}, {"t": 0.7, "kind": "grasp_end"}]
        assert post(unnamed).status_code == 422, "nothing was taught before this demo"
        demo.keypoints = unnamed  # as saved before the editor named the object
        ref_motion, problem = pregrasp._reference_motion(demo, None)
        assert ref_motion is None and "name no object" in problem, "no taught object to fall back on"
        bound = [
            {"t": 0.4, "kind": "pregrasp", "object": "gamepad"},
            {"t": 0.7, "kind": "grasp_end", "object": "gamepad"},
        ]
        assert post(bound).status_code == 200
        assert pregrasp._marks_object(demo) == "gamepad"
        ref_motion, problem = pregrasp._reference_motion(demo, None)
        assert ref_motion is None and "click gamepad in the camera view" in problem, (
            "the act needs the live find first"
        )
        assert client.post("/api/pregrasp/act", json={}).status_code == 409
        # The live click teaches with the demo's view of the object as its reference.
        monkeypatch.setattr(
            pregrasp,
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
        ref_motion, problem = pregrasp._reference_motion(demo, pregrasp._state.teach)
        assert problem == "" and np.allclose(
            ref_motion, ref_delta @ np.linalg.inv(demo.objects["gamepad"]["deltas"][2])
        )
        assert np.allclose(ref_motion, ref_delta)
        failed = dict(
            pregrasp._state.teach.keypoints["ref"],
            ok=False,
            delta=None,
            reason="the live view does not match",
        )
        pregrasp._state.teach.keypoints["ref"] = failed
        assert pregrasp._reference_motion(demo, pregrasp._state.teach) == (
            None,
            "the live view does not match",
        )
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.teach_job = None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


def test_the_editor_shows_the_followed_objects_own_visibility(client, tmp_path, monkeypatch):
    """The editor's "object hidden" came from the live tracker's frames during the recording, a few a second, so a
    sample between two of them read hidden while the object's outline was drawn on every frame. With marks on a
    designated object it is that object's own track on the recording, the one the outline comes from."""
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    demo = _demo_with_object(tmp_path, time.time())  # its object is hidden on stream frames 5 and 6
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
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
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_the_editor_draws_the_tracked_pose_and_reports_the_frame_the_act_reads_it_from(
    client, tmp_path, monkeypatch
):
    """The demo view drew the object's outline but not its tracked pose, so a pose the tracker had tipped could not
    be seen, and nothing showed which frame the act reads the object's demo pose from."""
    import cv2
    from scipy.spatial.transform import Rotation

    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
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

    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    try:
        assert red(8) == 0, "without the arm's camera calibration the frames render, without a pose"
        monkeypatch.setattr(pregrasp, "_t_base_cam", lambda: t_bc)
        times = np.loadtxt(pathlib.Path(demo.recording) / "times.txt")
        pose_t = client.get("/api/pregrasp/demo/curve").json()["pose_t"]
        assert pose_t == pytest.approx(times[2] - demo.t0), "the frame the object was clicked on"
        assert pregrasp._pose_frame(demo) == 2, "the same frame the act's reference motion uses"
        assert red(8) > 5 and red(2) > 5, "every frame carries the pose, the frame the act reads included"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


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
        pregrasp,
        "_frame",
        lambda: _async((np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.45, np.float32), dict(INTR))),
    )
    monkeypatch.setattr(pregrasp, "ACT_STEP_TIMEOUT_S", 0.5)
    taught = np.zeros((H, W), dtype=bool)
    taught[100:141, 100:141] = True  # where the last find had it
    seen = np.zeros((H, W), dtype=bool)
    seen[300:341, 500:541] = True  # where the tracker last saw it, since then moved by hand
    fresh = np.eye(4)
    fresh[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()

    def start_from_an_old_find() -> None:
        teach = pregrasp._Teach(
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
        live = pregrasp._Test(
            at="t", rgb=None, result={"ok": True, "live_mask": seen, "delta_cam": np.eye(4)}
        )
        with pregrasp._state.lock:
            pregrasp._state.demo, pregrasp._state.teach, pregrasp._state.test = demo, teach, live
            pregrasp._state.teach_job = None
            pregrasp._state.track.history.clear()
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()

    async def play_the_worker(meta: dict, then_tracked: bool) -> pregrasp._Job:
        for _ in range(200):
            await asyncio.sleep(0.01)
            with pregrasp._state.lock:
                job_id = pregrasp._state.teach_job
            if job_id is not None:
                break
        else:
            raise AssertionError("no find was asked of the worker")
        job = pregrasp._state.worker.jobs[job_id]
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
        pregrasp._apply_teach_result(job)
        if then_tracked:
            await asyncio.sleep(0.05)
            with pregrasp._state.lock:
                pregrasp._state.track.history.append((time.time(), True, np.eye(4)))
        return job

    async def find(meta: dict, then_tracked: bool) -> tuple[pregrasp._Job, str]:
        worker = asyncio.create_task(play_the_worker(meta, then_tracked))
        why = await pregrasp._find_afresh("gamepad", lambda: False)
        return await worker, why

    pregrasp._state.worker.proc = _FakeProc()
    try:
        start_from_an_old_find()
        job, why = asyncio.run(find({"ok": True, "ref_ok": True, "ref_inliers": 90}, then_tracked=True))
        assert why == "", why
        assert job.kind == "teach" and job.concept == "gamepad" and job.extra["ref_object"] == "gamepad"
        assert abs(job.click[0] - 520) <= 1 and abs(job.click[1] - 320) <= 1, (
            f"the click lands inside where the tracker last saw it, not the old find: {job.click}"
        )
        assert np.allclose(pregrasp._state.teach.keypoints["ref"]["delta"], fresh), (
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
        _, why = asyncio.run(find(mismatch, then_tracked=True))
        assert why == "the live view does not match the demo's view"

        start_from_an_old_find()
        act = pregrasp._state.act
        act.on, act.ok, act.reason, act.step, act.stop_requested = True, None, "", "starting", False

        async def act_with_a_failed_find() -> None:
            worker = asyncio.create_task(play_the_worker(mismatch, then_tracked=True))
            await pregrasp._act_task(0.5)
            await worker

        asyncio.run(act_with_a_failed_find())
        assert (act.ok, act.step, act.reason, act.on) == (
            False,
            "aborted",
            "the live view does not match the demo's view",
            False,
        ), "the act ends on its find, before planning or moving"
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.teach_job = None
            pregrasp._state.track.history.clear()
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


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
    return pregrasp._stream_times(demo.recording)


def test_a_place_follows_its_own_object_found_and_tracked_apart_from_the_picked_one(
    client, tmp_path, monkeypatch
):
    """The pre-grasps and the grasp follow the object picked; the pre-places and the place follow the one it goes
    onto. That object is found by a locate against the demo's view of it, which leaves the picked object's teach alone
    and has the box join the live session that follows the gamepad; the place moves by what that find says, against
    where the demo had the object at the first pre-place."""
    import asyncio

    from scipy.spatial.transform import Rotation

    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    demo, target = _two_object_demo(tmp_path, time.time())
    live_rgb = np.full((H, W, 3), 200, np.uint8)
    monkeypatch.setattr(
        pregrasp, "_frame", lambda: _async((live_rgb, np.full((H, W), 0.45, np.float32), dict(INTR)))
    )
    pregrasp._state.worker.proc = _FakeProc()
    try:
        with pregrasp._state.lock:
            pregrasp._state.demo, pregrasp._state.teach, pregrasp._state.test = demo, None, None
            pregrasp._state.located.clear()
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
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
        assert r.json()["place_object"] == "box" and pregrasp._marks_object(demo) == "gamepad"
        assert pregrasp._target_motion(demo, np.eye(4)) == (None, "click box in the camera view to find it")

        moved = np.eye(4)
        moved[:3, :3] = Rotation.from_euler("z", 15, degrees=True).as_matrix()
        moved[:3, 3] = [0.04, -0.02, 0.0]

        async def answer(inliers: int) -> pregrasp._Job:
            for _ in range(500):
                await asyncio.sleep(0.002)
                with pregrasp._state.lock:
                    pending = list(pregrasp._state.worker.pending)
                    pregrasp._state.worker.pending.clear()  # taken, as the worker's poll takes them
                if pending:
                    break
            job = pregrasp._state.worker.jobs[pending[0]]
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
            info = await pregrasp.locate(pregrasp.LocateBody(click=[550, 330], object="box"))
            return await worker, info

        job, info = asyncio.run(click(160))
        assert job.kind == "locate" and job.concept == "box" and job.click == [550, 330]
        assert job.extra == {
            "ref_recording": demo.recording,
            "ref_frame": 1,
            "ref_object": "box",
            "track": True,
            "scene": ["gamepad"],
        }, "the box joins the live session beside the gamepad"
        assert np.array_equal(job.arrays["ref_mask"], target), "the box's own view in the demo"
        assert info["ok"] and info["strong"] and info["inliers"] == 160
        assert pregrasp._state.teach is None and pregrasp._state.teach_job is None, (
            "the picked object's teach and track untouched"
        )
        # Placed onto where the box is now: the find, against where the demo had it on its pose frame, the frame it
        # was clicked on (1), which comes before the place.
        motion, problem = pregrasp._target_motion(demo, np.eye(4))
        assert problem == "" and np.allclose(motion, moved @ np.linalg.inv(demo.objects["box"]["deltas"][1]))
        assert client.get("/api/pregrasp/state").json()["located"]["box"]["strong"] is True

        asyncio.run(click(32))
        motion, problem = pregrasp._target_motion(demo, np.eye(4))
        assert motion is None and problem.startswith(
            "a weak find: box matched 32 of the demo view's 400 points"
        )
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.located.clear()
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


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


def _place_demo(tmp_path, t0):
    """A demo at 30 Hz with its stream at 15 Hz: the arm comes down by 0.83 s and stands there while the gripper
    closes on the gamepad, firm at 0.93 s; it lifts at 1.5 s, carries the gamepad above the box, holds it still there
    (1.93 to 2.6 s), sets it down and lets go at 2.93 s. The fake arm's tip is its first three joints in millimetres;
    the gamepad stops the fingers 3 units short of the closing command."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi = MOTOR_NAMES.index("gripper")
    n = 100
    s = np.arange(n)
    q = np.zeros((n, 7))
    q[:, 0] = np.interp(s, [0, 25, 52, 58, 99], [100, 120, 120, 180, 180])
    q[:, 2] = np.interp(s, [0, 25, 45, 52, 78, 86, 99], [60, 20, 20, 60, 60, 25, 25])
    gripping = (s >= PLACE_AT["grip"]) & (s < PLACE_AT["release"])
    q_cmd = q.copy()
    q_cmd[:, gi] = np.where(gripping, 85.0, 60.0)
    q[:, gi] = np.where(gripping, 82.0, 60.0)
    kin = _TipKinematics()
    rec = write_stream(tmp_path / "demos" / ".recordings" / "place", 50, t0=t0, hz=15.0)
    demo = pregrasp._Demo(
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


def _run_place_act(
    tmp_path,
    monkeypatch,
    empty_grip=False,
    slow_grip=False,
    slip=False,
    weak_carry=False,
    blind=False,
    inject=None,
    correct_hold=True,
    finds_wait_for_the_arm=False,
    box_moves_in_the_carry=None,
    covered_after_find=False,
    sag_deg=0.0,
):
    """Run the act on the place demo against a fake arm and a fake worker. The gamepad lies 10 mm from where the demo
    had it, the box 30 mm and 10 mm; the demo held the gamepad 20 mm below the fingertip and the act holds it 6 mm
    further along x; with ``slip`` the lift moves it another 6 mm. With ``weak_carry`` the demo's frames after the
    lift show too little of the gamepad to find it; with ``blind`` no view of it in the gripper matches at all.
    ``inject`` and ``correct_hold`` are what the operator asked the act for (:class:`pregrasp.InjectBody`). With
    ``finds_wait_for_the_arm`` the worker answers nothing until the arm has been given a target. With
    ``box_moves_in_the_carry``, a motion (camera frame), the session that follows the gamepad, which the box joins at
    the act's start, sees the box moved by it once the gamepad is carried. With ``covered_after_find`` the arm hides
    the gamepad from the camera once it is found again: the restarted track never sees it. With ``sag_deg`` the
    shoulder reads that much short of its command while the gripper holds the gamepad, as the load holds it down.
    Post: (demo, sim, views, the motions and holds); ``sim["sent_at"]`` has each streamed sample's time and how many
    streams had ended by then, ``sim["first_answer"]`` when the worker first answered."""
    import asyncio
    import time as _time

    from lerobot.gui.api import jog
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    demo, kin, box = _place_demo(tmp_path, _time.time())
    pick_moved, box_moved, hold_demo, hold_now = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
    pick_moved[:3, 3] = [0.010, 0.0, 0.0]
    box_moved[:3, 3] = [0.030, 0.010, 0.0]
    hold_demo[:3, 3] = [0.0, 0.0, -0.020]
    hold_now[:3, 3] = [0.006, 0.0, -0.020]
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
        return _async((rgb, np.full((H, W), 0.45, np.float32), dict(INTR)))

    monkeypatch.setattr(pregrasp, "_frame", frame)
    sim = {
        "q": np.array([80.0, -20.0, 90.0, 0, 0, 0, 60.0]),
        "grip": 60.0,
        "closed_at": None,
        "targets": [],
        "target_at": [],
        "streamed": [],
        "sent_at": [],
        "stops": 0,
        "first_answer": None,
    }

    def grip_obs():
        final = sim["grip"] if empty_grip or sim["grip"] < 82.0 else 82.0  # the gamepad stops the fingers
        if (
            slow_grip and sim["closed_at"] is not None
        ):  # still closing, 5 units behind and catching up over 0.3 s
            final -= 5.0 * max(0.0, 1.0 - (_time.monotonic() - sim["closed_at"]) / 0.3)
        return final

    def set_target_pose(pose):
        sim["targets"].append(np.array(pose))
        sim["target_at"].append(_time.monotonic())
        sim["q"][:3] += (np.asarray(pose)[:3, 3] * 1000.0 - sim["q"][:3]) * 0.34

    async def joints_start(q_first):
        sim["q"] = np.array([q_first[m] for m in MOTOR_NAMES])

    async def joints_stop():
        sim["stops"] += 1

    def set_target_joints(q):
        sim["q"] = np.array([q[m] for m in MOTOR_NAMES])
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
        return kin.forward_kinematics(sim["q"]), np.eye(4), q

    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    monkeypatch.setattr(jog, "current_tip_and_anchor", tip_and_anchor)
    monkeypatch.setattr(jog, "set_target_pose", set_target_pose)
    monkeypatch.setattr(jog, "current_status", lambda: {"connected": True, "halted": False, "holding": False})
    monkeypatch.setattr(jog, "current_gripper", grip_obs)
    monkeypatch.setattr(jog, "set_gripper", lambda g: sim.__setitem__("grip", g))
    monkeypatch.setattr(jog, "walk_limits", lambda: (0.04, np.radians(30)))
    monkeypatch.setattr(jog, "set_walk_limits", lambda lin, ang: None)
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    monkeypatch.setattr(jog, "joints_start", joints_start)
    monkeypatch.setattr(jog, "joints_stop", joints_stop)
    monkeypatch.setattr(jog, "set_target_joints", set_target_joints)
    monkeypatch.setattr(jog, "start_record", lambda: _time.time())
    monkeypatch.setattr(jog, "stop_record", lambda: [])
    monkeypatch.setattr(jog, "fk_tip", lambda q: np.eye(4))
    monkeypatch.setattr(pregrasp, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(pregrasp, "ACT_TICK_S", 0.002)
    monkeypatch.setattr(pregrasp, "HOLD_VIEW_GAP_S", 0.002)
    monkeypatch.setattr(pregrasp, "ACT_STEP_TIMEOUT_S", 5.0)
    monkeypatch.setattr(pregrasp, "TRIALS_PATH", tmp_path / "trials.jsonl")
    monkeypatch.setattr(pregrasp, "_trials", None, raising=False)
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    teach = pregrasp._Teach(
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
            "ref": {"object": "gamepad", "ok": True, "delta": pick_moved, "inliers": 150, "card_points": 400},
        },
    )  # a live track since an earlier find: the act starts on it while the find is started over beside the arm
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach, pregrasp._state.teach_job = demo, teach, None
        pregrasp._state.test = pregrasp._Test(
            at="t", rgb=None, result={"ok": True, "live_mask": box_mask(), "delta_cam": np.eye(4)}
        )
        pregrasp._state.located = {  # where an earlier find left the box, against the demo's view of it
            "box": {
                "object": "box",
                "ok": True,
                "delta": box_moved
                @ demo.objects["box"]["deltas"][pregrasp._pose_frame(demo, "box", "preplace")],
                "mask": box,
                "inliers": 150,
                "card_points": 400,
                "view": [demo.name, int(demo.objects["box"]["frame"])],
            }
        }
        pregrasp._state.track.on = True
        pregrasp._state.track.last = {"state": "tracking"}
        pregrasp._state.track.history = []
        pregrasp._state.worker.pending.clear()
        pregrasp._state.worker.jobs.clear()
        pregrasp._state.act = pregrasp._Act(on=True, speed=4.0, inject=inject, correct_hold=correct_hold)
    pregrasp._state.worker.proc = _FakeProc()
    views = {"demo": 0, "live": 0, "live_z": []}

    async def worker():
        """Answers the act's jobs as the real worker would, and keeps the live track certifying the gamepad."""
        taught = False
        while True:
            await asyncio.sleep(0.002)
            if finds_wait_for_the_arm and not sim["targets"]:
                continue
            with pregrasp._state.lock:
                pending = list(pregrasp._state.worker.pending)
                pregrasp._state.worker.pending.clear()
                if taught and not covered_after_find:
                    pregrasp._state.track.history.append((_time.time(), True, np.eye(4)))
                elif taught:
                    pregrasp._state.track.last = {"state": "lost"}
            if box_moves_in_the_carry is not None and taught:  # the box's share of each tracked frame
                carried = (
                    sim["grip"] >= 82.0 and kin.forward_kinematics(sim["q"])[2, 3] > 0.05
                )  # closed, lifted
                sim["box_moved"] = sim.get("box_moved") or carried
                share = {"name": "box", "ok": True, "lost": False, "n_visible": 50, "n_tracks": 50}
                moved = box_moves_in_the_carry if sim["box_moved"] else np.eye(4)
                pregrasp._apply_others({"others": [share], "other_delta_0": moved}, (H, W))
            if pending and sim["first_answer"] is None:
                sim["first_answer"] = _time.monotonic()
            for job_id in pending:
                job = pregrasp._state.worker.jobs[job_id]
                found = {
                    "ok": True,
                    "ref_ok": True,
                    "ref_inliers": 150,
                    "ref_card_points": 400,
                    "ref_turn_deg": 0.0,
                }
                if job.kind == "teach":
                    job.result = {
                        **found,
                        "n_points": 50,
                        "radius_mm": 40.0,
                        "shape_class": "box",
                        "yaw_observable": True,
                        "mask": box_mask(),
                        "uv": np.zeros((50, 2)),
                        "xyz": np.zeros((50, 3)),
                        "ref_delta": pick_moved,
                    }
                    pregrasp._apply_teach_result(job)
                    with pregrasp._state.lock:
                        pregrasp._state.test = pregrasp._Test(
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
                    if blind:
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
            await asyncio.wait_for(pregrasp._act_task(4.0), timeout=30.0)
        finally:
            feed.cancel()

    try:
        asyncio.run(run())
    finally:
        pregrasp._state.worker.proc = None
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


def _end_place_state():
    with pregrasp._state.lock:
        pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
        pregrasp._state.located.clear()
        pregrasp._state.track.on = False
        pregrasp._state.track.history = []
        pregrasp._state.track.last = {}
        pregrasp._state.worker.pending.clear()
        pregrasp._state.worker.jobs.clear()
        pregrasp._state.act = pregrasp._Act()


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
        act = pregrasp._state.act
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
        assert any(np.allclose(t, m["box"] @ demo.tips[pre]) for t in sim["targets"]), (
            "carried onto the moved box"
        )
        assert np.allclose(sim["targets"][-1], m["box"] @ demo.tips[pre] @ fix), (
            "then to the corrected pre-place"
        )
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
        target = json.loads((pathlib.Path(row["run"]) / "act.json").read_text())["target"]
        assert target["object"] == "box" and target["inliers"] == 150 and target["strong"] is True
        assert np.allclose(target["motion_base"], m["box"]) and target["pose_frame"] == 2
    finally:
        _end_place_state()


def test_a_shift_during_the_lift_is_caught_at_the_pre_place(tmp_path, monkeypatch):
    """The grip measured before the lift is not the grip after it when the lift moves the object in the fingers: the
    pre-place measurement, seen well on both sides, is the one the place uses, and the shift is reported."""
    end_at = PLACE_AT["place_end"]
    try:
        demo, sim, views, m = _run_place_act(tmp_path, monkeypatch, slip=True)
        act = pregrasp._state.act
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


def test_with_too_little_seen_while_carried_the_hold_at_the_grip_places_it(tmp_path, monkeypatch):
    """The stacking demo of 2026-10-07: carried, the gamepad was mostly behind the gripper and 2 of 7 frames matched;
    at the grip, before the lift, all 11 did. Without a carry measurement the hold at the grip is used, and said so."""
    end_at = PLACE_AT["place_end"]
    try:
        demo, sim, views, m = _run_place_act(tmp_path, monkeypatch, weak_carry=True)
        act = pregrasp._state.act
        assert act.ok, act.reason
        assert act.place["hold_used"] == "grip"
        assert "not visible enough" in act.place["demo_carry"] and act.place["live_place"].startswith(
            "not measured"
        )
        assert views["live"] == 5, "no live views at the pre-place without the demo's to compare"
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert (
            np.linalg.norm((end @ m["now"])[:3, 3] - (m["box"] @ demo.tips[end_at] @ m["demo"])[:3, 3])
            <= 0.0005
        )
    finally:
        _end_place_state()


@pytest.mark.parametrize("slow_grip", [False, True])
def test_the_act_stops_before_carrying_when_the_gripper_closed_on_nothing(tmp_path, monkeypatch, slow_grip):
    """The grasp check runs beside the stream, from the demo's firm grip: once the gripper's reading stops, an
    empty one is at its command, and the stream halts there. A gripper still closing reads short of its command with
    nothing in it, so the check waits for it to stop: 0.3 s here, longer than this test's grasp takes to stream at
    4x, so that act lifts and the check after the lift stops it (at 1x the demo stood still 0.57 s after its grip)."""
    try:
        demo, sim, views, m = _run_place_act(tmp_path, monkeypatch, empty_grip=True, slow_grip=slow_grip)
        act = pregrasp._state.act
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
    assert pregrasp._firm_grip(demo) == PLACE_AT["grip"], "the reading stops 3 short of the command at 0.93 s"
    assert [f for f, _ in pregrasp._still_held_frames(demo, "gamepad", "grip")] == [22, 19, 16], (
        "from 0.93 s until the arm moves again at 1.47 s, before the lift"
    )
    assert [f for f, _ in pregrasp._still_held_frames(demo, "gamepad", "carry")] == [38, 35, 32]
    demo.keypoints[2] = {
        "t": float(demo.t[59]),
        "kind": "preplace",
        "object": "box",
    }  # the pause's first sample
    assert [f for f, _ in pregrasp._still_held_frames(demo, "gamepad", "carry")] == [38, 35, 32], (
        "the pause runs on after the mark until the arm moves again"
    )
    demo.q_cmd[65:, -1] = (
        60.0  # let go in the middle of the pause, at 2.17 s: no longer the same hold from there
    )
    assert [f for f, _ in pregrasp._still_held_frames(demo, "gamepad", "carry")] == [32], (
        "frame 32 (2.13 s) is the last before letting go, and 30 and 31 too close to it to be other views"
    )


def test_a_locate_against_another_demo_or_view_does_not_move_the_place(client, tmp_path, monkeypatch):
    """A locate's motion is from the view it matched; kept by object name alone, one made against the previous demo
    moved this demo's place."""
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
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
        with pregrasp._state.lock:
            pregrasp._state.demo = demo
            pregrasp._state.located["box"] = {**found, "view": ["an earlier demo", 1]}
        assert pregrasp._target_motion(demo, np.eye(4)) == (None, "click box in the camera view to find it")
        assert client.get("/api/pregrasp/state").json()["located"] == {}
        pregrasp._state.located["box"] = {
            **found,
            "view": [demo.name, 4],
        }  # the box designated again, elsewhere
        assert pregrasp._target_motion(demo, np.eye(4))[0] is None
        pregrasp._state.located["box"] = {**found, "view": [demo.name, 1]}
        assert pregrasp._target_motion(demo, np.eye(4))[1] == ""
        assert set(client.get("/api/pregrasp/state").json()["located"]) == {"box"}
        r = client.post("/api/pregrasp/locate", json={"click": [10, 10], "object": "mug"})
        assert r.status_code == 409 and "mug" not in pregrasp._state.located, (
            "nothing kept for an unknown object"
        )
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None
            pregrasp._state.located.clear()


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

    async def locate(obj, rgb, depth, intr, click, stopped):
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

    blank = (np.zeros((H, W, 3), np.uint8), np.full((H, W), 0.45, np.float32), dict(INTR))
    monkeypatch.setattr(pregrasp, "_locate", locate)
    monkeypatch.setattr(pregrasp, "_frame", lambda: _async(blank))
    monkeypatch.setattr(jog, "current_tip_and_anchor", lambda: (tip, np.eye(4), {}))
    try:
        with pregrasp._state.lock:
            pregrasp._state.demo = demo
            pregrasp._state.test = pregrasp._Test(at="t", rgb=None, result={"ok": True, "live_mask": tracked})
            pregrasp._state.track.last = {"state": "tracking"}
        avg, problem = asyncio.run(pregrasp._live_hold(demo, "gamepad", hold, np.eye(4), lambda: False))
        assert problem == "" and avg["n"] == 5 and np.allclose(avg["hold"], hold), (problem, clicks)
        assert clicks[1:] == [clicks[1]] * 5 and clicks[0] != clicks[1], (
            "the predicted middle, then the track's"
        )
        with pregrasp._state.lock:
            pregrasp._state.track.last = {
                "state": "lost"
            }  # a lost track's mask is where it last was, not now
        clicks.clear()
        avg, problem = asyncio.run(pregrasp._live_hold(demo, "gamepad", hold, np.eye(4), lambda: False))
        assert avg is None and "0 of 7 views" in problem and len(set(clicks)) == 1, "nowhere else to click"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.test = None
            pregrasp._state.track.last = {}


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
    times = pregrasp._stream_times(demo.recording) - demo.t0
    grasp = [
        {"t": 0.4, "kind": "pregrasp", "object": "gamepad"},
        {"t": 0.7, "kind": "grasp_end", "object": "gamepad"},
    ]
    demo.keypoints = list(grasp)
    assert pregrasp._pose_choice(demo, "gamepad", "pregrasp") == (2, "clicked")
    demo.keypoints = [*grasp, {"t": float(times[8]), "kind": "pose", "object": "gamepad"}]
    assert pregrasp._pose_choice(demo, "gamepad", "pregrasp") == (8, "set")
    demo.objects["gamepad"]["frame"] = (
        9  # clicked after the last pre-grasp, at 0.6 s: the arm may have moved it
    )
    demo.keypoints = [{"t": 0.3, "kind": "pregrasp", "object": "gamepad"}, grasp[1]]
    assert pregrasp._pose_choice(demo, "gamepad", "pregrasp") == (4, "last seen by the first pre-grasp")
    demo.objects["gamepad"]["frame"] = 2
    demo.keypoints = []
    assert pregrasp._pose_choice(demo, "gamepad", "pregrasp") is None, "no stage yet"
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
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
        assert client.get("/api/pregrasp/demo/curve").json()["pose_t"] == pytest.approx(times[3])
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_a_held_object_never_seen_is_placed_by_the_grasp_pose(tmp_path, monkeypatch):
    """The pick-and-place demo of 2026-10-07: the gripper came straight down over the gamepad and hid it, and carried
    it tilted far from its view on the table, so no view of it in the gripper matched and the act refused. The grasp
    pose is the hold no camera is needed for: the act aims the grasp by the object's estimated pose, so the hold is
    the demo's, changed by how far the arm landed from that aim. What the closing fingers did stays unseen: here the
    gripper holds the gamepad 6 mm off, which only a view would show, and the place keeps that offset."""
    end_at = PLACE_AT["place_end"]
    try:
        demo, sim, views, m = _run_place_act(tmp_path, monkeypatch, blind=True)
        act = pregrasp._state.act
        assert act.ok, act.reason
        assert act.place["hold_used"] == "grasp pose"
        assert act.place["grasp_pose_mm"] == pytest.approx(0.0, abs=0.01), (
            "the fake arm landed where it aimed"
        )
        assert "hidden" not in act.place["demo_grip"] and "not visible enough" in act.place["demo_grip"]
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (m["box"] @ demo.tips[end_at])[:3, 3]) <= 0.0005, (
            "the demo's place"
        )
        offset = (end @ m["now"])[:3, 3] - (m["box"] @ demo.tips[end_at] @ m["demo"])[:3, 3]
        assert np.linalg.norm(offset) == pytest.approx(0.006, abs=0.0005), "the unseen 6 mm stays"
    finally:
        _end_place_state()


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
    fix = pregrasp._grasp_pose_fix(demo_tip, motion, live_tip)
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
    """Record what the act gives :func:`pregrasp._grasp_pose_fix` (demo tip, object motion, live tip) and the pivots
    it turns an injected error about."""
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pivots: list[np.ndarray] = []
    fix, transform = pregrasp._grasp_pose_fix, pregrasp._inject_transform

    def spy_fix(demo_tip, motion, live_tip):
        calls.append((np.array(demo_tip), np.array(motion), np.array(live_tip)))
        return fix(demo_tip, motion, live_tip)

    def spy_transform(inject, pivot):
        pivots.append(np.array(pivot)[:3])
        return transform(inject, pivot)

    monkeypatch.setattr(pregrasp, "_grasp_pose_fix", spy_fix)
    monkeypatch.setattr(pregrasp, "_inject_transform", spy_transform)
    return calls, pivots


@pytest.mark.parametrize("sag", [3.4, 8.0])
def test_an_arm_held_short_under_load_settles_once_it_stops(tmp_path, monkeypatch, sag):
    """The act of 2026-10-07 16:33: after the lift the shoulder sat 3.4 deg short of its last target with the gamepad
    in the gripper, and the settle waited 20 s for the 2 deg it wanted. An arm that has stopped within ACT_STALL_DEG
    has arrived; one stopped further off says so at once instead of waiting."""
    import time as _time

    t0 = _time.monotonic()
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, sag_deg=sag)
        act = pregrasp._state.act
        if sag <= pregrasp.ACT_STALL_DEG:
            assert act.ok, act.reason
            assert act.place["settled_short_deg"] == pytest.approx(sag, abs=0.01)
        else:
            assert not act.ok and act.reason == f"the end: the arm stopped {sag:.0f} deg short of it", (
                act.reason
            )
            assert _time.monotonic() - t0 < pregrasp.ACT_STEP_TIMEOUT_S, "not after the timeout"
    finally:
        _end_place_state()


def test_an_object_the_arm_covers_after_its_find_is_grasped_where_the_find_put_it(tmp_path, monkeypatch):
    """The act of 2026-10-07 16:12: the gamepad's fresh find came in while the arm hovered over it, the restarted track
    never saw it under the gripper, and the act gave up after 20 s waiting for it. The find's own view is where the
    object is until the track sees it again: the act grasps and places as usual."""
    end_at = PLACE_AT["place_end"]
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, covered_after_find=True)
        act = pregrasp._state.act
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
        span = pregrasp._hold_window(demo, window)
        key = json.dumps(["gamepad", int(o["frame"]), window, span, np.round(np.eye(4), 5).tolist()])
        demo.holds[key] = {"problem": "gamepad is not visible enough in the gripper in the demo"}
    try:
        with pregrasp._state.lock:
            pregrasp._state.demo = demo
        assert pregrasp._demo_hold_info(demo) is None
        assert client.get("/api/pregrasp/state").status_code == 200
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_the_arm_moves_while_the_act_finds_its_objects(tmp_path, monkeypatch):
    """Finding the box again, starting the gamepad's track over from a fresh find and measuring the demo's holds all
    take the worker's time; none of it holds the arm back. With a worker that answers nothing until the arm moves,
    the act walks toward the first pre-grasp on the track it began with and places as usual."""
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, finds_wait_for_the_arm=True)
        act = pregrasp._state.act
        assert act.ok, act.reason
        assert sim["target_at"][0] < sim["first_answer"], "the arm moved before any find was answered"
        first_pre = int(
            np.argmin(np.abs(demo.t - min(k["t"] for k in demo.keypoints if k["kind"] == "pregrasp")))
        )
        assert np.allclose(sim["targets"][0], m["pick"] @ demo.tips[first_pre]), "on the track it began with"
    finally:
        _end_place_state()


def test_a_demo_hold_the_worker_measured_is_kept_and_one_it_did_not_answer_is_asked_again(tmp_path):
    """The pick-and-place demo's carry never shows the gamepad well enough: every act asked the worker for the same
    7 failing finds before it moved. A failure the worker measured is kept with the demo like a hold; a worker that
    was off measured nothing, and is asked again."""
    import asyncio

    demo, _kin, _box = _place_demo(tmp_path, time.time())
    asked = {"n": 0}
    with pregrasp._state.lock:
        pregrasp._state.demo = demo

    async def measure():
        async def worker():
            while True:
                await asyncio.sleep(0.001)
                with pregrasp._state.lock:
                    pending = list(pregrasp._state.worker.pending)
                    pregrasp._state.worker.pending.clear()
                for job_id in pending:
                    asked["n"] += 1
                    pregrasp._state.worker.jobs[job_id].result = {
                        "ok": True,
                        "ref_ok": False,
                        "ref_reason": "the live view does not match",
                    }

        feed = asyncio.create_task(worker())
        try:
            return await pregrasp._demo_hold(demo, "gamepad", np.eye(4), lambda: False, "carry")
        finally:
            feed.cancel()

    try:
        hold, off = asyncio.run(measure())
        assert hold is None and "start the worker first" in off and asked["n"] == 0
        assert not demo.holds, "nothing measured, nothing kept"
        pregrasp._state.worker.proc = _FakeProc()
        hold, problem = asyncio.run(measure())
        assert hold is None and "not visible enough" in problem and asked["n"] > 0, (
            "asked again once it was on"
        )
        n = asked["n"]
        assert asyncio.run(measure()) == (None, problem) and asked["n"] == n, "kept: not asked a third time"
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.demo = None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


def test_the_place_object_is_drawn_where_its_find_put_it(tmp_path):
    """The box a place goes onto is not tracked: the live view draws it where its last find put it, its surface from
    the demo's view carried by the find's motion, so a find that missed shows against the real box."""
    import cv2

    demo, _kin, box = _place_demo(tmp_path, time.time())
    points = pregrasp._view_points(demo, "box")
    assert pregrasp._view_points(demo, "box") is points, "read once"
    spans = []
    for shift in ([0.0, 0.0, 0.0], [0.03, 0.0, 0.0]):
        delta = np.eye(4)
        delta[:3, 3] = shift
        mask = pregrasp._found_mask(dict(INTR), points, delta, (H, W))
        uv = pregrasp._project_cam(dict(INTR), points + shift)
        ys, xs = np.nonzero(mask)
        assert abs(xs.min() - uv[:, 0].min()) <= 4 and abs(xs.max() - uv[:, 0].max()) <= 4
        assert abs(ys.min() - uv[:, 1].min()) <= 4 and abs(ys.max() - uv[:, 1].max()) <= 4
        spans.append(xs.mean())
    z = float(points[:, 2].mean())
    assert spans[1] - spans[0] == pytest.approx(INTR["fx"] * 0.03 / z, abs=2.0), "moved with the find"
    if np.allclose(points[:, 2], z):  # a flat view at one depth: the outline is the box's own mask
        assert np.mean(pregrasp._found_mask(dict(INTR), points, np.eye(4), (H, W))[box]) > 0.95

    teach = pregrasp._Teach(
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
        with pregrasp._state.lock:
            pregrasp._state.demo, pregrasp._state.located = demo, {"box": found}
        drawn = orange(pregrasp._render_live(np.zeros((H, W, 3), np.uint8), {}, None, None, teach, status))
        edge = cv2.dilate(box.astype(np.uint8), np.ones((5, 5), np.uint8)) & ~cv2.erode(
            box.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
        assert drawn[edge.astype(bool)].mean() > 0.3, "the box outlined where it was found"
        with pregrasp._state.lock:
            pregrasp._state.located = {}
        assert not orange(
            pregrasp._render_live(np.zeros((H, W, 3), np.uint8), {}, None, None, teach, status)
        )[edge.astype(bool)].any(), "nothing drawn without a find"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None
            pregrasp._state.located = {}


def test_the_place_object_is_followed_from_its_find_and_the_place_aims_where_it_is_now(tmp_path, monkeypatch):
    """The box is moved 20 mm while the gamepad is carried: the find at the act's start began a track of the box,
    the track moves the find, and the carry and the place aim at the box where it is now, not where it was found."""
    end_at, pre = PLACE_AT["place_end"], PLACE_AT["preplace"]
    later = np.eye(4)
    later[:3, 3] = [0.0, 0.020, 0.0]
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, box_moves_in_the_carry=later)
        act = pregrasp._state.act
        assert act.ok, act.reason
        fix = np.asarray(act.place["fix"])
        moved_box = later @ m["box"]
        assert act.place["target_moved_mm"] == pytest.approx(20.0, abs=0.5)
        assert any(np.allclose(t, moved_box @ demo.tips[pre] @ fix, atol=1e-6) for t in sim["targets"]), (
            "the corrected pre-place on the moved box"
        )
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (moved_box @ demo.tips[end_at] @ fix)[:3, 3]) <= 0.0005
        found = pregrasp._located(demo, "box")
        assert np.allclose(found["delta"], later @ m["box"]), "the find moved by the track"
        assert pregrasp._located_info(found)["track"] == "tracking"
        record = json.loads((pathlib.Path(pregrasp._load_trials()[-1]["run"]) / "act.json").read_text())
        steps = record["target_track"]
        assert steps and all(s["trusted"] for s in steps), "the act's record keeps what the box's track said"
        assert steps[-1]["delta_mm"] == pytest.approx((later @ m["box"])[:3, 3] * 1000.0, abs=0.1)
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
        return {"others": [{"name": "box", "ok": True, "lost": False, "n_visible": 40, "n_tracks": 50, **kw}]}

    try:
        with pregrasp._state.lock:
            pregrasp._state.demo = demo
        pregrasp._store_located("box", {**found, "tracking": False})
        assert pregrasp._state.target.obj is None, "not followed without joining the session"
        pregrasp._store_located("box", dict(found))
        assert pregrasp._state.target.obj == "box"
        pregrasp._apply_others({**share(lost=True), "other_delta_0": moved}, (H, W))
        assert np.allclose(pregrasp._state.located["box"]["delta"], anchor), "lost: not moved"
        pregrasp._apply_others({**share(n_visible=3), "other_delta_0": moved}, (H, W))
        assert np.allclose(pregrasp._state.located["box"]["delta"], anchor), "too few tracks seen: not moved"
        assert pregrasp._state.target.last["state"] == "untrusted"
        pregrasp._apply_others({**share(), "other_delta_0": moved}, (H, W))
        assert np.allclose(pregrasp._state.located["box"]["delta"], moved @ anchor), "moved with its share"
        assert pregrasp._located_info(pregrasp._state.located["box"])["track"] == "tracking"
        pregrasp._store_located("box", {**found, "delta": np.eye(4)})
        pregrasp._apply_others({"others": [], "other_delta_0": moved}, (H, W))
        assert np.allclose(pregrasp._state.located["box"]["delta"], np.eye(4)), (
            "the newer find, not in this frame"
        )
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None
            pregrasp._state.located = {}
            pregrasp._state.target = pregrasp._TargetTrack()


def test_a_tracked_frame_moves_the_place_object_by_its_share(client, tmp_path):
    """The worker answers each tracked frame of the picked object with every other object of its Point2Pose session:
    the share of the object a place goes onto moves that object's find, through the same result as the picked one."""
    demo, _kin, box = _place_demo(tmp_path, time.time())
    anchor = np.eye(4)
    anchor[:3, 3] = [0.03, 0.01, 0.0]
    moved = np.eye(4)
    moved[:3, 3] = [0.0, 0.02, 0.0]
    rgb, depth = np.full((H, W, 3), 200, np.uint8), np.full((H, W), 0.45, np.float32)
    ref = {"object": "gamepad", "ok": True, "delta": np.eye(4), "inliers": 150, "card_points": 400}
    keypoints = {"mode": "features", "concept": "gamepad", "mask": box_mask(), "n_points": 50, "ref": ref}
    keypoints.update(xyz=np.zeros((50, 3)), radius_mm=40.0, shape_class="box", yaw_observable=True, face=None)
    try:
        with pregrasp._state.lock:
            pregrasp._state.demo = demo
            pregrasp._state.teach = pregrasp._Teach(
                at="t", box=(0, 0, 0, 0), rgb=rgb, depth_m=depth, intr=dict(INTR), keypoints=keypoints
            )
            pregrasp._state.track.on, pregrasp._state.track.algo = True, "p2p"
        view = [demo.name, int(demo.objects["box"]["frame"])]
        pregrasp._store_located(
            "box", {"object": "box", "ok": True, "delta": anchor, "mask": box, "tracking": True, "view": view}
        )
        job = pregrasp._queue_job("track", "gamepad", rgb, depth, dict(INTR), algo="p2p", compress=False)
        with pregrasp._state.lock:
            pregrasp._state.track.job = job.id
        others = [{"name": "box", "ok": True, "lost": False, "n_visible": 40, "n_tracks": 50}]
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
        assert np.allclose(pregrasp._state.located["box"]["delta"], moved @ anchor)
        assert client.get("/api/pregrasp/state").json()["located"]["box"]["track"] == "tracking"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.located = {}
            pregrasp._state.target = pregrasp._TargetTrack()
            pregrasp._state.track.on, pregrasp._state.track.job = False, None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


def test_where_an_object_was_last_seen_is_kept_beside_the_demo(tmp_path, monkeypatch):
    """A find or a tracked frame of an object keeps where it was seen, as the click a find there would use, beside the
    saved demo; a tracked object's point is written at most every LAST_SEEN_EVERY_S."""
    demo, _kin, box = _place_demo(tmp_path, time.time())
    demo.root = str(tmp_path / "saved")
    pathlib.Path(demo.root).mkdir()
    path = pathlib.Path(demo.root) / pregrasp.LAST_SEEN_FILE
    monkeypatch.setattr(pregrasp, "_last_seen_written", {})
    try:
        with pregrasp._state.lock:
            pregrasp._state.demo = demo
        pregrasp._remember_seen("box", box, box.shape)
        pregrasp._SEEN_EXECUTOR.submit(lambda: None).result()  # the write is done
        click = json.loads(path.read_text())["box"]["click"]
        assert box[click[1], click[0]], "a click on the box"
        moved = np.roll(box, 80, axis=1)
        pregrasp._remember_seen("box", moved, box.shape)
        pregrasp._SEEN_EXECUTOR.submit(lambda: None).result()
        assert json.loads(path.read_text())["box"]["click"] == click, "not again so soon"
        monkeypatch.setattr(pregrasp, "LAST_SEEN_EVERY_S", 0.0)
        pregrasp._remember_seen("box", moved, box.shape)
        pregrasp._SEEN_EXECUTOR.submit(lambda: None).result()
        new = json.loads(path.read_text())["box"]["click"]
        assert moved[new[1], new[0]] and not box[new[1], new[0]], "where it was seen last"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_a_loaded_demo_finds_its_objects_where_they_were_last_seen_without_a_click(tmp_path, monkeypatch):
    """After a restart nothing in memory says where the objects are: the load finds them again where they were last
    seen, once the worker and the camera are up: the one a place goes onto first, then the picked one by a teach
    against the demo's view whose Point2Pose session starts with both, one start for the two."""
    import asyncio

    from lerobot.gui.api import showservo

    demo, _kin, box = _place_demo(tmp_path, time.time())
    demo.root = str(tmp_path / "saved")
    pathlib.Path(demo.root).mkdir()
    (pathlib.Path(demo.root) / pregrasp.LAST_SEEN_FILE).write_text(
        json.dumps({"gamepad": {"click": [300, 250]}, "box": {"click": [560, 300]}})
    )
    live_rgb = np.full((H, W, 3), 200, np.uint8)
    monkeypatch.setattr(
        pregrasp, "_frame", lambda: _async((live_rgb, np.full((H, W), 0.45, np.float32), dict(INTR)))
    )
    camera = {"on": False}
    monkeypatch.setattr(showservo, "live_camera", lambda: object() if camera["on"] else None)
    asked: list[pregrasp._Job] = []

    async def run():
        async def worker():
            while True:
                await asyncio.sleep(0.002)
                with pregrasp._state.lock:
                    pending = list(pregrasp._state.worker.pending)
                    pregrasp._state.worker.pending.clear()
                for job_id in pending:
                    job = pregrasp._state.worker.jobs[job_id]
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
                        pregrasp._apply_teach_result(job)
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
        refind = asyncio.create_task(pregrasp._refind_last_seen(demo))
        await asyncio.sleep(0.6)
        assert not asked, "nothing before the worker and the camera are up"
        with pregrasp._state.lock:
            pregrasp._state.worker.log.append("worker ready")
        camera["on"] = True
        await asyncio.sleep(0.6)
        assert not asked, "nor while the load's own teach of the demo's object is in flight"
        with pregrasp._state.lock:
            pregrasp._state.teach_job = None
        await asyncio.wait_for(refind, timeout=5.0)
        await asyncio.sleep(0.05)
        feed.cancel()

    try:
        with pregrasp._state.lock:
            pregrasp._state.demo, pregrasp._state.teach, pregrasp._state.test = demo, None, None
            pregrasp._state.located = {}
            pregrasp._state.worker.log = []
            pregrasp._state.teach_job = "the load's own teach"
        pregrasp._state.worker.proc = _FakeProc()
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
        assert pregrasp._state.teach is not None and pregrasp._located(demo, "box")["ok"]
        assert pregrasp._state.target.obj == "box", "the box followed in that session"
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.located = {}
            pregrasp._state.target = pregrasp._TargetTrack()
            pregrasp._state.worker.log = []
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
            pregrasp._state.track.on = False


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


def test_a_mark_the_objects_move_carries_out_of_reach_says_how_far(tmp_path):
    """2026-10-07: "pre-place 1 is out of reach as the object lies now (57 mm short)", and the operator could not see
    why: the cube lay farther out than in the demo, and its move carried the demo's pre-place with it. The reason says
    how far the move carried the mark, and how much of that is away from the arm's base."""
    demo, _kin, _box = _place_demo(tmp_path, time.time())
    teach = pregrasp._Teach(
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
        with pregrasp._state.lock:
            pregrasp._state.teach = teach
        plan = pregrasp._plan_act(
            demo,
            np.eye(4),
            np.eye(4),
            _ReachKinematics(0.20),
            np.array([100.0, 0.0, 60.0, 0, 0, 0, 60.0]),
            (0.04, np.radians(30)),
            ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            1.0,
            0,
            farther,
        )
        assert not plan["ok"]
        assert plan["reason"].startswith("pre-place 1 is out of reach as the object lies now"), plan["reason"]
        assert plan["reason"].endswith(
            "box lies where it carries pre-place 1 50 mm from where the demo's arm was there, 50 mm farther out from "
            "the arm's base"
        ), plan["reason"]
        mark = next(m for m in plan["marks"] if m["label"] == "pre-place 1")
        assert mark["object"] == "box" and mark["moved_mm"] == pytest.approx(50.0, abs=0.5)
    finally:
        with pregrasp._state.lock:
            pregrasp._state.teach = None


def test_an_injected_error_turns_about_its_pivot_and_moves():
    pivot = np.array([0.20, -0.05, 0.03])
    e = pregrasp._inject_transform({"dx_mm": 5.0, "rz_deg": 10.0}, pivot)
    assert np.allclose(e[:3, :3] @ pivot + e[:3, 3], pivot + [0.005, 0.0, 0.0]), "the pivot only moves"
    p = pivot + [0.01, 0.0, 0.0]
    turned = pivot + [0.01 * np.cos(np.radians(10)), 0.01 * np.sin(np.radians(10)), 0.0]
    assert np.allclose(e[:3, :3] @ p + e[:3, 3], turned + [0.005, 0.0, 0.0])
    assert np.allclose(pregrasp._inject_transform({}, pivot), np.eye(4))


def test_an_injected_error_is_capped_and_named():
    from pydantic import ValidationError

    body = pregrasp.ActBody(inject={"at": "find", "dz_mm": -30, "rx_deg": 20}, correct_hold=False)
    assert body.inject.at == "find" and not body.correct_hold
    for bad in ({"at": "aim", "dx_mm": 31}, {"at": "aim", "ry_deg": -21}, {"at": "sideways"}):
        with pytest.raises(ValidationError):
            pregrasp.ActBody(inject=bad)
    assert not pregrasp._injects(pregrasp.InjectBody()), "all zeros injects nothing"


def test_an_aim_injected_into_the_grasp_is_measured_by_the_grasp_pose_and_corrected_at_the_place(
    tmp_path, monkeypatch
):
    """The arm misses its grasp by 5 mm along x and 3 mm down: the act knows where it aimed, so the grasp pose sees the
    miss, and the place takes it out. The miss turns about where the fingertip grips (the fake arm cannot turn, so the
    turn itself is the transform's own test)."""
    end_at = PLACE_AT["place_end"]
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
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, blind=True, inject=inject)
        act = pregrasp._state.act
        assert act.ok, act.reason
        g = demo.tips[pregrasp._grasp_index(demo)]
        assert pivots and np.allclose(pivots[-1], (m["pick"] @ g)[:3, 3]), "turned about the grip"
        miss = pregrasp._inject_transform(inject, (m["pick"] @ g)[:3, 3])
        demo_tip, motion, live_tip = calls[-1]
        assert np.allclose(motion, m["pick"]), "the act's own belief, unbent"
        assert pregrasp.core.pose_residual(live_tip, miss @ m["pick"] @ g)[0] <= 0.0005, (
            "the arm went off its aim"
        )
        expected = np.linalg.inv(g) @ np.linalg.inv(m["pick"]) @ miss @ m["pick"] @ g
        fix = np.asarray(act.place["fix"])
        assert act.place["hold_used"] == "grasp pose"
        assert pregrasp.core.pose_residual(fix, expected)[0] <= 0.0005
        assert pregrasp.core.pose_residual(fix, expected)[1] <= 0.2
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert pregrasp.core.pose_residual(end, m["box"] @ demo.tips[end_at] @ fix)[0] <= 0.0005, (
            "the place corrected by the miss"
        )
        row = pregrasp._load_trials()[-1]
        assert row["inject"] == inject and row["correct_hold"] is True
    finally:
        _end_place_state()


def test_a_find_injected_wrong_is_believed_and_not_corrected(tmp_path, monkeypatch):
    """The gamepad is found 6 mm from where it is: the act believes it, the arm lands where it aimed, the grasp pose
    sees no miss, and the place is not corrected. The error turns about the object's centre. Afterwards nothing is
    left believing the wrong pose."""
    end_at = PLACE_AT["place_end"]
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
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, blind=True, inject=inject)
        act = pregrasp._state.act
        assert act.ok, act.reason
        centre = pregrasp._object_points(demo, "gamepad", np.eye(4)).mean(axis=0)
        assert pivots and np.allclose(pivots[0], m["pick"][:3, :3] @ centre + m["pick"][:3, 3]), (
            "about its centre"
        )
        wrong = pregrasp._inject_transform(inject, m["pick"][:3, :3] @ centre + m["pick"][:3, 3])
        g = demo.tips[pregrasp._grasp_index(demo)]
        _demo_tip, motion, live_tip = calls[-1]
        assert np.allclose(motion, wrong @ m["pick"]), "the act believes the wrong find"
        assert pregrasp.core.pose_residual(live_tip, wrong @ m["pick"] @ g)[0] <= 0.0005
        assert pregrasp.core.pose_residual(np.asarray(act.place["fix"]), np.eye(4))[0] <= 0.0005
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (m["box"] @ demo.tips[end_at])[:3, 3]) <= 0.0005, "not corrected"
        assert act.find_error is None
        assert np.allclose(pregrasp._delta_base(demo, np.eye(4), np.eye(4)), m["pick"]), (
            "the act's belief ended"
        )
    finally:
        _end_place_state()


def test_the_hold_correction_switched_off_replays_the_place_against_the_target(tmp_path, monkeypatch):
    """A baseline for the injections: the place as the demo did it against where the target is now, the hold the
    act measured left out."""
    end_at = PLACE_AT["place_end"]
    try:
        demo, sim, _views, m = _run_place_act(tmp_path, monkeypatch, correct_hold=False)
        act = pregrasp._state.act
        assert act.ok, act.reason
        assert act.place["hold_used"] == "off" and np.allclose(act.place["fix"], np.eye(4))
        end = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (m["box"] @ demo.tips[end_at])[:3, 3]) <= 0.0005
        assert pregrasp._load_trials()[-1]["correct_hold"] is False
    finally:
        _end_place_state()


def test_an_object_the_gripper_hides_is_reported_hidden_not_unheld(tmp_path):
    """The pick-and-place demo held still at its grip for 0.7 s, but the gamepad's own track saw nothing there under
    the gripper; the act said the demo never held it still."""
    import asyncio

    demo, _kin, _box = _place_demo(tmp_path, time.time())
    demo.objects["gamepad"]["seen"][10:25] = False  # under the gripper from 0.67 s to past the lift
    hold, problem = asyncio.run(pregrasp._demo_hold(demo, "gamepad", np.eye(4), lambda: False, "grip"))
    assert hold is None and problem.startswith(
        "gamepad is hidden in the gripper after the gripper closed on it"
    ), problem
