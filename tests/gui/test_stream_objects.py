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
        }
        assert st["demo"]["objects"] == ["gamepad"]
        # The motion the act uses: the live track, times the find, times the inverse of where the demo had the
        # object when the first pre-grasp was shown. Frame 6 is hidden, so the last frame it was seen, 4.
        ref_motion, problem = pregrasp._reference_motion(demo, pregrasp._state.teach)
        assert problem == "" and np.allclose(
            ref_motion, ref_delta @ np.linalg.inv(demo.objects["gamepad"]["deltas"][4])
        )
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
        assert pose_t == pytest.approx(times[4] - demo.t0), (
            "the last frame seen at or before the first pre-grasp"
        )
        assert pregrasp._pose_frame(demo) == 4, "the same frame the act's reference motion uses"
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
