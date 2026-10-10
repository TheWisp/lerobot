"""The point groups live: TAPIR tracks the target objects and the points around them, the groups say what moves
with what, and a view shows it as it happens. Runs in Point2Pose's environment (TAPIR and its checkpoint live
there); the groups and the scene helpers are loaded from this checkout by path.

Live:     python benchmarks/group_live.py --live [--record] [--object gamepad=u,v ...] [--server URL] [--port 9141]
          Frames come from the GUI server's camera (a recording it writes frame by frame); the view is an MJPEG
          stream at http://<host>:9141/ , relayed by the GUI on the Approach tab's Groups panel, whose Start runs
          this with --record: colour, depth and the groups' state are kept for the offline replay below. Without
          objects the tracked points are stable corners balanced over the view. POST /objects with
          {"name": [u, v], ...} designates objects while it runs, on the frame after the request; GET /objects
          says how each went. GET /poses?since=T gives each designated object's pose (camera <- object, 16 floats)
          on every frame read after T, the time the GUI server stamped on it, what an act carries a hidden object
          by, and the newest frame's picture (groups_scene.drawing), which the act's camera view paints. POST
          /objects/held with {"name": [16 floats] or null} puts an object in the gripper at that pose, or releases
          it, from the next frame. Ctrl-C or SIGTERM stops it and the recordings.
Offline:  python benchmarks/group_live.py --recording DIR --out OUT_DIR --object name=u,v ... [--start K --end K]
          Writes OUT_DIR/groups.mp4 and timeline.json and prints, at each reappearance of an object, how far the
          group's estimate and a held pose were from the object's own points.

Each object is SAM 2.1's mask for its click;
around it the tracker takes corners on whatever has depth, nearest first: the points the object borrows while
covered. Nothing is assumed about planes, trays or heights."""

from __future__ import annotations

import argparse
import collections
import contextlib
import http.server
import importlib.util
import json
import pathlib
import queue
import signal
import socketserver
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "src" / "lerobot" / "showservo"
P2P_REPO = pathlib.Path.home() / ".cache/point2pose/point-to-pose"
SERVER = "http://127.0.0.1:9100"
RECORDINGS = pathlib.Path.home() / ".cache/huggingface/lerobot/demos/.recordings"
RING_PX = (10, 150)  # the borrowed points: clear of the object by the first, out to the second
N_OBJECT, N_RING = 60, 200
RESEED_EVERY = 30  # frames between checks that each object still has borrowed points enough
POSES_KEPT = 900  # frames of the objects' poses GET /poses serves: minutes at the view's rate


def segment_clicks(rgb: np.ndarray, clicks) -> list[np.ndarray]:
    """Each clicked object's mask from SAM 2.1 (Point2Pose's checkpoint), prompted by the click alone: the image
    decides where the object ends, nothing about planes or heights. The model is freed after."""
    if not clicks:
        return []
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(P2P_REPO / "checkpoints/sam2.1/sam2.1_hiera_large.pt"),
        device="cuda",
    )
    pred = SAM2ImagePredictor(model)
    masks = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(rgb)
        for c in clicks:
            out, scores, _ = pred.predict(
                point_coords=np.array([c], float), point_labels=np.array([1]), multimask_output=True
            )
            masks.append(out[int(np.argmax(scores))].astype(bool))
    del pred, model
    torch.cuda.empty_cache()
    return masks


def load_by_path():
    """The groups and the scene helpers without the lerobot package's own imports (torch and the rest)."""
    for name in ("lerobot", "lerobot.showservo"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name].__path__ = []  # a package, so submodules may be registered under it
    mods = {}
    for stem in ("pose", "placement", "groups", "groups_scene", "frame_ring"):
        spec = importlib.util.spec_from_file_location(f"lerobot.showservo.{stem}", SRC / f"{stem}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[stem] = mod
    return mods["groups"], mods["groups_scene"]


class Tapir:
    """TAPIR (causal BootsTAPIR) on the GPU: (x, y) in and out, visibility per point. The points in the resized
    image are what it tracks; 256 square was 83 ms a step for 300 points on the rig's GPU, 480 was 233."""

    def __init__(self, h: int, w: int, resize: int = 256, pips_iters: int = 4):
        import torch
        from omegaconf import OmegaConf

        sys.path.insert(0, str(P2P_REPO))
        from point2pose.data_types.frame import Frame
        from point2pose.modules.tracker.tapir_tracker import TapirTracker

        self.torch, self.Frame = torch, Frame
        cfg = OmegaConf.create(
            {
                "resize_height": resize,
                "resize_width": resize,
                "visible_threshold": 0.5,
                "device": "cuda",
                "checkpoint_path": str(P2P_REPO / "checkpoints/tapir/causal_bootstapir_checkpoint.pt"),
                "img_height": h,
                "img_width": w,
                "num_pips_iter": pips_iters,
            }
        )
        self.tracker = TapirTracker(cfg)
        # Point2Pose estimates the tracks 64 queries at a time, launching the whole refinement for each chunk: the
        # step was bound by kernel launches (27.5 ms for 200 points, 96 for 955); one chunk takes 9.3 and 14.3, the
        # tracks moving a median 0.05 px.
        model, estimate = self.tracker._model, self.tracker._model.estimate_trajectories
        model.estimate_trajectories = lambda *a, **kw: estimate(*a, **{**kw, "query_chunk_size": 4096})
        self.frame_id = 0
        self.n = 0

    def add(self, rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            frame = self.Frame(id=self.frame_id, rgb=rgb)
            if self.n == 0:
                self.tracker.initialize(frame)
            idx = self.tracker.add_query_points(frame, np.asarray(uv, dtype=np.float32))
        self.n += len(idx)
        return idx

    def step(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.frame_id += 1
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            uv, _unc, vis = self.tracker.track_once(self.Frame(id=self.frame_id, rgb=rgb))
        return np.asarray(uv, dtype=np.float32), np.asarray(vis, dtype=bool)


class RecordingSource:
    """Frames of a recording on disk (rgb jpg, depth png in mm, times.txt, cam_K.txt)."""

    def __init__(self, rec: pathlib.Path, start: int, end: int | None, every: int):
        self.rec = rec
        self.k = np.loadtxt(rec / "cam_K.txt")
        self.times = np.atleast_1d(np.loadtxt(rec / "times.txt"))
        self.frames = list(range(start, len(self.times) if end is None else min(end, len(self.times)), every))

    def __iter__(self):
        for n in self.frames:
            bgr = cv2.imread(str(self.rec / "rgb" / f"{n:06d}.jpg"))
            depth = cv2.imread(str(self.rec / "depth" / f"{n:06d}.png"), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth is None:
                continue
            yield (
                n,
                float(self.times[n]),
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                depth.astype(np.float32) / 1000.0,
            )


class LiveSource:
    """The GUI server's camera through shared memory (lerobot.showservo.frame_ring): the newest complete frame each
    time, never the same one twice, stamped with when the server's read of it returned, so the view can say how old a
    result is when it is shown. Waits for the camera, and attaches again when the server shares a new ring."""

    STALE_S = 2.0  # no new frame for this long: the camera stopped, or the server shares another ring

    def __init__(self, server: str):
        self.server = server
        self.ring = None
        self.k = None
        self.last = -1
        self.frames = 0
        self._attach()

    def _attach(self):
        waited = None
        while True:  # the camera may be off: wait, saying so once
            try:
                out = self._post("/api/pregrasp/camera/share/start")
                break
            except urllib.error.HTTPError as e:
                if e.code != 409:
                    raise
                reason = json.loads(e.read() or b"{}").get("detail", "")
                if reason != waited:
                    print(f"waiting for the camera: {reason}", flush=True)
                    waited = reason
                time.sleep(1.0)
        if waited is not None:
            print("the camera is shared again", flush=True)
        if self.ring is not None:
            self.ring.close()
        self.ring = sys.modules["lerobot.showservo.frame_ring"].FrameRing(out["name"])
        self.k = self.ring.k
        self.last = -1
        self.fresh = time.time()

    def _post(self, path: str) -> dict:
        req = urllib.request.Request(
            self.server + path, data=b"{}", headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def close(self):
        if self.ring is not None:
            self.ring.close()
        with contextlib.suppress(Exception):
            self._post("/api/pregrasp/camera/share/stop")

    def __iter__(self):
        while True:
            got = self.ring.read(after=self.last)
            if got is None:
                if time.time() - self.fresh > self.STALE_S:
                    self._attach()
                time.sleep(0.001)
                continue
            n, t_read, rgb, depth_mm = got
            self.last, self.fresh = n, time.time()
            self.frames += 1
            yield self.frames, t_read, rgb, depth_mm.astype(np.float32) / 1000.0


PAGE = """<html><head><title>point groups</title></head><body style='margin:0;background:#111'>
<img src='/stream' style='width:100%' onerror="setTimeout(()=>{this.src='/stream?'+Date.now()},1000)">
</body></html>"""  # the stream alone; the GUI's Groups panel has Start and Finish


class MjpegView:
    """The newest drawn frame as an MJPEG stream, a page that shows it, and the recorder: colour and depth frames
    in the camera recordings' layout (rgb/*.jpg, depth/*.png in mm, times.txt, cam_K.txt), so the replay reads
    them, plus groups.jsonl, the groups' state at each frame, for the analysis. /record/status says what it holds."""

    def __init__(self, port: int, record_root: pathlib.Path):
        view = self
        self.record_root = record_root
        self.recording: pathlib.Path | None = None
        self.last_recording: pathlib.Path | None = None
        self.recorded = 0
        self.requests: queue.Queue = queue.Queue()  # objects to designate, name -> click, for the live loop
        self.designated: dict[str, dict] = {}  # how each designation went, as the live loop reports it
        # Requests are numbered as they arrive; "served" is how many the live loop has carried out, so a caller
        # knows its own request is done when served reaches its ticket.
        self.received = self.served = 0
        self.ticket_lock = threading.Lock()
        self.poses: collections.deque = collections.deque(
            maxlen=POSES_KEPT
        )  # (frame time, {name: pose}), oldest first
        self.drawing: dict | None = (
            None  # the newest frame's picture (groups_scene.drawing), with its frame time
        )
        self.poses_lock = threading.Lock()
        # Objects in the gripper: name -> pose by the arm's joints (camera <- object), None once released; the newest
        # of each, for the live loop (GroupTracker.hold and release).
        self.held: dict[str, np.ndarray | None] = {}
        self.held_lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code: int, obj) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path.startswith("/objects/held"):
                    try:
                        raw = json.loads(
                            self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
                        )
                        held = {
                            str(n): None if m is None else np.asarray(m, dtype=float).reshape(4, 4)
                            for n, m in raw.items()
                        }
                    except (ValueError, TypeError, AttributeError) as e:
                        self._json(
                            400, {"detail": f'held objects are {{"name": [16 floats] or null, ...}}: {e}'}
                        )
                        return
                    with view.held_lock:
                        view.held.update(held)
                    self._json(202, {"held": sorted(n for n, m in held.items() if m is not None)})
                    return
                if not self.path.startswith("/objects"):
                    self._json(404, {"detail": "only /objects and /objects/held take a POST"})
                    return
                try:
                    raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                    asked = {str(name): (int(uv[0]), int(uv[1])) for name, uv in raw.items()}
                except (ValueError, TypeError, IndexError, KeyError, AttributeError) as e:
                    self._json(400, {"detail": f'objects are {{"name": [u, v], ...}}: {e}'})
                    return
                with view.ticket_lock:
                    view.received += 1
                    ticket = view.received
                    view.requests.put(asked)
                self._json(202, {"queued": sorted(asked), "ticket": ticket})

            def do_GET(self):
                if self.path.startswith("/objects"):
                    self._json(200, {"designated": dict(view.designated), "served": view.served})
                    return
                if self.path.startswith("/poses"):
                    query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    try:
                        since = float(query.get("since", ["0"])[0])
                    except ValueError:
                        self._json(400, {"detail": "since is a frame time, in seconds"})
                        return
                    with view.poses_lock:
                        frames = [[t, poses] for t, poses in view.poses if t > since]
                        drawing = view.drawing
                    self._json(200, {"frames": frames, "drawing": drawing})
                    return
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        last = None
                        while True:
                            jpg, stamp = view.jpg, view.stamp
                            if jpg is not None and stamp != last:
                                last = stamp
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                                )
                            time.sleep(0.02)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                elif self.path.startswith("/record/status"):
                    body = json.dumps(
                        {
                            "recording": str(view.recording) if view.recording else None,
                            "frames": view.recorded,
                            "last": str(view.last_recording) if view.last_recording else None,
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.jpg, self.stamp = None, 0.0
        self.server = Server(("0.0.0.0", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def show(self, bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.jpg, self.stamp = buf.tobytes(), time.time()

    def add_poses(self, stamp: float, poses: dict[str, list[float]], drawing: dict | None = None) -> None:
        """One frame's poses of the designated objects, for GET /poses (a frame with none says the view is alive),
        and its picture for another camera view (groups_scene.drawing), the newest kept."""
        with self.poses_lock:
            self.poses.append((stamp, poses))
            if drawing is not None:
                self.drawing = {**drawing, "stamp": stamp}

    def take_held(self) -> dict[str, np.ndarray | None]:
        """The objects held or released since the last call, the newest word on each."""
        with self.held_lock:
            held, self.held = self.held, {}
        return held

    def take_requests(self) -> tuple[dict[str, tuple[int, int]], int]:
        """Every designation asked for since the last call, merged (a later click for a name wins), and how many
        requests that was."""
        asked: dict[str, tuple[int, int]] = {}
        taken = 0
        while True:
            try:
                asked.update(self.requests.get_nowait())
                taken += 1
            except queue.Empty:
                return asked, taken

    def _writer(self) -> None:
        """Writes the recording's frames in order, off the live loop's path; a None ends it."""
        while True:
            item = self.queue.get()
            if item is None:
                return
            rec, i, stamp, rgb, depth_m, k, row, drawn = item
            if i == 0:
                np.savetxt(rec / "cam_K.txt", k)
            cv2.imwrite(
                str(rec / "rgb" / f"{i:06d}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            cv2.imwrite(str(rec / "depth" / f"{i:06d}.png"), np.rint(depth_m * 1000.0).astype(np.uint16))
            if drawn is not None:  # what the view showed, for a video of the live run
                cv2.imwrite(str(rec / "drawn" / f"{i:06d}.jpg"), drawn, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with open(rec / "times.txt", "a") as f:
                f.write(f"{stamp:.6f}\n")
            with open(rec / "groups.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

    def start_recording(self) -> None:
        if self.recording is not None:
            return
        self.queue = queue.Queue(
            maxsize=60
        )  # a bounded backlog: a slow disk slows the view rather than fill memory
        self.writer = threading.Thread(target=self._writer, name="groups-recorder", daemon=True)
        self.writer.start()
        rec = self.record_root / f"groups_{time.strftime('%Y%m%d_%H%M%S')}"
        (rec / "rgb").mkdir(parents=True, exist_ok=True)
        (rec / "depth").mkdir(parents=True, exist_ok=True)
        (rec / "drawn").mkdir(parents=True, exist_ok=True)
        self.recorded = 0
        self.recording = rec
        print(f"recording to {rec}", flush=True)

    def stop_recording(self) -> None:
        if self.recording is None:
            return
        self.last_recording, self.recording = self.recording, None
        self.queue.put(None)
        self.writer.join(timeout=30.0)  # every frame accepted is on disk before the recording is reported
        print(f"recorded {self.recorded} frames to {self.last_recording}", flush=True)

    def record(
        self, stamp: float, rgb: np.ndarray, depth_m: np.ndarray, k: np.ndarray, row: dict, drawn=None
    ) -> None:
        """One frame into the recording in progress (the camera recordings' layout, plus the groups' state), handed
        to the writer."""
        if self.recording is None:
            return
        self.queue.put((self.recording, self.recorded, stamp, rgb, depth_m, k, row, drawn))
        self.recorded += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument(
        "--record",
        action="store_true",
        help="live: keep colour, depth and the groups' state from the first frame",
    )
    ap.add_argument("--server", default=SERVER, help="the GUI server whose camera the live view reads")
    ap.add_argument("--recording")
    ap.add_argument("--out")
    ap.add_argument("--dump", help="replay: save every frame's 3D tracks (xyz, seen, t) to this .npz")
    ap.add_argument("--object", action="append", default=[], help="name=u,v on the start frame")
    ap.add_argument(
        "--designate-at",
        type=int,
        default=None,
        help="replay: designate the --object clicks on this frame instead of the first, as a live request would",
    )
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--port", type=int, default=9141)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--pips", type=int, default=4)
    args = ap.parse_args()
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    )  # a kill stops the recording too
    groups, scene = load_by_path()
    source = (
        LiveSource(args.server)
        if args.live
        else RecordingSource(pathlib.Path(args.recording), args.start, args.end, args.every)
    )
    view = MjpegView(args.port, RECORDINGS) if args.live else None
    if view and args.record:
        view.start_recording()
    out = pathlib.Path(args.out) if args.out else None
    if out:
        (out / "frames").mkdir(parents=True, exist_ok=True)
    names, clicks = [], []
    for spec in args.object:
        name, uv = spec.split("=")
        names.append(name)
        clicks.append(tuple(int(x) for x in uv.split(",")))
    later: dict = {}  # objects to designate on a later frame (--designate-at), through the live request's path
    if args.designate_at is not None:
        later, names, clicks = dict(zip(names, clicks, strict=True)), [], []
    k = source.k
    tracker = groups.GroupTracker()
    surfaces_memory = scene.SurfaceMemory()
    world = scene.World()
    tapir = None
    objects, outlines, centres, rings = {}, {}, {}, {}
    timeline, frame_no, t_wall = [], 0, time.time()
    prev_small, prev_depth, still_for, moving_frames = None, None, None, []
    known_groups: set[int] = set()
    lag_ms: list[float] = []
    dumped: list = []

    def designate(rgb, pts, asked: dict, taken=None) -> dict[str, dict]:
        """Each asked object (name -> click) designated on this frame: SAM 2.1's mask for its click, its own corners,
        and the corners around it that it borrows while covered, the cells nearest it served first; ``taken`` are
        pixels other tracks hold already. Says how each went: one that cannot be designated is reported, not raised.
        Called between frames, so every array of the frame being drawn stays the length it was."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        t_sam = time.time()
        masks = segment_clicks(rgb, list(asked.values()))
        print(f"SAM 2.1: {len(asked)} masks in {time.time() - t_sam:.1f} s, model load included", flush=True)
        # What the objects may borrow: corners on anything with depth but the objects themselves. Corners, so never a
        # smooth surface a tracker drifts on.
        field = np.isfinite(pts).all(axis=2)
        for m in masks:
            field &= ~cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        if taken is not None:
            for u, v in np.asarray(taken).astype(int):
                cv2.circle(field.view(np.uint8), (int(u), int(v)), 6, 0, -1)
        done: dict[str, dict] = {}
        for (name, click), mask in zip(asked.items(), masks, strict=True):
            own = scene.stable_points(gray, mask & np.isfinite(pts).all(axis=2), N_OBJECT, 24, spacing=6)
            borrowed = scene.stable_points(gray, field, N_RING, 48, near=np.asarray(click, float))
            for u, v in borrowed.astype(int):  # taken: the next object borrows other corners, not copies
                cv2.circle(field.view(np.uint8), (int(u), int(v)), 6, 0, -1)
            if len(own) < 4 or not len(borrowed):
                why = f"{len(own)} own and {len(borrowed)} borrowed corners; click it again"
                done[name] = {"ok": False, "reason": why}
                print(f"{name}: {why}", flush=True)
                continue
            idx_own = tapir.add(rgb, own)
            rings[name] = np.asarray(tapir.add(rgb, borrowed), dtype=int)
            centre = np.nanmedian(pts[mask], axis=0)
            pose = np.eye(4)
            pose[:3, 3] = centre
            tracker.add_object(name, idx_own, pose)
            centres[name], outlines[name] = centre, scene.contour_3d(mask, pts)
            objects[name] = (outlines[name], centre, None)
            if name not in names:
                names.append(name)
            done[name] = {"ok": True, "pixels": int(mask.sum()), "own": len(own), "around": len(borrowed)}
            print(f"{name}: {mask.sum()} px, {len(own)} own points, {len(borrowed)} around it", flush=True)
        return done

    try:
        for n, stamp, rgb, depth in source:
            pts = scene.points_3d(depth, k)
            # Two cheap signals a frame: did the picture move (the video's cut of still stretches), and for how many
            # frames each surface has held its depth (corners are seeded only on surfaces still for a second: a
            # hand passing, a sheet still being laid down, are no reference).
            small = cv2.GaussianBlur(
                cv2.resize(rgb[:, :, 1], (rgb.shape[1] // 4, rgb.shape[0] // 4)), (5, 5), 0
            ).astype(np.float32)
            moving_frames.append(
                prev_small is not None and float(np.mean(np.abs(small - prev_small) > 12)) > 0.005
            )
            prev_small = small
            d_half = depth[::2, ::2]
            held = (
                (np.abs(d_half - prev_depth) < 0.005) & (d_half > 0)
                if prev_depth is not None
                else np.zeros(d_half.shape, bool)
            )
            still_for = np.where(held, (0 if still_for is None else still_for) + 1, 0).astype(np.int32)
            prev_depth = d_half
            if tapir is None:  # the start frame: the bodies and the points to track
                h, w = depth.shape
                tapir = Tapir(h, w, resize=args.resize, pips_iters=args.pips)
                if names:
                    done = designate(rgb, pts, dict(zip(names, clicks, strict=True)))
                    failed = {name: d["reason"] for name, d in done.items() if not d["ok"]}
                    if failed:
                        raise SystemExit("; ".join(f"{name}: {why}" for name, why in failed.items()))
                    if view:
                        view.designated.update(done)
                else:  # nothing to expand from: corners balanced over the whole view
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    borrowed = scene.stable_points(gray, np.isfinite(pts).all(axis=2), N_RING, 48)
                    tapir.add(rgb, borrowed)
                    print(f"no objects: {len(borrowed)} points over the view", flush=True)
                budget = len(tapir.tracker.query_points)  # what was seeded: the field is kept near this
                uv = np.asarray(tapir.tracker.query_points.cpu().numpy()[:, [2, 1]], dtype=np.float32)
                uv[:, 0] *= w / args.resize
                uv[:, 1] *= h / args.resize
                vis = np.ones(len(uv), bool)
                t_track = 0.0
            else:
                t0 = time.time()
                uv, vis = tapir.step(rgb)
                t_track = (time.time() - t0) * 1000
            xyz, seen = scene.lookup_3d(uv, vis, pts)
            if args.dump:
                dumped.append((stamp, xyz.astype(np.float32), np.asarray(seen, bool)))
            if view is not None:  # an object in the gripper is where the arm has it
                for name, pose in view.take_held().items():
                    if name in tracker.objects and pose is None:
                        tracker.release(name)
                    elif name in tracker.objects:
                        tracker.hold(name, pose)
            t0 = time.time()
            tracker.update(xyz, seen)
            t_group = (time.time() - t0) * 1000
            base = world.update(tracker)
            for (
                gid
            ) in tracker.groups:  # a newborn group: where its members came from (the replay's diagnosis)
                if gid not in known_groups:
                    known_groups.add(gid)
                    m = np.flatnonzero(tracker.group_of == gid)
                    came = np.bincount(tracker.banned[m] + 1, minlength=1)
                    print(
                        f"frame {n}: g{gid} born with {len(m)} pts; last left: "
                        + ", ".join(f"g{i - 1}:{c}" if i else f"none:{c}" for i, c in enumerate(came) if c)
                        + f"; never left: {int((tracker.leaves[m] == 0).sum())}; hidden now: {int((tracker.unseen[m] >= 3).sum())}"
                        + f"; age min/max: {int(tracker.tenure[m].min())}/{int(tracker.tenure[m].max())}"
                        + f"; uv mean ({np.nanmean(uv[m, 0]):.0f},{np.nanmean(uv[m, 1]):.0f})",
                        flush=True,
                    )
            # Re-seeding: when fewer than half an object's borrowed points still stand in its group (retired, lost
            # to the arm, slid away), new corners near where the object is now take their place.
            if frame_no % RESEED_EVERY == 0 and frame_no > 0:
                gray = None
                for name in names:
                    obj = tracker.objects[name]
                    ring = rings[name]
                    # Standing: in some group and not retired. An object that moved on its own sits in a group of
                    # its own for a while, its borrowed points rightly left behind in the tray's; they are not lost.
                    standing = int(
                        (
                            (tracker.group_of[ring] >= 0)
                            & ~tracker.retired[ring]
                            & (tracker.unseen[ring] < 15)
                        ).sum()
                    )
                    if standing >= N_RING // 2:
                        continue
                    if gray is None:
                        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    at = scene.project(obj.pose[:3, 3], k)[0]
                    if not np.isfinite(at).all():
                        continue
                    field = np.isfinite(pts).all(axis=2)
                    field &= (
                        still_for.repeat(2, axis=0).repeat(2, axis=1)[: field.shape[0], : field.shape[1]]
                        >= 15
                    )
                    field &= ~scene.outline_mask(field.shape, outlines[name], centres[name], obj.pose, k)
                    for u, v in uv[seen].astype(int):  # nor where a track already is
                        cv2.circle(field.view(np.uint8), (int(u), int(v)), 6, 0, -1)
                    fresh = scene.stable_points(gray, field, N_RING - standing, 48, near=at)
                    if len(fresh):
                        rings[name] = np.concatenate([ring, np.asarray(tapir.add(rgb, fresh), dtype=int)])
                        print(
                            f"frame {n}: {name} had {standing} borrowed points standing; {len(fresh)} added",
                            flush=True,
                        )
                # The field as a whole: tracks are lost to hands, paper and drift for good (retired), and a world
                # thinning out paints and explains less. Below 70% of what was seeded, new corners where none is.
                # Hidden tracks (under a hand, a sheet of paper) do not stand: what covers them gets corners of
                # its own, so a body that arrives is tracked, not just the bodies that were there at the start.
                standing = int(((tracker.group_of >= 0) & ~tracker.retired & (tracker.unseen < 15)).sum())
                if standing < budget:
                    if gray is None:
                        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    field = np.isfinite(pts).all(axis=2)
                    field &= (
                        still_for.repeat(2, axis=0).repeat(2, axis=1)[: field.shape[0], : field.shape[1]]
                        >= 15
                    )
                    for u, v in uv[seen & ~tracker.retired].astype(int):
                        cv2.circle(field.view(np.uint8), (int(u), int(v)), 6, 0, -1)
                    fresh = scene.stable_points(gray, field, min(budget - standing, budget // 4), 48)
                    if len(fresh):
                        tapir.add(rgb, fresh)
                        print(
                            f"frame {n}: {standing} tracks standing of {budget}; {len(fresh)} added",
                            flush=True,
                        )
            row = {
                "frame": n,
                "t": stamp,
                "groups": len(tracker.groups),
                "free": int((tracker.group_of == -1).sum()),
                "hidden": int(((tracker.group_of >= 0) & (tracker.unseen >= 3)).sum()),
                "retired": int(tracker.retired.sum()),
                "world": base,
                "moving": bool(moving_frames[-1]),
                "ms": {"track": round(t_track, 1), "group": round(t_group, 1)},
                # each group's motion since its birth: how far its body moved and turned (mm, degrees)
                "motion": {
                    str(g): [
                        round(float(np.linalg.norm(grp.motion.trans)) * 1000, 1),
                        round(
                            float(np.degrees(np.arccos(np.clip((np.trace(grp.motion.rot) - 1) / 2, -1, 1)))),
                            2,
                        ),
                        int((tracker.group_of == g).sum()),
                    ]
                    for g, grp in tracker.groups.items()
                },
                "objects": {},
            }
            for name in names:
                obj = tracker.objects[name]
                row["objects"][name] = {
                    "n_seen": obj.n_seen,
                    "group": obj.group,
                    "own_ok": obj.own_ok,
                    "pose": obj.pose[:3, 3].tolist(),
                    "rot": np.round(obj.pose[:3, :3], 5).tolist(),
                    "carried": None if obj.carried is None else obj.carried[:3, 3].tolist(),
                }
            timeline.append(row)
            fps = frame_no / max(1e-3, time.time() - t_wall)
            header = (
                f"frame {n}  groups {len(tracker.groups)}  free {row['free']}  track {t_track:.0f} ms  "
                f"groups {t_group:.0f} ms  {fps:.1f} fps"
            )
            counts = "  ".join(
                f"g{g}:{int((tracker.group_of == g).sum())}" for g in sorted(tracker.groups)[:6]
            )
            surfaces = surfaces_memory.update(
                scene.group_surfaces(depth, uv, seen, tracker.group_of, base, quiet=world.quiet)
            )
            img = scene.draw(
                rgb,
                k,
                tracker,
                xyz,
                seen,
                objects,
                header,
                "white: the world, the stillest group; coloured: tracks and surfaces moving differently; hollow: hidden, where its group puts it; outline: the object by its group",
                surfaces=surfaces,
                base=base,
                quiet=world.quiet,
            )
            if view:  # the objects' poses for an act, and the picture for the act's own camera view
                view.add_poses(
                    stamp,
                    {name: np.round(tracker.objects[name].pose, 6).ravel().tolist() for name in names},
                    scene.drawing(k, tracker, xyz, seen, objects, surfaces, base, world.quiet),
                )
                view.show(img)
                lag_ms.append(
                    (time.time() - stamp) * 1000
                )  # from the camera read to the drawn frame, published
                row["lag_ms"] = round(lag_ms[-1], 1)
                view.record(stamp, rgb, depth, k, row, img)
            if out:
                cv2.imwrite(str(out / "frames" / f"{frame_no:06d}.jpg"), img)
            frame_no += 1
            if frame_no % 100 == 0:
                lag = f"  lag {np.median(lag_ms[-100:]):.0f} ms" if lag_ms else ""
                print(header + lag + "  " + counts, flush=True)
            if view is not None:  # objects asked for while it runs: designated between frames, on this one
                asked, taken = view.take_requests()
                if asked:
                    view.designated.update(designate(rgb, pts, asked, taken=uv[seen]))
                view.served += taken
            if later and n >= args.designate_at:
                designate(rgb, pts, later, taken=uv[seen])
                later = {}
    except KeyboardInterrupt:
        pass
    finally:
        if view:
            view.stop_recording()
        if args.live:
            source.close()
    if args.dump and dumped:  # the tracks as tracked, for offline comparisons of the grouping on equal input
        n_max = max(len(x) for _, x, _ in dumped)
        xyz_all = np.full((len(dumped), n_max, 3), np.nan, np.float32)
        seen_all = np.zeros((len(dumped), n_max), bool)
        for f, (_, x, v) in enumerate(dumped):
            xyz_all[f, : len(x)], seen_all[f, : len(v)] = x, v
        np.savez_compressed(args.dump, xyz=xyz_all, seen=seen_all, t=np.array([t for t, _, _ in dumped]))
        print(f"dumped {len(dumped)} frames x {n_max} tracks to {args.dump}", flush=True)
    if out:
        (out / "timeline.json").write_text(json.dumps(timeline))
        span = max(1e-3, timeline[-1]["t"] - timeline[0]["t"]) if len(timeline) > 1 else 1.0
        fps = max(1.0, min(30.0, frame_no / span))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                f"{fps:.2f}",
                "-i",
                str(out / "frames" / "%06d.jpg"),
                "-c:v",
                "libx264",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                str(out / "groups.mp4"),
            ],
            check=True,
        )
        # The same video with the still stretches cut: the frames within a second of the picture moving, and the
        # first second; a caption says how much was skipped. groups.mp4 keeps everything.
        keep = np.convolve(np.asarray(moving_frames, dtype=int), np.ones(31, dtype=int), mode="same") > 0
        keep[:15] = True
        cut = out / "cut"
        cut.mkdir(exist_ok=True)
        for f in cut.glob("*.jpg"):
            f.unlink()  # safe-destruct: this run's own cut directory, rewritten below
        j, skipped, show, skipped_s = 0, 0.0, 0, 0.0
        for i, k in enumerate(keep):
            if not k:
                skipped += 1.0 / fps
                continue
            img = cv2.imread(str(out / "frames" / f"{i:06d}.jpg"))
            if skipped > 0:
                show, skipped_s, skipped = 15, skipped, 0.0
            if show > 0:
                cv2.putText(
                    img,
                    f">> {skipped_s:.0f} s with nothing moving skipped",
                    (8, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                show -= 1
            cv2.imwrite(str(cut / f"{j:06d}.jpg"), img)
            j += 1
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                f"{fps:.2f}",
                "-i",
                str(cut / "%06d.jpg"),
                "-c:v",
                "libx264",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                str(out / "groups_cut.mp4"),
            ],
            check=True,
        )
        print(f"groups_cut.mp4: {j} of {len(keep)} frames", flush=True)
        report(timeline, names)


def report(timeline, names) -> None:
    """Per object: each stretch its own points did not place it, and at the reappearance how far the group had
    carried it from where its own points then put it, against a pose held from before the stretch. An object designated
    after the start is reported from the frame it was designated on."""
    for name in names:
        frames = [r["frame"] for r in timeline if name in r["objects"]]
        rows = [r["objects"][name] for r in timeline if name in r["objects"]]
        print(f"\n{name}:")
        held, since, found = None, None, 0
        for i, o in enumerate(rows):
            if o["own_ok"] and since is None:
                held = np.asarray(o["pose"])
            if not o["own_ok"] and since is None and held is not None:
                since = i
            if o["own_ok"] and since is not None:
                own, carried = np.asarray(o["pose"]), np.asarray(o["carried"])
                print(
                    f"  hidden {i - since} frames (from frame {frames[since]}): group carried it "
                    f"{np.linalg.norm(carried - own) * 1000:.1f} mm from its own points, a held pose {np.linalg.norm(held - own) * 1000:.1f} mm"
                )
                since, found = None, found + 1
        if not found:
            print("  never hidden and seen again")
    ms = np.array([[r["ms"]["track"], r["ms"]["group"]] for r in timeline[1:]])
    if len(ms):
        print(
            f"\ntiming: track median {np.median(ms[:, 0]):.0f} ms, groups median {np.median(ms[:, 1]):.0f} ms, max {ms[:, 1].max():.0f} ms"
        )


if __name__ == "__main__":
    main()
