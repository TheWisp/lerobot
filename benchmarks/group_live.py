"""The point groups live: TAPIR tracks the target objects and the points around them, the groups say what moves
with what, and a view shows it as it happens. Runs in Point2Pose's environment (TAPIR and its checkpoint live
there); the groups and the scene helpers are loaded from this checkout by path.

Live:     python benchmarks/group_live.py --live [--record] [--object gamepad=u,v ...] [--server URL] [--port 9141]
          Frames come from the GUI server's camera (a recording it writes frame by frame); the view is an MJPEG
          stream at http://<host>:9141/ , relayed by the GUI on the Approach tab's Groups panel, whose Start runs
          this with --record: colour, depth and the groups' state are kept for the offline replay below. Without
          objects the tracked points are the stable corners nearest the middle of the view. Ctrl-C or SIGTERM
          stops it and the recordings.
Offline:  python benchmarks/group_live.py --recording DIR --out OUT_DIR --object name=u,v ... [--start K --end K]
          Writes OUT_DIR/groups.mp4 and timeline.json and prints, at each reappearance of an object, how far the
          group's estimate and a held pose were from the object's own points.

Each object is the raised body under its click, at the click's height; around it, within RING_PX, the tracker gets
corners and grid points on whatever is there (the tray, its clutter): the points the object borrows while covered."""

from __future__ import annotations

import argparse
import contextlib
import http.server
import importlib.util
import json
import pathlib
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import time
import types
import urllib.error
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


def load_by_path():
    """The groups and the scene helpers without the lerobot package's own imports (torch and the rest)."""
    for name in ("lerobot", "lerobot.showservo"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name].__path__ = []  # a package, so submodules may be registered under it
    mods = {}
    for stem in ("pose", "groups", "groups_scene"):
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
    """The GUI server's camera, through a recording it writes frame by frame: the newest complete frame each time,
    never the same one twice. The recorder allows one recording at a time."""

    ROTATE_S = 30.0  # a recording grows at 12 MB/s: start a new one this often and delete the old (one filled a disk)

    def __init__(self, server: str):
        self.server = server
        self.rec = None
        self.k = None
        self.last = -1
        self.frames = 0
        self._start()

    def _start(self):
        old = self.rec
        if old is not None:
            with contextlib.suppress(urllib.error.HTTPError):  # already stopped with the camera
                self._post("/api/pregrasp/camera/record/stop")
        waited = None
        while True:  # the camera may be off, or a demo recording it: wait, saying so once
            try:
                out = self._post("/api/pregrasp/camera/record/start")
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
            print("the camera is recording again", flush=True)
        if out.get("status") != "recording":
            raise RuntimeError(f"the camera did not start recording: {out}")
        self.rec = pathlib.Path(out["out"])
        self.started = time.time()
        self.last = -1
        for _ in range(100):
            if (self.rec / "cam_K.txt").exists():
                break
            time.sleep(0.1)
        if self.k is None:
            self.k = np.loadtxt(self.rec / "cam_K.txt")
        if old is not None:
            shutil.rmtree(
                old, ignore_errors=True
            )  # safe-destruct: our own recording, every frame of it consumed

    def _post(self, path: str) -> dict:
        req = urllib.request.Request(
            self.server + path, data=b"{}", headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def close(self):
        with contextlib.suppress(Exception):
            self._post("/api/pregrasp/camera/record/stop")
        if self.rec is not None:
            shutil.rmtree(self.rec, ignore_errors=True)

    def __iter__(self):
        while True:
            if time.time() - self.started > self.ROTATE_S:
                self._start()
            depths = sorted((self.rec / "depth").glob("*.png"))
            if len(depths) < 2:
                time.sleep(0.02)
                continue
            n = int(depths[-2].stem)  # the newest may still be being written
            if n <= self.last:
                time.sleep(0.01)
                continue
            bgr = cv2.imread(str(self.rec / "rgb" / f"{n:06d}.jpg"))
            depth = cv2.imread(str(depths[-2]), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth is None:
                continue
            self.last = n
            self.frames += 1
            yield (
                self.frames,
                time.time(),
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                depth.astype(np.float32) / 1000.0,
            )


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

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
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

    def start_recording(self) -> None:
        if self.recording is not None:
            return
        rec = self.record_root / f"groups_{time.strftime('%Y%m%d_%H%M%S')}"
        (rec / "rgb").mkdir(parents=True, exist_ok=True)
        (rec / "depth").mkdir(parents=True, exist_ok=True)
        self.recorded = 0
        self.recording = rec
        print(f"recording to {rec}", flush=True)

    def stop_recording(self) -> None:
        if self.recording is None:
            return
        self.last_recording, self.recording = self.recording, None
        print(f"recorded {self.recorded} frames to {self.last_recording}", flush=True)

    def record(self, stamp: float, rgb: np.ndarray, depth_m: np.ndarray, k: np.ndarray, row: dict) -> None:
        """One frame into the recording in progress: the camera recordings' layout, plus the groups' state."""
        rec, i = self.recording, self.recorded
        if rec is None:
            return
        if i == 0:
            np.savetxt(rec / "cam_K.txt", k)
        cv2.imwrite(
            str(rec / "rgb" / f"{i:06d}.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        cv2.imwrite(str(rec / "depth" / f"{i:06d}.png"), np.rint(depth_m * 1000.0).astype(np.uint16))
        with open(rec / "times.txt", "a") as f:
            f.write(f"{stamp:.6f}\n")
        with open(rec / "groups.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        self.recorded = i + 1


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
    ap.add_argument("--object", action="append", default=[], help="name=u,v on the start frame")
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
    k = source.k
    tracker = groups.GroupTracker()
    surfaces_memory = scene.SurfaceMemory()
    world = scene.World()
    tapir = None
    objects, outlines, centres, rings = {}, {}, {}, {}
    timeline, frame_no, t_wall = [], 0, time.time()
    try:
        for n, stamp, rgb, depth in source:
            pts = scene.points_3d(depth, k)
            if tapir is None:  # the start frame: the bodies and the points to track
                h, w = depth.shape
                region = np.zeros((h, w), bool)
                region[h // 6 : 5 * h // 6, w // 8 : 7 * w // 8] = True
                plane = scene.tray_plane(pts, region)
                height = scene.heights(pts, plane)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                tapir = Tapir(h, w, resize=args.resize, pips_iters=args.pips)
                masks = [scene.object_mask(height, c) for c in clicks]
                # What the objects may borrow: corners anywhere on and around the tray (its rim, markers, clutter)
                # within 60 mm of its plane, the cells nearest each object served first; never the smooth tray
                # itself, which a tracker drifts on.
                field = np.isfinite(pts).all(axis=2) & (np.abs(height) < 0.06)
                for m in masks:
                    field &= ~cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
                for i, name in enumerate(names):
                    own = scene.stable_points(
                        gray, masks[i] & np.isfinite(pts).all(axis=2), N_OBJECT, 24, spacing=6
                    )
                    borrowed = scene.stable_points(gray, field, N_RING, 48, near=np.asarray(clicks[i], float))
                    for u, v in borrowed.astype(
                        int
                    ):  # taken: the next object borrows other corners, not copies
                        cv2.circle(field.view(np.uint8), (int(u), int(v)), 6, 0, -1)
                    idx_own = tapir.add(rgb, own)
                    idx_ring = tapir.add(rgb, borrowed)
                    rings[name] = np.asarray(idx_ring, dtype=int)
                    centre = np.nanmedian(pts[masks[i]], axis=0)
                    pose = np.eye(4)
                    pose[:3, 3] = centre
                    tracker.add_object(name, idx_own, pose)
                    centres[name], outlines[name] = centre, scene.contour_3d(masks[i], pts)
                    objects[name] = (outlines[name], centre, None)
                    print(
                        f"{name}: {masks[i].sum()} px, {len(own)} own points, {len(borrowed)} around it",
                        flush=True,
                    )
                if not names:  # nothing to expand from: the stable corners nearest the middle of the view
                    borrowed = scene.stable_points(gray, field, N_RING, 48, near=np.array([w / 2, h / 2]))
                    tapir.add(rgb, borrowed)
                    print(f"no objects: {len(borrowed)} points around the middle of the view", flush=True)
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
            t0 = time.time()
            tracker.update(xyz, seen)
            t_group = (time.time() - t0) * 1000
            base = world.update(tracker)
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
                    field = np.isfinite(pts).all(axis=2) & (np.abs(scene.heights(pts, plane)) < 0.06)
                    cv2.circle(
                        field.view(np.uint8), (int(at[0]), int(at[1])), 70, 0, -1
                    )  # not the object itself
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
                    field = np.isfinite(pts).all(axis=2) & (np.abs(scene.heights(pts, plane)) < 0.06)
                    for u, v in uv[seen & ~tracker.retired].astype(int):
                        cv2.circle(field.view(np.uint8), (int(u), int(v)), 6, 0, -1)
                    fresh = scene.stable_points(
                        gray, field, min(budget - standing, budget // 4), 48, near=np.array([w / 2, h / 2])
                    )
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
            img = scene.draw(
                rgb,
                k,
                tracker,
                xyz,
                seen,
                objects,
                header,
                "white: the world, the stillest group; coloured: tracks and surfaces moving differently; hollow: hidden, where its group puts it; outline: the object by its group",
                surfaces=surfaces_memory.update(
                    scene.group_surfaces(depth, uv, seen, tracker.group_of, base)
                ),
                base=base,
            )
            if view:
                view.show(img)
                view.record(stamp, rgb, depth, k, row)
            if out:
                cv2.imwrite(str(out / "frames" / f"{frame_no:06d}.jpg"), img)
            frame_no += 1
            if frame_no % 100 == 0:
                print(header + "  " + counts, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if view:
            view.stop_recording()
        if args.live:
            source.close()
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
        report(timeline, names)


def report(timeline, names) -> None:
    """Per object: each stretch its own points did not place it, and at the reappearance how far the group had
    carried it from where its own points then put it, against a pose held from before the stretch."""
    for name in names:
        rows = [r["objects"][name] for r in timeline]
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
                    f"  hidden {i - since} frames (from frame {timeline[since]['frame']}): group carried it "
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
