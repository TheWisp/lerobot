#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demonstration persistence, object designations and key-frame editing."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import pathlib
import threading
import time
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from lerobot.showservo import motion as core, symmetry

from . import acquisition, camera, holds, planning, recording, runtime, tracking, worker

router = APIRouter()


def _demos_root() -> pathlib.Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / runtime.DEMOS_NAMESPACE


def _demo_info(demo: runtime._Demo) -> dict[str, Any]:
    return {
        "name": demo.name,
        "concept": demo.concept,
        "n": int(len(demo.t)),
        "seconds": float(demo.t[-1] - demo.t[0]) if len(demo.t) > 1 else 0.0,
        "fps": demo.fps,
        "seen_fraction": float(demo.seen.mean()) if len(demo.seen) else 0.0,
        "frames": 0 if not demo.frames else len(demo.frames),
        "stream_frames": int(len(recording._stream_times(demo.recording))) if demo.recording else 0,
        "objects": [name for name, o in demo.objects.items() if o.get("status") == "done"],
        "root": demo.root,
        "repo_id": f"{runtime.DEMOS_NAMESPACE}/{demo.name}",
        "keypoints": list(demo.keypoints),
        "has_frames": _demo_has_frames(demo),
        "taught": demo.taught,
        "act_objects": planning._act_objects(
            demo
        ),  # the object whose find starts the session first, then the others
        "landing": demo.landing,
        "landing_axis": demo.landing_axis,
        "reference_constraints": dict(demo.reference_constraints),
    }


def _demo_from_samples(name: str, concept: str, samples: list, history: list, fk, t0: float) -> runtime._Demo:
    """A demo from the jog's samples and the tracker's history.

    Each sample's object pose is the tracker's nearest certified frame within
    :data:`HISTORY_MATCH_S` of it. The demo's start pose is the first seen
    one, or the identity when the object was never seen, which takes it to sit
    where it was taught. ``fk`` maps observed joints to the fingertip pose.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    t = np.array([s["t"] for s in samples], dtype=float)
    q_obs = np.array([[s["obs"][m] for m in MOTOR_NAMES] for s in samples], dtype=float)
    q_cmd = np.array([[s["cmd"].get(m, s["obs"][m]) for m in MOTOR_NAMES] for s in samples], dtype=float)
    tips = np.stack([fk(s["obs"]) for s in samples])
    grips = q_obs[:, MOTOR_NAMES.index("gripper")]
    seen_hist = [(w, np.asarray(d, dtype=float)) for (w, ok, d) in history if ok and d is not None]
    deltas = np.tile(np.eye(4), (len(samples), 1, 1))
    seen = np.zeros(len(samples), dtype=bool)
    if seen_hist:
        hw = np.array([w for w, _ in seen_hist])
        for i, ti in enumerate(t):
            k = int(np.argmin(np.abs(hw - (t0 + ti))))
            if abs(hw[k] - (t0 + ti)) <= runtime.HISTORY_MATCH_S:
                deltas[i], seen[i] = seen_hist[k][1], True
    first = np.flatnonzero(seen)
    delta0 = deltas[first[0]].copy() if len(first) else np.eye(4)
    fps = float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 30.0
    return runtime._Demo(
        name=name,
        concept=concept,
        fps=fps,
        t=t,
        tips=tips,
        grippers=grips,
        q_obs=q_obs,
        q_cmd=q_cmd,
        deltas=deltas,
        seen=seen,
        delta0=delta0,
        t0=t0,
    )


class DemoNameBody(BaseModel):
    name: str | None = None


@router.post("/demo/new")
async def demo_new() -> dict:
    """Prepare an independent demonstration without touching saved datasets or moving the arm."""
    with runtime._state.lock:
        s = runtime._state
        if s.act.on or s.recording is not None or s.camera_recording is not None:
            raise HTTPException(409, "stop the current act or recording before starting a new demo")
        if s.demo is not None and s.demo.root is None:
            raise HTTPException(409, "save the current demo before starting a new one")
    tasks = list(runtime._refinds)
    for task in tasks:
        task.cancel()
    await tracking.track_stop()
    if s.track.task is not None:
        s.track.task.cancel()
        tasks.append(s.track.task)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    with runtime._state.lock:
        s.demo = s.teach = s.test = None
        s.teach_job = s.find_job = None
        s.located.clear()
        s.targets.clear()
        s.track.follow = False
        s.track.history.clear()
        s.track.last = {}
        s.track.overlay = None
        s.track.job = None
        s.track.model = None
        s.track.depths = runtime._PointDepths()
        s.track.verification.clear()
        s.track.verification_conflicts.clear()
        s.track.hold_views.clear()
    return {"status": "new demo"}


@router.post("/demo/record/start")
async def demo_record_start() -> dict:
    """Record the arm (any mode: the leader or the gizmo) and the camera's colour and depth.

    Nothing has to be taught first: the objects that matter are designated on the
    recording afterwards. While tracking runs, the tracked object's motion is
    recorded too.
    """
    from .. import jog, showservo

    with runtime._state.lock:
        if runtime._state.camera_recording is not None:
            raise HTTPException(409, "the camera is being recorded on its own; stop that first")
    try:
        t0 = jog.start_record()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    camera = showservo.live_camera()
    stream = None
    if camera is not None:
        stream = runtime._StreamRecorder(out=_demos_root() / ".recordings" / time.strftime("%Y%m%d_%H%M%S"))
        stream.thread = threading.Thread(
            target=recording._record_stream, args=(camera, stream), name="pregrasp-demo-stream", daemon=True
        )
        stream.thread.start()
    with runtime._state.lock:
        runtime._state.recording = {"t0": t0, "frames": []}
        runtime._state.stream = stream
        tracking = runtime._state.track.on
    return {"status": "recording", "tracking": tracking, "camera": stream is not None}


@router.post("/demo/record/stop")
async def demo_record_stop(body: DemoNameBody) -> dict:
    """End the recording and keep it as the current demo (not yet saved)."""
    from .. import jog, showservo

    with runtime._state.lock:
        rec, teach, history = (
            runtime._state.recording,
            runtime._state.teach,
            list(runtime._state.track.history),
        )
        stream, previous = runtime._state.stream, runtime._state.demo
        runtime._state.recording, runtime._state.stream = None, None
    if rec is None:
        raise HTTPException(409, "not recording")
    if stream is not None and stream.thread is not None:
        stream.stop.set()
        await asyncio.get_event_loop().run_in_executor(runtime._RENDER_EXECUTOR, stream.thread.join, 5.0)
    try:
        samples = jog.stop_record()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    if len(samples) < 2:
        raise HTTPException(409, "the recording is empty")
    name = _safe_name(body.name) or time.strftime("demo_%Y%m%d_%H%M%S")
    concept = teach.keypoints["concept"] if teach is not None else "demo"
    _discard_unsaved_stream(previous)

    def build() -> runtime._Demo:
        return _demo_from_samples(name, concept, samples, history, jog.fk_tip, rec["t0"])

    demo = await asyncio.get_event_loop().run_in_executor(runtime._RENDER_EXECUTOR, build)
    demo.frames = rec["frames"] or None
    demo.taught = teach is not None
    demo.camera = _camera_label()
    camera = showservo.live_camera()
    if teach is not None:
        demo.intr = dict(teach.intr)
    elif camera is not None:
        demo.intr = dict(camera.color_intrinsics())
    if stream is not None and stream.n:
        demo.recording = str(stream.out)
    with runtime._state.lock:
        runtime._state.demo = demo
        if teach is not None:
            teach.tip_pose, teach.gripper = demo.tips[0].copy(), float(demo.grippers[0])
    info = _demo_info(demo)
    if stream is not None and stream.error:
        info["stream_error"] = stream.error
    return info


def _discard_unsaved_stream(demo: runtime._Demo | None) -> None:
    """A recorded demo that was never saved is replaced by the next one; its working stream goes with it."""
    import shutil

    if demo is None or demo.root is not None or demo.recording is None:
        return
    work = pathlib.Path(demo.recording)
    if work.parent == _demos_root() / ".recordings" and work.exists():
        shutil.rmtree(work)  # safe-destruct: our own working copy of a demo the operator never saved


def _camera_label() -> str:
    """The live camera's name, as the dataset's image feature is named after it."""
    from .. import showservo

    cam = showservo.live_camera()
    raw = str(getattr(getattr(cam, "config", None), "serial_number_or_name", "") or "camera")
    return _safe_name(raw) or "camera"


def _safe_name(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name.strip())
    return cleaned or None


def _write_demo(demo: runtime._Demo, teach: runtime._Teach | None) -> pathlib.Path:
    """The demo as a LeRobot dataset the Data tab can play, plus a sidecar with what the act needs.

    The camera stream, when one was recorded, moves into the demo as ``recording/``;
    the dataset's video is made from it. ``teach`` is the object taught before the
    demo, if any; a demo recorded without one designates its objects afterwards.
    """
    import shutil

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
    from lerobot.utils.constants import OBS_IMAGES

    root = _demos_root() / demo.name
    if root.exists() and (demo.root is None or pathlib.Path(demo.root).resolve() != root.resolve()):
        raise HTTPException(409, f"a different demo named {demo.name!r} already exists; choose a new name")
    stream = pathlib.Path(demo.recording) if demo.recording else None
    if stream is not None and root in stream.parents:
        # A re-save replaces the folder below; the stream steps aside first.
        parked = _demos_root() / ".recordings" / f"{demo.name}_resave_{time.strftime('%H%M%S')}"
        parked.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stream), str(parked))  # safe-destruct: our own stream, moved aside, not deleted
        stream = parked
    if root.exists():
        # safe-destruct: our own demos folder; saving under a taken name replaces that demo
        shutil.rmtree(root)
    nj = len(MOTOR_NAMES)
    matrix_names = [f"m{i}" for i in range(16)]
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (nj,), "names": list(MOTOR_NAMES)},
        "action": {"dtype": "float32", "shape": (nj,), "names": list(MOTOR_NAMES)},
        "tip.pose": {"dtype": "float32", "shape": (16,), "names": matrix_names},
        "object.delta": {"dtype": "float32", "shape": (16,), "names": matrix_names},
        "object.seen": {"dtype": "float32", "shape": (1,), "names": ["seen"]},
    }
    # The video's frames are read one at a time: a long stream does not fit in memory at once.
    frame_times, video_frame = _video_source(demo, stream)
    image_key = f"{OBS_IMAGES}.{demo.camera}"
    frames = frame_times is not None and len(frame_times) > 0
    if frames:
        h, w = video_frame(0).shape[:2]
        features[image_key] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    ds = LeRobotDataset.create(
        f"{runtime.DEMOS_NAMESPACE}/{demo.name}",
        fps=max(1, int(round(demo.fps))),
        features=features,
        root=root,
        use_videos=True,
    )
    for i in range(len(demo.t)):
        frame: dict[str, Any] = {
            "observation.state": demo.q_obs[i].astype(np.float32),
            "action": demo.q_cmd[i].astype(np.float32),
            "tip.pose": demo.tips[i].reshape(16).astype(np.float32),
            "object.delta": demo.deltas[i].reshape(16).astype(np.float32),
            "object.seen": np.array([float(demo.seen[i])], dtype=np.float32),
            "task": demo.concept,
        }
        if frames:
            k = int(np.argmin(np.abs(frame_times - (demo.t0 + demo.t[i]))))
            frame[image_key] = np.ascontiguousarray(video_frame(k), dtype=np.uint8)
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()
    taught: dict[str, Any] = {}
    if teach is not None and demo.taught:
        taught = {
            "teach_rgb": teach.rgb,
            "teach_depth": teach.depth_m,
            "teach_mask": np.asarray(teach.keypoints.get("mask", np.zeros(teach.depth_m.shape, dtype=bool))),
        }
    np.savez_compressed(
        root / runtime.DEMO_FILE,
        name=demo.name,
        concept=demo.concept,
        fps=demo.fps,
        t=demo.t,
        tips=demo.tips,
        grippers=demo.grippers,
        q_obs=demo.q_obs,
        q_cmd=demo.q_cmd,
        deltas=demo.deltas,
        seen=demo.seen,
        delta0=demo.delta0,
        t0=demo.t0,
        camera=demo.camera,
        intr=json.dumps(teach.intr if teach is not None else (demo.intr or {})),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        taught=demo.taught,
        **taught,
    )
    _write_keypoints(root, demo.keypoints, demo.landing, demo.landing_axis)
    _write_objects(root, demo.objects)
    if stream is not None and stream.exists():
        # safe-destruct: our own stream, moved into its demo
        shutil.move(str(stream), str(root / runtime.DEMO_RECORDING))
        demo.recording = str(root / runtime.DEMO_RECORDING)
    return root


def _video_source(demo: runtime._Demo, stream: pathlib.Path | None) -> tuple[np.ndarray | None, Any]:
    """The dataset video's frame times and a reader for frame ``k`` at half size: the stream, else the small frames.

    The reader keeps the last frame, since consecutive arm samples usually fall on the same camera frame.
    """

    if stream is not None and (stream / "times.txt").exists():

        @functools.lru_cache(maxsize=1)
        def from_stream(k: int) -> np.ndarray:
            return recording._stream_frame(str(stream), k)[0][::2, ::2].copy()

        return recording._stream_times(str(stream)), from_stream
    if demo.frames:
        return np.array([f[0] for f in demo.frames]), lambda k: demo.frames[k][1]
    return None, None


def _write_keypoints(
    root: pathlib.Path,
    keypoints: list[dict[str, Any]],
    landing: str = "exact",
    landing_axis: str = "surface_normal",
) -> None:
    """The operator's marks and the place's landing rule as a sidecar the act reads back; nothing is written when
    there are no marks and the landing is the default."""
    f = pathlib.Path(root) / runtime.KEYPOINTS_FILE
    if keypoints or landing != "exact" or landing_axis != "surface_normal":
        f.write_text(
            json.dumps(
                {
                    "keypoints": keypoints,
                    **({"landing": landing} if landing != "exact" else {}),
                    **({"landing_axis": landing_axis} if landing_axis != "surface_normal" else {}),
                },
                indent=1,
            )
        )
    elif f.exists():
        f.unlink()  # safe-destruct: the marks sidecar we wrote ourselves; the operator cleared the marks


def _read_landing(root: pathlib.Path) -> str:
    """The place's landing rule saved beside a demo: "exact" when none was saved or it is not one this version reads."""
    f = pathlib.Path(root) / runtime.KEYPOINTS_FILE
    try:
        landing = json.loads(f.read_text()).get("landing", "exact") if f.exists() else "exact"
    except (OSError, ValueError, AttributeError):
        return "exact"
    return landing if landing in core.LANDINGS else "exact"


def _read_landing_axis(root: pathlib.Path) -> str:
    """Old demos turn around the measured face normal."""
    f = pathlib.Path(root) / runtime.KEYPOINTS_FILE
    try:
        axis = (
            json.loads(f.read_text()).get("landing_axis", "surface_normal")
            if f.exists()
            else "surface_normal"
        )
    except (OSError, ValueError, AttributeError):
        return "surface_normal"
    return axis if axis in symmetry.AXES else "surface_normal"


def _read_keypoints(root: pathlib.Path) -> list[dict[str, Any]]:
    f = pathlib.Path(root) / runtime.KEYPOINTS_FILE
    if not f.exists():
        return []
    try:
        return list(json.loads(f.read_text()).get("keypoints", []))
    except (OSError, ValueError):
        return []


@router.post("/demo/save")
async def demo_save(body: DemoNameBody) -> dict:
    """Write the current demo as a dataset under the demos namespace; a name given here renames it."""
    with runtime._state.lock:
        demo, teach = runtime._state.demo, runtime._state.teach
    if demo is None:
        raise HTTPException(409, "record a demo first")
    name = _safe_name(body.name)
    if name:
        demo.name = name
    root = await asyncio.get_event_loop().run_in_executor(runtime._RENDER_EXECUTOR, _write_demo, demo, teach)
    demo.root = str(root)
    return _demo_info(demo)


@router.get("/demos")
async def demos() -> dict:
    out = []
    for f in sorted(_demos_root().glob(f"*/{runtime.DEMO_FILE}")):
        with contextlib.suppress(Exception):
            z = np.load(f, allow_pickle=False)
            t = z["t"]
            out.append(
                {
                    "name": str(z["name"]),
                    "concept": str(z["concept"]),
                    "n": int(len(t)),
                    "seconds": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
                    "created": str(z["created"]),
                    "root": str(f.parent),
                    "repo_id": f"{runtime.DEMOS_NAMESPACE}/{z['name']}",
                }
            )
    return {"demos": out}


class DemoLoadBody(BaseModel):
    name: str


@router.post("/demo/load")
async def demo_load(body: DemoLoadBody) -> dict:
    """A saved demo becomes the current one; its object is re-taught to the worker from the saved frame."""
    f = _demos_root() / body.name / runtime.DEMO_FILE
    if not f.exists():
        raise HTTPException(404, f"no demo named {body.name!r}")
    z = np.load(f, allow_pickle=False)
    demo = runtime._Demo(
        name=str(z["name"]),
        concept=str(z["concept"]),
        fps=float(z["fps"]),
        t=np.asarray(z["t"], dtype=float),
        tips=np.asarray(z["tips"], dtype=float),
        grippers=np.asarray(z["grippers"], dtype=float),
        q_obs=np.asarray(z["q_obs"], dtype=float),
        q_cmd=np.asarray(z["q_cmd"], dtype=float),
        deltas=np.asarray(z["deltas"], dtype=float),
        seen=np.asarray(z["seen"]).astype(bool),
        delta0=np.asarray(z["delta0"], dtype=float),
        t0=float(z["t0"]),
        camera=str(z["camera"]) if "camera" in z.files else "camera",
        root=str(f.parent),
        intr=json.loads(str(z["intr"])),
        keypoints=_read_keypoints(f.parent),
        landing=_read_landing(f.parent),
        landing_axis=_read_landing_axis(f.parent),
        recording=str(f.parent / runtime.DEMO_RECORDING)
        if (f.parent / runtime.DEMO_RECORDING / "times.txt").exists()
        else None,
        objects=_read_objects(f.parent),
        taught=bool(z["taught"]) if "taught" in z.files else "teach_rgb" in z.files,
    )
    if core.keypoints_problem(demo.keypoints, float(demo.t[0]), float(demo.t[-1])):
        demo.keypoints = []  # marks in a shape this version does not read
    await asyncio.get_running_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, holds._prepare_demo_references, demo
    )
    if "teach_rgb" not in z.files:  # recorded without a taught object: nothing to re-teach
        with runtime._state.lock:
            _discard_unsaved_stream(runtime._state.demo)
            runtime._state.demo = demo
            runtime._state.test = None
        acquisition._start_refind(demo)
        return {**_demo_info(demo), "teach_pending": False}
    with runtime._state.lock:
        running = runtime._state.worker.running
    if not running:
        raise HTTPException(409, "start the worker first")
    mask = np.asarray(z["teach_mask"]).astype(bool) if "teach_mask" in z.files else None
    click = None
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        click = [int(np.median(xs)), int(np.median(ys))]
    with runtime._state.lock:
        algo = runtime._state.track.algo
    job = worker._queue_job(
        "teach",
        demo.concept,
        np.asarray(z["teach_rgb"]),
        np.asarray(z["teach_depth"]),
        json.loads(str(z["intr"])),
        algo=algo,
        click=click,
    )
    job.log = {"why": "demo loaded: the demo's own view of the object", "came_from": {"demo_frame": "teach"}}
    with runtime._state.lock:
        _discard_unsaved_stream(runtime._state.demo)
        runtime._state.teach_job = job.id
        runtime._state.demo = demo
        runtime._state.test = None
    acquisition._start_refind(
        demo
    )  # after that teach: the objects designated on the demo, where they were last seen
    return {**_demo_info(demo), "teach_pending": True}


def _demo_video_file(root: str) -> pathlib.Path | None:
    files = sorted(pathlib.Path(root).glob("videos/**/*.mp4"))
    return files[0] if files else None


def _decode_demo_video(path: pathlib.Path) -> list[bytes]:
    """Every frame of the saved demo video as JPEG, decoded once; the dataset's AV1 needs PyAV's dav1d."""
    import av
    import cv2

    out: list[bytes] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            rgb = frame.to_ndarray(format="rgb24")
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if ok:
                out.append(buf.tobytes())
    return out


def _demo_has_frames(demo: runtime._Demo) -> bool:
    return (
        demo.recording is not None
        or bool(demo.frames)
        or bool(demo.video)
        or (demo.root is not None and _demo_video_file(demo.root) is not None)
    )


def _demo_frame_rgb(demo: runtime._Demo, i: int) -> np.ndarray | None:
    """The camera frame nearest sample ``i``: the recorded stream at full size, else the small frames, else the video."""
    import cv2

    if demo.recording is not None:
        times = recording._stream_times(demo.recording)
        if len(times):
            return recording._stream_frame(
                demo.recording, int(np.argmin(np.abs(times - (demo.t0 + demo.t[i]))))
            )[0]
    if demo.frames:
        times = np.array([f[0] for f in demo.frames])
        k = int(np.argmin(np.abs(times - (demo.t0 + demo.t[i]))))
        return np.asarray(demo.frames[k][1])
    if demo.video is None and demo.root is not None:
        f = _demo_video_file(demo.root)
        demo.video = _decode_demo_video(f) if f is not None else []
    if demo.video:
        bgr = cv2.imdecode(np.frombuffer(demo.video[min(i, len(demo.video) - 1)], np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


@router.get("/demo/curve")
async def demo_curve() -> dict:
    """What the editor draws under its slider: the demo's time axis, gripper, visibility, marks and fingertip path."""
    with runtime._state.lock:
        demo = runtime._state.demo
    if demo is None:
        raise HTTPException(404, "no demo")
    return {
        "name": demo.name,
        "n": int(len(demo.t)),
        "t": demo.t.tolist(),
        "gripper": demo.grippers.tolist(),
        "seen": _demo_seen(demo).astype(int).tolist(),
        "keypoints": list(demo.keypoints),
        "landing": demo.landing,
        "landing_axis": demo.landing_axis,
        "has_frames": _demo_has_frames(demo),
        "recording": demo.recording is not None,
        "taught": demo.taught,
        "image_size": None if demo.intr is None else [demo.intr["width"], demo.intr["height"]],
        "uv": _demo_path_uv(demo),
        "grip_t": None if (g := holds._firm_grip(demo)) is None else float(demo.t[g]),
    }


def _demo_seen(demo: runtime._Demo) -> np.ndarray:
    """Per sample, whether the object the act follows was seen: the designated object's own track on the recording
    when the marks name one (the track whose outline the editor draws), else the live tracker's frames during the
    recording, which come a few times a second and leave samples between them unseen."""
    obj = planning._session_object(demo)
    o = demo.objects.get(obj) if obj else None
    times = recording._stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    if o is None or o.get("status") != "done" or not len(times):
        return demo.seen
    frames = np.abs(times[None, :] - (demo.t0 + demo.t)[:, None]).argmin(axis=1)
    return np.asarray(o["seen"], dtype=bool)[frames]


def _demo_path_uv(demo: runtime._Demo) -> list[list[int] | None] | None:
    """The fingertip path in the camera image (full-resolution pixels), or None without a camera calibration."""
    try:
        t_bc = camera._t_base_cam()
    except HTTPException:
        return None
    if demo.intr is None:
        return None
    return [
        None if (uv := camera._project(t_bc, demo.intr, tip[:3, 3])) is None else [uv[0], uv[1]]
        for tip in demo.tips
    ]


@router.get("/demo/frame.jpg")
async def demo_frame(i: int = 0) -> Response:
    """Frame ``i`` of the current demo with the designated objects' outlines; the page draws path and marks."""
    with runtime._state.lock:
        demo = runtime._state.demo
    if demo is None:
        raise HTTPException(404, "no demo")
    if not _demo_has_frames(demo):
        raise HTTPException(404, "this demo has no camera frames")
    i = int(np.clip(i, 0, len(demo.t) - 1))

    def render() -> bytes:
        import cv2

        rgb = _demo_frame_rgb(demo, i)
        assert rgb is not None, "has_frames said so"
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        if demo.recording is not None and demo.objects:
            times = recording._stream_times(demo.recording)
            _draw_objects(bgr, demo, int(np.argmin(np.abs(times - (demo.t0 + demo.t[i])))))
        return camera._jpeg(bgr)

    jpg = await asyncio.get_event_loop().run_in_executor(runtime._RENDER_EXECUTOR, render)
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


class ObjectBody(BaseModel):
    i: int  # the demo sample the editor shows
    x: float  # the clicked pixel, in the stream's full-size frame
    y: float
    name: str


@router.post("/demo/objects")
async def demo_object_add(body: ObjectBody) -> dict:
    """Designate an object by clicking it on a frame of the recording; it is then tracked through the whole stream."""
    with runtime._state.lock:
        demo, running = runtime._state.demo, runtime._state.worker.running
    if demo is None:
        raise HTTPException(409, "record or load a demo first")
    if demo.recording is None:
        raise HTTPException(409, "this demo has no camera stream to designate on")
    if not running:
        raise HTTPException(409, "start the worker first")
    name = _safe_name(body.name)
    if not name:
        raise HTTPException(422, "name the object")
    times = recording._stream_times(demo.recording)
    i = int(np.clip(body.i, 0, len(demo.t) - 1))
    k = int(np.argmin(np.abs(times - (demo.t0 + demo.t[i]))))
    rgb, depth = await asyncio.get_event_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, recording._stream_frame, demo.recording, k
    )
    if not (0 <= body.x < rgb.shape[1] and 0 <= body.y < rgb.shape[0]):
        raise HTTPException(422, "the click is outside the frame")
    job = worker._queue_job(
        "stream_object",
        name,
        rgb,
        depth,
        dict(demo.intr or {}),
        click=[int(round(body.x)), int(round(body.y))],
    )
    job.extra = {"recording": demo.recording, "frame": k, "name": name}
    with runtime._state.lock:
        demo.objects[name] = {"frame": k, "click": job.click, "status": "tracking", "job": job.id}
    return _objects_info(demo)


class ObjectNameBody(BaseModel):
    name: str


@router.post("/demo/objects/remove")
async def demo_object_remove(body: ObjectNameBody) -> dict:
    with runtime._state.lock:
        demo = runtime._state.demo
        if demo is None or body.name not in demo.objects:
            raise HTTPException(404, "no such object")
        del demo.objects[body.name]
    if demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            runtime._RENDER_EXECUTOR, _write_objects, pathlib.Path(demo.root), demo.objects
        )
    return _objects_info(demo)


class ObjectSymmetryBody(BaseModel):
    name: str
    # An order alone was implicitly about the table normal. Keep old requests
    # readable; new declarations name both the observed axis and turn family.
    order: int = Field(1, ge=1, le=12)
    kind: Literal["none", "discrete", "continuous"] | None = None
    axis: Literal["surface_normal", "long_axis"] = "surface_normal"
    end_swap: bool = False


@router.post("/demo/objects/symmetry")
async def demo_object_symmetry(body: ObjectSymmetryBody) -> dict:
    """Declare appearance symmetry around an axis measured in the object's depth surface.

    This resolves ambiguous object *poses*. It does not by itself grant freedom
    to change a contact relation or to grasp a shape from another side.
    """
    try:
        spec = symmetry.normalise(
            {
                "kind": body.kind or ("discrete" if body.order > 1 else "none"),
                "axis": body.axis,
                "order": body.order,
                "end_swap": body.end_swap,
            }
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with runtime._state.lock:
        demo = runtime._state.demo
        if demo is None or body.name not in demo.objects:
            raise HTTPException(404, "no such object")
        demo.objects[body.name]["symmetry"] = spec
    if demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            runtime._RENDER_EXECUTOR, _write_objects, pathlib.Path(demo.root), demo.objects
        )
    return _objects_info(demo)


@router.get("/demo/objects")
async def demo_objects() -> dict:
    with runtime._state.lock:
        demo = runtime._state.demo
    if demo is None:
        raise HTTPException(404, "no demo")
    return _objects_info(demo)


def _objects_info(demo: runtime._Demo) -> dict[str, Any]:
    """The designated objects for the editor: name, where it was clicked, how far tracking got and how much it saw."""
    times = recording._stream_times(demo.recording) if demo.recording else np.zeros(0)
    with runtime._state.lock:
        jobs = dict(runtime._state.worker.jobs)
    out = []
    for n, (name, o) in enumerate(demo.objects.items()):
        job = jobs.get(o.get("job", ""))
        b, g, r = runtime.OBJECT_COLOURS[n % len(runtime.OBJECT_COLOURS)]
        out.append(
            {
                "name": name,
                "colour": f"#{r:02x}{g:02x}{b:02x}",
                "frame": int(o["frame"]),
                "t": float(times[o["frame"]] - demo.t0) if len(times) > o["frame"] else None,
                "status": o["status"],
                "progress": 1.0 if o["status"] != "tracking" else (job.progress if job else 0.0),
                "seen_fraction": float(np.mean(o["seen"])) if o.get("seen") is not None else None,
                "reason": o.get("reason", ""),
                "symmetry": symmetry.normalise(o.get("symmetry")),
                "shape_evidence": _shape_evidence(demo, name) if o["status"] == "done" else None,
                **(planning._object_pose_info(demo, name) if o["status"] == "done" else {}),
            }
        )
    return {"objects": out}


def _shape_evidence(demo: runtime._Demo, name: str) -> dict[str, Any] | None:
    """A cached, visible-surface suggestion, never a declaration of full symmetry.

    A single RGB-D view can support a round long axis without showing the
    underside, texture or grip contact. Keep the measured dimensions and fit
    error visible to the operator; do not silently change the saved symmetry.
    """
    obj = demo.objects[name]
    if "shape_evidence" in obj:
        return obj["shape_evidence"]
    if demo.recording is None:
        return None
    try:
        points = planning._object_points(demo, name, np.eye(4))
        fit = symmetry.round_long_axis(points)
    except (OSError, ValueError, IndexError):
        return None
    axis_uv = None
    if fit["ok"]:
        try:
            k = np.loadtxt(pathlib.Path(demo.recording) / "cam_K.txt")
            ends = (
                np.asarray(fit["point"])[None, :]
                + np.array([-0.45, 0.45])[:, None] * fit["length_m"] * np.asarray(fit["axis"])[None, :]
            )
            if np.all(ends[:, 2] > 0):
                pixels = ends[:, :2] / ends[:, 2, None] @ k[:2, :2].T + k[:2, 2]
                axis_uv = pixels.tolist()
        except (OSError, ValueError):
            pass
    evidence = (
        {
            "kind": "round_long_axis",
            "length_mm": fit["length_m"] * 1000.0,
            "radius_mm": fit["radius_m"] * 1000.0,
            "radial_error_mm": fit["median_radial_error_m"] * 1000.0,
            "axis_camera": np.asarray(fit["axis"], float).tolist(),
            "axis_uv": axis_uv,
        }
        if fit["ok"]
        else None
    )
    obj["shape_evidence"] = evidence
    return evidence


async def _apply_object_result(job: runtime._Job) -> None:
    """Keep what the worker tracked for an object on the demo's stream, and write it beside a saved demo.

    Post on success: the object's entry has ``deltas`` (K, 4, 4), its motion from the
    clicked frame in camera coordinates; ``seen`` (K,); ``masks`` (K, h/4, w/4); ``mask``,
    full size on the clicked frame; one row per stream frame.
    """
    r = job.result or {}
    with runtime._state.lock:
        demo = runtime._state.demo
        entry = None
        if demo is not None:
            entry = next((o for o in demo.objects.values() if o.get("job") == job.id), None)
        if entry is None:
            return  # the object was removed, or another demo is current
        if r.get("ok"):
            entry.update(
                status="done",
                deltas=np.asarray(r["deltas"], dtype=float),
                seen=np.asarray(r["seen"]).astype(bool),
                masks=np.asarray(r["masks"]).astype(bool),
                mask=np.asarray(r["mask"]).astype(bool),
            )
        else:
            entry.update(status="failed", reason=str(r.get("reason", "the worker gave no reason")))
    if demo is not None and demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            runtime._RENDER_EXECUTOR, _write_objects, pathlib.Path(demo.root), demo.objects
        )


def _write_objects(root: pathlib.Path, objects: dict[str, dict[str, Any]]) -> None:
    """The tracked objects beside a saved demo; objects still tracking, or failed, are not kept."""
    done = {n: o for n, o in objects.items() if o.get("status") == "done"}
    f = pathlib.Path(root) / runtime.OBJECTS_FILE
    if not done:
        if f.exists():
            f.unlink()  # safe-destruct: our own sidecar; the operator removed the last object
        return
    arrays: dict[str, Any] = {
        "names": json.dumps(
            [
                [n, int(o["frame"]), list(o["click"]), symmetry.normalise(o.get("symmetry"))]
                for n, o in done.items()
            ]
        )
    }
    for n, o in enumerate(done.values()):
        for key in ("deltas", "seen", "masks", "mask"):
            arrays[f"o{n}_{key}"] = o[key]
    np.savez_compressed(f, **arrays)


def _read_objects(root: pathlib.Path) -> dict[str, dict[str, Any]]:
    f = pathlib.Path(root) / runtime.OBJECTS_FILE
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=False)
    out: dict[str, dict[str, Any]] = {}
    for n, (name, frame, click, *rest) in enumerate(json.loads(str(z["names"]))):
        out[name] = {
            "frame": int(frame),
            "click": list(click),
            "symmetry": symmetry.normalise(rest[0] if rest else None),
            "status": "done",
            **{key: np.asarray(z[f"o{n}_{key}"]) for key in ("deltas", "seen", "masks", "mask")},
        }
    return out


def _draw_objects(bgr: np.ndarray, demo: runtime._Demo, k: int) -> None:
    """Each tracked object's pose, outline and name on stream frame ``k``, in its colour."""
    import cv2

    try:
        t_bc = camera._t_base_cam()
    except HTTPException:
        t_bc = None  # without the arm's camera calibration there is no vertical to draw the pose against
    for n, (name, o) in enumerate(demo.objects.items()):
        if o.get("status") != "done":
            continue
        if t_bc is not None and demo.intr is not None:
            _draw_pose(bgr, demo, o, k, t_bc)
        if not o["seen"][k]:
            continue
        m = o["masks"][k].astype(np.uint8)
        if not m.any():
            continue
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = runtime.OBJECT_COLOURS[n % len(runtime.OBJECT_COLOURS)]
        cv2.drawContours(bgr, contours, -1, colour, 2, cv2.LINE_AA)
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.putText(bgr, name, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def _draw_pose(bgr: np.ndarray, demo: runtime._Demo, o: dict[str, Any], k: int, t_bc: np.ndarray) -> None:
    """The object's tracked pose on stream frame ``k``: its axes at its tracked centre, built from the arm base's at
    the frame it was clicked on (x red, y green, up blue), with true vertical as a thin white line beside up. A
    hidden object's held pose is drawn dimmed."""
    import cv2

    if "centre0" not in o:  # the object's centre where it was clicked, from that frame's depth
        depth = recording._stream_frame(demo.recording, int(o["frame"]))[1]
        mask = cv2.resize(
            np.asarray(o["mask"], dtype=np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        ys, xs = np.nonzero(mask & (depth > 0.05))
        if not len(xs):
            o["centre0"] = None
        else:
            i, z = demo.intr, depth[ys, xs]
            o["centre0"] = np.array(
                [((xs - i["cx"]) * z / i["fx"]).mean(), ((ys - i["cy"]) * z / i["fy"]).mean(), z.mean()]
            )
    c0 = o["centre0"]
    if c0 is None:
        return
    d = np.asarray(o["deltas"][k], dtype=float)
    c = d[:3, :3] @ c0 + d[:3, 3]
    axes_cam = t_bc[:3, :3].T  # the base's x, y, z as columns, in camera coordinates
    i = demo.intr

    def px(p: np.ndarray) -> tuple[int, int] | None:
        return (
            None
            if p[2] <= 1e-6
            else (int(round(i["fx"] * p[0] / p[2] + i["cx"])), int(round(i["fy"] * p[1] / p[2] + i["cy"])))
        )

    origin = px(c)
    if origin is None:
        return
    seen = bool(o["seen"][k])
    up = px(c + runtime.POSE_UP_M * axes_cam[:, 2])
    if up is not None:
        cv2.line(bgr, origin, up, (255, 255, 255), 1, cv2.LINE_AA)
    for axis, colour in zip(range(3), ((0, 0, 255), (0, 200, 0), (255, 80, 0)), strict=True):
        tip = px(
            c + (runtime.POSE_UP_M if axis == 2 else runtime.POSE_AXIS_M) * (d[:3, :3] @ axes_cam[:, axis])
        )
        if tip is not None:
            cv2.arrowedLine(
                bgr,
                origin,
                tip,
                colour if seen else tuple(v // 2 for v in colour),
                2,
                cv2.LINE_AA,
                tipLength=0.2,
            )


class KeypointsBody(BaseModel):
    keypoints: list[dict[str, Any]]


@router.post("/demo/keypoints")
async def demo_keypoints(body: KeypointsBody) -> dict:
    """Replace the demo's marks (:func:`_marks_problem` says why not); kept beside a saved demo."""
    with runtime._state.lock:
        demo = runtime._state.demo
    if demo is None:
        raise HTTPException(409, "record or load a demo first")
    kps = planning._normal_marks(body.keypoints)
    problem = await asyncio.get_event_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, planning._marks_problem, demo, kps
    )
    if problem:
        raise HTTPException(422, problem)
    with runtime._state.lock:
        demo.keypoints = kps
    if demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            runtime._RENDER_EXECUTOR,
            _write_keypoints,
            pathlib.Path(demo.root),
            kps,
            demo.landing,
            demo.landing_axis,
        )
    return _demo_info(demo)


@router.post("/demo/key_frames/check")
async def demo_key_frames_check(body: KeypointsBody) -> dict:
    """What the editor shows of marks it has not saved: why they cannot be saved ("" when they can), the key frames
    they make with whether the demo made each holding something, when the demo held something, and the object a
    landing rule would turn."""
    with runtime._state.lock:
        demo = runtime._state.demo
    if demo is None:
        raise HTTPException(409, "record or load a demo first")
    kps = planning._normal_marks(body.keypoints)
    problem = await asyncio.get_event_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, planning._marks_problem, demo, kps
    )
    shape = core.keypoints_problem(kps, float(demo.t[0]), float(demo.t[-1]))
    trial = runtime._Demo(**{**demo.__dict__, "keypoints": kps}) if not shape else demo
    frames = core.key_frames(trial.keypoints) if not shape else []
    holdings = planning._holdings(trial) if frames else []
    holding = next(iter(holdings), None)
    return {
        "problem": problem,
        "key_frames": [
            {
                **k,
                "held": core.holding_at(holdings, trial.t, frames, n) is not None,
                "held_object": (core.holding_at(holdings, trial.t, frames, n) or {}).get("frame"),
            }
            for n, k in enumerate(frames)
        ],
        "holds": [
            {
                "object": h["frame"],
                "grip_t": float(trial.t[h["grip"]]),
                "release_t": None if h["release"] is None else float(trial.t[h["release"]]),
            }
            for h in holdings
        ],
        "hold": None
        if holding is None
        else {
            "object": holding["frame"],
            "grip_t": float(trial.t[holding["grip"]]),
            "release_t": None if holding["release"] is None else float(trial.t[holding["release"]]),
        },
        "landing_object": planning._landing_object(trial) if frames else None,
    }


class LandingBody(BaseModel):
    landing: str
    axis: str | None = None


@router.post("/demo/landing")
async def demo_landing(body: LandingBody) -> dict:
    """How the place may land on its object: "exact" as shown, "symmetry" turned by any of that object's symmetric
    turns, "turn" turned any amount about its middle. The act takes, of those, a landing the servos reach whose joints
    stay nearest the demo's. Kept beside a saved demo."""
    if body.landing not in core.LANDINGS:
        raise HTTPException(422, f"a landing is one of {', '.join(core.LANDINGS)}")
    if body.axis is not None and body.axis not in symmetry.AXES:
        raise HTTPException(422, f"a landing axis is one of {', '.join(symmetry.AXES)}")
    with runtime._state.lock:
        demo = runtime._state.demo
        if demo is None:
            raise HTTPException(409, "record or load a demo first")
        demo.landing = body.landing
        if body.axis is not None:
            demo.landing_axis = body.axis
        kps = list(demo.keypoints)
    if demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            runtime._RENDER_EXECUTOR,
            _write_keypoints,
            pathlib.Path(demo.root),
            kps,
            demo.landing,
            demo.landing_axis,
        )
    return _demo_info(demo)
