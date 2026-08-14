# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Show-and-Servo capture sessions: the GUI face of the real-rig first-contact bench.

The split keeps the server light. Camera work (connect, preview, capture-to-disk) runs
IN PROCESS — it is pyrealsense2 only. The analysis (SAM3 designation + DINOv3 binding)
runs `benchmarks/showservo_real.py` AS A SUBPROCESS over the captured directory, so the
models never load into the server and the GUI exercises exactly the code path the
command line does — same script, same numbers.

Sessions write the same on-disk format as `benchmarks/showservo_capture.py`
(rgb.png + depth.npy float32 metres + preview.jpg per scene, intrinsics.json once);
either tool can continue a directory the other started. One session at a time.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Dedicated single worker: camera reads are exclusive-device operations that can
# stall for a frame interval — they must never queue behind (or starve) the GUI's
# shared pool. One worker also serialises capture vs preview by construction.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="showservo")
# The arm bus gets its own single worker: a serial retry must not stall frame grabs,
# and one worker serialises commands with position reads by construction.
_ARM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="showservo-arm")

router = APIRouter(prefix="/api/showservo", tags=["showservo"])

# Repo root (…/src/lerobot/gui/api -> repo). Captures live under <repo>/captures.
_REPO = pathlib.Path(__file__).resolve().parents[4]
_CAPTURES = _REPO / "captures"
_BENCH = _REPO / "benchmarks" / "showservo_real.py"
_M1_BENCH = _REPO / "benchmarks" / "showservo_m1.py"


@dataclass
class _Session:
    out: pathlib.Path
    camera: Any | None = None  # RealSenseCamera when live; None when opened from disk
    scenes: list[dict] = field(default_factory=list)


@dataclass
class _Bind:
    proc: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)
    done: bool = False
    ok: bool = False


@dataclass
class _Live:
    """The live worker: a bench subprocess (live fit or M1 servo), fed frames by
    /live/frame.npz and posting annotated JPEGs back to /live/result."""

    proc: subprocess.Popen | None = None
    overlay: bytes | None = None
    log: list[str] = field(default_factory=list)
    kind: str = "live"  # "live" (fit only) or "m1" (fit + arm)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


@dataclass
class _Arm:
    """One connected SO-107 arm, bus only — the M1 worker's actuator, owned here so
    the SERVER holds the safety authority (clamps, stop flag), not the worker."""

    robot: Any | None = None
    arm: str = ""  # "left" | "right"
    start_pos: dict[str, float] = field(default_factory=dict)  # pose at connect
    last_pos: dict[str, float] = field(default_factory=dict)  # last commanded targets
    stopped: bool = False
    moves: int = 0

    @property
    def connected(self) -> bool:
        return self.robot is not None


_session: _Session | None = None
_bind: _Bind = _Bind()
_live: _Live = _Live()
_arm: _Arm = _Arm()
_lock = threading.Lock()

# The three positioning joints M1 may drive. The wrist and gripper are structurally
# out of reach of this API — no clamp needed for a joint that cannot be named. This
# is a safety ALLOWLIST, not a robot description: deriving it from action_features
# would grant exactly the joints it exists to withhold.
M1_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")  # hardcode-ok: M1 safety allowlist
ARM_STEP_LIMIT = 3.0  # per command per joint, normalized units
ARM_EXCURSION_LIMIT = 30.0  # total travel from the connect pose, per joint


def check_arm_move(
    deltas: dict[str, float],
    last_pos: dict[str, float],
    start_pos: dict[str, float],
    *,
    step_limit: float = ARM_STEP_LIMIT,
    excursion_limit: float = ARM_EXCURSION_LIMIT,
) -> dict[str, float]:
    """Validate a relative move and return absolute targets. Pure — the unit-testable
    safety core. Raises ValueError with the reason on ANY violation; a violating
    command is rejected whole, never partially applied or silently clamped, because a
    worker asking for too much is a bug that must surface, not be smoothed over.

    Pre: ``last_pos``/``start_pos`` hold every joint in M1_JOINTS (from connect).
    Post: targets stay within ``excursion_limit`` of the connect pose per joint.
    """
    if not deltas:
        raise ValueError("empty move")
    targets = {}
    for joint, delta in deltas.items():
        if joint not in M1_JOINTS:
            raise ValueError(f"joint {joint!r} is not servoable (allowed: {', '.join(M1_JOINTS)})")
        d = float(delta)
        if not np.isfinite(d):
            raise ValueError(f"{joint}: non-finite delta")
        if abs(d) > step_limit:
            raise ValueError(f"{joint}: step {d:+.2f} exceeds the {step_limit} unit limit")
        target = last_pos[joint] + d
        if abs(target - start_pos[joint]) > excursion_limit:
            raise ValueError(
                f"{joint}: target {target:+.1f} leaves the +/-{excursion_limit} unit "
                "excursion budget around the connect pose"
            )
        targets[joint] = target
    return targets


def _scene_info(path: pathlib.Path) -> dict:
    """Post: one thumbnail-strip entry; depth stats read lazily from the npy."""
    info: dict[str, Any] = {"name": path.name}
    depth_file = path / "depth.npy"
    if depth_file.exists():
        depth = np.load(depth_file, mmap_mode="r")
        valid = np.asarray(depth) > 0
        info["depth_valid"] = float(valid.mean())
        info["median_m"] = float(np.median(np.asarray(depth)[valid])) if valid.any() else 0.0
    info["has_overlay"] = (path / "overlay.jpg").exists()
    # A directory made by hand (or by the sim synthesizer) may lack preview.jpg;
    # the frontend then falls back to the raw rgb.png rather than a broken image.
    info["has_preview"] = (path / "preview.jpg").exists()
    return info


@router.get("/cameras")
async def list_cameras() -> list[dict]:
    """RealSense devices currently attached, for the session dropdown."""
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

    def scan() -> list[dict]:
        return [
            {"serial": str(c.get("serial_number", c.get("id", "?"))), "name": str(c.get("name", ""))}
            for c in RealSenseCamera.find_cameras()
        ]

    return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, scan)


@router.get("/sessions")
async def list_sessions() -> list[dict]:
    """Existing capture directories, newest first, for the reopen dropdown."""
    if not _CAPTURES.exists():
        return []
    out = []
    for p in sorted(_CAPTURES.iterdir(), key=lambda q: q.stat().st_mtime, reverse=True):
        if p.is_dir() and (p / "intrinsics.json").exists():
            out.append({"name": p.name, "scenes": len(list(p.glob("scene_*")))})
    return out


class StartBody(BaseModel):
    serial: str
    name: str = ""
    width: int = 848
    height: int = 480
    fps: int = 30


@router.post("/session/start")
async def start_session(body: StartBody) -> dict:
    """Connect the camera and open a capture directory."""
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    global _session
    with _lock:
        if _session is not None and _session.camera is not None:
            raise HTTPException(409, "a camera session is already live — stop it first")
        name = body.name.strip() or time.strftime("session_%Y%m%d_%H%M%S")
        out = _CAPTURES / name

        # Decimation would halve depth resolution AFTER alignment (breaking the
        # pixel-for-pixel match with color); hole filling invents depth exactly at
        # object boundaries. Both stay off — same choices as the CLI capture tool.
        config = RealSenseCameraConfig(
            serial_number_or_name=body.serial,
            fps=body.fps,
            width=body.width,
            height=body.height,
            use_depth=True,
            enable_decimation=False,
            enable_hole_filling=False,
        )
        camera = RealSenseCamera(config)

    def connect() -> dict:
        camera.connect()
        out.mkdir(parents=True, exist_ok=True)
        intr = camera.color_intrinsics()
        (out / "intrinsics.json").write_text(json.dumps(intr, indent=2))
        return intr

    try:
        intr = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, connect)
    except Exception as e:
        raise HTTPException(500, f"camera connect failed: {e}") from e
    with _lock:
        _session = _Session(out=out, camera=camera)
        _session.scenes = [_scene_info(p) for p in sorted(out.glob("scene_*"))]
    return {"name": out.name, "intrinsics": intr, "scenes": _session.scenes, "live": True}


class OpenBody(BaseModel):
    name: str


@router.post("/session/open")
async def open_session(body: OpenBody) -> dict:
    """Open an existing capture directory, RESUMING capture when possible.

    The camera is re-attached when exactly one RealSense is present, it connects
    (i.e. teleop has released it), and its intrinsics match the session's — mixing
    intrinsics across scenes would silently corrupt the 3D lift, so a mismatched
    camera is left unattached rather than trusted. Otherwise the session opens
    camera-less for re-analysis, exactly as before. This exists because the teleop
    stack and a capture session share one physical camera: the workflow "capture →
    close session → teleop → stop → reopen" must land back in a capturing session,
    not a dead end without a Capture button.
    """
    global _session
    out = _CAPTURES / body.name
    if not (out / "intrinsics.json").exists():
        raise HTTPException(404, f"no capture session named {body.name!r}")
    with _lock:
        if _session is not None and _session.camera is not None:
            raise HTTPException(409, "a camera session is live — stop it first")

    def attach():
        import cv2

        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        cams = RealSenseCamera.find_cameras()
        if len(cams) != 1:
            return None  # zero or ambiguous: the user picks explicitly via New session
        sample = next(iter(sorted(out.glob("scene_*/rgb.png"))), None)
        if sample is None:
            h, w = 480, 848
        else:
            h, w = cv2.imread(str(sample)).shape[:2]
        camera = RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=str(cams[0].get("serial_number", cams[0].get("id", ""))),
                fps=30,
                width=int(w),
                height=int(h),
                use_depth=True,
                enable_decimation=False,
                enable_hole_filling=False,
            )
        )
        camera.connect()
        stored = json.loads((out / "intrinsics.json").read_text())
        intr = camera.color_intrinsics()
        if any(abs(float(intr[k]) - float(stored[k])) > 1.0 for k in ("fx", "fy", "cx", "cy")):
            camera.disconnect()
            return None
        return camera

    try:
        camera = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, attach)
    except Exception:
        camera = None  # busy (teleop still holds it) or absent: re-analysis mode

    with _lock:
        _session = _Session(out=out, camera=camera)
        _session.scenes = [_scene_info(p) for p in sorted(out.glob("scene_*"))]
        return {"name": out.name, "scenes": _session.scenes, "live": camera is not None}


def _stop_live_locked() -> None:
    """Pre: _lock held. Kills the live worker; it also exits by itself on the next
    frame request once the camera/session is gone."""
    if _live.proc is not None and _live.proc.poll() is None:
        _live.proc.terminate()
    _live.proc = None
    _live.overlay = None
    _live.kind = "live"


@router.post("/session/stop")
async def stop_session() -> dict:
    global _session
    with _lock:
        _stop_live_locked()
        session, _session = _session, None
    if session is not None and session.camera is not None:
        await asyncio.get_event_loop().run_in_executor(_EXECUTOR, session.camera.disconnect)
    return {"status": "ok"}


@router.get("/state")
async def state() -> dict:
    with _lock:
        if _session is None:
            return {"session": None}
        return {
            "session": {
                "name": _session.out.name,
                "live": _session.camera is not None,
                "scenes": _session.scenes,
            },
            "bind": {"running": _bind.proc is not None and not _bind.done, "done": _bind.done},
        }


@router.get("/preview.jpg")
async def preview() -> Response:
    """One RGB|depth side-by-side JPEG; the frontend polls this while live."""
    import cv2

    with _lock:
        session = _session
    if session is None or session.camera is None:
        raise HTTPException(409, "no live camera session")

    def grab() -> bytes:
        rgb, depth_mm = session.camera.read_color_and_aligned_depth()
        depth_m = depth_mm.astype(np.float32) / 1000.0
        colored = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth_m, 0, 1.5), alpha=170), cv2.COLORMAP_TURBO
        )
        colored[depth_m <= 0] = 0
        side = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colored], axis=1)
        _ok, jpeg = cv2.imencode(".jpg", side, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    try:
        data = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, grab)
    except Exception as e:
        raise HTTPException(500, f"preview failed: {e}") from e
    return Response(content=data, media_type="image/jpeg")


@router.post("/capture")
async def capture() -> dict:
    """Save one scene — median of 5 depth frames, same format as the CLI tool."""
    import cv2

    with _lock:
        session = _session
    if session is None or session.camera is None:
        raise HTTPException(409, "no live camera session")

    def snap() -> dict:
        rgb, first = session.camera.read_color_and_aligned_depth()
        depths = [first]
        for _ in range(4):
            _rgb, d = session.camera.read_color_and_aligned_depth()
            depths.append(d)
        depth_m = (np.median(np.stack(depths), axis=0) / 1000.0).astype(np.float32)
        assert depth_m.shape == rgb.shape[:2], "depth/color resolution mismatch"

        index = len(list(session.out.glob("scene_*")))
        scene = session.out / f"scene_{index:03d}"
        scene.mkdir(exist_ok=True)
        cv2.imwrite(str(scene / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(scene / "depth.npy", depth_m)
        valid = depth_m > 0
        colored = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth_m, 0, 1.5), alpha=170), cv2.COLORMAP_TURBO
        )
        colored[~valid] = 0
        preview_img = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colored], axis=1)
        cv2.imwrite(str(scene / "preview.jpg"), preview_img)
        return _scene_info(scene)

    info = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, snap)
    with _lock:
        if _session is session:
            session.scenes.append(info)
    return info


@router.get("/scene/{name}/{kind}")
async def scene_file(name: str, kind: str) -> FileResponse:
    """Serve a scene's preview/overlay/rgb image."""
    if kind not in ("preview.jpg", "overlay.jpg", "rgb.png"):
        raise HTTPException(404, "unknown file kind")
    with _lock:
        session = _session
    if session is None:
        raise HTTPException(409, "no session")
    path = (session.out / name / kind).resolve()
    if not path.is_relative_to(session.out.resolve()) or not path.exists():
        raise HTTPException(404, f"{name}/{kind} not found")
    return FileResponse(path)


class BindBody(BaseModel):
    concept: str = ""
    teach: list[int] = [0]
    mask: str = "sam3"  # or "files"


@router.post("/bind")
async def bind(body: BindBody) -> dict:
    """Run the first-contact bench over the session directory, as a subprocess."""
    global _bind
    with _lock:
        session = _session
        if session is None:
            raise HTTPException(409, "no session")
        if _bind.proc is not None and not _bind.done:
            raise HTTPException(409, "a bind run is already in progress")
        if body.mask == "sam3" and not body.concept.strip():
            raise HTTPException(422, "concept is required with SAM3 designation")

        cmd = [
            sys.executable,
            str(_BENCH),
            "--captures",
            str(session.out),
            "--mask",
            body.mask,
            "--teach",
            *[str(i) for i in body.teach],
        ]
        if body.concept.strip():
            cmd += ["--concept", body.concept.strip()]
        env = {"PYTHONPATH": str(_REPO / "src")}
        import os

        env = {**os.environ, **env}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(_REPO)
        )
        _bind = _Bind(proc=proc)

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with _lock:
                _bind.log.append(line.rstrip("\n"))
        code = proc.wait()
        with _lock:
            _bind.done = True
            _bind.ok = code == 0
            if session is _session:
                session.scenes = [_scene_info(p) for p in sorted(session.out.glob("scene_*"))]

    threading.Thread(target=pump, daemon=True, name="showservo-bind").start()
    return {"status": "started"}


@router.get("/bind/log")
async def bind_log() -> dict:
    with _lock:
        return {
            "running": _bind.proc is not None and not _bind.done,
            "done": _bind.done,
            "ok": _bind.ok,
            "log": "\n".join(_bind.log),
        }


class LiveBody(BaseModel):
    concept: str
    teach: list[int] = [0]


@router.post("/live/start")
async def live_start(body: LiveBody, request: Request) -> dict:
    """Spawn the live-fit worker: teach from the marked scenes, then fit every frame."""
    with _lock:
        session = _session
        if session is None or session.camera is None:
            raise HTTPException(409, "live fit needs a live camera session")
        if _bind.proc is not None and not _bind.done:
            raise HTTPException(409, "a batch bind is running — wait for it to finish")
        if _live.running:
            raise HTTPException(409, "live fit is already running")
        if not body.concept.strip():
            raise HTTPException(422, "concept is required for live fit")
        if not body.teach:
            raise HTTPException(422, "mark at least one captured scene as teach")

        cmd = [
            sys.executable,
            str(_BENCH),
            "--captures",
            str(session.out),
            "--concept",
            body.concept.strip(),
            "--teach",
            *[str(i) for i in body.teach],
            "--live",
            str(request.base_url),
        ]
        import os

        env = {**os.environ, "PYTHONPATH": str(_REPO / "src")}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(_REPO)
        )
        _live.proc = proc
        _live.overlay = None
        _live.log = []
        _live.kind = "live"

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with _lock:
                _live.log.append(line.rstrip("\n"))
                del _live.log[:-250]  # teach + probe + transitions; bounded

    threading.Thread(target=pump, daemon=True, name="showservo-live").start()
    return {"status": "started"}


@router.post("/live/stop")
async def live_stop() -> dict:
    with _lock:
        _stop_live_locked()
    return {"status": "ok"}


@router.get("/live/status")
async def live_status() -> dict:
    with _lock:
        return {
            "running": _live.running,
            "kind": _live.kind,
            "has_overlay": _live.overlay is not None,
            # A generous tail: teach diagnosis, probe measurements and state
            # transitions must all survive to the UI — an 8-line tail once showed
            # only the traceback and cost two debugging rounds.
            "log": "\n".join(_live.log[-60:]),
        }


@router.get("/live/frame.npz")
async def live_frame() -> Response:
    """The worker's frame feed: current RGB + aligned depth (float32 metres), as NPZ."""
    with _lock:
        session = _session
        live_running = _live.running
    if session is None or session.camera is None or not live_running:
        raise HTTPException(409, "live fit is not active")  # tells the worker to exit

    def grab() -> bytes:
        import io

        rgb, depth_mm = session.camera.read_color_and_aligned_depth()
        buf = io.BytesIO()
        np.savez_compressed(buf, rgb=rgb, depth=(depth_mm.astype(np.float32) / 1000.0))
        return buf.getvalue()

    data = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, grab)
    return Response(content=data, media_type="application/octet-stream")


@router.post("/live/result")
async def live_result(request: Request) -> dict:
    """The worker's annotated JPEG for the current frame."""
    body = await request.body()
    with _lock:
        if not _live.running:
            raise HTTPException(409, "live fit is not active")
        _live.overlay = body
    return {"status": "ok"}


@router.get("/live/overlay.jpg")
async def live_overlay() -> Response:
    with _lock:
        overlay = _live.overlay
    if overlay is None:
        raise HTTPException(404, "no live overlay yet — the worker is still teaching")
    return Response(content=overlay, media_type="image/jpeg")


# --- the arm (M1) ---------------------------------------------------------------
# Bus only, no cameras: frames keep flowing through the capture session above, so
# the RealSense is never opened twice. The single-arm follower with the bimanual
# profile's per-arm id loads the same calibration file the teleop stack uses.


class ArmConnectBody(BaseModel):
    profile: str = "white"
    arm: str = "right"  # "left" | "right"


def _arm_positions(robot: Any) -> dict[str, float]:
    obs = robot.get_observation()
    return {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}


@router.post("/arm/connect")
async def arm_connect(body: ArmConnectBody) -> dict:
    """Connect ONE arm of a bimanual profile for servoing. The arm should be at rest
    and may briefly relax while motors are configured — same as every teleop connect."""
    from .robot import ROBOT_PROFILES_DIR

    global _arm
    if body.arm not in ("left", "right"):
        raise HTTPException(422, "arm must be 'left' or 'right'")
    with _lock:
        if _arm.connected:
            raise HTTPException(409, "an arm is already connected — disconnect it first")
    profile_path = ROBOT_PROFILES_DIR / f"{body.profile}.json"
    if not profile_path.exists():
        raise HTTPException(404, f"no robot profile named {body.profile!r}")
    profile = json.loads(profile_path.read_text())
    if profile.get("type") != "bi_so107_follower":
        raise HTTPException(
            422, f"profile {body.profile!r} is {profile.get('type')!r}, need bi_so107_follower"
        )
    fields = profile.get("fields", {})

    def connect() -> tuple[Any, dict[str, float]]:
        # The parent package is the live export; the so107_follower SUBPACKAGE is a
        # stale refactor leftover whose config import is broken.
        from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

        cfg = SO107FollowerConfig(
            id=f"{fields.get('id', body.profile)}_{body.arm}",
            port=str(fields[f"{body.arm}_arm_port"]),
            use_degrees=bool(fields.get(f"{body.arm}_arm_use_degrees", False)),
            disable_torque_on_disconnect=bool(
                fields.get(f"{body.arm}_arm_disable_torque_on_disconnect", False)
            ),
            # Robot-layer clamp UNDER the API's own per-step check: two independent
            # belts between a worker bug and a fast move.
            max_relative_target=4.0,
            cameras={},
        )
        robot = SO107Follower(cfg)
        # calibrate=False: the interactive calibrate() would block a server thread on
        # stdin forever. An uncalibrated arm is a hard refusal, not a prompt.
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError(
                f"arm {cfg.id!r} reports uncalibrated — run its calibration from the robot flow first"
            )
        # A motor whose overload protection latched torque off ignores every goal
        # while gear friction holds the pose — the arm LOOKS fine and silently
        # discards commands (field-diagnosed on shoulder_lift after a day of
        # phantom stalls). Refuse loudly instead of servoing a dead joint.
        dead = [j for j in M1_JOINTS if int(robot.bus.read("Torque_Enable", j)) != 1]
        if dead:
            robot.disconnect()
            raise RuntimeError(
                f"torque is OFF on {', '.join(dead)} (overload latch?) — "
                "use Recover in the robot tab, then reconnect"
            )
        return robot, _arm_positions(robot)

    try:
        robot, pos = await asyncio.get_event_loop().run_in_executor(_ARM_EXECUTOR, connect)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"arm connect failed: {e}") from e
    with _lock:
        _arm = _Arm(robot=robot, arm=body.arm, start_pos=dict(pos), last_pos=dict(pos))
    return {"arm": body.arm, "positions": pos}


@router.get("/arm/state")
async def arm_state() -> dict:
    """``positions`` is the REAL encoder read; ``commanded`` is the excursion
    ledger. They were once the same field, and a stalled motor then read back its
    own commands as if executing them — a full debugging day lost to an instrument
    reporting the instruction as the measurement."""
    with _lock:
        if not _arm.connected:
            return {"connected": False}
        robot = _arm.robot
        arm_name, stopped, moves, commanded = _arm.arm, _arm.stopped, _arm.moves, dict(_arm.last_pos)
    try:
        pos = await asyncio.get_event_loop().run_in_executor(_ARM_EXECUTOR, _arm_positions, robot)
    except Exception as e:
        raise HTTPException(500, f"arm read failed: {e}") from e
    return {
        "connected": True,
        "arm": arm_name,
        "stopped": stopped,
        "moves": moves,
        "positions": pos,
        "commanded": commanded,
    }


class ArmMoveBody(BaseModel):
    deltas: dict[str, float]


@router.post("/arm/move")
async def arm_move(body: ArmMoveBody) -> dict:
    """One relative move, validated by `check_arm_move`. 409 once stopped — the M1
    worker treats any non-200 as a hard halt."""
    with _lock:
        if not _arm.connected:
            raise HTTPException(409, "no arm connected")
        if _arm.stopped:
            raise HTTPException(409, "arm is stopped — reconnect to re-arm")
        try:
            targets = check_arm_move(body.deltas, _arm.last_pos, _arm.start_pos)
        except ValueError as e:
            _arm.stopped = True  # a violating command means the worker is buggy: halt
            raise HTTPException(409, f"move refused and arm stopped: {e}") from e
        robot = _arm.robot
        # Excursion accounting uses COMMANDED targets: where the arm will go, known
        # before it gets there — conservative against the read-back racing the motion.
        _arm.last_pos.update(targets)
        _arm.moves += 1

    def act() -> dict[str, float]:
        robot.send_action({f"{j}.pos": v for j, v in targets.items()})
        return _arm_positions(robot)

    try:
        pos = await asyncio.get_event_loop().run_in_executor(_ARM_EXECUTOR, act)
    except Exception as e:
        with _lock:
            _arm.stopped = True
        raise HTTPException(500, f"arm command failed (arm stopped): {e}") from e
    return {"positions": pos, "commanded": targets}


@router.post("/arm/stop")
async def arm_stop() -> dict:
    """Latch the stop flag: every further /arm/move 409s until reconnect. Idempotent;
    the arm HOLDS its position (torque stays on) — it does not go limp."""
    with _lock:
        _arm.stopped = True
    return {"status": "stopped"}


@router.post("/arm/disconnect")
async def arm_disconnect() -> dict:
    global _arm
    with _lock:
        arm, _arm = _arm, _Arm()
    if arm.robot is not None:
        await asyncio.get_event_loop().run_in_executor(_ARM_EXECUTOR, arm.robot.disconnect)
    return {"status": "ok"}


class M1Body(BaseModel):
    concept: str
    held_concept: str
    teach: list[int] = [0]
    arm: str = "right"


@router.post("/m1/start")
async def m1_start(body: M1Body, request: Request) -> dict:
    """Spawn the M1 servo worker into the live slot: same frame feed and overlay
    channel as the live fit, plus the arm endpoints above."""
    with _lock:
        session = _session
        if session is None or session.camera is None:
            raise HTTPException(409, "M1 needs a live camera session")
        if not _arm.connected:
            raise HTTPException(409, "connect the arm first")
        if _arm.stopped:
            raise HTTPException(409, "arm is stopped — reconnect to re-arm")
        if _arm.arm != body.arm:
            raise HTTPException(409, f"the connected arm is {_arm.arm!r}, not {body.arm!r}")
        if _bind.proc is not None and not _bind.done:
            raise HTTPException(409, "a batch bind is running — wait for it to finish")
        if _live.running:
            raise HTTPException(409, "a live worker is already running — stop it first")
        if not body.concept.strip() or not body.held_concept.strip():
            raise HTTPException(422, "both the target and the held concept are required")
        if not body.teach:
            raise HTTPException(422, "mark at least one captured scene as teach")

        cmd = [
            sys.executable,
            str(_M1_BENCH),
            "--captures",
            str(session.out),
            "--concept",
            body.concept.strip(),
            "--held-concept",
            body.held_concept.strip(),
            "--teach",
            *[str(i) for i in body.teach],
            "--arm",
            body.arm,
            "--server",
            str(request.base_url),
        ]
        import os

        env = {**os.environ, "PYTHONPATH": str(_REPO / "src")}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(_REPO)
        )
        _live.proc = proc
        _live.overlay = None
        _live.log = []
        _live.kind = "m1"

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with _lock:
                _live.log.append(line.rstrip("\n"))
                del _live.log[:-250]

    threading.Thread(target=pump, daemon=True, name="showservo-m1").start()
    return {"status": "started"}
