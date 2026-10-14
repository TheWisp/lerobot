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

"""Camera and act recordings, trial persistence and raw/control evidence."""

from __future__ import annotations

import contextlib
import json
import logging
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import acquisition, demos, groups, planning, runtime

logger = logging.getLogger(__name__)
router = APIRouter()


def _share_frames(camera: Any, share: runtime._FrameShare) -> None:
    """Copy every new camera frame into the ring until stopped, taking none from the camera's other readers
    (``read_newer_color_and_depth``; a camera without it is read through the camera executor like every other reader).
    The time stamped on a frame is when its read returned."""
    from .. import showservo

    newer = getattr(camera, "read_newer_color_and_depth", None)
    last: float | None = None
    try:
        while not share.stop.is_set():
            try:
                if newer is not None:
                    rgb, depth_mm, last = newer(last)
                else:
                    rgb, depth_mm = showservo._EXECUTOR.submit(camera.read_color_and_aligned_depth).result()
            except TimeoutError:
                continue  # no new frame yet, or another reader took it
            share.ring.write(rgb, depth_mm.astype(np.uint16), time.time())
            share.n += 1
    except Exception as e:  # the stop endpoint reports it
        logger.exception("camera frame sharing failed")
        share.error = str(e)


def _record_stream(camera: Any, rec: runtime._StreamRecorder) -> None:
    """Write every new camera frame until stopped: rgb/%06d.jpg, depth/%06d.png (uint16 mm), cam_K.txt, times.txt.

    Reads go through the camera executor like every other reader. A read waits for a
    new frame, so no frame is written twice; when the live tracker also reads, the two
    share the camera's frames.
    """
    import cv2

    from .. import showservo

    out = rec.out
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    intr = camera.color_intrinsics()
    np.savetxt(
        out / "cam_K.txt", [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    )
    times: list[float] = []
    try:
        while not rec.stop.is_set():
            try:
                rgb, depth_mm = showservo._EXECUTOR.submit(camera.read_color_and_aligned_depth).result()
            except TimeoutError:
                continue  # the other reader took that frame
            times.append(time.time())
            cv2.imwrite(
                str(out / "rgb" / f"{rec.n:06d}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            cv2.imwrite(
                str(out / "depth" / f"{rec.n:06d}.png"),
                depth_mm.astype(np.uint16),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
            )
            rec.n += 1
    except Exception as e:  # the stop endpoint reports it; the arm's own recording goes on
        logger.exception("camera stream recording failed")
        rec.error = str(e)
    finally:
        np.savetxt(out / "times.txt", times, fmt="%.6f")


def _stream_times(recording: str) -> np.ndarray:
    f = pathlib.Path(recording) / "times.txt"
    return np.atleast_1d(np.loadtxt(f)) if f.exists() and f.stat().st_size else np.zeros(0)


def _stream_frame(recording: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame ``k`` of a recorded stream: (RGB uint8, depth in metres)."""
    import cv2

    root = pathlib.Path(recording)
    bgr = cv2.imread(str(root / "rgb" / f"{k:06d}.jpg"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
    assert bgr is not None and depth is not None, f"frame {k} missing from {root}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth.astype(np.float32) / 1000.0


def _load_trials() -> list[dict[str, Any]]:
    if runtime._trials is None:
        rows: list[dict[str, Any]] = []
        if runtime.TRIALS_PATH.exists():
            for line in runtime.TRIALS_PATH.read_text().splitlines():
                with contextlib.suppress(ValueError):
                    rows.append(json.loads(line))
        runtime._trials = rows
    return runtime._trials


def _save_trials(rows: list[dict[str, Any]]) -> None:
    runtime.TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runtime.TRIALS_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _record_trial(run_dir: str | None = None) -> dict[str, Any]:
    """Append the act that just ended: what was found, which demo was replayed, how it ended, its recording."""
    with runtime._state.lock:
        teach, test, act, track, demo = (
            runtime._state.teach,
            runtime._state.test,
            runtime._state.act,
            runtime._state.track,
            runtime._state.demo,
        )
    r = test.result if test is not None else {}
    row: dict[str, Any] = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "object": (teach.keypoints.get("concept") if teach else None) or "box",
        "source": f"track:{track.algo}" if track.on else "find",
        "n_inliers": r.get("n_inliers"),
        "n_matches": r.get("n_matches"),
        "rms_mm": None if r.get("rms_m") is None else r["rms_m"] * 1000.0,
        "axis_source": r.get("axis_source"),
        "yaw_deg": r.get("yaw_deg"),
        "surface_tilt_deg": r.get("surface_tilt_deg"),
        "face_tilt_deg": r.get("face_tilt_deg"),
        "arm_turn_deg": r.get("arm_turn_deg"),
        "arm_lean_deg": r.get("arm_lean_deg"),
        "demo": None if demo is None else demo.name,
        "speed": act.speed,
        "result": "done" if act.ok else act.step,
        "reason": act.reason,
        "progress": act.progress,
        "verdict": None,
        "run": run_dir,
        "place": act.place,
        "inject": act.inject,
        "correct_hold": act.correct_hold,
        "allow_grasp_prediction": act.allow_grasp_prediction,
    }
    if teach is not None and r.get("ok") and r.get("delta_cam") is not None:
        d = np.asarray(r["delta_cam"])
        c = np.asarray(teach.keypoints["xyz"]).mean(axis=0)
        row["centre_shift_mm"] = float(np.linalg.norm(d[:3, :3] @ c + d[:3, 3] - c) * 1000.0)
    rows = _load_trials()
    rows.append(row)
    _save_trials(rows)
    return row


def _jsonable(o: Any) -> Any:
    return o.tolist() if hasattr(o, "tolist") else str(o)


@dataclass
class _Run:
    """One act's recording: every tracker frame from the arm's first move (image, depth, mask, the pose and
    the points it was fitted on), every target the act gave the arm, and the arm's joints at the loop rate."""

    root: pathlib.Path
    meta: dict[str, Any]
    frames: list[dict[str, Any]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    arm_recording: bool = False

    def frame(
        self,
        job: runtime._Job,
        r: dict[str, Any],
        status: dict[str, Any],
        step: str,
        delta_used: np.ndarray | None,
        transported: np.ndarray | None,
    ) -> None:
        """One tracker result, on the loop; its files are written on the run's own thread."""
        i = len(self.frames)
        self.frames.append(
            {
                "i": i,
                "t_frame": job.created,
                "t_result": time.time(),
                "step": step,
                "used": bool(status.get("ok")),  # the act follows only a frame the server trusted
                **{k: r.get(k) for k in runtime.RUN_FRAME_FIELDS},
                **({"depth_seen": status["depth_seen"]} if "depth_seen" in status else {}),
                **(
                    {"pose_verification": status["pose_verification"]}
                    if "pose_verification" in status
                    else {}
                ),
            }
        )
        for check in status.get("pose_verification") or ():
            if check.get("disagreement"):
                self.meta.setdefault("pose_disagreements", []).append({"frame": i, **check})
        # The session object's fit and every track and mask of it and of each other object of the session (number,
        # pixel, seen), which the act's checks of a view read; the others' names in the order their arrays are numbered.
        self.frames[-1]["others"] = [o.get("name") for o in r.get("others") or ()]
        arrays = {
            k: np.asarray(v)
            for k, v in r.items()
            if v is not None
            and (
                k
                in (
                    "delta",
                    "fit_uv",
                    "fit_inlier",
                    "live_uv",
                    "track_idx",
                    "track_uv",
                    "track_vis",
                    "model_xyz",
                    "model_valid",
                )
                or k.startswith(("other_track_", "other_mask_", "other_delta_", "other_model_", "other_fit_"))
            )
        }
        if delta_used is not None:
            arrays["delta_used"] = np.asarray(delta_used)
        if transported is not None:
            arrays["transported"] = np.asarray(transported)
        # Where the act holds each other object as of this frame (its last placing view, carried by the point groups),
        # the pose a plan made now would aim at: what a replay draws without rebuilding it. ``held`` names them in the
        # order their ``held_k`` arrays are numbered; one not found yet has no array.
        with runtime._state.lock:
            demo = runtime._state.demo
        held_from = []
        for k, name in enumerate([] if demo is None else planning._other_objects(demo)):
            located = acquisition._located(demo, name)
            held_from.append({"object": name})
            if located is not None and located.get("ok") and located.get("delta") is not None:
                held = np.asarray(located["delta"], dtype=float)
                if located.get("stamp") is not None:
                    held = groups._carried(name, float(located["stamp"])) @ held
                arrays[f"held_{k}"] = held
                held_from[-1].update({key: located.get(key) for key in ("find_id", "stamp", "tracked_at")})
        self.frames[-1]["held"] = held_from
        # What carried each object in the point groups as of this frame (their newest picture): every track's pixel and
        # group, each object's group, whether its own points placed it and which tracks are its own, each group's fit.
        with runtime._state.lock:
            drawing = runtime._state.groups_feed.drawing if runtime._state.groups_with_acts else None
        if drawing is not None:
            arrays["groups_points"] = np.asarray(drawing.get("points") or [], dtype=np.int32).reshape(-1, 4)
            if drawing.get("links") is not None:  # the groups' wireframe, as the camera view painted it
                arrays["groups_links"] = np.asarray(drawing["links"], dtype=np.int32).reshape(-1, 2)
                arrays["groups_joins"] = np.asarray(drawing.get("joins") or [], dtype=np.int32).reshape(-1, 2)
                arrays["groups_depth"] = np.asarray(drawing.get("depth") or [], dtype=np.uint16)
            # with the surfaces' label map, so a replay tints what the live view did
            self.frames[-1]["groups"] = {
                k: drawing.get(k)
                for k in ("stamp", "base", "quiet", "objects", "groups", "depth_range", "surfaces", "scale")
            }
        runtime._RUN_EXECUTOR.submit(
            _write_run_frame, self.root, i, job.rgb, job.depth_m, r.get("mask"), arrays
        )

    def target(
        self,
        step: str,
        pose: np.ndarray | None = None,
        joints: np.ndarray | None = None,
        at: float | None = None,
        phase: int | None = None,
        hold_revision: int | None = None,
        composition: dict | None = None,
    ) -> None:
        """A target the act gave the arm, at ``at`` (wall clock) when it went out, or now."""
        entry: dict[str, Any] = {"t": time.time() if at is None else at, "step": step}
        if pose is not None:
            entry["pose"] = np.asarray(pose, dtype=float).tolist()
        if joints is not None:
            entry["joints"] = np.asarray(joints, dtype=float).tolist()
        if phase is not None:
            entry["phase"] = phase
        if hold_revision is not None:
            entry["hold_revision"] = hold_revision
        if composition is not None:
            entry["composition"] = composition
        self.targets.append(entry)


def _write_run_frame(
    root: pathlib.Path,
    i: int,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray | None,
    arrays: dict[str, np.ndarray],
) -> None:
    import cv2

    d = root / "frames"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{i:06d}.jpg"), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    depth_mm = np.round(np.nan_to_num(np.asarray(depth_m, dtype=np.float64)) * 1000.0)
    cv2.imwrite(str(d / f"{i:06d}_depth.png"), np.clip(depth_mm, 0, 65535).astype(np.uint16))
    if mask is not None:
        cv2.imwrite(str(d / f"{i:06d}_mask.png"), np.asarray(mask, dtype=np.uint8) * 255)
    np.savez_compressed(d / f"{i:06d}.npz", **arrays)


def _begin_run(
    demo: runtime._Demo, speed: float, delta: np.ndarray, t_bc: np.ndarray, plan: dict[str, Any]
) -> _Run:
    """Start recording the act, beside its demo when it is saved. Post: ``_state.run`` is the recording."""
    from .. import jog

    with runtime._state.lock:
        teach, track = runtime._state.teach, runtime._state.track
        ref = dict((teach.keypoints.get("ref") or {}) if teach is not None else {})
        algo, fps, last = track.algo, track.fps, dict(track.last or {})
    base = pathlib.Path(demo.root) / runtime.ACTS_DIR if demo.root else demos._demos_root() / ".acts"
    meta: dict[str, Any] = {
        "demo": demo.name,
        "demo_root": demo.root,
        "keypoints": list(demo.keypoints),
        "demo_reference_constraints": dict(demo.reference_constraints),
        "speed": speed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "t_started": time.time(),
        "t_bc": np.asarray(t_bc, dtype=float).tolist(),
        "intr": None if teach is None else dict(teach.intr),
        "delta_at_start": np.asarray(delta, dtype=float).tolist(),
        # the act's objects: the one whose find starts the session (its "find"), then the others joining it
        "objects": [
            planning._session_object(demo)
            or str(((teach.keypoints if teach else {}) or {}).get("concept") or "object"),
            *planning._other_objects(demo),
        ],
        "find": ref,
        "tracker_at_start": {"algo": algo, "fps": fps, **{k: last.get(k) for k in runtime.RUN_FRAME_FIELDS}},
        "plan": {k: plan.get(k) for k in ("ok", "reason", "marks", "summary")},
        "finds": runtime._state.act.finds,  # the same list: what it finds from here on is added
    }
    run = _Run(root=base / time.strftime("%Y%m%d_%H%M%S"), meta=meta)
    try:
        meta["arm_t0"] = jog.start_record()
        run.arm_recording = True
    except RuntimeError as e:
        meta["arm_error"] = str(e)
    with runtime._state.lock:
        runtime._state.run = run
    return run


def _finish_run(run: _Run, samples: list[dict[str, Any]], fk: Any, result: dict[str, Any]) -> str:
    """Write the act's summary and the arm's joints, after every queued frame (the run's thread is in order)."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    run.root.mkdir(parents=True, exist_ok=True)
    if samples:
        arm: dict[str, np.ndarray] = {
            "t": np.array([s["t"] for s in samples], dtype=float) + float(run.meta.get("arm_t0", 0.0)),
            "q_obs": np.array([[s["obs"][m] for m in MOTOR_NAMES] for s in samples], dtype=float),
            "q_cmd": np.array(
                [[s["cmd"].get(m, s["obs"][m]) for m in MOTOR_NAMES] for s in samples], dtype=float
            ),
            "motors": np.array(MOTOR_NAMES),
        }
        for key in ("load", "temps"):  # each motor's load and temperature at every tick, as the jog read them
            if all(key in s for s in samples):
                arm[key] = np.array(
                    [[s[key].get(m, np.nan) for m in MOTOR_NAMES] for s in samples], dtype=float
                )
        with contextlib.suppress(Exception):  # the arm may have gone away; the joints are still the record
            arm["tip_obs"] = np.stack([fk(s["obs"]) for s in samples])
        np.savez_compressed(run.root / "arm.npz", **arm)
    summary = {
        **run.meta,
        "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
        "frames": run.frames,
        "targets": run.targets,
    }
    (run.root / "act.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    return str(run.root)


class VerdictBody(BaseModel):
    index: int
    verdict: str | None = None  # null clears a verdict given by mistake


@router.get("/trials")
async def trials() -> dict:
    return {"rows": _load_trials()}


@router.post("/trials/verdict")
async def trial_verdict(body: VerdictBody) -> dict:
    """The operator's word on a run: what the camera cannot see once the gripper covers the object."""
    if body.verdict is not None and body.verdict not in runtime.TRIAL_VERDICTS:
        raise HTTPException(422, f"verdict must be one of {runtime.TRIAL_VERDICTS}, or null to clear it")
    rows = _load_trials()
    if not 0 <= body.index < len(rows):
        raise HTTPException(404, "no such trial")
    rows[body.index]["verdict"] = body.verdict
    _save_trials(rows)
    return {"index": body.index, "verdict": body.verdict}
