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

"""The act engine run from the GUI, beside the key-frame runner and chosen per act (``runner="engine"``).

What the engine reads, filled from the GUI (src/lerobot/showservo/docs/act_execution.md, "What the engine connects
to"): the arm from the jog, the motor loop's live playback, and a scene log the tracker writes into on every frame it
places: each object's 2D tracks, the frame's depth, and the views the pose rule accepted.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import logging
import time
from typing import Any

import numpy as np

from lerobot.showservo import motion as core
from lerobot.showservo.engine.engine import Arm, Engine
from lerobot.showservo.engine.objects import Objects
from lerobot.showservo.engine.perturb import Perturbations
from lerobot.showservo.engine.scene import Camera, Depth, SceneLog, Tracks, View
from lerobot.showservo.engine.timeline import Demo, Timeline

from . import acquisition, camera, planning, recording, runtime

logger = logging.getLogger(__name__)


class _JogPlayback:
    """The jog's live playback as the engine's queue; commands go out as the jog's motor dictionaries."""

    def __init__(self, names: list[str]):
        self.names = names

    def _dict(self, q: np.ndarray) -> dict[str, float]:
        return {m: float(v) for m, v in zip(self.names, q, strict=True)}

    def sent(self) -> int:
        from .. import jog

        return jog.playback_state()[0]

    def clock(self) -> float:
        from .. import jog

        return jog.playback_clock()

    def append(self, expected_tail: int, samples: list[np.ndarray], due_s: list[float]) -> bool:
        from .. import jog

        return jog.append_playback_chunk(expected_tail, [self._dict(q) for q in samples], due_s)

    def cancel(self, expected_sent: int) -> bool:
        from .. import jog

        return jog.replace_playback_tail(expected_sent, [], [])


class _ArmScene(SceneLog):
    """The scene log, the fingertip read from the jog's own history of checked reads (the camera's stamps and the
    jog's reads share the host's perf-counter clock)."""

    def __init__(self, cam: Camera, kin: Any, names: list[str]):
        super().__init__(cam, keep_s=120.0)
        self.kin, self.names = kin, names

    def tip_at(self, t: float, max_gap_s: float = 0.2) -> np.ndarray | None:
        from .. import jog

        q = jog.joints_at(t, 2.0 / jog.HZ)
        if q is None:
            return None
        return np.asarray(self.kin.forward_kinematics(np.array([q[m] for m in self.names])), dtype=float)


class _Feed:
    """Each frame the tracker placed, into the scene log: the session object's and the others' tracks, the depth,
    and the views the pose rule accepted, each at the frame's camera stamp."""

    def __init__(
        self, scene: SceneLog, demo: runtime._Demo, t_bc: np.ndarray, session: str, others: list[str]
    ):
        self.scene, self.demo, self.t_bc, self.session, self.others = scene, demo, t_bc, session, others
        self.frames = 0

    def __call__(self, job: runtime._Job, r: dict[str, Any], result: dict[str, Any] | None) -> None:
        t = job.camera_stamp_mono
        if t is None:
            return
        self.frames += 1
        self.scene.add_depth(Depth(t=t, metres=np.asarray(job.depth_m, dtype=np.float32)))
        if r.get("track_uv") is not None:
            self.scene.add_tracks(self.session, _tracks(t, r, "track_idx", "track_uv", "track_vis"))
        if result is not None and result.get("delta_cam") is not None:
            self.scene.add_view(
                self.session,
                View(t=t, motion=planning._delta_base(self.demo, result["delta_cam"], self.t_bc)),
            )
        names = [o.get("name") for o in r.get("others") or ()]
        for name in self.others:
            if name not in names:
                continue
            k = names.index(name)
            if r.get(f"other_track_uv_{k}") is not None:
                self.scene.add_tracks(
                    name, _tracks(t, r, f"other_track_idx_{k}", f"other_track_uv_{k}", f"other_track_vis_{k}")
                )
            with runtime._state.lock:
                found = runtime._state.located.get(name)
            if found is not None and found.get("stamp") == t:  # this frame's view placed it
                motion, problem = planning._object_motion(self.demo, name, self.t_bc)
                if motion is not None and not problem:
                    self.scene.add_view(name, View(t=t, motion=motion))


def _tracks(t: float, r: dict[str, Any], idx: str, uv: str, vis: str) -> Tracks:
    points = np.asarray(r[uv], dtype=float).reshape(-1, 2)
    ids = np.asarray(r.get(idx, np.arange(len(points))), dtype=int).reshape(-1)
    seen = np.asarray(r.get(vis, np.ones(len(points), dtype=bool)), dtype=bool).reshape(-1)
    n = min(len(points), len(ids), len(seen))
    return Tracks(t=t, ids=ids[:n], uv=points[:n], visible=seen[:n])


def _camera(teach: runtime._Teach, t_bc: np.ndarray, shape: tuple[int, ...]) -> Camera:
    i = teach.intr
    k = np.array([[i["fx"], 0.0, i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]])
    return Camera(K=k, t_bc=np.asarray(t_bc, dtype=float), size=(int(shape[0]), int(shape[1])))


async def _find_objects(demo: runtime._Demo, obj: str | None, others: list[str], stopped) -> str:
    """Find every object of the act afresh before the arm moves, as the key-frame runner does: the others first,
    then the session object, whose find restarts the live session with the others joining it. "" or why not."""
    act = runtime._state.act
    relocate = {
        name: asyncio.create_task(acquisition._locate_afresh(name, stopped, track=obj is None))
        for name in others
    }
    founds = {}
    for name in others:
        act.step = f"finding {name}"
        why = await relocate[name]
        if why:
            return why
        founds[name] = acquisition._located(demo, name)
    if obj is None:
        return ""
    more = {n: f["mask"] for n, f in founds.items() if f and f["ok"] and f.get("mask") is not None}
    act.step = f"finding {obj}"
    why = await acquisition._find_afresh(obj, stopped, more or None)
    if why:
        return why
    for name in more:
        if founds[name].get("delta") is not None:
            acquisition._store_located(name, {**founds[name], "tracking": True})
    return ""


async def _act_engine_task(
    speed: float, server: str | None = None, perturb: list[dict[str, Any]] | None = None
) -> None:
    """One act run by the engine: find the objects, build the engine from the demo, play it on the jog's live
    playback while the tracker feeds the scene log, and record it as the key-frame runner records its acts."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from .. import jog

    act = runtime._state.act
    names = list(MOTOR_NAMES)
    gi = names.index("gripper")

    def fail(reason: str) -> None:
        act.ok, act.reason, act.step = False, reason, "aborted"

    def stopped() -> bool:
        return act.stop_requested

    run: recording._Run | None = None
    streaming = False
    feed: _Feed | None = None
    engine: Engine | None = None
    try:
        with runtime._state.lock:
            demo, teach = runtime._state.demo, runtime._state.teach
        if demo is None:
            fail("record or load a demo first")
            return
        frames = core.key_frames(demo.keypoints)
        if not frames:
            fail("mark a key frame first")
            return
        obj, others = planning._session_object(demo), planning._other_objects(demo)
        why = await _find_objects(demo, obj, others, stopped)
        if why:
            fail(why)
            return
        with runtime._state.lock:
            test, teach = runtime._state.test, runtime._state.teach
        if test is None or not test.result.get("ok") or teach is None:
            fail("find the object first")
            return
        try:
            t_bc = camera._t_base_cam()
        except Exception as e:  # noqa: BLE001
            fail(str(getattr(e, "detail", e)))
            return
        kin, cur = jog.kinematics(), jog.current_tip_and_anchor()
        if kin is None or cur is None:
            fail("connect the arm first")
            return
        kin = copy.deepcopy(kin)  # the engine's solves must not touch the motor loop's model
        session = obj or str(teach.keypoints.get("concept") or "object")
        starts = {
            session: planning._delta_base(demo, np.asarray(test.result["delta_cam"], dtype=float), t_bc)
        }
        for name in others:
            motion, problem = planning._object_motion(demo, name, t_bc)
            if problem:
                fail(problem)
                return
            starts[name] = motion
        surfaces = {}
        for name in starts:
            with contextlib.suppress(Exception):
                surfaces[name] = np.asarray(
                    planning._object_points(demo, name if name != session else obj, t_bc)
                )
        shape = np.asarray(teach.rgb).shape
        cam = _camera(teach, t_bc, shape)
        scene = _ArmScene(cam, kin, names)
        lin, ang = jog.walk_limits()
        holdings = planning._holdings(demo)
        arm = Arm(
            kin,
            gripper=gi,
            closing=float(holdings[0]["closing"]) if holdings else 1.0,
            lo=None if jog.servo_ranges() is None else jog.servo_ranges()[0],
            hi=None if jog.servo_ranges() is None else jog.servo_ranges()[1],
            step_deg=jog.MAX_STEP_DEG,
            workspace=jog.workspace_box(),
            hz=jog.HZ,
            walk_m_s=lin,
            walk_rad_s=ang,
            grip_units_s=jog.GRIP_UNITS_S,
        )
        demo_record = Demo(
            t=np.asarray(demo.t, dtype=float),
            tips=np.asarray(demo.tips, dtype=float),
            grip=np.asarray(demo.q_cmd, dtype=float)[:, gi],
            grip_obs=np.asarray(demo.q_obs, dtype=float)[:, gi],
            q=np.asarray(demo.q_obs, dtype=float),
        )
        timeline = Timeline(demo_record, frames, session_object=session)
        objects = Objects(scene, starts, surfaces, lin, ang)
        engine = Engine(
            timeline,
            arm,
            scene,
            _JogPlayback(names),
            objects,
            speed=speed,
            perturb=Perturbations.from_spec(perturb),
            clock=time.perf_counter,
        )
        feed = _Feed(scene, demo, t_bc, session, others)
        with runtime._state.lock:
            runtime._state.frame_listeners.append(feed)
        run = recording._begin_run(
            demo,
            speed,
            np.asarray(test.result["delta_cam"], dtype=float),
            t_bc,
            {"ok": True, "summary": "act engine"},
        )
        run.meta["runner"] = "engine"
        run.meta["perturb"] = perturb
        q0 = np.array([float(cur[2][m]) for m in names])
        act.step = "starting"
        await jog.joints_start({m: float(v) for m, v in zip(names, q0, strict=True)})
        streaming = True
        jog.play_joints([{m: float(v) for m, v in zip(names, q0, strict=True)}], [0.0])
        engine.start(q0, time.perf_counter())
        loop = asyncio.get_running_loop()
        recorded = 0
        while not engine.done:
            st = jog.current_status()
            why = (
                "stopped"
                if act.stop_requested
                else "the arm went away"
                if not st.get("connected")
                else f"arm frozen: {st.get('reason')}"
                if st.get("halted")
                else ""
            )
            if why:
                fail(why)
                with contextlib.suppress(Exception):
                    jog.playback_stop()
                break
            await loop.run_in_executor(
                runtime._ACT_EXECUTOR, functools.partial(engine.step, time.perf_counter())
            )
            sent, stamps = jog.playback_state()
            for k in range(recorded + 1, min(sent, len(engine.cmds) - 1) + 1):
                c = engine.cmds[k]
                leg = engine.tl.legs[c.leg]
                run.target(
                    f"{c.kind}: {leg.label}",
                    pose=c.pose,
                    joints=c.q,
                    at=stamps[k] if k < len(stamps) else None,
                )
                act.step = f"{leg.label}" if c.kind == "leg" else f"going back to {leg.label}"
                act.progress = (c.leg + c.s) / len(engine.tl.legs)
            recorded = max(recorded, sent)
            await asyncio.sleep(0.25 / jog.HZ)
        if engine.done and act.ok is None:
            ok = bool(engine.result["ok"])
            act.ok, act.reason = ok, engine.result["why"]
            act.step = "done" if ok else "aborted"
    except Exception as e:  # the arm holds its last target; the operator sees why
        logger.exception("engine act failed")
        fail(f"act error: {e}")
    finally:
        if feed is not None:
            with runtime._state.lock, contextlib.suppress(ValueError):
                runtime._state.frame_listeners.remove(feed)
        if streaming:
            with contextlib.suppress(Exception):
                await jog.joints_stop()
        run_dir = None
        if run is not None:
            with runtime._state.lock:
                runtime._state.run = None
            if run.timeline is not None:
                with contextlib.suppress(Exception):
                    run.meta["timeline"] = await asyncio.get_running_loop().run_in_executor(
                        runtime._RUN_EXECUTOR, run.timeline.close
                    )
            if engine is not None:
                run.meta["engine"] = {
                    "result": engine.result,
                    "progress": list(engine.progress),
                    "milestones": [
                        {"name": m.name, "status": m.status, "failures": m.failures}
                        for m in engine.milestones
                    ],
                    "stretched_s": engine.stretched_s,
                    "records": engine.records,
                    "frames_fed": None if feed is None else feed.frames,
                    "slowest_solve_ms": engine.solve_s * 1000.0,
                }
            samples: list[dict[str, Any]] = []
            if run.arm_recording:
                with contextlib.suppress(Exception):
                    samples = jog.stop_record()
            result = {"ok": act.ok, "step": act.step, "reason": act.reason, "progress": act.progress}
            try:
                run_dir = await asyncio.get_event_loop().run_in_executor(
                    runtime._RUN_EXECUTOR, recording._finish_run, run, samples, jog.fk_tip, result
                )
            except Exception:
                logger.exception("writing the engine act's recording failed")
            with contextlib.suppress(Exception):
                recording._record_trial(run_dir)
        act.on = False
