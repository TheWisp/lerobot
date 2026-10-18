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
import collections
import contextlib
import copy
import functools
import logging
import time
from typing import Any

import numpy as np

from lerobot.showservo import motion as core, symmetry
from lerobot.showservo.engine.engine import Arm, Engine
from lerobot.showservo.engine.perturb import Perturbations
from lerobot.showservo.engine.scene import Camera, Measurement, SceneLog
from lerobot.showservo.engine.timeline import Demo, Timeline
from lerobot.showservo.engine.world import World
from lerobot.showservo.estimation import OLD_RUNNER, POINT2POSE, Candidate, PoseEstimator, tracker_verdict

from . import acquisition, camera, holds, planning, recording, runtime

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

    def tip_now(self) -> np.ndarray | None:
        from .. import jog

        latest = jog.latest_joints()
        if latest is None:
            return None
        return np.asarray(
            self.kin.forward_kinematics(np.array([latest[1][m] for m in self.names])), dtype=float
        )

    def tip_at(self, t: float, max_gap_s: float = 0.2) -> np.ndarray | None:
        from .. import jog

        q = jog.joints_at(t, 2.0 / jog.HZ)
        if q is None:
            # No read after ``t`` yet: ``t`` is now, and the latest read is the arm now, up to two ticks old. Without
            # it the engine took its grasp's hold from the commanded pose (act_execution.md E24).
            latest = jog.latest_joints()
            if latest is None or not 0.0 <= t - latest[0] <= 2.0 / jog.HZ:
                return None
            q = latest[1]
        return np.asarray(self.kin.forward_kinematics(np.array([q[m] for m in self.names])), dtype=float)


class _Feed:
    """Each tracked frame, into the scene log (act_execution.md, "The loops, and who writes what"): the frame's time,
    and each object's pose the estimator takes (:mod:`lerobot.showservo.estimation`). The feed only turns the trackers'
    output into candidates, base frame, each with its source's verdict: Point2Pose's own (its fit found, the object not
    lost) and the key-frame runner's checks on that view, which ran on the frame before the feed is called; which of
    them is a measurement is the estimator's."""

    def __init__(
        self,
        scene: SceneLog,
        demo: runtime._Demo,
        t_bc: np.ndarray,
        session: str,
        others: list[str],
        estimator: PoseEstimator | None = None,
    ):
        self.scene, self.demo, self.t_bc, self.session, self.others = scene, demo, t_bc, session, others
        self.estimator = tracker_verdict() if estimator is None else estimator
        self.reference: dict[
            str, np.ndarray | None
        ] = {}  # each other object's demo pose its measurement is against
        self.frames = 0
        self.refused: collections.Counter[str] = collections.Counter()  # "object: why", for the act's record

    def _other_motion(self, name: str, delta: np.ndarray) -> np.ndarray | None:
        """``name``'s motion since the demo, base frame, from the tracker's delta for it (as ``planning._object_motion``
        composes a locate's)."""
        with runtime._state.lock:
            target = runtime._state.targets.get(name)
        if target is None or target.anchor is None:
            return None
        return self._view_motion(name, np.asarray(delta, dtype=float) @ target.anchor)

    def _view_motion(self, name: str, view: np.ndarray) -> np.ndarray | None:
        """``name``'s motion since the demo, base frame, from its view: its motion from the demo's camera view."""
        if name not in self.reference:
            f = planning._pose_frame(self.demo, name)
            self.reference[name] = None if f is None else holds._demo_control_pose(self.demo, name, f)[0]
        reference = self.reference[name]
        if reference is None:
            return None
        cam = np.asarray(view, dtype=float) @ np.linalg.inv(reference)
        return self.t_bc @ cam @ np.linalg.inv(self.t_bc)

    def _candidates(
        self, r: dict[str, Any], certified: dict[str, Any] | None, stamp: float
    ) -> dict[str, list[Candidate]]:
        """Each object's candidates. ``certified`` is the runner's result for the session object (None when its checks
        refused the view); ``stamp`` the frame's read time, which stamps the views they certified of the others."""
        out = {}
        if r.get("delta") is not None:
            motion = planning._delta_base(self.demo, np.asarray(r["delta"], dtype=float), self.t_bc)
            out[self.session] = [
                Candidate(POINT2POSE, motion, bool(r.get("ok")), "" if r.get("ok") else "no fit")
            ]
            if certified is not None and certified.get("delta_cam") is not None:
                view = planning._delta_base(
                    self.demo, np.asarray(certified["delta_cam"], dtype=float), self.t_bc
                )
                out[self.session].append(Candidate(OLD_RUNNER, view, True))
            else:
                out[self.session].append(Candidate(OLD_RUNNER, motion, False, "not certified"))
        names = [o.get("name") for o in r.get("others") or ()]
        for name in self.others:
            if name not in names or r.get(f"other_delta_{names.index(name)}") is None:
                continue
            k = names.index(name)
            share = r["others"][k]
            motion = self._other_motion(name, r[f"other_delta_{k}"])
            if motion is None:
                continue
            ok = bool(share.get("ok")) and not share.get("lost")
            why = "" if ok else ("lost" if share.get("lost") else "no fit")
            out[name] = [Candidate(POINT2POSE, motion, ok, why)]
            with runtime._state.lock:
                target = runtime._state.targets.get(name)
                last = dict(target.last) if target is not None else {}
                views = list(target.hold_views[-1:]) if target is not None else []
            view = self._view_motion(name, views[0][1]) if views and views[0][0] == stamp else None
            if last.get("state") == "tracking" and view is not None:
                out[name].append(Candidate(OLD_RUNNER, view, True))
            else:
                out[name].append(
                    Candidate(OLD_RUNNER, motion, False, str(last.get("reason") or "not certified"))
                )
        return out

    def __call__(self, job: runtime._Job, r: dict[str, Any], result: dict[str, Any] | None) -> None:
        t = job.camera_stamp_mono
        if t is None:
            return
        self.frames += 1
        self.scene.add_frame(t)
        for name, e in self.estimator.estimate(t, self._candidates(r, result, job.created)).items():
            if e.motion is not None:
                self.scene.add_measurement(name, Measurement(t=t, motion=e.motion))
            else:
                self.refused[f"{name}: {e.why}"] += 1


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


def object_geometry(
    objects: dict[str, dict[str, Any]], surfaces: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, tuple[dict, np.ndarray]]]:
    """Each object's centre and declared symmetry for the world, from its demo surface points (base frame): with a
    symmetry the centre is a point on its measured axis, so a turn about the axis moves no centre; otherwise the
    points' mean. ``objects`` are the demo's, by the same names as ``surfaces``."""
    centres, symmetries = {}, {}
    for name, points in surfaces.items():
        centres[name] = np.asarray(points).mean(axis=0)
        spec = symmetry.normalise(objects.get(name, {}).get("symmetry"))
        if spec["kind"] == "none":
            continue
        try:
            axis, point = symmetry.observed_axis(points, spec["axis"])
        except ValueError as exc:
            logger.warning(
                "%s: its declared symmetry has no measurable axis, so it is not used: %s", name, exc
            )
            continue
        centres[name], symmetries[name] = point, (spec, axis)
    return centres, symmetries


def arm_refusal() -> str | None:
    """Why the act engine cannot drive the connected arm, or None. It commands joints from the calibrated kinematics
    without watching the fingertip, so it drives only the arm held on its commanded pose against gravity: the
    ``so107_follower_gravity`` robot type, never a plain follower that settles short under load."""
    from lerobot.robots.so107_follower_gravity import ROBOT_TYPE

    from .. import jog

    kind = jog.connected_robot_type()
    if kind == ROBOT_TYPE:
        return None
    if kind is None:
        return "connect the arm first"
    return f"the act engine drives only a {ROBOT_TYPE} arm; the connected profile is {kind!r}"


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
        centres, symmetries = object_geometry(
            {n: demo.objects.get(n if n != session else obj, {}) for n in surfaces}, surfaces
        )
        world = World(scene, starts, centres, symmetry=symmetries)
        engine = Engine(
            timeline,
            arm,
            scene,
            _JogPlayback(names),
            world,
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
                    "refused": None if feed is None else dict(feed.refused),
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
