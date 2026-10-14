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

"""The single key-frame act runner and manual approach controls."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from lerobot.showservo import motion as core

from . import acquisition, camera, groups, held_views, holds, planning, recording, runtime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/go")
async def go(body: runtime.GoBody) -> dict:
    """Walk the fingertip to the transported pre-grasp, ``hover_mm`` above it (0 to go exactly there)."""
    from .. import jog

    with runtime._state.lock:
        test, teach = runtime._state.test, runtime._state.teach
    if test is None or test.transported is None:
        raise HTTPException(409, "no transported pose; capture the test frame first")
    check = test.result.get("camera_check") or {}
    if check.get("moved"):
        raise HTTPException(
            409,
            f"the camera or the tray moved since the calibration (markers shifted {check['max_px']:.1f} px); "
            "redo the camera calibration before going anywhere",
        )
    pose = test.transported.copy()
    pose[2, 3] += body.hover_mm / 1000.0
    try:
        jog.set_target_pose(pose)
        # The gripper opening is part of the taught pose.
        if teach is not None and teach.gripper is not None:
            jog.set_gripper(teach.gripper)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"target_mm": (pose[:3, 3] * 1000.0).tolist(), "gripper": None if teach is None else teach.gripper}


class InjectBody(BaseModel):
    """An error to inject into one act, base frame: moved by dx, dy, dz and turned by rx, ry, rz about the base's
    axes. ``at`` "aim": the grasp is aimed off, turned about the grasp point, as an arm that misses its plan; the
    act knows where it aimed. "find": the object's found pose is wrong, turned about its centre; the act believes
    it, as a find that erred."""

    at: Literal["aim", "find"] = "aim"
    dx_mm: float = Field(0.0, ge=-runtime.INJECT_MAX_MM, le=runtime.INJECT_MAX_MM)
    dy_mm: float = Field(0.0, ge=-runtime.INJECT_MAX_MM, le=runtime.INJECT_MAX_MM)
    dz_mm: float = Field(0.0, ge=-runtime.INJECT_MAX_MM, le=runtime.INJECT_MAX_MM)
    rx_deg: float = Field(0.0, ge=-runtime.INJECT_MAX_DEG, le=runtime.INJECT_MAX_DEG)
    ry_deg: float = Field(0.0, ge=-runtime.INJECT_MAX_DEG, le=runtime.INJECT_MAX_DEG)
    rz_deg: float = Field(0.0, ge=-runtime.INJECT_MAX_DEG, le=runtime.INJECT_MAX_DEG)


class ActBody(BaseModel):
    speed: float = (
        1.0  # scales the straight lines (from the jog's walk speed) and the grasp (from the demo's clock)
    )
    inject: InjectBody | None = None
    correct_hold: bool = True  # False: a place replays the demo against the target, uncorrected for the hold
    allow_grasp_prediction: bool = False  # use the grasp-time hold if post-carry vision cannot verify it
    model_config = ConfigDict(extra="forbid")


def _grasp_index(demo: runtime._Demo) -> int:
    """The demo sample where its fingertip grips the object: the firm grip, else the last key frame before the grasp
    check (:func:`_grasp_split`). Pre: a key frame is marked."""
    i = holds._firm_grip(demo)
    if i is not None:
        return i
    frames = core.key_frames(demo.keypoints)
    return int(np.argmin(np.abs(demo.t - frames[planning._grasp_split(frames, None) - 1]["t"])))


def _injects(inject: InjectBody) -> bool:
    """Whether ``inject`` moves or turns anything."""
    return any(getattr(inject, k) for k in ("dx_mm", "dy_mm", "dz_mm", "rx_deg", "ry_deg", "rz_deg"))


def _inject_transform(inject: dict[str, Any], pivot: np.ndarray) -> np.ndarray:
    """The injected error as a base-frame motion: turned by ``rx, ry, rz`` (degrees, about the base's x, y, z) about
    ``pivot``, then moved by ``dx, dy, dz`` (mm)."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler(
        "xyz", [float(inject.get(k, 0.0)) for k in ("rx_deg", "ry_deg", "rz_deg")], degrees=True
    ).as_matrix()
    p = np.asarray(pivot, dtype=float)[:3]
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = (
        p - rot @ p + np.array([float(inject.get(k, 0.0)) for k in ("dx_mm", "dy_mm", "dz_mm")]) / 1000.0
    )
    return out


async def _act_key_frames_task(speed: float, server: str | None = None) -> None:
    """The act: the demo's key frames in order (:func:`core.key_frames`), each carried by its frame's motion since
    the demo, the hold read from the demo's gripper (:func:`core.demo_holding`) the one exception. ``server`` is the
    GUI server's own address, for a point groups view the act starts.

    The act's objects are those its key frames are held in (:func:`_act_objects`): each is found where it was last
    seen, the first one's find restarting the live session with the others joining it, and the act's pose of each is
    the last view that placed it (:func:`_view_places`), moved on with what it rests on (the point groups, when acts
    use them). The whole act is planned and judged from the arm's present joints before anything moves.

    Before the grasp check (:func:`_grasp_split`) a line is walked, re-aimed every tick at its frame's object as it
    moves, the gripper set first or moved along as the key frame says; a follow is planned from where the arm stands
    and streamed. During the follow the demo's hold begins in, what the grip shows is read on the fly, as the old act
    did (the grasp check, the grasp pose, views of the held object while the arm stands still). From the grasp check
    on, while the demo held something: the lines made holding are streamed as one plan with the grasp's closing kept;
    at the first held line, the arm stands still while the held object is found in the gripper, reaches it corrected for
    how the object sits there now, and the rest is planned from there and streamed, the held object let go of in the
    point groups at the end of the follow its hold ends in. Without a hold, the rest is streamed as planned.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from .. import jog

    act = runtime._state.act

    def fail(reason: str) -> None:
        act.ok, act.reason, act.step = False, reason, "aborted"

    def q_dict(q: np.ndarray) -> dict[str, float]:
        return {m: float(q[k]) for k, m in enumerate(MOTOR_NAMES)}

    def stopped() -> bool:
        return act.stop_requested

    halt = ""  # why the act stops, found on the fly by something running beside the stream

    def interrupted() -> str:
        st = jog.current_status()
        if act.stop_requested:
            return "stopped"
        if halt:
            return halt
        if not st.get("connected"):
            return "the arm went away"
        if st.get("halted"):
            return f"arm frozen: {st.get('reason')}"
        return ""

    moved = streaming = False
    sent = -1  # the last sample of the plan being streamed that went to the arm
    limits_before: tuple[float, float] | None = None
    run: recording._Run | None = None
    run_dir: str | None = None
    beside: list[asyncio.Task] = []  # what runs beside the arm; cancelled however the act ends
    held: tuple[str, asyncio.Task] | None = (
        None  # the object in the gripper, its pose sent to the point groups
    )
    with_groups = False
    followed: dict[str, float] = {}  # the objects the point groups carry for this act, and since when
    try:
        with runtime._state.lock:
            with_groups = runtime._state.groups_with_acts
            finish_view = runtime._state.groups.running and not with_groups
        if finish_view:  # the switch is off: the act's tracker gets the GPU and the camera to itself
            act.step = "finishing the point groups view"
            await groups._finish_groups()
        with runtime._state.lock:
            demo, test, teach, tracking = (
                runtime._state.demo,
                runtime._state.test,
                runtime._state.teach,
                runtime._state.track.on,
            )
        if demo is None:
            fail("record or load a demo first")
            return
        frames = core.key_frames(demo.keypoints)
        if not frames:
            fail("mark a key frame first")
            return
        obj, others = planning._session_object(demo), planning._other_objects(demo)
        holdings = planning._holdings(demo)
        phases = core.interaction_phases(holdings, demo.t, frames)
        for holding in holdings:
            made_held = [core.held_at(holding, demo.t, frames, n) for n in range(len(frames))]
            split = holding["begins"] + 1
            first_line = next(
                (n for n in range(split, len(frames)) if made_held[n] and frames[n]["move"] != "lift"), None
            )
            if any(made_held[split:]):
                if holding["frame"] not in demo.objects:
                    fail("the object held must be one clicked on the recording, to find it in the gripper")
                    return
                if first_line is None or frames[first_line]["move"] != "line":
                    fail("after the grasp the arm needs a straight line before it follows the demo again")
                    return
        with runtime._state.lock:
            act.find_error = None
            live = bool(
                tracking
                and (runtime._state.track.last or {}).get("state") == "tracking"
                and test is not None
                and test.result.get("ok")
            )
        live = (
            live and not planning._reference_motion(demo, teach)[1]
        )  # a track to start from, against the demo's view
        # With the session's own object to follow, the others join its session there, not on their own: one restart
        # of the session for all.
        relocate = {
            name: asyncio.create_task(
                acquisition._locate_afresh(name, stopped, track=not (obj is not None and live))
            )
            for name in others
        }
        beside.extend(relocate.values())
        if obj is not None and not live:  # nothing to start from: the find comes first
            act.step = f"finding {obj}"
            why = await acquisition._find_afresh(obj, stopped)
            if why:
                fail(why)
                return
            with runtime._state.lock:
                test, teach, tracking = runtime._state.test, runtime._state.teach, runtime._state.track.on
        if test is None or not test.result.get("ok"):
            fail("find the object first")
            return
        if teach is None:
            fail("teach the object first")
            return
        try:
            t_bc = camera._t_base_cam()
        except HTTPException as e:
            fail(e.detail)
            return
        kin, cur = jog.kinematics(), jog.current_tip_and_anchor()
        if kin is None or cur is None:
            fail("connect the arm first")
            return
        ranges = jog.servo_ranges()
        landing_obj = planning._landing_object(demo)
        landing_turn: tuple[np.ndarray, float] | None = (
            None  # the middle and the turn it lands at, once planned
        )
        if tracking:
            with runtime._state.lock:
                state = (runtime._state.track.last or {}).get("state")
            if state != "tracking":
                fail(
                    f"the tracker does not see the object ({state or 'no frame yet'}): "
                    "load the demo again with the object where it was taught"
                )
                return
        problem = planning._reference_motion(demo, teach)[1]
        if problem:
            fail(problem)
            return
        motions: dict[str, np.ndarray] = {}
        for name in others:
            motion, problem = planning._object_motion(demo, name, t_bc)
            if problem:  # the last find is no use: this one is needed before anything moves
                act.step = f"finding {name}"
                why = await relocate[name]
                if why:
                    fail(why)
                    return
                motion, problem = planning._object_motion(demo, name, t_bc)
                if problem:
                    fail(problem)
                    return
            motions[name] = motion
        # Assigned separately for each attachment, never reused across a regrasp.
        demo_grip = demo_carry = None
        grip_problem = carry_problem = ""
        hold_task: asyncio.Task | None = None
        picked = obj or str(teach.keypoints.get("concept") or "object")  # its name in the point groups
        if with_groups:
            # The point groups carry the objects the act follows while no view places them: designated where the act's
            # own views have them now, before anything moves (src/lerobot/showservo/docs/act_loop.md).
            seen_mask = test.result.get("live_mask")
            clicks = {
                picked: acquisition._deepest_pixel(
                    seen_mask if seen_mask is not None else teach.keypoints.get("mask"), teach.rgb.shape
                )
            }
            for name in others:
                mask = (acquisition._located(demo, name) or {}).get("mask")
                clicks[name] = None if mask is None else acquisition._deepest_pixel(mask, mask.shape)
            unknown = [n for n, c in clicks.items() if c is None]
            if unknown:
                fail(f"where {' and '.join(unknown)} is, for the point groups: click it in the camera view")
                return
            act.step = f"designating {' and '.join(clicks)} in the point groups"
            why, groups_since = await groups._designate_in_groups(clicks, server, stopped)
            if why:
                fail(why)
                return
            followed = dict.fromkeys(clicks, groups_since)
            designated_at = time.monotonic()

            async def watch_groups() -> None:
                """Beside the arm: a view that draws nothing for GROUPS_STALE_S stops the act, since the objects would
                stand still in it whatever happened to them; the feed takes its frames (:func:`_groups_feed_pump`)."""
                nonlocal halt
                while True:
                    with runtime._state.lock:
                        on, fresh = runtime._state.groups_with_acts, runtime._state.groups_feed.fresh
                    if not on:  # turned off during the act: the last views are held from here
                        return
                    if time.monotonic() - max(fresh, designated_at) > runtime.GROUPS_STALE_S:
                        halt = f"the point groups view drew no frame for {runtime.GROUPS_STALE_S:g} s"
                        return
                    await asyncio.sleep(runtime.GROUPS_POLL_S)

            beside.append(asyncio.create_task(watch_groups()))

        def carried(name: str, since: float) -> np.ndarray:
            """How the point groups moved ``name`` since the frame read at ``since``, camera frame; none without
            them."""
            return groups._carried(name, since) if with_groups else np.eye(4)

        def index(tk: float) -> int:
            return int(np.argmin(np.abs(demo.t - tk)))

        gi = MOTOR_NAMES.index("gripper")
        delta = np.asarray(test.result["delta_cam"], dtype=float)
        inject = act.inject
        if inject and inject.get("at") == "find":
            # The act believes the object turned about its centre and moved: everything it does follows that belief.
            truth = planning._delta_base(demo, delta, t_bc)
            centre = (
                planning._object_points(demo, obj, t_bc).mean(axis=0)
                if obj is not None and obj in demo.objects
                else demo.tips[_grasp_index(demo)][:3, 3]
            )
            with runtime._state.lock:
                act.find_error = _inject_transform(inject, truth[:3, :3] @ centre + truth[:3, 3])
        grasp_at = _grasp_index(demo)

        def aim_error(d: np.ndarray) -> np.ndarray | None:
            """The injected miss of the session object's key frames, turned about where the fingertip grips the
            object as it lies now; None without one."""
            if not inject or inject.get("at") != "aim":
                return None
            return _inject_transform(
                inject, (planning._delta_base(demo, d, t_bc) @ demo.tips[grasp_at])[:3, 3]
            )

        # The view the act holds of the session's object: ``delta``, of the frame read at ``seen_at``; views of frames
        # read after ``newer_than`` replace it.
        seen_at = max((w for w, _ in planning._certified_since(None)), default=0.0)
        newer_than = seen_at
        limits_before = jog.walk_limits()
        act.step = "planning"
        plan = await asyncio.get_event_loop().run_in_executor(
            runtime._ACT_EXECUTOR,
            functools.partial(planning._plan_act, aim=aim_error(delta), landing=demo.landing, ranges=ranges),
            demo,
            delta,
            t_bc,
            kin,
            np.array([float(cur[2][m]) for m in MOTOR_NAMES]),
            limits_before,
            jog.workspace_box(),
            speed,
            0,
            motions,
        )
        act.plan = {k: plan[k] for k in ("ok", "reason", "marks", "summary", "landing") if k in plan}
        if not plan["ok"]:
            fail(plan["reason"])
            return
        if (plan.get("landing") or {}).get("turn_deg") is not None:
            # It lands at this turn from here on: every later key frame on that object, and the hold's check, aims by it.
            landing_turn = (plan["landing_centre"], float(plan["landing"]["turn_deg"]))
        jog.set_walk_limits(limits_before[0] * speed, limits_before[1] * speed)
        moved = True
        run = recording._begin_run(demo, speed, delta, t_bc, plan)
        run.meta["inject"], run.meta["correct_hold"] = inject, act.correct_hold
        run.meta["allow_grasp_prediction"] = act.allow_grasp_prediction
        run.meta["runner"] = "key_frames"

        def replanned(stage: str, p: dict[str, Any]) -> None:
            """A plan made after the start's: the page shows it, and the act's record keeps it beside the start's, so
            a refusal there names its marks."""
            act.plan = {k: p[k] for k in ("ok", "reason", "marks", "summary", "landing") if k in p}
            run.meta.setdefault("plans", []).append({"stage": stage, **act.plan})

        # The walk starts on the track the act began with; its find is started over beside the arm, and until the
        # fresh track certifies a view the old one's last motion stands (its views are against the old teach).
        motion0, seen_at0 = planning._delta_base(demo, delta, t_bc), seen_at

        async def refind_with_others() -> str:
            """The session started over from a fresh find of its own object, the others (found just before, where
            their tracks last had them) joining it from the start."""
            for name in others:
                why = await relocate[name]
                if why:
                    return why
            nonlocal delta, seen_at, newer_than
            founds = {name: acquisition._located(demo, name) for name in others}
            more = {
                n: f["mask"] for n, f in founds.items() if f and f["ok"] and f.get("mask") is not None
            } or None
            # The arm is on its way by now and may cover the object from the camera, so the restarted track is not
            # waited for: the find's own view is where the object is (no motion since it, against the new teach)
            # until the track sees it again.
            why = await acquisition._find_afresh(obj, stopped, more, need_track=False)
            if not why:  # views from before the new teach was applied are against the old one
                with runtime._state.lock:
                    fresh = runtime._state.teach
                delta, newer_than = np.eye(4), time.time()
                seen_at = fresh.stamp if fresh is not None and fresh.stamp else newer_than
                for name in more or ():
                    acquisition._store_located(name, {**founds[name], "tracking": True})
            return why

        refind = asyncio.create_task(refind_with_others()) if obj is not None and live else None
        if refind is not None:
            beside.append(refind)

        def follow() -> np.ndarray:
            """The session object's motion as the act holds it, camera frame: the newest view that placed it, moved on
            with what it rests on since that view's frame was read."""
            nonlocal seen_at, newer_than, delta
            if tracking:
                newer = planning._certified_since(newer_than)
                if newer:
                    seen_at, delta = newer[-1]
                    newer_than = seen_at
            return carried(picked, seen_at) @ delta

        def approach() -> np.ndarray:
            """The session object's motion for a walk, base frame: the track the act began with until its fresh find
            is in, then the fresh one's newest view; each moved on with what the object rests on."""
            if refind is not None and not refind.done():
                return t_bc @ carried(picked, seen_at0) @ np.linalg.inv(t_bc) @ motion0
            return planning._delta_base(demo, follow(), t_bc)

        def other_now(name: str) -> np.ndarray:
            """Another object's motion as its track has it now, moved on with what it rests on, base frame, the last
            one when it has no newer, turned by the landing the plan took when it is the object landed on."""
            motion, _problem = planning._object_motion(demo, name, t_bc, carry=lambda s: carried(name, s))
            if motion is not None:
                motions[name] = motion
            if name == landing_obj and landing_turn is not None:
                return core.landed(motions[name], *landing_turn)
            return motions[name]

        def others_now() -> dict[str, np.ndarray]:
            return {name: other_now(name) for name in others}

        def frame_now(frame: str) -> np.ndarray:
            """A key frame's frame's motion for a walk, base frame, asked again every tick."""
            if frame == core.WORLD_FRAME:
                return np.eye(4)
            if frame == (obj or ""):
                return approach()
            return other_now(frame)

        async def walk_to(label: str, aim: Callable[[], np.ndarray]) -> str:
            """Walk the arm to ``aim()``, asked again every tick, until it arrives: "" or why not. Arrived is within
            ACT_ARRIVE_M of it, or, as for a stream's end, with the commanded joints on it and the arm stopped with
            every joint within ACT_STALL_DEG of its command: held short by the servos, which waiting does not close."""
            act.step = label
            t0 = time.monotonic()
            kin = jog.kinematics()
            arm = [m for m in MOTOR_NAMES if m != "gripper"]
            seen: list[tuple[float, np.ndarray]] = []
            while True:
                why = interrupted()
                if why:
                    return why
                target = aim()
                jog.set_target_pose(target)
                run.target(act.step, pose=target)
                cur, st = jog.current_tip_and_anchor(), jog.current_status()
                if cur is not None and not st.get("holding"):
                    e_m, e_deg = core.pose_residual(cur[0], target)
                    if e_m <= runtime.ACT_ARRIVE_M and e_deg <= core.ACT_REACH_TOL_DEG:
                        return ""
                    if kin is not None and st.get("q_cmd") and st.get("q_obs"):
                        now = time.monotonic()
                        q_obs = np.array([st["q_obs"][m] for m in arm])
                        seen = [(w, v) for w, v in seen if now - w <= runtime.ACT_STILL_S] + [(now, q_obs)]
                        stopped = (
                            now - seen[0][0] >= 0.8 * runtime.ACT_STILL_S
                            and np.ptp([v for _w, v in seen], axis=0).max() < runtime.ACT_STILL_DEG
                        )
                        lag = max(abs(st["q_cmd"][m] - st["q_obs"][m]) for m in arm)
                        c_m, c_deg = core.pose_residual(
                            kin.forward_kinematics(np.array([st["q_cmd"][m] for m in MOTOR_NAMES])), target
                        )
                        commanded = c_m <= runtime.ACT_ARRIVE_M and c_deg <= core.ACT_REACH_TOL_DEG
                        if stopped and lag <= runtime.ACT_STALL_DEG and commanded:
                            shorts = {
                                **(act.place or {}).get("walk_short_mm", {}),
                                label: round(e_m * 1000.0, 1),
                            }
                            act.place = {**(act.place or {}), "walk_short_mm": shorts}
                            return ""
                if time.monotonic() - t0 > runtime.ACT_STEP_TIMEOUT_S:
                    return f"{label}: not there after {runtime.ACT_STEP_TIMEOUT_S:.0f} s"
                await asyncio.sleep(runtime.ACT_TICK_S)

        async def stream(planned: dict[str, Any]) -> str:
            """Stream a plan's joints, then wait for the arm to settle on the last: "" or why not. The jog's loop plays
            them back on its own clock (:func:`jog.play_joints`); ``sent`` follows the samples that went out."""
            nonlocal streaming, sent
            q, times, stage = planned["q"], planned["times"], planned["stage"]
            sent = -1
            try:
                await jog.joints_start(q_dict(q[0]))
            except RuntimeError as e:
                return str(e)
            streaming = True
            n = len(q)
            try:
                jog.play_joints([q_dict(x) for x in q], [float(t) for t in times])
            except RuntimeError as e:
                return str(e)
            while sent < n - 1:
                await asyncio.sleep(runtime.ACT_TICK_S)
                why = interrupted()
                if why:
                    jog.playback_stop()
                    return why
                done, at = jog.playback_state()
                for i in range(sent + 1, done + 1):
                    run.target(stage[i], joints=q[i], at=at[i])
                if done > sent:
                    sent = done
                    act.step = stage[done]
                    act.progress = (done + 1) / n
            act.step = "settling"
            t0 = time.monotonic()
            arm = [k for k, m in enumerate(MOTOR_NAMES) if m != "gripper"]
            seen: list[tuple[float, np.ndarray]] = []
            while True:
                await asyncio.sleep(runtime.ACT_TICK_S)
                cur = jog.current_tip_and_anchor()
                why = interrupted()
                if why or cur is None:
                    return why or "the arm went away"
                q_now = np.array([float(cur[2][MOTOR_NAMES[k]]) for k in arm])
                lag = float(np.abs(q_now - np.asarray(q[-1], dtype=float)[arm]).max())
                if lag <= runtime.ACT_SETTLE_DEG:
                    return ""
                now = time.monotonic()
                seen = [(w, v) for w, v in seen if now - w <= runtime.ACT_STILL_S] + [(now, q_now)]
                stopped = (
                    now - seen[0][0] >= 0.8 * runtime.ACT_STILL_S
                    and np.ptp([v for _w, v in seen], axis=0).max() < runtime.ACT_STILL_DEG
                )
                if (
                    stopped and lag <= runtime.ACT_STALL_DEG
                ):  # held short under load: waiting will not close it
                    act.place = {**(act.place or {}), "settled_short_deg": lag}
                    return ""
                if stopped:
                    return f"the end: the arm stopped {lag:.0f} deg short of it"
                if now - t0 > runtime.ACT_STEP_TIMEOUT_S:
                    return f"the end: not there after {runtime.ACT_STEP_TIMEOUT_S:.0f} s ({lag:.0f} deg off)"

        async def gripper_still() -> float | None:
            """The gripper's reading once it has stopped moving, as a closing on the object or on nothing ends."""
            g_obs, t0 = jog.current_gripper(), time.monotonic()
            while g_obs is not None and time.monotonic() - t0 < runtime.GRIP_SETTLE_S:
                await asyncio.sleep(0.1)
                g_prev, g_obs = g_obs, jog.current_gripper()
                if g_obs is not None and abs(g_obs - g_prev) < core.GRIP_STILL_UNITS:
                    break
            return g_obs

        def grasp_missed(cmd: float, g_obs: float, i_demo: int, closing: float) -> str:
            """Why the grasp missed, or "": closing on nothing reaches the command, an object stops it short."""
            held, short = core.grasp_held(
                cmd, g_obs, float(demo.q_cmd[i_demo, gi]), float(demo.q_obs[i_demo, gi]), closing
            )
            act.place = {**(act.place or {}), "grasp_held": held, "grasp_short": short}
            if held is not False:
                return ""
            return (
                f"the grasp missed: the gripper closed to {g_obs:.1f}, {short:.1f} short of its command; "
                f"an object between the fingers stops it more than {core.GRASP_HELD_SHORT:.1f} short"
            )

        async def still() -> str:
            """Wait until the fingertip has moved less than :data:`core.HOLD_STILL_M_S` allows over half a second: ""
            or why not. Over a single tick, one encoder count of jitter would read as motion."""
            window_s, seen, t0 = 0.5, [], time.monotonic()
            while True:
                why = interrupted()
                if why:
                    return why
                cur = jog.current_tip_and_anchor()
                if cur is None:
                    return "the arm went away"
                now = time.monotonic()
                seen = [(w, p) for w, p in seen if now - w <= window_s] + [(now, cur[0][:3, 3].copy())]
                span = now - seen[0][0]
                if (
                    span >= 0.8 * window_s
                    and np.linalg.norm(seen[-1][1] - seen[0][1]) / span < core.HOLD_STILL_M_S
                ):
                    return ""
                if now - t0 > runtime.ACT_STEP_TIMEOUT_S:
                    return f"the arm did not hold still for {runtime.ACT_STEP_TIMEOUT_S:.0f} s"
                await asyncio.sleep(runtime.ACT_TICK_S)

        async def plan_span(
            label: str,
            span: tuple[int, int],
            fix: np.ndarray | None = None,
            delta_cam: np.ndarray | None = None,
            grip_cmd: float | None = None,
            **kw: Any,
        ) -> dict:
            """Plan key frames ``span`` from the arm's present joints, every object where the act holds it now (the
            session object by ``delta_cam`` when given)."""
            cur = jog.current_tip_and_anchor()
            if cur is None:
                return {"ok": False, "reason": "the arm went away"}
            if delta_cam is None and any(frames[n]["frame"] == (obj or "") for n in range(*span)):
                delta_cam = follow()
            q_now = np.array([float(cur[2][m]) for m in MOTOR_NAMES])
            if grip_cmd is not None:
                q_now[gi] = grip_cmd  # preserve the closing while a held line corrects the pose
            p = await asyncio.get_event_loop().run_in_executor(
                runtime._ACT_EXECUTOR,
                functools.partial(planning._plan_act, part=span, ranges=ranges, **kw),
                demo,
                delta_cam,
                t_bc,
                kin,
                q_now,
                limits_before,
                jog.workspace_box(),
                speed,
                0,
                others_now(),
                fix,
            )
            replanned(label, p)
            return p

        async def caught_up() -> str:
            """What ran beside the arm is needed from here: the fresh track, the others' finds and the demo's holds,
            each waited for only if it is not in yet."""
            nonlocal demo_grip, grip_problem, demo_carry, carry_problem, hold_task
            if refind is not None:
                act.step = f"finding {obj}"
                why = await refind
                if why:
                    return why
            for name in others:
                act.step = f"finding {name}"
                why = await relocate[name]
                if why:
                    return why
                motion, problem = planning._object_motion(
                    demo, name, t_bc, carry=lambda s, name=name: carried(name, s)
                )
                if problem:
                    return problem
                motions[name] = motion
            if landing_obj is not None and landing_obj in motions:  # what the landing is aimed by
                found = acquisition._located(demo, landing_obj) or {}
                run.meta["target"] = {
                    **acquisition._located_info(found),
                    "delta": found.get("delta"),
                    "motion_base": motions[landing_obj],
                    "pose_frame": planning._pose_frame(demo, landing_obj),
                    "landing": plan.get("landing"),
                    "landing_centre": None if landing_turn is None else landing_turn[0],
                }
            if hold_task is not None:
                act.step = f"measuring how the demo holds {held_obj}"
                (demo_grip, grip_problem), (demo_carry, carry_problem) = await hold_task
                hold_task = None
                if stopped():
                    return "stopped"
                if act.correct_hold and not act.allow_grasp_prediction and demo_carry is None:
                    # This reference is known before grasping. Do not pick up an
                    # object whose required post-carry comparison cannot be made.
                    direct_refs = await asyncio.get_running_loop().run_in_executor(
                        runtime._RENDER_EXECUTOR, held_views.references, demo, held_obj
                    )
                    if not direct_refs:
                        act.place = {
                            "hold_used": "unverified",
                            "demo_carry_problem": carry_problem,
                            "reference_preflight": "failed before grasp",
                        }
                        return (
                            f"cannot verify {held_obj}'s demo carry reference before grasp: {carry_problem}"
                        )
                if demo_grip is None and demo_carry is None and holding is None:
                    return (
                        f"how the demo holds {held_obj} cannot be known: no firm grip in the demo; at its grip, "
                        f"{grip_problem}; while carried, {carry_problem}"
                    )
            return ""

        original_demo = demo
        for phase_index, (phase_start, phase_stop) in enumerate(phases):
            if streaming:
                await jog.joints_stop()
                streaming = False
            demo = dataclasses.replace(
                original_demo, hold_index=phase_index, interaction_span=(phase_start, phase_stop)
            )
            if phase_start:
                # A reused object may have moved in the demonstration too. Compare
                # it with its pose at this interaction, not its original position.
                demo.reference_constraints = {}
                why = await asyncio.get_event_loop().run_in_executor(
                    runtime._RENDER_EXECUTOR, holds._prepare_demo_references, demo
                )
                if why:
                    fail(why)
                    return
            holding = planning._holding(demo)
            split = min(phase_stop, max(phase_start, planning._grasp_split(frames, holding)))
            made_held = [core.held_at(holding, demo.t, frames, n) for n in range(len(frames))]
            carries = any(made_held[split:phase_stop])
            held_obj = None if holding is None else holding["frame"]
            landing_obj = planning._landing_object(demo)
            if phase_start:
                landing_turn = None
            demo_grip = demo_carry = None
            grip_problem = carry_problem = ""
            act.place = None
            hold_task = None
            if carries:

                async def demo_holds(
                    demo=demo, held_obj=held_obj
                ) -> tuple[tuple[dict | None, str], tuple[dict | None, str]]:
                    return (
                        await holds._demo_hold(demo, held_obj, t_bc, stopped, "grip"),
                        await holds._demo_hold(demo, held_obj, t_bc, stopped, "carry"),
                    )

                hold_task = asyncio.create_task(demo_holds())
                beside.append(hold_task)
            # ── up to the grasp check: lines walked, re-aimed at their objects; follows planned where the arm stands ──
            grip: dict[str, Any] = {}
            grip_watch: asyncio.Task | None = None
            closing, i_grasp_end = 1.0, None
            last_follow: dict[str, Any] | None = None
            waited = False
            for n in range(phase_start, split):
                k = frames[n]
                i = index(k["t"])
                if k["move"] == "lift":
                    cur = jog.current_tip_and_anchor()
                    if cur is None:
                        fail("the arm went away")
                        return
                    up = np.array(cur[0], dtype=float)
                    up[2, 3] += k["up_mm"] / 1000.0
                    why = await walk_to(k["label"], lambda up=up: up)
                    if why:
                        fail(why)
                        return
                    continue
                if k["move"] == "line":
                    g = float(demo.q_cmd[i, gi])
                    if k["grip"] == "first":
                        g_now = jog.current_gripper()
                        jog.set_gripper(g)
                        act.step = f"{k['label']}: setting the gripper"
                        await asyncio.sleep(
                            abs(g - (g if g_now is None else g_now)) / jog.GRIP_UNITS_S + runtime.ACT_TICK_S
                        )
                    else:
                        jog.set_gripper(g)  # it moves on its own clock while the arm walks
                    why = await walk_to(k["label"], lambda f=k["frame"], tip=demo.tips[i]: frame_now(f) @ tip)
                    if why:
                        fail(why)
                        return
                    continue
                if not waited:
                    why = await caught_up()
                    if why:
                        fail(why)
                        return
                    waited = True
                # The follow is aimed by the pose the act holds for its object, as every walk before it was; the line to
                # the key frame before it is planned again from where the arm stands.
                act.step = f"planning {k['label']}"
                aim_then = follow() if k["frame"] == (obj or "") else None
                planned = await plan_span(
                    k["label"],
                    (n - 1, n + 1),
                    delta_cam=aim_then,
                    aim=None if aim_then is None else aim_error(aim_then),
                )
                if not planned["ok"]:
                    fail(planned["reason"])
                    return
                i0 = index(frames[n - 1]["t"])
                closing = float(np.sign(demo.q_cmd[i, gi] - demo.q_cmd[i0, gi])) or 1.0
                i_grasp_end, last_follow = i, planned
                if (
                    carries
                    and holding is not None
                    and n == holding["begins"]
                    and k["label"] in planned["stage"]
                ):
                    grip_i = holding["grip"]
                    at = list(planned["stage"]).index(k["label"]) + (grip_i - i0 - 1)
                    if 0 < at < len(planned["q"]) - 1:
                        # Where the plan put the held object: the grasp pose is judged against it.
                        session_held = holding["frame"] == (obj or "")
                        held_then = (
                            planning._delta_base(demo, aim_then, t_bc)
                            if session_held
                            else motions.get(holding["frame"])
                        )
                        held_name = picked if session_held else holding["frame"]

                        async def watch_grip(
                            at: int = at,
                            grip_i: int = grip_i,
                            planned: dict = planned,
                            closing: float = closing,
                            held_then: np.ndarray | None = held_then,
                            held_name: str = held_name,
                            demo: runtime._Demo = demo,
                            grip: dict = grip,
                            demo_grip: dict | None = demo_grip,
                            held_obj: str | None = held_obj,
                        ) -> None:
                            """Beside the stream, from its sample where the demo's grip became firm: once the gripper's
                            reading stops, the grasp check (a miss halts the stream) and the fingertip then, the grasp pose;
                            with the demo's grip showing the object, views of it in the gripper while the arm stands still,
                            read while the act goes on. Post: ``grip`` holds ``fix`` and ``live`` as far as measured."""
                            nonlocal halt, held
                            while sent < at:
                                if interrupted():
                                    return
                                await asyncio.sleep(runtime.ACT_TICK_S)
                            g = g_prev = jog.current_gripper()
                            t_prev = t0 = time.monotonic()
                            while g is not None and time.monotonic() - t0 < runtime.GRIP_SETTLE_S:
                                await asyncio.sleep(runtime.ACT_TICK_S)
                                g = jog.current_gripper()
                                if g is not None and time.monotonic() - t_prev >= core.GRIP_STILL_S:
                                    if abs(g - g_prev) < core.GRIP_STILL_UNITS:
                                        break
                                    g_prev, t_prev = g, time.monotonic()
                            cur = jog.current_tip_and_anchor()
                            if g is None or cur is None or interrupted():
                                return
                            why = grasp_missed(float(planned["q"][at][gi]), g, grip_i, closing)
                            if why:
                                halt = why
                                return
                            # Without seeing the object: the grasp was aimed by its estimated pose, so the hold is the
                            # demo's, changed by however far the arm landed from that aim.
                            if held_then is not None:
                                grip["fix"] = planning._grasp_pose_fix(demo.tips[grip_i], held_then, cur[0])
                            if (
                                with_groups
                            ):  # the camera cannot see it in the fingers; the arm's joints say where it is
                                with runtime._state.lock:
                                    frames_g = list(runtime._state.groups_feed.frames.get(held_name, ()))
                                if frames_g:
                                    tip_obj = (
                                        np.linalg.inv(cur[0]) @ t_bc @ frames_g[-1][1]
                                    )  # how it sits in the fingers
                                    held = (
                                        held_name,
                                        asyncio.create_task(groups._feed_held(held_name, tip_obj, t_bc)),
                                    )
                                    beside.append(held[1])
                            if demo_grip is None:
                                return
                            views: list[Any] = []
                            last, t_last = cur[0][:3, 3].copy(), time.monotonic()
                            while len(views) < core.HOLD_VIEWS + 2:
                                await asyncio.sleep(runtime.HOLD_VIEW_GAP_S)
                                view = await holds._held_view(held_obj)
                                if view is None or interrupted():
                                    break
                                now = time.monotonic()
                                if (
                                    np.linalg.norm(view[3][:3, 3] - last) / max(now - t_last, 1e-6)
                                    > core.HOLD_STILL_M_S
                                ):
                                    break  # the lift has begun
                                views.append(view)
                                last, t_last = view[3][:3, 3].copy(), now
                            if len(views) < core.HOLD_MIN_VIEWS:
                                grip["live"] = (
                                    None,
                                    f"not measured: the arm stood still at the grip for {len(views)} views",
                                )
                                return
                            grip["live"] = await holds._live_hold(
                                demo,
                                held_obj,
                                demo_grip["hold"],
                                t_bc,
                                stopped,
                                views=views,
                                method=demo_grip.get("method", "find"),
                            )

                        grip_watch = asyncio.create_task(watch_grip())
                        beside.append(grip_watch)
                why = await stream(planned)
                if why:
                    fail(why)
                    return
            if split == phase_stop:
                act.step, act.ok = "done", True
                continue
            if not waited:
                why = await caught_up()
                if why:
                    fail(why)
                    return

            # ── from the grasp check on ──
            if not carries:  # nothing held from here: the rest as planned
                await jog.joints_stop()
                streaming = False
                rest = await plan_span("the rest", (split, phase_stop), landing=demo.landing)
                if not rest["ok"]:
                    fail(rest["reason"])
                    return
                why = await stream(rest)
                if why:
                    fail(why)
                    return
                act.step, act.ok = "done", True
                continue
            act.step = "checking the grasp"
            g_obs = await gripper_still()
            if g_obs is None:
                fail("the arm went away")
                return
            if last_follow is not None and i_grasp_end is not None:
                why = grasp_missed(float(last_follow["q"][-1][gi]), g_obs, i_grasp_end, closing)
                if why:
                    fail(why)
                    return
            closing_cmd = (
                float(last_follow["q"][-1][gi]) if last_follow is not None else float(jog.current_gripper())
            )
            await jog.joints_stop()  # the grasp's closing kept
            streaming = False
            # Check the carry as one plan, but execute only to its first held line before measuring the hold.
            # Later lines may already touch the support: correcting there can slide or rotate it underneath the
            # object. Mark the first held line at a clear observation pose; preserve the grasp throughout correction.
            held_follow = next(
                (n for n in range(split, phase_stop) if made_held[n] and frames[n]["move"] == "follow"), None
            )
            end_lines = held_follow if held_follow is not None else phase_stop
            hold_line = min(n for n in range(split, end_lines) if frames[n]["move"] == "line")
            act.step = "planning the carry"
            carry = await plan_span("carry", (split, phase_stop))
            if not carry["ok"] and landing_turn is not None:
                # The turn the start took no longer plans from where the arm and the object are now, but any turn the
                # demo's landing rule allows is the same place: rank them again.
                saved, landing_turn = landing_turn, None
                carry = await plan_span(
                    "carry, the landing chosen again", (split, phase_stop), landing=demo.landing
                )
                landing_turn = (
                    (carry["landing_centre"], float(carry["landing"]["turn_deg"]))
                    if carry["ok"] and (carry.get("landing") or {}).get("turn_deg") is not None
                    else saved
                )
            if not carry["ok"]:
                fail(carry["reason"])
                return
            last_label = frames[hold_line]["label"]
            last = max(j for j, s in enumerate(carry["stage"]) if s == last_label)
            q_carry = np.array(carry["q"][: last + 1], dtype=float)
            q_carry[:, gi] = closing_cmd  # the grasp's closing: on the object the fingers stand short of it
            why = await stream(
                {"q": q_carry, "times": carry["times"][: last + 1], "stage": carry["stage"][: last + 1]}
            )
            if why:
                fail(why)
                return
            await jog.joints_stop()  # stand still to measure the hold, the closing kept
            streaming = False
            # Re-check at the first held line, before later approach/contact lines. An earlier grip estimate cannot
            # replace a missing current visual hold when correction is enabled.
            live_place, live_place_problem = None, "designation-based demo carry reference unavailable"
            if demo_carry is not None or act.correct_hold:
                act.step = f"checking how {held_obj} sits in the gripper"
                why = await still()
                if why:
                    fail(f"at {last_label}: {why}")
                    return
            if demo_carry is not None:
                live_place, live_place_problem = await holds._live_hold(
                    demo,
                    held_obj,
                    demo_carry["hold"],
                    t_bc,
                    lambda: act.stop_requested,
                    method=demo_carry.get("method", "find"),
                )
                if act.stop_requested:
                    fail("stopped")
                    return
            if grip_watch is not None:
                await grip_watch
            fix_grasp = grip.get("fix")
            live_grip, live_grip_problem = grip.get(
                "live", (None, "not measured: the demo's grip shows too little of it")
            )
            fix_grip = (
                demo_grip["hold"] @ np.linalg.inv(live_grip["hold"])
                if demo_grip is not None and live_grip is not None
                else None
            )
            fix_place = (
                demo_carry["hold"] @ np.linalg.inv(live_place["hold"])
                if demo_carry is not None and live_place is not None
                else None
            )
            direct, direct_problem, direct_attempts = None, "", []
            if act.correct_hold and fix_place is None:
                act.step = f"matching {held_obj} against the demo's held views"
                prediction = fix_grip if fix_grip is not None else fix_grasp
                direct, direct_problem, direct_attempts = await held_views.measure(
                    demo, held_obj, t_bc, prediction, lambda: act.stop_requested
                )
                if act.stop_requested:
                    fail("stopped")
                    return
                if (
                    direct is not None
                    and direct["orientation"] == "predicted"
                    and not act.allow_grasp_prediction
                ):
                    direct_problem = "held position measured, but orientation still requires grasp prediction"
                    direct = None
                if direct is not None:
                    fix_place = direct["fix"]
                else:
                    live_place_problem += f"; {direct_problem}"
            fix, used, evidence = fix_place, "pre-place", "measured"
            if direct is not None:
                used = "held-view registration"
                if direct["orientation"] == "predicted":
                    used, evidence = "held-view position; rotation predicted", "partly measured"
            if not act.correct_hold:
                fix, used, evidence = np.eye(4), "off", "off"
            elif fix is None and act.allow_grasp_prediction:
                # Carry the best grasp-time attachment forward under an explicit
                # no-slip assumption. A group pose driven by the same arm joints
                # does not make this an independent post-lift measurement.
                fix, used = (
                    (fix_grip, "grip prediction") if fix_grip is not None else (fix_grasp, "grasp prediction")
                )
                evidence = "predicted"
            if fix is None:
                # A grasp-time estimate cannot observe slip or a wrong initial find. Keep the closing and
                # refuse further approach/release when the held pose cannot be verified after the carry.
                act.place = {
                    **(act.place or {}),
                    "hold_used": "unverified",
                    "demo_carry_problem": carry_problem,
                    "live_place": live_place_problem,
                    "grasp_estimate_available": fix_grasp is not None,
                    "grip_measurement_available": fix_grip is not None,
                    "allow_grasp_prediction": act.allow_grasp_prediction,
                    "held_view_attempts": direct_attempts,
                }
                fail(
                    f"cannot verify {held_obj}'s current held pose: {carry_problem or live_place_problem}; "
                    "placement stopped with the gripper closed"
                )
                return
            i_last = index(frames[hold_line]["t"])
            aimed = frame_now(frames[hold_line]["frame"]) @ demo.tips[i_last]
            shift_m, shift_deg = core.pose_residual(aimed @ fix, aimed)

            def summary(hold: dict | None, problem: str) -> dict[str, Any] | str:
                return (
                    problem
                    if hold is None
                    else {
                        **{k: hold[k] for k in ("n", "views", "spread_mm", "spread_deg")},
                        "hold": np.asarray(hold["hold"]).tolist(),
                    }
                )

            act.place = {
                **(act.place or {}),
                "hold_used": used,
                "hold_evidence": evidence,
                "fallback_reason": (carry_problem or live_place_problem) if evidence == "predicted" else "",
                "held_view_comparison": None
                if direct is None
                else {k: v for k, v in direct.items() if k != "fix"},
                "held_view_problem": direct_problem,
                "held_view_attempts": direct_attempts,
                "demo_grip": summary(demo_grip, grip_problem),
                "live_grip": summary(live_grip, live_grip_problem),
                "demo_carry": summary(demo_carry, carry_problem),
                "live_place": summary(live_place, live_place_problem),
                "shift_mm": shift_m * 1000.0,
                "shift_deg": shift_deg,
                "fix": fix.tolist(),
            }
            run.meta["hold_selection"] = {
                "object": held_obj,
                "interaction": phase_index,
                "at": time.time(),
                "source": used,
                "evidence": evidence,
                "reason": direct["reason"] if evidence == "partly measured" else act.place["fallback_reason"],
            }
            run.meta.setdefault("hold_selections", []).append(dict(run.meta["hold_selection"]))
            run.meta.setdefault("holds", []).append(
                {"interaction": phase_index, "object": held_obj, **act.place}
            )
            if (
                fix_grasp is not None
            ):  # what the camera saw against the grasp pose: what the closing and the lift did
                act.place["grasp_pose_mm"], act.place["grasp_pose_deg"] = (
                    v * k
                    for v, k in zip(core.pose_residual(aimed @ fix_grasp, aimed), (1000.0, 1.0), strict=True)
                )
                if evidence == "measured":
                    seen_m, seen_deg = core.pose_residual(aimed @ fix, aimed @ fix_grasp)
                    act.place.update(seen_vs_grasp_pose_mm=seen_m * 1000.0, seen_vs_grasp_pose_deg=seen_deg)
            if (
                fix_grip is not None and fix_place is not None
            ):  # how far the lift and the carry moved the hold
                moved_m, moved_deg = core.pose_residual(aimed @ fix_grip, aimed @ fix_place)
                act.place.update(grip_vs_place_mm=moved_m * 1000.0, grip_vs_place_deg=moved_deg)
            # The hold changes the target after the carry has already been checked. Plan its correction together with
            # the remaining key frames before commanding it: a walk can otherwise stall at a wrist limit for 20 s.
            act.step = "planning the hold correction and place"
            placing = await plan_span(
                "hold correction and place", (hold_line, phase_stop), fix, grip_cmd=closing_cmd
            )
            if not placing["ok"] and landing_turn is not None:
                saved, landing_turn = landing_turn, None
                placing = await plan_span(
                    "hold correction and place, the landing chosen again",
                    (hold_line, phase_stop),
                    fix,
                    grip_cmd=closing_cmd,
                    landing=demo.landing,
                )
                landing_turn = (
                    (placing["landing_centre"], float(placing["landing"]["turn_deg"]))
                    if placing["ok"] and (placing.get("landing") or {}).get("turn_deg") is not None
                    else saved
                )
            if landing_obj is not None and landing_obj in motions and held_follow is not None:
                end_at = index(frames[held_follow]["t"])
                then = run.meta.get("target", {}).get("motion_base")
                if then is not None:  # how far the object landed on moved the place since the grasp
                    act.place["target_moved_mm"] = 1000.0 * float(
                        np.linalg.norm(
                            (motions[landing_obj] @ demo.tips[end_at])[:3, 3]
                            - (np.asarray(then, dtype=float) @ demo.tips[end_at])[:3, 3]
                        )
                    )
            if not placing["ok"]:
                fail(placing["reason"])
                return
            # Settle at the corrected held line before descending or releasing; a feasible command does not prove
            # that the loaded arm followed it. Both pieces use the very same checked joint path.
            corrected = max(j for j, s in enumerate(placing["stage"]) if s == last_label)
            why = await stream(
                {
                    "q": placing["q"][: corrected + 1],
                    "times": placing["times"][: corrected + 1],
                    "stage": placing["stage"][: corrected + 1],
                }
            )
            if why:
                fail(why)
                return
            if corrected == len(placing["q"]) - 1:
                act.step, act.ok = "done", True
                continue
            placing = {
                **placing,
                "q": placing["q"][corrected:],
                "times": placing["times"][corrected:] - placing["times"][corrected],
                "stage": placing["stage"][corrected:],
            }
            # The follow the hold ends in streams to its end, the release included, and the held object is let go of there;
            # what comes after streams on from that sample with the gripper empty.
            release_t = None if holding["release"] is None else float(demo.t[holding["release"]])
            release_label = next(
                (
                    k["label"]
                    for k in frames[split:phase_stop]
                    if release_t is not None and k["t"] >= release_t
                ),
                None,
            )
            ends = [j for j, s in enumerate(placing["stage"]) if s == release_label]
            end = ends[-1] if ends else len(placing["q"]) - 1
            why = await stream(
                {
                    "q": placing["q"][: end + 1],
                    "times": placing["times"][: end + 1],
                    "stage": placing["stage"][: end + 1],
                }
            )
            if not why and release_label is not None:
                run.meta["hold_selection"]["until"] = time.time()
                run.meta["hold_selections"][-1]["until"] = run.meta["hold_selection"]["until"]
            if (
                held is not None and release_label is not None
            ):  # the recorded release, after all held interactions
                held[1].cancel()
                await groups._groups_call("POST", "/objects/held", {held[0]: None})
                held = None
            if why:
                fail(why)
                return
            if end < len(placing["q"]) - 1:
                why = await stream(
                    {
                        "q": placing["q"][end:],
                        "times": placing["times"][end:] - placing["times"][end],
                        "stage": [placing["stage"][end + 1], *placing["stage"][end + 1 :]],
                    }
                )
                if why:
                    fail(why)
                    return
            act.step, act.ok = "done", True
    except Exception as e:  # the arm holds its last target; the operator sees why
        logger.exception("act failed")
        fail(f"act error: {e}")
    finally:
        for task in beside:
            task.cancel()
        if held is not None:  # the act ended holding it: the point groups take it back where the arm has it
            with contextlib.suppress(Exception):
                await groups._groups_call("POST", "/objects/held", {held[0]: None})
        with runtime._state.lock:
            act.find_error = None
        if limits_before is not None:
            with contextlib.suppress(Exception):
                jog.set_walk_limits(*limits_before)
        if streaming:
            # Back to the Cartesian walk where the arm stopped, still commanding the grasp's closing.
            with contextlib.suppress(Exception):
                await jog.joints_stop()
        if run is not None:
            with runtime._state.lock:
                runtime._state.run = None
            with (
                runtime._state.lock
            ):  # what carried the objects: each one's pose in the point groups, per frame
                kept = {
                    n: [(t, m) for t, m in runtime._state.groups_feed.frames.get(n, ()) if t > t0]
                    for n, t0 in followed.items()
                }
            run.meta["groups"] = {
                "on": with_groups,
                "frames": {n: [[t, *np.round(m, 6).ravel().tolist()] for t, m in f] for n, f in kept.items()},
            }
            samples: list[dict[str, Any]] = []
            if run.arm_recording:
                with contextlib.suppress(Exception):
                    samples = jog.stop_record()
            # Keep the correction and both measured holds beside this act, even if its trial index is lost.
            run.meta["place"] = act.place
            result = {"ok": act.ok, "step": act.step, "reason": act.reason, "progress": act.progress}
            try:
                run_dir = await asyncio.get_event_loop().run_in_executor(
                    runtime._RUN_EXECUTOR, recording._finish_run, run, samples, jog.fk_tip, result
                )
            except Exception:  # the act's outcome stands without its recording
                logger.exception("writing the act's recording failed")
        if moved:
            with contextlib.suppress(Exception):
                recording._record_trial(run_dir)
        act.on = False


async def _act_task(speed: float, server: str | None = None) -> None:
    """Validate demo references, then execute the single key-frame runner."""
    with runtime._state.lock:
        demo, act = runtime._state.demo, runtime._state.act
    if demo is not None:
        try:
            problem = await asyncio.get_running_loop().run_in_executor(
                runtime._RENDER_EXECUTOR, holds._prepare_demo_references, demo
            )
        except Exception as exc:
            logger.exception("Demo surface validation failed before motion")
            problem = f"demo surface validation error: {type(exc).__name__}: {exc}"
        with runtime._state.lock:
            stale = runtime._state.demo is not demo or runtime._state.act is not act or act.stop_requested
        if stale or problem:
            act.on, act.ok, act.step = False, False, "aborted"
            act.reason = "demo or act changed during surface validation" if stale else problem
            return
    await _act_key_frames_task(speed, server)


@router.post("/act")
async def act_start(body: ActBody, request: Request) -> dict:
    """Replay the demo on the object where it is now: the whole recorded path, carried by the object's motion."""
    from .. import jog

    with runtime._state.lock:
        demo, test, teach, act = (
            runtime._state.demo,
            runtime._state.test,
            runtime._state.teach,
            runtime._state.act,
        )
        if act.on:
            raise HTTPException(409, "an act is in progress")
        if demo is None:
            raise HTTPException(409, "record or load a demo first")
        if not planning._has_key_frames(demo):
            raise HTTPException(409, "mark a key frame first (Edit demo)")
        # No trusted live view is needed: the act finds the object afresh without one. A track kept through an act
        # has added the points it saw while carried, more than a view of the object at rest can show.
        if teach is None or runtime._state.teach_job is not None:
            raise HTTPException(409, "find the object first")
        if test is not None and (test.result.get("camera_check") or {}).get("moved"):
            raise HTTPException(409, "the camera or the tray moved since the calibration; recalibrate first")
        runtime._state.track.follow = False  # the act owns the target now
        act.plan = act.place = None
        act.inject = body.inject.model_dump() if body.inject is not None and _injects(body.inject) else None
        act.correct_hold = body.correct_hold
        act.allow_grasp_prediction = body.allow_grasp_prediction
        act.finds = []
        act.on, act.ok, act.reason, act.step, act.stop_requested, act.progress, act.speed = (
            True,
            None,
            "",
            "starting",
            False,
            0.0,
            body.speed,
        )
    if jog.current_robot_id() is None:
        with runtime._state.lock:
            act.on = False
        raise HTTPException(409, "connect the arm first")
    act.task = asyncio.create_task(_act_task(body.speed, str(request.base_url).rstrip("/")))
    return {"status": "acting", "n": int(len(demo.t))}


@router.post("/act/stop")
async def act_stop() -> dict:
    """Stop the replay and hold the arm where it is."""
    from .. import jog

    with runtime._state.lock:
        runtime._state.act.stop_requested = True
    cur = jog.current_tip_and_anchor()
    if cur is not None:
        with contextlib.suppress(RuntimeError):
            jog.set_target_pose(cur[0])
    return {"status": "stopping"}
