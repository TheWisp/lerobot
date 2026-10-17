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
import copy
import dataclasses
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from lerobot.showservo import motion as core, reactive, symmetry

from . import acquisition, camera, groups, held_views, holds, live_execution, planning, recording, runtime

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
    speed: float = 1.0  # scales the straight lines (from the demo's own pace between key frames) and the follows (its clock)
    inject: InjectBody | None = None
    correct_hold: bool = True  # False: a place replays the demo against the target, uncorrected for the hold
    allow_grasp_prediction: bool = False  # use the grasp-time hold if post-carry vision cannot verify it
    queue_horizon_s: float = Field(1.0, ge=0.5, le=5.0)  # checked, replaceable motor lookahead
    confirm_recovered_pose: str | None = None  # token from the warning for this find and demo
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

    The act's objects are those its key frames are held in (:func:`_act_objects`): each is found before motion,
    the first one's find restarting the live session with the others joining it. The act's pose of each is the last
    view that placed it (:func:`_view_places`), moved on with what it rests on (the point groups, when acts use
    them). The whole act is planned and judged from the arm's present joints before anything moves.

    Before the grasp check (:func:`_grasp_split`) each line and lift is planned from the observed joints and streamed
    through the same live, checked path as a follow. Object-frame goals are re-aimed every cycle. During the follow
    the demo's hold begins in, what the grip shows is read on the fly, as the old act
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
    pregrasp_streaming = False

    def interrupted() -> str:
        st = jog.current_status()
        if act.stop_requested:
            return "stopped"
        if halt:
            return halt
        if pregrasp_streaming and (tracking or with_groups):
            with runtime._state.lock:
                state = (runtime._state.track.last or {}).get("state")
            # A free object uses the same carried observation whether it is
            # about to be picked or is a placement target. Motion itself is
            # not failure: the live queue already retargets it each update.
            supported = state == "tracking" or (with_groups and groups._motion_supported(picked))
            if not supported and not with_groups:
                pose = jog.current_tip_and_anchor()
                if pose is not None:
                    relative = follow() @ inv_start_delta
                    observed_points = start_points @ relative[:3, :3].T + relative[:3, 3]
                    tip_cam = np.linalg.inv(t_bc) @ pose[0]
                    if reactive.nearby_object_extent(observed_points, tip_cam[:3, 3], core.ACT_REACH_TOL_M):
                        return (
                            f"{picked}'s pose is unverified as the gripper enters its measured extent; "
                            "stopped before an unobserved pregrasp contact"
                        )
            if not supported and with_groups:
                current = follow()
                relative = current @ inv_start_delta
                shift = reactive.unverified_object_shift(relative, start_points, core.ACT_REACH_TOL_M)
                if shift is not None:
                    return (
                        f"{picked} moved {shift * 1000:.1f} mm while its pregrasp pose was unverified; "
                        "stopped before following a stale target"
                    )
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
        if act.force_recovered_pose and (
            teach is None or planning._recovered_pose_token(demo, teach) != act.recovered_pose_token
        ):
            fail("the proposed grasp changed; review its preview and confirm it again")
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
        # Find every object before the arm enters the scene. An in-flight
        # restart sees only the fragment exposed beside the gripper and can
        # reject an otherwise well-observed grasp midway through motion.
        relocate = {
            name: asyncio.create_task(acquisition._locate_afresh(name, stopped, track=obj is None))
            for name in others
        }
        beside.extend(relocate.values())
        if obj is not None:
            founds = {}
            for name in others:
                act.step = f"finding {name}"
                why = await relocate[name]
                if why:
                    fail(why)
                    return
                founds[name] = acquisition._located(demo, name)
            more = {
                name: found["mask"]
                for name, found in founds.items()
                if found and found["ok"] and found.get("mask") is not None
            }
            act.step = f"finding {obj}"
            why = await acquisition._find_afresh(obj, stopped, more or None)
            if why:
                fail(why)
                return
            with runtime._state.lock:
                test, teach, tracking = runtime._state.test, runtime._state.teach, runtime._state.track.on
            for name in more:
                if founds[name].get("delta") is not None:
                    acquisition._store_located(name, {**founds[name], "tracking": True})
            # A recovered find has passed _find_afresh only where the act's starting find put the object
            # (planning._recovered_pose_problem). One that told the object's ends apart leaves no ambiguity to bypass.
            fresh = ((teach.keypoints.get("ref") if teach is not None else None) or {}).get(
                "surface_constraint"
            )
            if act.force_recovered_pose and (fresh or {}).get("state") != "recovered":
                act.force_recovered_pose = False
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
        kin = copy.deepcopy(kin)  # planning FK/IK must not mutate the motor loop's model
        plan_kin = copy.deepcopy(kin)  # plans made ahead run beside the stream's solves: a model of their own
        ranges = jog.servo_ranges()
        landing_obj = planning._landing_object(demo)
        landing_turn: tuple[np.ndarray, float, np.ndarray] | None = (
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

        symmetry_lines: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
        hold_live = False
        held_obj = None

        def carried(name: str, since: float) -> np.ndarray:
            """How the point groups moved ``name`` since the frame read at ``since``, camera frame; none without
            them."""
            if not with_groups:
                return np.eye(4)
            line = symmetry_lines.get(name) if not (hold_live and name == held_obj) else None
            return (
                groups._carried(name, since, (line[0], line[1]), line[2])
                if line is not None
                else groups._carried(name, since)
            )

        def index(tk: float) -> int:
            return int(np.argmin(np.abs(demo.t - tk)))

        gi = MOTOR_NAMES.index("gripper")
        delta = np.asarray(test.result["delta_cam"], dtype=float)
        start_points = np.asarray(teach.keypoints["xyz"], dtype=float)
        inv_start_delta = np.linalg.inv(delta)
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
        seen_at = max(
            (w for w, _ in planning._certified_since(None)),
            default=teach.stamp if teach is not None and teach.stamp else time.time(),
        )
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
            landing_turn = (
                plan["landing_centre"],
                float(plan["landing"]["turn_deg"]),
                plan["landing_axis"],
            )
        jog.set_walk_limits(limits_before[0] * speed, limits_before[1] * speed)
        moved = True
        run = recording._begin_run(demo, speed, delta, t_bc, plan)
        run.meta["inject"], run.meta["correct_hold"] = inject, act.correct_hold
        run.meta["allow_grasp_prediction"] = act.allow_grasp_prediction
        run.meta["recovered_pose_override"] = act.force_recovered_pose
        run.meta["runner"] = "key_frames"

        def replanned(stage: str, p: dict[str, Any]) -> None:
            """A plan made after the start's: the page shows it, and the act's record keeps it beside the start's, so
            a refusal there names its marks."""
            act.plan = {k: p[k] for k in ("ok", "reason", "marks", "summary", "landing") if k in p}
            run.meta.setdefault("plans", []).append({"stage": stage, **act.plan})

        motion0 = planning._delta_base(demo, delta, t_bc)
        # Use the declared appearance equivalence only while the object rests
        # in the scene. Its visible RGB-D reference supplies an axis and point;
        # a held object instead follows the gripper's full rotation.
        if with_groups:
            inv_bc = np.linalg.inv(t_bc)
            for name, motion in {**motions, picked: motion0}.items():
                if name not in demo.objects:
                    continue
                spec = symmetry.normalise(demo.objects[name].get("symmetry"))
                if spec["kind"] != "continuous":
                    continue
                try:
                    points = planning._object_points(demo, name, np.eye(4))
                    cam_motion = inv_bc @ motion @ t_bc
                    live_points = points @ cam_motion[:3, :3].T + cam_motion[:3, 3]
                    axis, centre = symmetry.observed_axis(live_points, spec["axis"])
                    symmetry_lines[name] = (axis, centre, bool(spec.get("end_swap")))
                except (OSError, ValueError, IndexError) as exc:
                    logger.warning("cannot measure %s's declared symmetry axis: %s", name, exc)
            run.meta["symmetry_carry"] = {
                name: {"axis_cam": axis.tolist(), "point_cam": point.tolist(), "end_swap": swap}
                for name, (axis, point, swap) in symmetry_lines.items()
            }

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
            """The current measured motion, carried by its point group between certified views."""
            return planning._delta_base(demo, follow(), t_bc)

        def other_now(name: str, landed: bool = True) -> np.ndarray:
            """Another object's motion as its track has it now, moved on with what it rests on, base frame, the last
            one when it has no newer, turned by the landing the plan took when it is the object landed on (and
            ``landed``)."""
            motion, _problem = planning._object_motion(demo, name, t_bc, carry=lambda s: carried(name, s))
            if motion is not None:
                motions[name] = motion
            if landed and name == landing_obj and landing_turn is not None:
                return core.landed(motions[name], *landing_turn)
            return motions[name]

        def others_now(landed: bool = True) -> dict[str, np.ndarray]:
            return {name: other_now(name, landed) for name in others}

        def frame_now(frame: str) -> np.ndarray:
            """A key frame's frame's motion for a walk, base frame, asked again every tick."""
            if frame == core.WORLD_FRAME:
                return np.eye(4)
            if frame == (obj or ""):
                return approach()
            return other_now(frame)

        live_fix = np.eye(4)
        hold_watch = None
        hold_generation = 0
        held_attachment = {}

        def hold_prediction():
            return live_fix.copy()

        def hold_stopped(generation):
            return generation != hold_generation or not hold_live or bool(interrupted())

        def begin_hold_observation():
            """One observer per attachment; late results cannot survive release/regrasp."""
            nonlocal hold_watch
            if hold_watch is not None:
                return

            def publish_hold(result, held_obj=held_obj, phase_index=phase_index, generation=hold_generation):
                nonlocal live_fix
                if hold_stopped(generation):
                    return
                partial = result["orientation"] == "predicted"
                if partial and not act.allow_grasp_prediction:
                    return
                live_fix = np.asarray(result["fix"]).copy()
                at = result["accepted_at"]
                source = (
                    ("tracked position; rotation predicted" if partial else "verified tracking")
                    if result.get("source") == "verified tracking"
                    else ("held-view position; rotation predicted" if partial else "held-view registration")
                )
                evidence = "partly measured" if partial else "measured"
                run.meta["hold_selections"][-1]["until"] = at
                selection = {
                    "object": held_obj,
                    "interaction": phase_index,
                    "at": at,
                    "observed_at": result["observed_at"],
                    "source": source,
                    "evidence": evidence,
                    "fix": live_fix.tolist(),
                    "reason": result["reason"],
                }
                run.meta["hold_selection"] = selection
                run.meta["hold_selections"].append(dict(selection))
                act.place.update(
                    hold_used=source,
                    hold_evidence=evidence,
                    fix=live_fix.tolist(),
                    held_view_comparison={k: v for k, v in result.items() if k != "fix"},
                )
                if held_attachment:
                    # F = H_demo @ inv(H_live), hence H_new = inv(F_new) @ F_old @ H_old.
                    held_attachment["tip_obj"] = (
                        np.linalg.inv(live_fix) @ held_attachment["grasp_fix"] @ held_attachment["tip_obj"]
                    )
                    held_attachment["grasp_fix"] = live_fix.copy()

            def log_hold(problem, attempts, held_obj=held_obj):
                run.meta.setdefault("live_held_attempts", []).append(
                    {"at": time.time(), "object": held_obj, "problem": problem, "attempts": attempts}
                )

            if act.correct_hold:
                hold_watch = asyncio.create_task(
                    held_views.watch(
                        demo,
                        held_obj,
                        t_bc,
                        hold_prediction,
                        functools.partial(hold_stopped, hold_generation),
                        publish_hold,
                        log_hold,
                        prepared_references=held_bank_task,
                        demo_attachment=None if demo_carry is None else np.asarray(demo_carry["hold"]),
                    )
                )
                beside.append(hold_watch)

        async def reselect_live(planned, i, snapshot, correction, previous):
            nonlocal landing_turn
            path = reactive.slice_plan(planned, i)
            if landing_obj is None or landing_turn is None or landing_obj not in path["live_path"]["frames"]:
                return False
            turns = core.landing_turns(demo.landing, demo.objects[landing_obj].get("symmetry"))
            if len(turns) == 1:
                return False
            centre, old_turn, axis = landing_turn
            current_centre = snapshot[landing_obj][:3, :3] @ centre + snapshot[landing_obj][:3, 3]
            current_axis = snapshot[landing_obj][:3, :3] @ axis
            result = await asyncio.get_running_loop().run_in_executor(
                runtime._ACT_EXECUTOR,
                functools.partial(
                    reactive.choose_turn,
                    kin,
                    path["live_path"],
                    snapshot,
                    correction,
                    landing_obj,
                    current_centre,
                    turns,
                    path["q"],
                    previous,
                    gi,
                    ranges,
                    jog.workspace_box(),
                    (limits_before[0] * speed, limits_before[1] * speed),
                    jog.HZ,
                    axis=current_axis,
                ),
            )
            run.meta.setdefault("live_replans", []).append({"at": time.time(), "phase": i, **result})
            if interrupted() or result["turn_deg"] is None:
                return False
            # Recheck the current step on the next cycle. Changed observations
            # invalidate the search result; no command from this search is sent.
            fresh = {name: frame_now(name) for name in snapshot}
            changes = [core.pose_residual(snapshot[name], fresh[name]) for name in snapshot]
            changes.append(core.pose_residual(correction, live_fix))
            changed = any(m > core.ACT_SOLVE_TOL_M or deg > core.ACT_SOLVE_TOL_DEG for m, deg in changes)
            if changed:
                run.meta["live_replans"][-1]["discarded"] = "observation changed during search"
                return True
            landing_turn = (centre, (old_turn + result["turn_deg"]) % 360.0, axis)
            return True

        async def stream(
            planned: dict[str, Any], queued_last: Callable[[np.ndarray], None] | None = None
        ) -> str:
            """Stream a plan's joints, then wait for the arm to settle on the last: "" or why not. The jog's loop plays
            them back on its own clock (:func:`jog.play_joints`); ``sent`` follows the samples that went out.
            ``queued_last`` as :func:`live_execution.stream` has it."""
            nonlocal streaming, sent
            q, times, stage = planned["q"], planned["times"], planned["stage"]
            sent = -1
            if "live_path" in planned:
                current = jog.current_tip_and_anchor()
                if current is None:
                    return "the arm went away"
                # A prior slice may have been retargeted; never restart at its old world-space endpoint.
                q = np.array(q, copy=True)
                q[0] = [current[2][m] if m != "gripper" else q[0, gi] for m in MOTOR_NAMES]
                planned = {**planned, "q": q}
            try:
                await jog.joints_start(q_dict(q[0]))
            except RuntimeError as e:
                return str(e)
            streaming = True
            n = len(q)
            if "live_path" in planned:

                def observed(phase, command, stamp, goal, composition):
                    nonlocal sent, hold_live
                    sent = phase
                    act.step, act.progress = stage[phase], (phase + 1) / n
                    # Intermediate motor samples move toward the next phase
                    # without completing it. Label the command by its intended
                    # target, while progress and gripper gating keep the last
                    # completed phase until the chunk's final sample.
                    target_phase = phase if composition is None else composition["phase"]
                    run.target(
                        stage[target_phase],
                        joints=command,
                        pose=goal,
                        at=stamp,
                        phase=phase,
                        hold_revision=None if composition is None else composition["hold_revision"],
                        composition=composition,
                    )
                    if hold_live and (command[gi] - closing_cmd) * closing < -1e-6:
                        # The attachment ends when opening starts, before withdrawal.
                        hold_live = False
                        if hold_watch is not None:
                            hold_watch.cancel()

                def live_motions():
                    snapshots = {name: frame_now(name) for name in set(planned["live_path"]["frames"])}
                    if obj in snapshots and inject and inject.get("at") == "aim":
                        motion = snapshots[obj]
                        snapshots[obj] = (
                            _inject_transform(inject, (motion @ demo.tips[grasp_at])[:3, 3]) @ motion
                        )
                    return snapshots

                def settled(lag):
                    act.place = {**(act.place or {}), "settled_short_deg": lag}

                def best_effort_stopped(label, reason, joints):
                    event = {"at": time.time(), "stage": label, "reason": reason, "joints": joints}
                    run.meta.setdefault("best_effort_stops", []).append(event)
                    act.place = {**(act.place or {}), "best_effort_stop": event}

                stream_diagnostics = {"stage": stage[-1]}
                run.meta.setdefault("stream_diagnostics", []).append(stream_diagnostics)
                why, _last = await live_execution.stream(
                    planned,
                    kin,
                    live_motions,
                    lambda: live_fix.copy(),
                    interrupted,
                    observed,
                    reselect_live,
                    limits_before,
                    speed,
                    settled=settled,
                    revision=lambda: len(run.meta.get("hold_selections", [])),
                    best_effort_stopped=best_effort_stopped,
                    diagnostics=stream_diagnostics,
                    queue_horizon_s=act.queue_horizon_s,
                    queued_last=queued_last,
                )
                return why
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
            """Reject an unobstructed closure; a blocked gripper alone cannot verify object lift."""
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

        def aimed(
            span: tuple[int, int], delta_cam: np.ndarray | None, objects: dict[str, np.ndarray]
        ) -> list:
            """The demo's fingertip at each key frame of ``span`` moved with its frame, by ``delta_cam`` (the session
            object, camera frame) and ``objects`` (base frame), uncorrected for the hold: what a plan of them aims at."""
            motion = planning._frame_motion(demo, delta_cam, t_bc, objects)
            out = []
            for n in range(*span):
                m = motion(frames[n]["frame"])
                out.append(None if m is None else m @ demo.tips[index(frames[n]["t"])])
            return out

        def plan_ahead(label: str, span: tuple[int, int], q_from: np.ndarray, fix: np.ndarray | None) -> dict:
            """Plan key frames ``span`` beside the arm from ``q_from``, the joints the stream now running will leave it
            at, aimed by what the act holds now, the landing chosen again when the turn the start took no longer plans,
            as the act would once the stream ends; but on the plan executor and its own model, so the stream's solves
            never wait for it. :func:`take_ahead` uses it when the arm gets there and what it was made from still
            holds."""
            q_from = np.array(q_from, dtype=float)
            session = any(frames[n]["frame"] == (obj or "") for n in range(*span))
            delta_cam = follow() if session else None
            planned_demo = demo
            ahead: dict[str, Any] = {"span": span, "q_from": q_from, "session": session, "fix": fix}

            async def attempts() -> list[tuple[str, dict, list, bool]]:
                made: list[tuple[str, dict, list, bool]] = []
                for suffix, landed, extra in (
                    ("", True, {}),
                    (", the landing chosen again", False, {"landing": planned_demo.landing}),
                ):
                    objects = others_now(landed)
                    p = await asyncio.get_running_loop().run_in_executor(
                        runtime._PLAN_EXECUTOR,
                        functools.partial(planning._plan_act, part=span, ranges=ranges, **extra),
                        planned_demo,
                        delta_cam,
                        t_bc,
                        plan_kin,
                        q_from,
                        limits_before,
                        jog.workspace_box(),
                        speed,
                        0,
                        objects,
                        fix,
                    )
                    made.append((label + suffix, p, aimed(span, delta_cam, objects), landed))
                    if p["ok"] or landing_turn is None:
                        break
                return made

            ahead["task"] = asyncio.create_task(attempts())
            beside.append(ahead["task"])
            return ahead

        async def take_ahead(ahead: dict | None, fix: np.ndarray | None) -> dict | None:
            """The plan made ahead (:func:`plan_ahead`), kept in the act's record as if made now, when what it was made
            from still holds: the arm stands within ACT_SETTLE_DEG of where it was planned from, and each key frame's
            target, corrected by the hold (``fix``, the act's now, against the plan's), moved less than the live view's
            path is planned again for (planning.PREVIEW_REPLAN_M, _DEG). None otherwise, the act then planning from where
            the arm stands as it would without; ``run.meta["ahead"]`` says which and why."""
            nonlocal landing_turn
            if ahead is None:
                return None
            why, made = "", []
            try:
                made = await ahead["task"]
            except Exception as e:  # a plan that could not be made ahead is made now
                why = f"planning ahead failed: {type(e).__name__}: {e}"
            cur = jog.current_tip_and_anchor()
            if not why and cur is None:
                why = "the arm went away"
            if not why:
                q_now = np.array([float(cur[2][m]) for m in MOTOR_NAMES])
                arm = np.arange(len(q_now)) != gi
                off = float(np.max(np.abs(q_now[arm] - ahead["q_from"][arm])))
                if off > runtime.ACT_SETTLE_DEG:
                    why = f"the arm stands {off:.1f} deg from where it was planned from"
            most = (0.0, 0.0)  # the largest change of a target since it was planned, mm and deg
            if not why:
                then = made[-1][2]
                now = aimed(ahead["span"], follow() if ahead["session"] else None, others_now(made[-1][3]))
                fix_then = np.eye(4) if ahead["fix"] is None else np.asarray(ahead["fix"], float)
                fix_now = np.eye(4) if fix is None else np.asarray(fix, float)
                for n, a, b in zip(range(*ahead["span"]), then, now, strict=True):
                    if (a is None) != (b is None):
                        why = f"{frames[n]['label']}'s frame was found or lost since"
                        break
                    if a is None:
                        continue
                    moved_m, moved_deg = core.pose_residual(a @ fix_then, b @ fix_now)
                    most = (max(most[0], moved_m * 1000.0), max(most[1], moved_deg))
                    if moved_m > planning.PREVIEW_REPLAN_M or moved_deg > planning.PREVIEW_REPLAN_DEG:
                        why = (
                            f"{frames[n]['label']} moved {moved_m * 1000.0:.1f} mm, {moved_deg:.1f} deg since"
                        )
                        break
            entry = {
                "at": time.time(),
                "plans": [m[0] for m in made],
                "used": not why,
                "why": why,
                "moved_mm": round(most[0], 2),
                "moved_deg": round(most[1], 2),
            }
            run.meta.setdefault("ahead", []).append(entry)
            if why:
                return None
            for label, p, _then, _landed in made:
                replanned(label, p)
            p, landed = made[-1][1], made[-1][3]
            if not landed and p["ok"] and (p.get("landing") or {}).get("turn_deg") is not None:
                landing_turn = (p["landing_centre"], float(p["landing"]["turn_deg"]), p["landing_axis"])
            entry["landing_turn_deg"] = None if landing_turn is None else float(landing_turn[1])
            return p

        async def caught_up() -> str:
            """Use the pre-motion finds and finish reading the demo's hold reference if it is still running."""
            nonlocal demo_grip, grip_problem, demo_carry, carry_problem, hold_task
            for name in others:
                if not relocate[name].done():
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
                    "landing_axis": None if landing_turn is None else landing_turn[2],
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
            held_bank_task = None
            if carries:
                if act.correct_hold:

                    async def prepare_held_bank(demo=demo, held_obj=held_obj):
                        return await asyncio.get_running_loop().run_in_executor(
                            runtime._RENDER_EXECUTOR, held_views.prepare_references, demo, held_obj, t_bc
                        )

                    held_bank_task = asyncio.create_task(prepare_held_bank())
                    beside.append(held_bank_task)

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
            hold_generation += 1
            live_fix, hold_live = np.eye(4), False
            if hold_watch is not None:
                hold_watch.cancel()
            hold_watch = None
            held_attachment.clear()
            grip: dict[str, Any] = {}
            grip_watch: asyncio.Task | None = None
            closing, i_grasp_end = 1.0, None
            last_follow: dict[str, Any] | None = None
            carry_ahead: dict | None = None
            waited = False
            for n in range(phase_start, split):
                k = frames[n]
                pregrasp_streaming = k["move"] in ("line", "lift")
                i = index(k["t"])
                if k["move"] in ("line", "lift"):
                    # Preflight and execution must take the same IK branch. A free-running Cartesian walk could
                    # wind a joint into a physical stop even when the separately checked plan had a safe branch.
                    act.step = f"planning {k['label']}"
                    aim_then = follow() if k["frame"] == (obj or "") else None
                    planned = await plan_span(
                        k["label"],
                        (n, n + 1),
                        delta_cam=aim_then,
                        aim=None if aim_then is None else aim_error(aim_then),
                    )
                    if not planned["ok"]:
                        fail(planned["reason"])
                        return
                    why = await stream(planned)
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
                if carries and n == split - 1:
                    # The carry is planned from where this grasp will leave the arm while it streams (take_ahead).
                    carry_ahead = plan_ahead("carry", (split, phase_stop), planned["q"][-1], None)
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
                                    held_attachment.update(
                                        tip_obj=tip_obj, grasp_fix=grip.get("fix", np.eye(4))
                                    )
                                    held = (
                                        held_name,
                                        asyncio.create_task(
                                            groups._feed_held(
                                                held_name, lambda: held_attachment["tip_obj"], t_bc
                                            )
                                        ),
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
            pregrasp_streaming = False
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
            continuous_hold = False
            if act.correct_hold and act.allow_grasp_prediction:
                live_grip, _ = grip.get("live", (None, "not yet measured"))
                seed = (
                    demo_grip["hold"] @ np.linalg.inv(live_grip["hold"])
                    if demo_grip is not None and live_grip is not None
                    else grip.get("fix")
                )
                if seed is not None:
                    continuous_hold = True
                    live_fix, hold_live = np.asarray(seed).copy(), True
                    source = "grip prediction" if live_grip is not None else "grasp prediction"
                    act.place = {
                        **(act.place or {}),
                        "hold_used": source,
                        "hold_evidence": "predicted",
                        "fallback_reason": "grasp attachment predicted until fresh held evidence arrives",
                        "fix": live_fix.tolist(),
                        "held_view_attempts": [],
                        "held_observer_started_before_carry": True,
                    }
                    run.meta["hold_selection"] = {
                        "object": held_obj,
                        "interaction": phase_index,
                        "at": time.time(),
                        "source": source,
                        "evidence": "predicted",
                        "fix": live_fix.tolist(),
                        "reason": act.place["fallback_reason"],
                    }
                    run.meta.setdefault("hold_selections", []).append(dict(run.meta["hold_selection"]))
                    run.meta.setdefault("holds", []).append(
                        {"interaction": phase_index, "object": held_obj, **act.place}
                    )
                    if held_attachment:
                        held_attachment["tip_obj"] = (
                            np.linalg.inv(live_fix)
                            @ held_attachment["grasp_fix"]
                            @ held_attachment["tip_obj"]
                        )
                        held_attachment["grasp_fix"] = live_fix.copy()
                    begin_hold_observation()
            # Check the carry as one plan, with the existing boundary before
            # approach/contact. Prediction-enabled acts observe alongside this
            # motion; strict acts still require verification at the boundary.
            held_follow = next(
                (n for n in range(split, phase_stop) if made_held[n] and frames[n]["move"] == "follow"), None
            )
            end_lines = held_follow if held_follow is not None else phase_stop
            hold_line = min(n for n in range(split, end_lines) if frames[n]["move"] == "line")
            act.step = "planning the carry"
            # What the carry planned ahead need not still match is the hold's correction: the carry's stream applies
            # the newest every step, as it does each observation of the hold, and the place is planned again with it.
            carry = await take_ahead(carry_ahead, None)
            if carry is None:
                carry = await plan_span("carry", (split, phase_stop), live_fix if continuous_hold else None)
                if not carry["ok"] and landing_turn is not None:
                    # The turn the start took no longer plans from where the arm and the object are now, but any turn
                    # the demo's landing rule allows is the same place: rank them again.
                    saved, landing_turn = landing_turn, None
                    carry = await plan_span(
                        "carry, the landing chosen again",
                        (split, phase_stop),
                        live_fix if continuous_hold else None,
                        landing=demo.landing,
                    )
                    landing_turn = (
                        (carry["landing_centre"], float(carry["landing"]["turn_deg"]), carry["landing_axis"])
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
            carried_plan = reactive.slice_plan(carry, stop=last + 1)
            carried_plan["q"] = q_carry
            if "live_path" in carried_plan:
                carried_plan["live_path"]["grips"] = np.full(last + 1, closing_cmd)
            # With the hold carried on, the place is planned from the carry's end while it streams (take_ahead).
            placing_ahead: dict | None = None

            def place_ahead(q_last: np.ndarray, span: tuple[int, int] = (hold_line, phase_stop)) -> None:
                """The place, planned from the command the carry ends at once that is queued, while the carry streams,
                with the hold as the act has it then."""
                nonlocal placing_ahead
                placing_ahead = plan_ahead("hold correction and place", span, q_last, hold_prediction())

            why = await stream(carried_plan, queued_last=place_ahead if continuous_hold else None)
            if why:
                fail(why)
                return
            if continuous_hold:
                # The observer already ran beside the carry. Preserve its newest
                # accepted attachment; never overwrite it with an earlier seed.
                fix = live_fix.copy()
            else:
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
                        direct_problem = (
                            "held position measured, but orientation still requires grasp prediction"
                        )
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
                        (fix_grip, "grip prediction")
                        if fix_grip is not None
                        else (fix_grasp, "grasp prediction")
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
                    "fallback_reason": (carry_problem or live_place_problem)
                    if evidence == "predicted"
                    else "",
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
                    "reason": direct["reason"]
                    if evidence == "partly measured"
                    else act.place["fallback_reason"],
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
                        for v, k in zip(
                            core.pose_residual(aimed @ fix_grasp, aimed), (1000.0, 1.0), strict=True
                        )
                    )
                    if evidence == "measured":
                        seen_m, seen_deg = core.pose_residual(aimed @ fix, aimed @ fix_grasp)
                        act.place.update(
                            seen_vs_grasp_pose_mm=seen_m * 1000.0, seen_vs_grasp_pose_deg=seen_deg
                        )
                if (
                    fix_grip is not None and fix_place is not None
                ):  # how far the lift and the carry moved the hold
                    moved_m, moved_deg = core.pose_residual(aimed @ fix_grip, aimed @ fix_place)
                    act.place.update(grip_vs_place_mm=moved_m * 1000.0, grip_vs_place_deg=moved_deg)
                live_fix = np.asarray(fix).copy()
                if held_attachment:
                    held_attachment["tip_obj"] = (
                        np.linalg.inv(live_fix) @ held_attachment["grasp_fix"] @ held_attachment["tip_obj"]
                    )
                    held_attachment["grasp_fix"] = live_fix.copy()
                hold_live = True

            begin_hold_observation()
            # The hold changes the target after the carry has already been checked. Plan its correction together with
            # the remaining key frames before commanding it: a walk can otherwise stall at a wrist limit for 20 s.
            act.step = "planning the hold correction and place"
            placing = await take_ahead(placing_ahead, fix)
            if placing is None:
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
                        (
                            placing["landing_centre"],
                            float(placing["landing"]["turn_deg"]),
                            placing["landing_axis"],
                        )
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
            why = await stream(reactive.slice_plan(placing, stop=corrected + 1))
            if why:
                fail(why)
                return
            if corrected == len(placing["q"]) - 1:
                act.step, act.ok = "done", True
                continue
            placing = reactive.slice_plan(placing, corrected)
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
            why = await stream(reactive.slice_plan(placing, stop=end + 1))
            hold_live = False
            if hold_watch is not None:
                hold_watch.cancel()
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
                withdrawal = reactive.slice_plan(placing, end)
                withdrawal["stage"] = [placing["stage"][end + 1], *placing["stage"][end + 1 :]]
                # A later key frame may close the fingers again while the just-released object is still between
                # them. In the 2026-10-14 insertion replay this coincided with the shoulder straining and freezing
                # at the first withdrawal step. Do not reclose until a new grasp interaction.
                if release_label is not None:
                    before = np.asarray(withdrawal["q"])[:, gi].copy()
                    withdrawal = reactive.keep_released(withdrawal, gi, closing)
                    run.meta["release_open_guard"] = {
                        "at": time.time(),
                        "gripper_at_release": float(before[0]),
                        "suppressed_samples": int(np.count_nonzero(before != withdrawal["q"][:, gi])),
                    }
                why = await stream(withdrawal)
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
            if run.timeline is not None:
                try:
                    run.meta["timeline"] = await asyncio.get_running_loop().run_in_executor(
                        runtime._RUN_EXECUTOR, run.timeline.close
                    )
                except Exception as e:
                    logger.exception("closing the Act camera timeline failed")
                    run.meta["timeline_error"] = f"{type(e).__name__}: {e}"
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
        reference = teach.keypoints.get("ref") or {}
        problem = planning._recovered_pose_problem(planning._session_object(demo) or "object", reference)
        token = planning._recovered_pose_token(demo, teach) if problem else None
        if problem and body.confirm_recovered_pose != token:
            raise HTTPException(
                409,
                {
                    "code": "recovered_pose_requires_confirmation",
                    "message": f"{problem}. The displayed path is a projection, not a collision check; "
                    "the selected end assignment could drive the gripper into the scene.",
                    "confirmation": token,
                    "preview": "/api/pregrasp/act/recovered-preview.jpg",
                },
            )
        if body.confirm_recovered_pose is not None and not problem:
            raise HTTPException(409, "the proposed pose changed; press Act to review it again")
        if test is not None and (test.result.get("camera_check") or {}).get("moved"):
            raise HTTPException(409, "the camera or the tray moved since the calibration; recalibrate first")
        runtime._state.track.follow = False  # the act owns the target now
        act.plan = act.place = None
        act.inject = body.inject.model_dump() if body.inject is not None and _injects(body.inject) else None
        act.correct_hold = body.correct_hold
        act.allow_grasp_prediction = body.allow_grasp_prediction
        act.queue_horizon_s = body.queue_horizon_s
        act.force_recovered_pose = bool(problem)
        act.recovered_pose_token = token if problem else None
        act.recovered_pose_object = planning._session_object(demo) if problem else None
        # Finds whose ends are settled: measured, or (the object just checked) confirmed by the operator.
        settled = {
            name: found
            for name, found in runtime._state.located.items()
            if (found.get("surface_constraint") or {}).get("state") != "recovered"
        }
        settled[planning._session_object(demo) or ""] = reference
        act.start_poses = {
            name: np.asarray(found["delta"], dtype=float)
            for name, found in settled.items()
            if name and found.get("ok") and found.get("delta") is not None
        }
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
