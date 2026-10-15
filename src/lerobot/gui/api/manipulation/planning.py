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

"""Object-relative key-frame composition, reach checks and placement planning."""

from __future__ import annotations

import functools
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import HTTPException

from lerobot.showservo import motion as core, symmetry

from . import acquisition, holds, recording, runtime


def _recovered_pose_problem(obj: str, found: dict[str, Any]) -> str:
    """A recovered axis has alternatives but no collision-checked grasp choice yet."""
    if (found.get("surface_constraint") or {}).get("state") != "recovered":
        return ""
    return (
        f"{obj}'s measured pose has equivalent end assignments; collision-checked grasp selection is required"
    )


def _object_motion(
    demo: runtime._Demo, obj: str, t_bc: np.ndarray, carry: Callable[[float], np.ndarray] | None = None
) -> tuple[np.ndarray | None, str]:
    """``(motion, problem)``: how ``obj``, one of :func:`_other_objects`, moved from the demo to now, base frame. Its
    last locate against the demo's view of it, times the inverse of where the demo's track had it on its pose frame
    (:func:`_pose_choice`). With ``carry``, the locate is moved on by ``carry(stamp)``, how the object moved since the
    locate's frame was read (camera frame). Pre: a key frame is held in ``obj``'s frame."""
    o = demo.objects.get(obj)
    if o is None or o.get("status") != "done" or demo.recording is None:
        return None, f"{obj!r} is not tracked in this demo"
    found = acquisition._located(demo, obj)
    if found is None:
        return None, f"click {obj} in the camera view to find it"
    if not found["ok"]:
        return None, found["reason"]
    if problem := _recovered_pose_problem(obj, found):
        return None, problem
    if acquisition._weak(found):
        return None, acquisition._weak(found)
    f = _pose_frame(demo, obj)
    if f is None:
        return None, f"{obj}'s demo pose is not visible before this interaction"
    with runtime._state.lock:  # a placing view replaces both at once
        pin, stamp = np.asarray(found["delta"], dtype=float), found.get("stamp")
    if carry is not None and stamp is not None:
        pin = carry(float(stamp)) @ pin
    reference, problem = holds._demo_control_pose(demo, obj, f)
    if reference is None:
        return None, problem
    cam = pin @ np.linalg.inv(reference)
    return t_bc @ cam @ np.linalg.inv(t_bc), ""


def _object_points(demo: runtime._Demo, obj: str, t_bc: np.ndarray) -> np.ndarray:
    """The designated object's surface on the frame it was clicked on, base frame (N, 3): its depth under its mask
    there, every fourth pixel. The finds' motions are from that view. Pre: the object is done on a stream demo."""
    o = demo.objects[obj]
    _rgb, depth = recording._stream_frame(demo.recording, int(o["frame"]))
    ys, xs = np.nonzero(np.asarray(o["mask"], dtype=bool) & (depth > 0))
    ys, xs = ys[::4], xs[::4]
    k = np.loadtxt(pathlib.Path(demo.recording) / "cam_K.txt")
    z = depth[ys, xs]
    cam = np.stack([(xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z], axis=1)
    return cam @ t_bc[:3, :3].T + t_bc[:3, 3]


def _normal_marks(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Marks as the server keeps them, in time order: a time and a kind, with an object, a frame (always, on a key
    frame mark), a move, a gripper and a lift's height when given. Raises HTTPException for a mark without a time or a kind."""
    try:
        return sorted(
            (
                {
                    "t": float(k["t"]),
                    "kind": str(k["kind"]),
                    **({"object": str(k["object"])} if k.get("object") else {}),
                    **(
                        {"frame": str(k.get("frame") or "")}
                        if k["kind"] == core.KEY_KIND or k.get("frame")
                        else {}
                    ),
                    **{key: str(k[key]) for key in ("move", "grip", "constraint") if k.get(key)},
                    **({"up_mm": float(k["up_mm"])} if k.get("up_mm") is not None else {}),
                }
                for k in marks
            ),
            key=lambda k: k["t"],
        )
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"a mark needs a time and a kind: {e}") from e


def _marks_problem(demo: runtime._Demo, kps: list[dict[str, Any]]) -> str:
    """Why ``kps`` cannot be the demo's marks, or "" when they can: their own rules (:func:`core.keypoints_problem`);
    every object they name tracked in the demo; a pose read on a frame its object is seen on; an unnamed frame only on
    a demo recorded after a teach; and no key frame the demo made holding an object held in that object's own frame,
    which the gripper carries."""
    problem = core.keypoints_problem(kps, float(demo.t[0]), float(demo.t[-1]))
    if problem:
        return problem
    frames = core.key_frames(kps)
    framed = {k["frame"] for k in frames} - {"", core.WORLD_FRAME}
    for named in sorted(({k.get("object", "") for k in kps} | framed) - {""}):
        if demo.objects.get(named, {}).get("status") != "done":
            return f"{named!r} is not a tracked object of this demo"
    times = recording._stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    for k in kps:
        if k["kind"] == "pose" and len(times):
            f = int(np.argmin(np.abs(times - (demo.t0 + k["t"]))))
            if not demo.objects[k["object"]]["seen"][f]:
                return (
                    f"{k['object']} is hidden at {k['t']:.2f} s: read its pose on a frame where it is in view"
                )
    if any(k["frame"] == "" for k in frames) and not demo.taught:
        return "nothing was taught before this demo: the key frames follow an object clicked on its recording"
    trial = runtime._Demo(**{**demo.__dict__, "keypoints": kps})
    holdings = _holdings(trial)
    try:
        core.interaction_phases(holdings, trial.t, frames)
    except ValueError as exc:
        return str(exc)
    for n, k in enumerate(frames):
        holding = core.holding_at(holdings, trial.t, frames, n)
        if (
            holding is not None
            and k["frame"] == holding["frame"]
            and core.held_at(holding, trial.t, frames, n)
        ):
            who = holding["frame"] or "the object"
            return (
                f"{k['label']} at {k['t']:.2f} s is held in {who}'s frame while the gripper holds {who}: "
                "hold it in another object's frame or the world's"
            )
    return ""


def _has_key_frames(demo: runtime._Demo) -> bool:
    return bool(core.key_frames(demo.keypoints))


def _act_objects(demo: runtime._Demo) -> list[str]:
    """The designated objects the demo's key frames are held in, in the order the first key frame in each comes."""
    frames = core.key_frames(demo.keypoints)
    return list(dict.fromkeys(k["frame"] for k in frames if k["frame"] not in ("", core.WORLD_FRAME)))


def _session_object(demo: runtime._Demo) -> str | None:
    """The object whose find starts the live Point2Pose session, the first the key frames are held in (the others
    join it: :func:`_session_names`); None for the object taught before the demo, which the teach follows."""
    return next(iter(_act_objects(demo)), None)


def _other_objects(demo: runtime._Demo) -> list[str]:
    """The act's objects that join the session started from :func:`_session_object`'s find, followed by
    :func:`_apply_others` and moved since the demo by :func:`_object_motion`."""
    first = _session_object(demo)
    return [n for n in _act_objects(demo) if n != first]


def _landing_object(demo: runtime._Demo) -> str | None:
    """The object the demo's landing rule turns its key frames about: the first a key frame made holding is held
    in; None when no key frame is made holding in a designated object's frame."""
    frames = core.key_frames(demo.keypoints)
    holding = _holding(demo)
    return next(
        (
            k["frame"]
            for n, k in enumerate(frames)
            if core.held_at(holding, demo.t, frames, n) and k["frame"] not in ("", core.WORLD_FRAME)
        ),
        None,
    )


def _reference_motion(demo: runtime._Demo, teach: runtime._Teach | None) -> tuple[np.ndarray | None, str]:
    """``(R, problem)``: the live track's motion times ``R`` is the object's motion from the demo to now.

    The live track follows the object whose find started the session (:func:`_session_object`). For the object
    taught before the demo, ``R`` undoes where the object was when the demo began. For a designated object it is the
    live view's registration against the demo's view of it, times the inverse of where the demo's track had the
    object on its pose frame (:func:`_pose_choice`). ``R`` is None, with the reason, when that object has not been
    found live yet.
    """
    obj = _session_object(demo)
    if obj is None:
        if not demo.taught:
            return None, "the marks name no object: open Edit demo and save them for the object clicked there"
        return np.linalg.inv(demo.delta0), ""
    o = demo.objects.get(obj)
    if o is None or o.get("status") != "done" or demo.recording is None:
        return None, f"{obj!r} is not tracked in this demo"
    ref = (teach.keypoints.get("ref") if teach is not None else None) or {}
    if ref.get("object") != obj:
        return None, f"click {obj} in the camera view to find it"
    if not ref.get("ok") or ref.get("delta") is None:
        return None, ref.get("reason") or f"{obj} was not found in the live view"
    if problem := _recovered_pose_problem(obj, ref):
        return None, problem
    # A certified live track only follows motion since its find. It cannot
    # strengthen that find's registration to the demo. Apply the same gate as
    # _find_afresh before either runner reuses an already-running session.
    if core.find_strength(ref.get("inliers"), ref.get("card_points"))[0] is False:
        return None, (
            f"a weak find: {obj} matched {ref['inliers']} of the demo view's {ref['card_points']} points; "
            "turn it closer to how it lay in the demo, then press Act"
        )
    f = _pose_frame(demo, obj)
    if f is None:
        return None, f"{obj}'s demo pose is not visible before this interaction"
    reference, problem = holds._demo_control_pose(demo, obj, f)
    if reference is None:
        return None, problem
    return np.asarray(ref["delta"], dtype=float) @ np.linalg.inv(reference), ""


def _pose_frame(demo: runtime._Demo, obj: str | None) -> int | None:
    """The stream frame the act reads a designated object's demo pose from, as :func:`_pose_choice` picks it."""
    choice = None if obj is None else _pose_choice(demo, obj)
    return None if choice is None else choice[0]


def _approach_times(demo: runtime._Demo, obj: str) -> list[float]:
    """The moments of the lines held in ``obj``'s frame before the arm first follows the demo in it, while nothing
    the act does has touched it."""
    frames = core.key_frames(demo.keypoints)
    out = []
    for k in frames:
        if k["frame"] != obj:
            continue
        if k["move"] == "follow":
            break
        out.append(k["t"])
    return out


def _pose_choice(demo: runtime._Demo, obj: str) -> tuple[int, str] | None:
    """``(stream frame, where it comes from)``: the frame the act reads ``obj``'s demo pose from.

    The frame the operator set for it with a pose mark ("set"). Else the frame it was clicked on ("clicked"): its
    view there is the one every find matches against, so its pose there is exact, and the operator clicked it where
    it could be seen; taken when it comes no later than the arm first follows the demo in its frame. Else the last
    frame it was seen at or before its first key frame. None when ``obj`` is not tracked in the demo or no key frame
    is held in its frame yet (:func:`_approach_times`).
    """
    o = demo.objects.get(obj)
    times = recording._stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    if o is not None and len(times) and demo.interaction_span and demo.interaction_span[0]:
        a, b = demo.interaction_span
        local = [k for k in core.key_frames(demo.keypoints)[a:b] if k["frame"] == obj]
        if local:
            # After an earlier contact, the object's demo pose need not equal its
            # designation pose. Anchor this interaction before its first motion.
            at = int(np.argmin(np.abs(times - (demo.t0 + local[0]["t"]))))
            after = core.key_frames(demo.keypoints)[a - 1]["t"]
            seen = np.flatnonzero(np.asarray(o["seen"])[: at + 1] & (times[: at + 1] >= demo.t0 + after))
            if not len(seen):
                return None
            return int(seen[-1]), "before this interaction"
    pre = _approach_times(demo, obj)
    if o is None or o.get("status") != "done" or not len(times) or not pre:
        return None
    chosen = next(
        (float(k["t"]) for k in demo.keypoints if k["kind"] == "pose" and k.get("object") == obj), None
    )
    if chosen is not None:
        return int(np.argmin(np.abs(times - (demo.t0 + chosen)))), "set"
    click = int(o["frame"])
    if times[click] - demo.t0 <= max(pre):
        return click, "clicked"
    f0 = int(np.argmin(np.abs(times - (demo.t0 + min(pre)))))
    seen = np.flatnonzero(np.asarray(o["seen"])[: f0 + 1])
    return (int(seen[-1]) if len(seen) else click), "last seen by its first key frame"


def _object_pose_info(demo: runtime._Demo, name: str) -> dict[str, Any]:
    """Where the editor says ``name``'s demo pose is read: ``pose_t`` (seconds into the demo) and ``pose_from``. An
    object no mark follows yet reports the frame set for it, else the frame it was clicked on."""
    choice = _pose_choice(demo, name)
    times = recording._stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    if choice is None:
        o = demo.objects[name]
        chosen = next(
            (float(k["t"]) for k in demo.keypoints if k["kind"] == "pose" and k.get("object") == name), None
        )
        choice = (
            (int(o["frame"]), "clicked")
            if chosen is None
            else (int(np.argmin(np.abs(times - (demo.t0 + chosen)))), "set")
        )
    frame, where = choice
    return {"pose_t": float(times[frame] - demo.t0) if len(times) > frame else None, "pose_from": where}


def _delta_base(demo: runtime._Demo, delta_cam: np.ndarray, t_bc: np.ndarray) -> np.ndarray:
    """The object's motion from the demo to now, in the base frame: the tracker's fit as it is.

    Pre: :func:`_reference_motion` has no problem for the current teach.
    """
    with runtime._state.lock:
        teach = runtime._state.teach
    ref, problem = _reference_motion(demo, teach)
    assert ref is not None, problem
    rel = np.asarray(delta_cam, dtype=float) @ ref
    with runtime._state.lock:
        wrong = (
            runtime._state.act.find_error
        )  # an injected error: the act believes the object is here instead
    base = t_bc @ rel @ np.linalg.inv(t_bc)
    return base if wrong is None else wrong @ base


def _certified_since(since: float | None) -> list[tuple[float, np.ndarray]]:
    """The tracker's certified motions with their frames' times, oldest first, of frames read after ``since`` (wall
    clock), or all of them."""
    with runtime._state.lock:
        hist = list(runtime._state.track.history)
    return [
        (w, np.asarray(d, dtype=float))
        for (w, ok, d) in hist
        if ok and d is not None and (since is None or w > since)
    ]


def _frame_motion(
    demo: runtime._Demo,
    delta_cam: np.ndarray | None,
    t_bc: np.ndarray,
    motions: dict[str, np.ndarray] | None,
    aim: np.ndarray | None = None,
) -> Callable[[str], np.ndarray | None]:
    """Each key frame's frame's motion since the demo, base frame, read when asked: the world's none; the object the
    live track follows (:func:`_delta_base`, ``aim`` before it); another object's from ``motions`` (by name); None
    for one not found yet."""
    picked = _session_object(demo) or ""

    def motion(frame: str) -> np.ndarray | None:
        if frame == core.WORLD_FRAME:
            return np.eye(4)
        if frame == picked:
            base = _delta_base(demo, delta_cam, t_bc)
            return base if aim is None else aim @ base
        return (motions or {}).get(frame)

    return motion


def _found_motions(
    demo: runtime._Demo, t_bc: np.ndarray, carry: Callable[[str, float], np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Each of :func:`_other_objects` found so far, by name: its motion since the demo (:func:`_object_motion`),
    moved on by ``carry(name, stamp)`` when given."""
    out = {}
    for name in _other_objects(demo):
        motion = _object_motion(demo, name, t_bc, None if carry is None else functools.partial(carry, name))[
            0
        ]
        if motion is not None:
            out[name] = motion
    return out


def _act_preview(demo: runtime._Demo, delta_cam: np.ndarray, t_bc: np.ndarray) -> np.ndarray | None:
    """The act's fingertip path from its first key frame on, for the camera view, through every key frame whose
    object has been found (as if held as in the demo); None until a key frame is marked."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from .. import jog

    if not _has_key_frames(demo):
        return None
    with runtime._state.lock:
        teach = runtime._state.teach
    if _reference_motion(demo, teach)[0] is None:
        return None
    gi = MOTOR_NAMES.index("gripper")
    steps = _act_steps(demo, "all", 0, _frame_motion(demo, delta_cam, t_bc, _found_motions(demo, t_bc)), None)
    if not steps:
        return None
    i = int(np.argmin(np.abs(demo.t - steps[0]["t"])))
    plan = core.plan_key_frames(
        steps,
        demo.t,
        demo.tips,
        demo.q_cmd[:, gi],
        demo.q_obs,
        steps[0]["motion"] @ demo.tips[i] @ steps[0]["fix"],
        float(demo.q_cmd[i, gi]),
        jog.MAX_LINEAR_M_S,
        jog.MAX_ANGULAR_RAD_S,
        jog.GRIP_UNITS_S,
        1.0,
        jog.HZ,
    )
    return plan["poses"]


def _plan_act(
    demo: runtime._Demo,
    delta_cam: np.ndarray | None,
    t_bc: np.ndarray,
    kin: Any,
    q_now: np.ndarray,
    limits: tuple[float, float],
    box: tuple[tuple[float, float, float], tuple[float, float, float]],
    speed: float,
    skip: int = 0,
    motions: dict[str, np.ndarray] | None = None,
    hold_fix: np.ndarray | None = None,
    part: str | tuple[int, int] = "all",
    aim: np.ndarray | None = None,
    landing: str = "exact",
    ranges: tuple[np.ndarray, np.ndarray] | None = None,
    grasp_adjust: np.ndarray | None = None,
) -> dict[str, Any]:
    """:func:`_plan_act_once`, landed at the turn its rule ``landing`` allows that the arm makes best.

    With "exact", or no object to land on (:func:`_landing_object`) found in ``motions``, the demo's own landing.
    Otherwise the turns the rule allows about the vertical through that object's middle (:func:`core.landing_turns`)
    are ranked on the samples of the key frames in its frame (:func:`_landing_samples`), reachable within the servos'
    ``ranges`` with joints nearest the demo's (:func:`core.rank_landings`), and the cheapest that plans in full is
    taken. Post: as :func:`_plan_act_once`, plus ``landing`` (the rule, how many turns the arm reaches, the turn taken
    and its cost; ``turn_deg`` None when none plans) when a rule other than "exact" was searched, and
    ``landing_centre``, the middle the turn is about, when a turn was taken.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    onto = _landing_object(demo)
    target_base = (motions or {}).get(onto) if onto is not None else None
    once = functools.partial(
        _plan_act_once,
        demo,
        delta_cam,
        t_bc,
        kin,
        q_now,
        limits,
        box,
        speed,
        skip,
        part=part,
        aim=aim,
        ranges=ranges,
        grasp_adjust=grasp_adjust,
    )
    if isinstance(part, tuple):  # a span lands only when a key frame in it is held in the object landed on
        lands = any(k["frame"] == onto for k in core.key_frames(demo.keypoints)[part[0] : part[1]])
    else:
        lands = part in ("all", "place")
    if landing == "exact" or onto is None or target_base is None or not lands:
        return once(motions, hold_fix)
    axis, centre = _landing_axis(demo, onto, t_bc, landing=landing)
    fix = np.eye(4) if hold_fix is None else np.asarray(hold_fix, dtype=float)
    samples = _landing_samples(demo, onto)
    idx = [i for i, _held in samples]
    lo, hi = ranges if ranges is not None else (None, None)

    def turned(deg: float) -> dict[str, np.ndarray]:
        return {**(motions or {}), onto: core.landed(target_base, centre, deg, axis)}

    ranked = core.rank_landings(
        kin,
        core.landing_turns(landing, demo.objects[onto].get("symmetry")),
        lambda deg: np.stack(
            [
                core.landed(target_base, centre, deg, axis) @ demo.tips[i] @ (fix if held else np.eye(4))
                for i, held in samples
            ]
        ),
        demo.q_obs[idx],
        lo,
        hi,
        MOTOR_NAMES.index("gripper"),
    )
    info: dict[str, Any] = {"rule": landing, "reachable": len(ranked), "turn_deg": None}
    tried = []
    for cost, deg in ranked[: core.LANDING_TRIES]:
        plan = once(turned(deg), hold_fix)
        if plan["ok"]:
            return {
                **plan,
                "landing": {**info, "turn_deg": deg, "cost": cost},
                "landing_centre": centre,
                "landing_axis": axis,
            }
        tried.append((deg, plan))
    if tried:
        deg, plan = tried[0]
        why = f"of the {len(ranked)} landings on {onto} within reach, none plans: turned {deg:.0f} deg, {plan['reason']}"
    else:
        plan = once(motions, hold_fix)
        why = f"no landing on {onto} is within reach of the arm" + (
            f": as shown, {plan['reason']}" if plan["reason"] else ""
        )
    return {**plan, "ok": False, "reason": why, "landing": info}


def _landing_axis(
    demo: runtime._Demo, obj: str, t_bc: np.ndarray, *, landing: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Observed axis and a point on it, in the demo's pose frame in base coordinates.

    The old top-surface centre is retained for a face normal. A long-axis turn
    needs its fitted centreline instead of the visible upper-half centroid.
    """
    f = _pose_frame(demo, obj)
    assert f is not None, "a done object with a key frame in its frame has a pose frame"
    reference, problem = holds._demo_control_pose(demo, obj, f)
    if reference is None:
        raise ValueError(problem)
    move = t_bc @ reference @ np.linalg.inv(t_bc)
    pts = _object_points(demo, obj, t_bc) @ move[:3, :3].T + move[:3, 3]
    # "symmetry" explicitly uses appearance-equivalent turns. "turn" is a
    # task-level freedom and must not change when pose-estimation symmetry does.
    selector = (
        symmetry.normalise(demo.objects[obj].get("symmetry"))["axis"]
        if (landing or demo.landing) == "symmetry"
        else demo.landing_axis
    )
    axis, centre = symmetry.observed_axis(pts, selector)
    if selector == "surface_normal":
        # A plane has two equally valid normals. Keep positive base z as the
        # turn's sign convention for supported objects and old demos.
        if axis[2] < 0:
            axis = -axis
        centre = pts[pts[:, 2] >= np.percentile(pts[:, 2], 95) - runtime.LANDING_TOP_M].mean(axis=0)
    return axis, centre


def _landing_centre(demo: runtime._Demo, obj: str, t_bc: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for the observed landing pivot."""
    return _landing_axis(demo, obj, t_bc)[1]


def _holding(demo: runtime._Demo) -> dict[str, Any] | None:
    """The attachment selected by the current interaction context, or None."""
    holdings = _holdings(demo)
    return holdings[demo.hold_index] if demo.hold_index < len(holdings) else None


def _holdings(demo: runtime._Demo) -> list[dict[str, Any]]:
    """All recorded attachments; object identity comes from the key frames."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi = MOTOR_NAMES.index("gripper")
    return core.demo_holdings(demo.t, demo.q_cmd[:, gi], demo.q_obs[:, gi], core.key_frames(demo.keypoints))


def _grasp_split(frames: list[dict[str, Any]], holding: dict[str, Any] | None) -> int:
    """How many of the key frames the act does before it checks the grasp: those up to the follow the demo's hold
    began in, else up to the first follow, else all of them."""
    if holding is not None:
        return holding["begins"] + 1
    return next((n + 1 for n, k in enumerate(frames) if k["move"] == "follow"), len(frames))


def _act_steps(
    demo: runtime._Demo,
    part: str | tuple[int, int],
    skip: int,
    motion: Callable[[str], np.ndarray | None],
    hold_fix: np.ndarray | None,
    grasp_adjust: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """The demo's key frames an act plans, resolved for where things are now (:func:`core.plan_key_frames`).

    ``part`` "grasp" takes those up to the grasp check (:func:`_grasp_split`), "place" the rest, "all" both, and
    ``(start, stop)`` those numbered so in time order; ``skip`` leaves out that many of the part's leading lines,
    already reached. ``motion(frame)`` is that frame's motion since the demo, base frame, or None while unknown (an
    object not found yet): the steps stop before the first such. Those the demo made holding something are corrected
    by ``hold_fix`` (the identity when None)."""
    frames = core.key_frames(demo.keypoints)
    holding = _holding(demo)
    split = _grasp_split(frames, holding)
    holdings = _holdings(demo)
    if isinstance(part, tuple):
        lo, hi = part
    else:
        lo, hi = {"grasp": (0, split), "place": (split, len(frames)), "all": (0, len(frames))}[part]
    assert skip == 0 or all(k["move"] == "line" for k in frames[lo : lo + skip + 1]), (
        "a line is left to reach"
    )
    fix = np.eye(4) if hold_fix is None else np.asarray(hold_fix, dtype=float)
    steps: list[dict[str, Any]] = []
    for n in range(lo + skip, hi):
        m = motion(frames[n]["frame"])
        if m is None:
            break
        attached = core.holding_at(holdings, demo.t, frames, n)
        held = attached is not None
        steps.append(
            {
                **frames[n],
                "since": frames[n - 1]["t"] if n else None,
                "motion": m,
                "fix": fix if attached is not None and attached == holding else np.eye(4),
                "adjust": (
                    np.asarray(grasp_adjust, float)
                    if grasp_adjust is not None
                    and holding is not None
                    and n <= holding["begins"]
                    and frames[n]["frame"] == holding["frame"]
                    else np.eye(4)
                ),
                "held": held,
                "held_object": None if attached is None else attached["frame"],
            }
        )
    return steps


def _landing_samples(demo: runtime._Demo, obj: str) -> list[tuple[int, bool]]:
    """The demo samples a landing turn on ``obj`` is judged on, each with whether the demo made it holding something
    (:func:`core.held_at`): each line key frame in its frame, and each follow in it every LANDING_EVERY_S from the key
    frame before to its end. Pre: a key frame is held in ``obj``'s frame."""
    frames = core.key_frames(demo.keypoints)
    holding = _holding(demo)
    step = max(1, int(round(runtime.LANDING_EVERY_S * demo.fps)))
    out: list[tuple[int, bool]] = []
    for n, k in enumerate(frames):
        if k["frame"] != obj:
            continue
        held = core.held_at(holding, demo.t, frames, n)
        i = int(np.argmin(np.abs(demo.t - k["t"])))
        if k["move"] == "follow":
            out += [
                (j, held) for j in range(int(np.argmin(np.abs(demo.t - frames[n - 1]["t"]))) + step, i, step)
            ]
        out.append((i, held))
    return out


def _plan_act_once(
    demo: runtime._Demo,
    delta_cam: np.ndarray | None,
    t_bc: np.ndarray,
    kin: Any,
    q_now: np.ndarray,
    limits: tuple[float, float],
    box: tuple[tuple[float, float, float], tuple[float, float, float]],
    speed: float,
    skip: int = 0,
    motions: dict[str, np.ndarray] | None = None,
    hold_fix: np.ndarray | None = None,
    part: str | tuple[int, int] = "all",
    aim: np.ndarray | None = None,
    ranges: tuple[np.ndarray, np.ndarray] | None = None,
    grasp_adjust: np.ndarray | None = None,
) -> dict[str, Any]:
    """What the act will do on the objects where they are now, judged before the arm moves.

    Pre: the demo has a key frame. From the arm's present joints ``q_now``, the key frames in order
    (:func:`core.plan_key_frames`), each carried by its frame's motion since the demo: the object the live track
    follows by the tracker's (``aim``, an injected error, before it), another object by its entry in ``motions``
    (none: the plan stops before its first key frame), the world by none. Those the demo made holding something are
    corrected by ``hold_fix`` (the identity until the hold is measured). ``part`` "grasp" plans the key frames up to
    the grasp check only, "place" the rest only, from the arm's present joints, ``skip`` counting the part's leading
    lines already reached. Every sample is solved by IK from the one before. Refuses, naming the
    reason, when a mark or a replayed sample is out of reach, when it needs a joint past
    its servo's range (``ranges``: each joint's (lo, hi) in motor degrees, the gripper
    NaN), when a sample leaves the workspace or goes lower than the table floor (or than
    the demo itself went at that sample), or when the arm would jump between two samples.
    Post: ``ok``, ``reason``, ``times`` (N,), ``q`` (N, J), ``stage`` (N,), ``marks``
    (label, t, residual_mm, ok) and ``summary``, JSON-safe apart from the arrays.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from .. import jog

    gi = MOTOR_NAMES.index("gripper")
    q_now = np.asarray(q_now, dtype=float)
    steps = _act_steps(
        demo, part, skip, _frame_motion(demo, delta_cam, t_bc, motions, aim), hold_fix, grasp_adjust
    )
    assert steps, "a key frame to plan"
    best_effort = [s for s in steps if s.get("constraint") == "best_effort"]
    online_candidate = len(steps) == 1 and len(best_effort) == 1 and steps[0]["move"] in ("line", "lift")
    for _ in range(1 + (0 if online_candidate else len(best_effort))):
        plan = core.plan_key_frames(
            steps,
            demo.t,
            demo.tips,
            demo.q_cmd[:, gi],
            demo.q_obs,
            kin.forward_kinematics(q_now),
            float(q_now[gi]),
            limits[0],
            limits[1],
            jog.GRIP_UNITS_S,
            speed,
            jog.HZ,
        )
        online_best_effort = online_candidate and np.ptp(plan["grips"]) < core.GRIP_STILL_UNITS
        sol = core.solve_plan_joints(
            kin,
            plan["poses"],
            plan["grips"],
            plan["hints"],
            q_now,
            gi,
            *(ranges if ranges is not None else (None, None)),
        )
        # A plan containing later strict stages still needs a modeled feasible
        # prefix. A standalone best-effort stage can instead be tried online:
        # checked_step guards each incremental command and stops at the measured
        # prefix. The next key frame is planned from those observed joints.
        p = plan["poses"][:, :3, 3]
        low_box, high_box = np.asarray(box[0], float), np.asarray(box[1], float)
        floor_limit = np.minimum(np.minimum(low_box[2], plan["floor_ref"]), p[0, 2])
        invalid = (
            (sol["residual_m"] > core.ACT_REACH_TOL_M)
            | (sol["residual_deg"] > core.ACT_REACH_TOL_DEG)
            | (sol["step_deg"] > core.ACT_MAX_JOINT_STEP_DEG)
            | (p[:, 2] < floor_limit - core.ACT_REACH_TOL_M)
            | np.any(p[:, :2] < low_box[:2] - core.ACT_REACH_TOL_M, axis=1)
            | np.any(p > high_box + core.ACT_REACH_TOL_M, axis=1)
        )
        shortened = False
        for s in [] if online_best_effort else best_effort:
            if "best_effort_fraction" in s:
                continue
            samples = np.flatnonzero(np.asarray(plan["stage"]) == s["label"])
            grip_here = plan["grips"][max(0, samples[0] - 1) : samples[-1] + 1]
            if np.ptp(grip_here) >= core.GRIP_STILL_UNITS:
                continue  # a closing/opening event must keep its planned contact position
            failed = next((int(i) for i in samples if invalid[i]), None)
            if failed is None:
                continue
            safe = max(0, failed - 1)
            target = np.asarray(plan["poses"][safe], float)
            if s["move"] == "lift":
                before = np.asarray(plan["poses"][samples[0] - 1], float)
                rise = max(0.0, float((target[2, 3] - before[2, 3]) * 1000))
                if abs(rise - s["up_mm"]) < 1e-6:
                    continue
                s["up_mm"] = rise
            else:
                current_target = np.asarray(plan["poses"][samples[-1]], float)
                if max(core.pose_residual(current_target, target)) < 1e-9:
                    continue
                s["target_local"] = (
                    np.linalg.inv(s["adjust"]) @ np.linalg.inv(s["motion"]) @ target @ np.linalg.inv(s["fix"])
                )
            s["best_effort_fraction"] = max(0.0, (safe - samples[0] + 1) / len(samples))
            shortened = True
            break
        if not shortened:
            break
    # Retain the reference relation for every sample. Recomposition can update
    # either side (receiving object and held attachment) without replaying an
    # earlier phase or confusing a new live pose with the old planned world pose.
    by_label = {s["label"]: s for s in steps}
    local, names, attached = [], [], []
    for label, pose in zip(plan["stage"], plan["poses"], strict=True):
        s = by_label.get(label)
        if s is None or s["move"] == "lift":
            names.append(core.WORLD_FRAME)
            local.append(pose)
            attached.append(False)
        else:
            names.append(s["frame"])
            local.append(np.linalg.inv(s["motion"]) @ pose @ np.linalg.inv(s["fix"]))
            attached.append(s["held"])
    fine = (sol["residual_m"] <= core.ACT_REACH_TOL_M) & (sol["residual_deg"] <= core.ACT_REACH_TOL_DEG)
    pos = plan["poses"][:, :3, 3]
    lo, hi = np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)
    floor = np.minimum(
        np.minimum(lo[2], plan["floor_ref"]), pos[0, 2]
    )  # never lower than the arm already stands
    low = pos[:, 2] < floor - core.ACT_REACH_TOL_M
    outside = np.any(pos[:, :2] < lo[:2] - core.ACT_REACH_TOL_M, axis=1) | np.any(
        pos > hi + core.ACT_REACH_TOL_M, axis=1
    )
    required = (
        np.asarray(plan["constraint"]) != "best_effort" if online_best_effort else np.ones(len(pos), bool)
    )
    marks = []
    lines = [s for s in steps if s["move"] in ("line", "lift")]
    for s, idx in zip(lines, plan["arrive"], strict=True):
        # How far the object's move carried this mark from where the demo's arm was, and how much of that is away from
        # the arm's base: what puts a mark the demo reached out of reach now. A lift has no demo pose: none.
        now = np.asarray(plan["poses"][idx], dtype=float)[:3, 3]
        was = now if s["move"] == "lift" else demo.tips[int(np.argmin(np.abs(demo.t - s["t"])))][:3, 3]
        marks.append(
            {
                "label": s["label"],
                "t": s["t"],
                "residual_mm": float(sol["residual_m"][idx] * 1000.0),
                "residual_deg": float(sol["residual_deg"][idx]),
                "ok": bool(fine[idx]),
                "object": s["frame"] if s["frame"] not in ("", core.WORLD_FRAME) else None,
                "moved_mm": float(np.linalg.norm(now - was) * 1000.0),
                "out_mm": float((np.linalg.norm(now[:2]) - np.linalg.norm(was[:2])) * 1000.0),
                "best_effort_fraction": s.get("best_effort_fraction"),
                "target_pose": np.asarray(plan["poses"][idx], dtype=float).tolist(),
                "solved_joints": np.asarray(sol["q"][idx], dtype=float).tolist(),
                "held_joint": int(sol["held"][idx]),
            }
        )
    follows = [s for s in steps if s["move"] == "follow"]
    for s, span in zip(follows, plan["spans"], strict=True):
        if span is None:
            continue
        a, b = span
        marks.append(
            {
                "label": s["label"],
                "t": s["t"],
                "residual_mm": float(sol["residual_m"][a : b + 1].max() * 1000.0),
                "ok": bool(fine[a : b + 1].all()),
            }
        )
    reason = ""
    bad = [m for m in marks if not m["ok"] and (not online_best_effort or m["label"] != steps[0]["label"])]
    far = np.flatnonzero(~fine & required)
    held = np.flatnonzero(~fine & (sol["held"] >= 0) & required)
    jumps = np.flatnonzero((sol["step_deg"] > core.ACT_MAX_JOINT_STEP_DEG) & required)
    low &= required
    outside &= required
    # Out of reach for a joint held at its servo's range: say which, and how far the stage would need it.
    if len(held) and ranges is not None:
        n = max(  # the farthest that stage needs the joint, not just where it first leaves the range
            (int(k) for k in held if plan["stage"][k] == plan["stage"][held[0]]),
            key=lambda k: max(
                sol["residual_m"][k] / core.ACT_REACH_TOL_M, sol["residual_deg"][k] / core.ACT_REACH_TOL_DEG
            ),
        )
        j = int(sol["held"][n])
        lo_j, hi_j = float(ranges[0][j]), float(ranges[1][j])
        free = core.solve_plan_joints(
            kin,
            plan["poses"][n : n + 1],
            plan["grips"][n : n + 1],
            np.zeros((1, len(q_now))),
            sol["q"][n],
            gi,
        )
        reason = (
            f"the planned path to {plan['stage'][n]} drives {MOTOR_NAMES[j]} toward {free['q'][0, j]:.0f} deg, past its servo's range of "
            f"{lo_j:.0f} to {hi_j:.0f} deg"
        )
    elif bad:
        m = bad[0]
        # A sequential IK failure does not prove the pose itself is outside the
        # arm's reach. Another joint configuration or approach can solve it.
        reason = (
            f"the planned approach to {m['label']} has no solved joint path "
            f"({m['residual_mm']:.0f} mm / {m['residual_deg']:.0f} deg pose residual)"
        )
        if m.get("object") and m.get("moved_mm") is not None:
            reason += (
                f"; {m['object']}'s motion shifts that target {m['moved_mm']:.0f} mm from the demo,"
                f" including {m['out_mm']:.0f} mm radially away from the arm base"
            )
    elif len(far):
        n = int(far[0])
        reason = (
            f"the straight line to {plan['stage'][n]} has no solved joint path "
            f"({sol['residual_m'][n] * 1000.0:.0f} mm / {sol['residual_deg'][n]:.0f} deg pose residual)"
        )
    elif np.any(low):
        n = int(np.argmax(np.where(low, floor - pos[:, 2], -np.inf)))
        reason = f"{plan['stage'][n]} would go {(floor[n] - pos[n, 2]) * 1000.0:.0f} mm below the table"
    elif np.any(outside):
        n = int(np.flatnonzero(outside)[0])
        reason = f"{plan['stage'][n]} leaves the arm's workspace"
    elif len(jumps):
        n = int(jumps[0])
        reason = f"the arm would jump {sol['step_deg'][n]:.0f} deg during {plan['stage'][n]}"
    return {
        "ok": not reason,
        "reason": reason,
        "times": plan["times"],
        "q": sol["q"],
        "stage": plan["stage"],
        "live_path": {
            "local": np.stack(local),
            "frames": names,
            "held": np.asarray(attached),
            "grips": plan["grips"],
            "hints": plan["hints"],
            "floor": plan["floor_ref"],
            "constraint": plan["constraint"],
        },
        "marks": marks,
        "summary": {
            "samples": len(plan["times"]),
            "seconds": float(plan["times"][-1]),
            "worst_residual_mm": float(sol["residual_m"].max() * 1000.0),
            "worst_step_deg": float(sol["step_deg"].max()),
        },
    }


def _grasp_pose_fix(demo_tip: np.ndarray, object_motion: np.ndarray, live_tip: np.ndarray) -> np.ndarray:
    """The hold's change when the held object cannot be seen: ``demo_tip^-1 . object_motion^-1 . live_tip``.

    ``demo_tip`` is the demo's fingertip at its firm grip, ``object_motion`` the object's motion since the demo that
    the grasp was planned with (base frame), ``live_tip`` the fingertip where the arm actually stands at the grip.
    The grasp was aimed at ``object_motion . demo_tip``, so the result is the identity when the arm landed there,
    and otherwise how far it missed, on the gripper's side. Whatever the closing fingers or the lift did to the
    object is not in it.
    """
    return np.linalg.inv(demo_tip) @ np.linalg.inv(object_motion) @ live_tip
