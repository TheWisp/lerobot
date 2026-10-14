"""Compare a carried object directly with the demonstration's carried views.

The reference is an observed surface, not a pose registered to the designation
view. Each reference has its own recorded fingertip frame. Never average their
camera-frame transforms together or install them in the designation's tracker.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import numpy as np

from lerobot.showservo import motion as core, surface_constraint

from . import acquisition, holds, planning, recording, runtime


def references(demo: runtime._Demo, obj: str) -> list[dict]:
    """View-diverse observations from this grasp, ending before opening.

    A segmentation proposes a reference click, not a trusted object pose. SAM
    resegments it and registration still has to pass the normal evidence checks.
    Include moving views: limiting this bank to a stationary checkpoint made
    all references show essentially the same side of the object.
    """
    if demo.recording is None or "masks" not in demo.objects.get(obj, {}):
        return []
    holding = planning._holding(demo)
    if holding is None or holding["frame"] != obj:
        return []
    start = holding["grip"]
    stop = len(demo.t) - 1 if holding["release"] is None else holding["release"]
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    grip = demo.q_cmd[:, MOTOR_NAMES.index("gripper")]
    # Release detection can describe the end of an opening. Reference material
    # must remain attached, so exclude its beginning too.
    closing = grip[start : stop + 1] * holding["closing"]
    opening = np.flatnonzero(np.maximum.accumulate(closing) - closing > core.GRIP_STILL_UNITS)
    if len(opening):
        stop = max(start, start + int(opening[0]) - 1)
    times = recording._stream_times(demo.recording) - demo.t0
    masks = demo.objects[obj]["masks"]
    frames = []
    for f in np.flatnonzero((times >= demo.t[start]) & (times <= demo.t[stop])):
        if f >= len(masks) or not np.any(masks[f]):
            continue
        i = int(np.clip(np.searchsorted(demo.t, times[f]), 1, len(demo.t) - 1))
        weight = (times[f] - demo.t[i - 1]) / (demo.t[i] - demo.t[i - 1])
        tip = core.interp_rigid(demo.tips[i - 1], demo.tips[i], weight)
        frames.append((int(f), i, tip))
    if not frames:
        return []
    return [
        {"frame": f, "sample": i, "tip": tip, "mask": demo.objects[obj]["masks"][f], "refresh_mask": True}
        for f, i, tip in frames
    ]


def prepare_references(demo, obj, t_bc):
    """Bound the bank by view diversity; load geometry once outside the act loop."""
    refs = references(demo, obj)
    if not refs:
        return []
    directions = []
    for ref in refs:
        tip = ref.get("tip", demo.tips[ref["sample"]])
        direction = tip[:3, :3].T @ (np.asarray(t_bc)[:3, 3] - tip[:3, 3])
        directions.append(direction / max(np.linalg.norm(direction), 1e-12))
    directions = np.asarray(directions)
    chosen = [max(range(len(refs)), key=lambda i: np.count_nonzero(refs[i]["mask"]))]
    # Reuse the existing per-window work budget, rather than adding an
    # object-specific number of views. Near-duplicate views add no coverage.
    while len(chosen) < min(len(refs), core.HOLD_VIEWS + 2):
        angle = np.degrees(np.arccos(np.clip(directions @ directions[chosen].T, -1, 1))).min(axis=1)
        candidate = int(np.argmax(angle))
        if angle[candidate] <= core.ACT_REACH_TOL_DEG:
            break
        chosen.append(candidate)
    intr = holds._recording_intr(demo)
    bank = []
    for i in chosen:
        ref = refs[i]
        _, depth = recording._stream_frame(demo.recording, ref["frame"])
        import cv2

        mask = cv2.resize(
            np.asarray(ref["mask"], np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST
        )
        points = surface_constraint.depth_points(depth, mask, intr)
        if len(points):
            bank.append({**ref, "mask": mask, "depth": depth, "points": points})
    return bank


def rank_references(refs, demo, t_bc, tip, prediction, depth, intr):
    """Rank proposals by viewing direction and depth overlap; this is not acceptance."""
    prediction = np.eye(4) if prediction is None else np.asarray(prediction)
    camera_in_live_hand = np.linalg.inv(tip) @ t_bc
    camera_in_demo_hand = prediction @ camera_in_live_hand
    ranked = []
    with runtime._state.lock:
        tolerance = runtime._state.depth_tol_m
    for ref in refs:
        points = ref["points"]
        if not len(points):
            continue
        reference_hand = np.linalg.inv(ref.get("tip", demo.tips[ref["sample"]])) @ t_bc
        centre = (reference_hand @ np.r_[points.mean(axis=0), 1])[:3]
        before, now = reference_hand[:3, 3] - centre, camera_in_demo_hand[:3, 3] - centre
        cosine = float(
            np.clip(before @ now / max(np.linalg.norm(before) * np.linalg.norm(now), 1e-12), -1, 1)
        )
        motion = np.linalg.inv(t_bc) @ tip @ np.linalg.inv(prediction) @ reference_hand
        moved = points @ motion[:3, :3].T + motion[:3, 3]
        z = moved[:, 2]
        uv = np.column_stack(
            (
                intr["fx"] * moved[:, 0] / np.maximum(z, 1e-9) + intr["cx"],
                intr["fy"] * moved[:, 1] / np.maximum(z, 1e-9) + intr["cy"],
            )
        )
        uv = np.rint(uv).astype(int)
        valid = (
            (z > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < depth.shape[1])
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < depth.shape[0])
        )
        measured = np.asarray(depth)[uv[valid, 1], uv[valid, 0]]
        support = float(
            np.count_nonzero((measured > 0) & (abs(measured - z[valid]) <= tolerance)) / len(points)
        )
        sampled_depth = np.zeros(len(points))
        sampled_depth[valid] = measured
        ranked.append(
            (
                ref,
                {
                    "view_angle_deg": float(np.degrees(np.arccos(cosine))),
                    "predicted_depth_support": support,
                    "_sampled_depth": sampled_depth,
                    "score": support * max(0.0, cosine),
                },
            )
        )
    return sorted(ranked, key=lambda item: (-item[1]["score"], item[1]["view_angle_deg"]))


def consensus(observations: list[dict], predicted_fix: np.ndarray | None) -> dict | None:
    """Compare attachments in fingertip coordinates, retaining ambiguous rotation.

    Registration supplies a material patch centre and a proposed rotation. The
    rotation must agree across reference views within the planner's angular
    tolerance AND across observations within the hold tolerance. Otherwise keep
    the predicted rotation and solve translation at the matched patch centre,
    not at the arbitrary origin of a camera transform.
    """
    from scipy.spatial.transform import Rotation

    if len(observations) < core.HOLD_MIN_VIEWS:
        return None
    rotations = Rotation.from_matrix(np.stack([o["rotation"] for o in observations]))
    mean = rotations.mean()
    refs = sorted({o["frame"] for o in observations})
    by_ref = [rotations[[i for i, o in enumerate(observations) if o["frame"] == f]].mean() for f in refs]
    between = max((float(np.degrees((a * b.inv()).magnitude())) for a in by_ref for b in by_ref), default=0.0)
    rotation_verified = (
        len(refs) >= 2
        and between <= core.ACT_REACH_TOL_DEG
        and np.max(np.degrees((rotations * mean.inv()).magnitude())) <= core.HOLD_AGREE_DEG
    )
    if not rotation_verified and predicted_fix is None:
        return None
    rotation = mean.as_matrix() if rotation_verified else np.asarray(predicted_fix)[:3, :3]
    shifts = np.stack([o["desired"] - rotation @ o["observed"] for o in observations])
    keep = np.linalg.norm(shifts - np.median(shifts, axis=0), axis=1) <= core.HOLD_AGREE_M
    if keep.sum() < core.HOLD_MIN_VIEWS or 2 * keep.sum() <= len(observations):
        return None
    fix = np.eye(4)
    fix[:3, :3], fix[:3, 3] = rotation, shifts[keep].mean(axis=0)
    result = {
        "fix": fix,
        "n": int(keep.sum()),
        "views": len(observations),
        "spread_mm": float(np.median(np.linalg.norm(shifts[keep] - fix[:3, 3], axis=1)) * 1000),
        "spread_deg": float(np.median(np.degrees((rotations * mean.inv()).magnitude()))),
        "reference_frames": refs,
        "reference_rotation_disagreement_deg": between,
        "orientation": "measured" if rotation_verified else "predicted",
        "reason": "" if rotation_verified else "reference views do not independently agree on rotation",
    }
    if all("stamp" in o for o in observations):
        result["observed_stamp"] = max(
            o["stamp"] for o, valid in zip(observations, keep, strict=True) if valid
        )
    return result


async def measure(
    demo: runtime._Demo,
    obj: str,
    t_bc: np.ndarray,
    predicted_fix: np.ndarray | None,
    stopped: Callable[[], bool],
    *,
    capture=None,
    publish=None,
    reference_bank=None,
    failed_views=None,
    observation_window=None,
    observation_clock=None,
) -> tuple[dict | None, str, list[dict]]:
    """Fresh held-view registration, stationary by default or with a timed capture provider.

    Prediction proposes the search region; fresh segmentation, matching, depth
    constraints and multi-frame agreement determine the measured correction.
    Failures remain failures and share one bounded wait budget.
    """
    refs = reference_bank
    if refs is None:
        refs = await asyncio.get_running_loop().run_in_executor(
            runtime._RENDER_EXECUTOR, prepare_references, demo, obj, t_bc
        )
    if not refs:
        return None, "no segmented held views in this demo attachment", []
    intr_demo = holds._recording_intr(demo)
    until = time.monotonic() + runtime.HOLD_TRACK_WAIT_S
    observations = [] if observation_window is None else observation_window
    previous_stamp = max((o["stamp"] for o in observations), default=-np.inf)
    attempts = []
    added = 0
    failed_views = {} if failed_views is None else failed_views
    tried = set()

    def cancelled() -> bool:
        return stopped() or time.monotonic() >= until

    for _attempt in range(core.HOLD_VIEWS + 2):
        if cancelled():
            break
        view = await (holds._held_view(obj) if capture is None else capture(obj))
        if view is None:
            return None, "the arm went away", attempts
        rgb, depth, intr, tip, mask = view
        observed_stamp = time.perf_counter() if observation_clock is None else observation_clock()
        observations[:] = [
            o for o in observations if observed_stamp - o["stamp"] <= runtime.HOLD_TRACK_WAIT_S
        ]
        ranked = rank_references(refs, demo, t_bc, tip, predicted_fix, depth, intr)
        eligible = []
        for candidate, score in ranked:
            previous = failed_views.get(candidate["frame"])
            if previous is not None:
                moved, turned = core.pose_residual(previous["tip"], tip)
                seed = np.eye(4) if predicted_fix is None else predicted_fix
                shifted, rotated = core.pose_residual(previous["prediction"], seed)
                old_depth, new_depth = previous["depth"], score["_sampled_depth"]
                same_surface = (
                    len(old_depth) == len(new_depth)
                    and np.quantile(abs(old_depth - new_depth), 0.95) <= core.HOLD_AGREE_M
                )
                if (
                    moved <= core.HOLD_AGREE_M
                    and turned <= core.HOLD_AGREE_DEG
                    and shifted <= core.HOLD_AGREE_M
                    and rotated <= core.HOLD_AGREE_DEG
                    and same_surface
                ):
                    continue
            eligible.append((candidate, score))
        if not eligible:
            break  # no new view to test; another identical image cannot resolve it
        untried = [item for item in eligible if item[0]["frame"] not in tried]
        if not untried:
            tried.clear()
            untried = eligible
        ref, score = untried[0]
        failed_view = {
            "tip": np.asarray(tip).copy(),
            "prediction": np.eye(4) if predicted_fix is None else np.asarray(predicted_fix).copy(),
            "depth": score.pop("_sampled_depth"),
        }
        tried.add(ref["frame"])
        points = ref["points"]
        click = acquisition._deepest_pixel(mask, rgb.shape)
        if click is None and predicted_fix is not None and len(points):
            base_points = points @ t_bc[:3, :3].T + t_bc[:3, 3]
            motion = (
                tip @ np.linalg.inv(predicted_fix) @ np.linalg.inv(ref.get("tip", demo.tips[ref["sample"]]))
            )
            with runtime._state.lock:
                tolerance = runtime._state.depth_tol_m
            click = holds._held_click(base_points, motion, t_bc, intr, depth, tolerance)
        if click is None:
            attempts.append(
                {
                    "reference_frame": ref["frame"],
                    "selection": score,
                    "reason": "no supported held-object click",
                }
            )
            failed_views[ref["frame"]] = failed_view
            continue
        found = await acquisition._locate(
            obj,
            rgb,
            depth,
            intr,
            click,
            cancelled,
            reference=ref,
            log={
                "why": "act: direct held-view comparison",
                "came_from": {"held_reference_frame": ref["frame"], "selection": score},
            },
        )
        check = found.get("surface_constraint") or {}
        attempts.append(
            {
                "reference_frame": ref["frame"],
                "selection": score,
                "find_id": found.get("find_id"),
                "reason": found.get("reason") or acquisition._weak(found),
                "constraint": check,
                "tip": np.asarray(tip).tolist(),
                "accepted": False,
            }
        )
        if found.get("reference_mask") is not None:
            ref["mask"], ref["refresh_mask"] = found["reference_mask"], False
            ref["points"] = surface_constraint.depth_points(ref["depth"], ref["mask"], intr_demo)
        if cancelled():
            break
        if not found["ok"] or acquisition._weak(found) or check.get("pose") is None:
            failed_views[ref["frame"]] = failed_view
            continue
        failed_views.pop(ref["frame"], None)
        # Both points describe the SAME reference material patch, expressed in
        # the demo and current fingertip frames. Neither uses its original pose.
        points = surface_constraint.depth_points(ref["depth"], ref["mask"], intr_demo)
        if not len(points):
            continue
        desired = np.linalg.inv(ref.get("tip", demo.tips[ref["sample"]])) @ t_bc
        observed = np.linalg.inv(tip) @ t_bc @ np.asarray(found["delta"])
        # Use the material patch that actually passed verification. Extrapolating
        # a partial-view rotation to the whole reference cloud's centre can move
        # the inferred attachment even when the visible patch did not move.
        centre = np.r_[check.get("reference_point", points.mean(axis=0)), 1.0]
        observations.append(
            {
                "stamp": observed_stamp,
                "frame": ref["frame"],
                "desired": (desired @ centre)[:3],
                "observed": (observed @ centre)[:3],
                "rotation": desired[:3, :3] @ observed[:3, :3].T,
            }
        )
        added += 1
        del observations[: -core.HOLD_VIEWS]
        attempts[-1].update(
            accepted=True,
            reference_point=centre[:3].tolist(),
            observation={
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in observations[-1].items()
            },
        )
        result = consensus(observations, predicted_fix)
        if result is not None and result["observed_stamp"] <= previous_stamp:
            result = None  # a new outlier did not refresh the accepted evidence
        # Sample different references and at least the normal hold view count;
        # an early three-view cluster must not win before the alternatives run.
        if result is not None and len(observations) >= core.HOLD_VIEWS:
            if publish is not None and not cancelled():
                publish(result)
            return result, "", attempts
    result = consensus(observations, predicted_fix) if added else None
    if result is not None and result["observed_stamp"] <= previous_stamp:
        result = None
    if result is not None and publish is not None and not stopped():
        publish(result)
    problem = (
        ""
        if result is not None
        else f"held-view comparison: {len(observations)} accepted views did not establish a current attachment"
    )
    return result, "stopped" if stopped() else problem, attempts


async def watch(demo, obj, t_bc, prediction, stopped, publish, log):
    """Keep measuring during motion; one job at a time, using capture-time FK.

    Camera publication and encoder times share the host perf-counter clock.
    Sensor exposure-to-publication latency remains a calibration uncertainty;
    late jobs never substitute the arm's completion-time pose for the snapshot.
    """
    import copy

    from .. import jog
    from . import camera

    original = jog.kinematics()
    if original is None:
        return
    kin = copy.deepcopy(original)
    last = None
    try:
        bank = await asyncio.get_running_loop().run_in_executor(
            runtime._RENDER_EXECUTOR, prepare_references, demo, obj, t_bc
        )
    except Exception as exc:
        log(f"held reference preparation: {type(exc).__name__}: {exc}", [])
        return
    if not bank:
        log("no segmented held views in this demo attachment", [])
        return
    failed_views = {}
    observations = []

    async def capture(_obj):
        nonlocal last
        snapshot = await camera.timed_frame(last)
        if snapshot is None:
            return None
        rgb, depth, intr, stamp = snapshot
        last = stamp
        # Wait for a right bracket, at most the existing history matching budget.
        until = time.monotonic() + runtime.HISTORY_MATCH_S
        q = jog.joints_at(stamp, 2.0 / jog.HZ)
        while q is None and time.monotonic() < until and not stopped():
            await asyncio.sleep(runtime.ACT_TICK_S)
            q = jog.joints_at(stamp, 2.0 / jog.HZ)
        if q is None:
            return None
        from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

        tip = kin.forward_kinematics(np.array([q[m] for m in MOTOR_NAMES]))
        # A stale track mask is not a prompt on this newer image. Use the
        # depth-supported attachment prediction to propose the click instead.
        return rgb, depth, intr, tip, None

    def deliver(result):
        if not stopped():
            measured_at = time.time() - (time.perf_counter() - result["observed_stamp"])
            publish({**result, "observed_at": measured_at, "accepted_at": time.time()})

    while not stopped():
        try:
            _result, problem, attempts = await measure(
                demo,
                obj,
                t_bc,
                prediction(),
                stopped,
                capture=capture,
                publish=deliver,
                reference_bank=bank,
                failed_views=failed_views,
                observation_window=observations,
                observation_clock=lambda: last,
            )
            log(problem, attempts)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log(f"live held measurement: {type(exc).__name__}: {exc}", [])
        await asyncio.sleep(runtime.HOLD_VIEW_GAP_S)
