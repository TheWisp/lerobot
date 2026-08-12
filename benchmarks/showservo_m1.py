# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""M1 — the naked servo: the SO-107 re-achieves a taught relation from live fits.

The sim loop's exact structure on the real rig. A teach scene shows BOTH ends — the
target object and an object held in the gripper — through one camera. The taught
relation is implicit in that scene: both fits are identity there, so transporting the
held card's cloud through the target's live fit says where the held end SHOULD be,
and the held card's own live fit says where it IS. `servo_error_3d` is that
difference; nothing here knows where the camera or the arm base is.

Control is uncalibrated by design: three small probe moves on three joints seed an
empirical joint→camera Jacobian, Broyden updates keep it honest, damped least squares
inverts it, a PI shapes the error. Position only (`e_t`) — the one channel trusted on
every object class; orientation waits for the wrist joints and D2.

The arm is commanded through the GUI's /api/showservo/arm endpoints, never a serial
port from here: the server owns the safety authority (per-step and total-excursion
clamps, the stop flag), so a bug in this loop cannot out-shout it. Every refused
measurement is an abstention — no fit, no command (invariant 5).

Pre: a live GUI capture session on the taught rig, the arm connected via the Servo
tab, the held object gripped as in the teach scene. The worker exits when the server
stops serving frames or refuses a move (stop pressed, limits hit).

Usage (normally spawned by the GUI):
    PYTHONPATH=src python benchmarks/showservo_m1.py \\
        --captures captures/<session> --concept "green ring" \\
        --held-concept "blue box" --teach 0 --arm right \\
        --server http://127.0.0.1:9100/
"""

from __future__ import annotations

import argparse
import io
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo_m0 import DinoTier  # noqa: E402
from showservo_real import (  # noqa: E402
    Card,
    Designator,
    Recruits,
    _LiveFrame,
    bind_rigid3d,
    draw_ghost,
    draw_gizmo,
    draw_matches,
    load_captures,
)

from lerobot.showservo.servo import (  # noqa: E402
    ConvergenceCertificate,
    JacobianEstimator,
    PIController,
    servo_error_3d,
)

M1_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
PROBE_U = 1.2  # probe move per joint, normalized units (~1.3 degrees on a shoulder)
STEP_LIMIT_U = 1.5  # worker-side per-step clamp; the server enforces its own above this
SETTLE_FRAMES = 1  # frames to discard after a command before measuring its effect
CONVERGED_MM = 4.0
CONVERGED_STREAK = 3
ABSTAIN_LIMIT = 20
SERVO_BUDGET = 240


class ArmClient:
    """The GUI's arm endpoints, from the worker's side. Post: ``move`` returns None
    exactly when the server refused (stop pressed, clamp exceeded, disconnected) OR
    could not be reached — either way the worker must hard-halt, never retry: a
    command whose fate is unknown must not be followed by another."""

    def __init__(self, http, server: str):
        self.http = http
        self.server = server

    def move(self, deltas: dict[str, float]) -> dict | None:
        import requests

        try:
            r = self.http.post(self.server + "api/showservo/arm/move", json={"deltas": deltas}, timeout=10)
        except requests.RequestException:
            return None
        if r.status_code != 200:
            return None
        return r.json()


class Pair:
    """One demo: (target card, held card) whose poses share ONE camera frame, plus
    the goal relation that sharing implies. The frames coincide either because both
    cards come from the same photo, or because the two photos bracket an interval in
    which neither the camera nor the TARGET moved — only the arm did."""

    def __init__(self, target: Card, held: Card):
        self.target = target
        self.held = held


def plan_pairs(flags: list[tuple[bool, bool]]) -> list[tuple[int, int]]:
    """Which teach photos form demos. Pure — the teaching rule, testable by itself.

    ``flags[k] = (target designated, held designated)`` for the k-th teach photo.
    Post: (a, b) index pairs — a supplies the target card, b the held card:

    * a photo showing BOTH is its own pair (a == b);
    * a held-only photo (goal pose, target hidden behind the gripper) pairs with the
      MOST RECENT earlier photo that showed the target — valid because the target's
      pose in b equals its pose in a whenever it has not been moved in between,
      which is the operator's one precondition;
    * a target-only photo teaches nothing by itself; it waits as the `a` of a later
      goal photo. A held-only photo with no earlier target photo is skipped.
    """
    pairs = []
    last_target = None
    for k, (has_target, has_held) in enumerate(flags):
        if has_target and has_held:
            pairs.append((k, k))
            last_target = k
        elif has_target:
            last_target = k
        elif has_held and last_target is not None:
            pairs.append((last_target, k))
    return pairs


def teach_pairs(scenes, teach, target_designator, held_designator, tier, intr) -> list[Pair]:
    """Pre: the target must not move between the photos of any (a, b) pair — the
    goal relation is composed across them on that assumption. Post: >= 1 Pair."""
    masks = []
    for i in teach:
        tmask = target_designator.mask(scenes[i])
        hmask = held_designator.mask(scenes[i])
        overlapping = (
            tmask is not None
            and hmask is not None
            and (tmask & hmask).sum() > 0.3 * min(tmask.sum(), hmask.sum())
        )
        if overlapping:
            print(f"teach {scenes[i].name}: concepts overlap heavily — held mask dropped", flush=True)
            hmask = None
        masks.append((tmask, hmask))
        seen = [n for n, m in (("target", tmask), ("held", hmask)) if m is not None]
        print(f"teach {scenes[i].name}: designated {' + '.join(seen) if seen else 'NOTHING'}", flush=True)

    pairs = []
    for a, b in plan_pairs([(t is not None, h is not None) for t, h in masks]):
        target = Card(scenes[teach[a]], masks[a][0], tier, intr)
        held = Card(scenes[teach[b]], masks[b][1], tier, intr)
        pairs.append(Pair(target, held))
        print(
            f"demo {len(pairs) - 1}: target from {scenes[teach[a]].name} ({len(target.uv)} pts), "
            f"held from {scenes[teach[b]].name} ({len(held.uv)} pts)",
            flush=True,
        )
    assert pairs, (
        "no demo could be formed: need one photo showing both concepts, or a "
        "target-only photo followed by a goal photo where the held end designates"
    )
    return pairs


def held_centroid(fit, card: Card) -> np.ndarray:
    return fit.transform.apply(card.xyz.mean(axis=0).reshape(1, 3))[0]


def annotate(vis, mask_t, mask_h, t_fit, t_uv, h_fit, pair, intr, state, err, extra=""):
    """One frame's evidence: target ghost green, held ghost cyan, error vector magenta."""
    import cv2

    for mask, col in ((mask_t, (255, 0, 255)), (mask_h, (80, 200, 255))):
        if mask is not None:
            edge = mask ^ (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            vis[edge] = col
    if t_fit is not None:
        if t_uv is not None:
            draw_matches(vis, t_uv)
        draw_ghost(vis, t_fit, pair.target, intr)
        draw_gizmo(vis, t_fit, pair.target, intr)
    if h_fit is not None:
        draw_ghost(vis, h_fit, pair.held, intr, color=(80, 200, 255))
    if t_fit is not None and h_fit is not None and err is not None and err.ok:
        # Where the held end IS -> where it SHOULD be, in pixels: the error the
        # loop drives, drawn as the user will judge it.
        cur = held_centroid(h_fit, pair.held)
        want = t_fit.transform.apply(pair.held.xyz.mean(axis=0).reshape(1, 3))[0]
        a, b = intr.project(np.stack([cur, want]))
        cv2.arrowedLine(
            vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 60, 255), 2, cv2.LINE_AA, tipLength=0.2
        )
    hdr = f"[M1:{state}]"
    if err is not None and err.ok:
        hdr += f"  |e| {err.norm * 1000:6.1f} mm"
    if extra:
        hdr += f"  {extra}"
    color = {
        "SERVO": (60, 230, 60),
        "DONE": (60, 230, 60),
        "PROBE": (250, 240, 80),
        "WAIT": (255, 190, 90),
    }.get(state, (255, 70, 70))
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, hdr, (10, 24), 0, 0.62, color, 2, cv2.LINE_AA)


def m1_loop(server: str, pairs: list[Pair], target_designator, held_designator, tier, intr) -> None:
    """WAIT -> PROBE -> SERVO -> DONE/HALTED, one frame at a time.

    The target end is armored like the live fit: recruits carry its frame when the
    arm occludes it (which the arm will, constantly). The held end NEVER coasts and
    never falls back — a stale held fit post-dates a command by construction, so a
    refused held bind is an abstention and the frame commands nothing.
    """
    import cv2
    import requests

    http = requests.Session()
    arm = ArmClient(http, server)
    recruits = Recruits(tier, intr)

    # Damping must be judged against J's own scale: these columns are METRES PER
    # UNIT (~0.008), so JJ^T ~ 6e-5 and a lambda of 1e-2 would rival it, silently
    # attenuating every command ~2.5x — measured in the loop rehearsal as a freeze
    # at 9 mm, right where attenuated commands sink under the backlash floor.
    est = JacobianEstimator(n_joints=len(M1_JOINTS), m=3, damping=1e-3)
    # The integral is the backlash insurance: commands under the joint-side floor
    # are suppressed, so the integral must be able to grow one that breaks through.
    pi = PIController(kp=0.5, ki=0.3, v_max=0.010, integral_limit=0.03)
    cert = ConvergenceCertificate(window=25, min_improvement=0.05)

    state = "WAIT"
    ready_streak = 0
    done_streak = 0
    abstains = 0
    servo_frames = 0
    settle = 0
    probe_dqs: list[np.ndarray] = []
    probe_des: list[np.ndarray] = []
    probe_ref: np.ndarray | None = None  # held centroid before the outstanding probe move
    dq_pending: np.ndarray | None = None  # executed command awaiting its measured effect
    prev_centroid: np.ndarray | None = None
    halt_reason = ""

    while True:
        try:
            r = http.get(server + "api/showservo/live/frame.npz", timeout=10)
        except requests.RequestException:
            return  # server gone: nothing left to measure or command
        if r.status_code != 200:
            return
        data = np.load(io.BytesIO(r.content))
        frame = _LiveFrame(data["rgb"], data["depth"])

        # Measure both ends, best target demo wins, held follows the SAME demo.
        mask_t = target_designator.mask(frame)
        t_fit, t_uv, demo = None, None, 0
        if mask_t is not None:
            for d, pair in enumerate(pairs):
                fit, uv = bind_rigid3d(pair.target, frame, mask_t, tier, intr)
                if fit is not None and (t_fit is None or fit.n_inliers > t_fit.n_inliers):
                    t_fit, t_uv, demo = fit, uv, d
        if t_fit is not None:
            recruits.refresh(frame, mask_t, t_fit, demo)
        else:
            rfit, _uv, _in = recruits.fallback(frame)
            if rfit is not None:
                t_fit, demo = rfit, recruits.anchor_demo
        pair = pairs[demo]
        mask_h = held_designator.mask(frame)
        h_fit = bind_rigid3d(pair.held, frame, mask_h, tier, intr)[0] if mask_h is not None else None
        err = (
            servo_error_3d(pair.held.xyz, t_fit, h_fit) if (t_fit is not None and h_fit is not None) else None
        )
        measured = err is not None and err.ok
        centroid = held_centroid(h_fit, pair.held) if h_fit is not None else None

        if settle > 0:
            settle -= 1
        elif state == "WAIT":
            ready_streak = ready_streak + 1 if measured else 0
            if ready_streak >= 3:
                state = "PROBE"
                probe_dqs, probe_des, probe_ref = [], [], None
        elif state == "PROBE":
            # One joint at a time: command, settle, measure the held-centroid
            # displacement it caused. At most one move is ever outstanding
            # (len(probe_des) < len(probe_dqs) marks it). Losing sight mid-probe
            # restarts the probe — a half-measured Jacobian is worse than none.
            if centroid is None:
                state, ready_streak = "WAIT", 0
            elif len(probe_des) < len(probe_dqs):
                assert probe_ref is not None
                probe_des.append(centroid - probe_ref)
            elif len(probe_dqs) < len(M1_JOINTS):
                if arm.move({M1_JOINTS[len(probe_dqs)]: PROBE_U}) is None:
                    state, halt_reason = "HALTED", "arm refused during probe"
                else:
                    dq = np.zeros(len(M1_JOINTS))
                    dq[len(probe_dqs)] = PROBE_U
                    probe_dqs.append(dq)
                    probe_ref = centroid
                    settle = SETTLE_FRAMES
            if state == "PROBE" and len(probe_des) == len(M1_JOINTS):
                est.seed_from_probe(np.array(probe_dqs), np.array(probe_des))
                pi.reset()
                cert.reset()
                prev_centroid = centroid
                state = "SERVO"
        elif state == "SERVO":
            servo_frames += 1
            if not measured:
                abstains += 1
                if abstains >= ABSTAIN_LIMIT:
                    state, halt_reason = "HALTED", f"{ABSTAIN_LIMIT} consecutive abstentions"
            else:
                abstains = 0
                cert.update(err.norm)
                if dq_pending is not None and prev_centroid is not None and centroid is not None:
                    est.update(dq_pending, centroid - prev_centroid)
                    dq_pending = None
                prev_centroid = centroid
                if err.norm * 1000.0 < CONVERGED_MM:
                    done_streak += 1
                    if done_streak >= CONVERGED_STREAK:
                        state = "DONE"
                else:
                    done_streak = 0
                if state == "SERVO":
                    if not cert.progressing:
                        state, halt_reason = "HALTED", "no progress over the window"
                    elif servo_frames > SERVO_BUDGET:
                        state, halt_reason = "HALTED", "frame budget exhausted"
                    else:
                        dq = est.solve(pi.step(err.e_t, dt=0.4))
                        dq = np.clip(dq, -STEP_LIMIT_U, STEP_LIMIT_U)
                        # Commands below the gear train's backlash wind it up without
                        # moving the joint and teach Broyden that commands do nothing.
                        if float(np.abs(dq).max()) > 0.2:
                            if arm.move(dict(zip(M1_JOINTS, dq.tolist(), strict=True))) is None:
                                state, halt_reason = "HALTED", "arm refused (stop or limit)"
                            else:
                                dq_pending = dq
                                settle = SETTLE_FRAMES

        vis = frame.rgb.copy()
        extra = ""
        if t_fit is not None and h_fit is not None:
            extra = f"inl t{t_fit.n_inliers}/h{h_fit.n_inliers}  demo {demo}"
        if halt_reason:
            extra = (extra + "  " + halt_reason).strip()
        annotate(vis, mask_t, mask_h, t_fit, t_uv, h_fit, pair, intr, state, err, extra)
        _ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        try:
            p = http.post(
                server + "api/showservo/live/result",
                data=jpg.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=10,
            )
        except requests.RequestException:
            return
        if p.status_code != 200:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True, type=pathlib.Path)
    ap.add_argument("--concept", required=True, help="SAM3 prompt for the TARGET object")
    ap.add_argument("--held-concept", required=True, help="SAM3 prompt for the object IN the gripper")
    ap.add_argument("--teach", type=int, nargs="+", default=[0])
    ap.add_argument("--arm", choices=("left", "right"), required=True)
    ap.add_argument("--server", required=True, help="GUI base URL serving frames and the arm")
    ap.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    intr, scenes = load_captures(args.captures)
    target_designator = Designator("sam3", args.concept, args.device)
    held_designator = Designator("sam3", args.held_concept, args.device)
    tier = DinoTier(args.dino_model, device=args.device)
    pairs = teach_pairs(scenes, args.teach, target_designator, held_designator, tier, intr)

    server = args.server if args.server.endswith("/") else args.server + "/"
    print(f"M1 servo on the {args.arm} arm against {server} — stop from the GUI", flush=True)
    m1_loop(server, pairs, target_designator, held_designator, tier, intr)


if __name__ == "__main__":
    main()
