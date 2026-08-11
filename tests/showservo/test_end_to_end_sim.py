# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The pipeline, end to end, on rendered pixels — the seam the unit tests leave open.

Everything upstream of here is exercised in isolation: the 3D tests hand exact
correspondences to Kabsch, the tracker tests never touch 3D. This joins them. The
system's only inputs are an RGB frame and a depth map from a camera at a deliberately
awkward angle; ground truth is used ONLY in assertions and to stand in for SAM3's
designation mask.

The chain under test: SIFT bind -> KLT track -> depth lift -> RANSAC Kabsch ->
servo_error_3d -> PI -> empirically probed Jacobian -> Cartesian move. No object model,
no hand-eye calibration, no knowledge of where the camera is.

Run with: ``uv run --with mujoco pytest tests/showservo/test_end_to_end_sim.py``
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="sim bench needs mujoco (uv run --with mujoco)")

from lerobot.showservo.binder import SiftBinder, _import_cv2, sift_keypoints  # noqa: E402
from lerobot.showservo.card import Budget, GoalRelation, Keypoint, Stage, Termination  # noqa: E402
from lerobot.showservo.pose import ransac_fit_rigid, sample_depth  # noqa: E402
from lerobot.showservo.servo import JacobianEstimator, PIController, servo_error_3d  # noqa: E402
from lerobot.showservo.tracker import KLTTracker, shi_tomasi_points  # noqa: E402

from .simrig import SimRig  # noqa: E402

HALF = {"block": np.array([0.05, 0.05, 0.04]), "held": np.array([0.032, 0.032, 0.026])}
TAUGHT_BLOCK = np.array([0.0, 0.0, 0.04])
TAUGHT_HELD = np.array([0.010, -0.008, 0.125])  # the taught "grasp": just above the block


_ERODE_PX = 3


def _designation_mask(rig: SimRig, name: str, shrink: float = 0.55, silhouette: bool = False) -> np.ndarray:
    """Stand-in for SAM3. Post: HxW bool over the region a card is about.

    Ground truth is acceptable HERE because designation is a separate, already-solved
    problem in-house — what is under test is the geometry downstream of the mask.

    Two stand-ins, because they are not interchangeable and the difference is not
    cosmetic. The default projected BOX is what these tests have always used. A
    ``silhouette`` is strictly the more faithful stand-in — SAM3 returns silhouettes, and
    a box hands the tier under test a rectangle containing table, which a corner detector
    largely ignores and a dense patch grid samples in full, so a box quietly favours
    sparse tiers. It is not the default only because switching it breaks binding on this
    fixture for reasons not yet run down, and a bench that fails for an unexplained
    reason measures nothing. The benchmark sweep exposes it as a flag so the tier
    comparison can be made on the fair mask once that is understood.
    """
    if silhouette:
        cv2 = _import_cv2()
        mask = rig.silhouette(name)
        k = 2 * _ERODE_PX + 1
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        return eroded if eroded.any() else mask

    pos, rot = rig.pose_of(name)
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    corners_w = pos + (signs * HALF[name] * shrink) @ rot.T
    uv = rig.intrinsics.project(rig.world_to_camera(corners_w))

    mask = np.zeros((rig.height, rig.width), dtype=bool)
    c0, c1 = int(np.floor(uv[:, 0].min())), int(np.ceil(uv[:, 0].max()))
    r0, r1 = int(np.floor(uv[:, 1].min())), int(np.ceil(uv[:, 1].max()))
    mask[max(r0, 0) : r1 + 1, max(c0, 0) : c1 + 1] = True
    assert mask.any(), f"{name} projected outside the frame"
    return mask


def _team(
    rig: SimRig, rgb, depth, name: str, max_points: int, describe=None, silhouette: bool = False
) -> list[Keypoint]:
    """Detect, describe and lift — the offline compiler's keypoint step.

    ``describe`` overrides the descriptor tier as ``(rgb, mask) -> (uv, desc)``; the
    benchmark sweep uses it to compile the same card with dense patch features. Left as
    a parameter rather than a module switch so the tests keep exercising exactly the
    GPU-free path they always did.
    """
    mask = _designation_mask(rig, name, silhouette=silhouette)
    uv, desc = describe(rgb, mask) if describe else sift_keypoints(rgb, mask, max_points=max_points)
    assert len(uv) >= 8, f"only {len(uv)} features on {name}; the fixture must be richer"
    z, valid = sample_depth(depth, uv)
    uv, desc, z = uv[valid], desc[valid], z[valid]
    xyz = rig.intrinsics.deproject(uv, z)
    return [Keypoint(uv=p, descriptor=d, xyz=x) for p, d, x in zip(uv, desc, xyz, strict=True)]


def _teach(rig: SimRig, describe=None, silhouette: bool = False) -> Stage:
    """One demonstration keyframe, compiled into a stage."""
    rig.place("block", TAUGHT_BLOCK, yaw=0.0)
    rig.place("held", TAUGHT_HELD, yaw=0.0)
    rgb, depth = rig.render()

    target = _team(rig, rgb, depth, "block", max_points=60, describe=describe, silhouette=silhouette)
    held = _team(rig, rgb, depth, "held", max_points=40, describe=describe, silhouette=silhouette)
    return Stage(
        name="align-over-the-block",
        camera="rig",
        teams={"target": target, "held": held},
        goal_relation=GoalRelation(held_uv=np.stack([kp.uv for kp in held]), n_demos=1),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("pose_hold", {"tolerance_m": 0.003, "frames": 3}),
        budget=Budget(seconds=30.0, retries=2),
    )


class _Tracked:
    """One team: bound once, then tracked, lifted and REPLENISHED every frame.

    **Taught points are binding evidence, not tracking targets.** The bind establishes
    one rigid transform; what the tracker then follows is corners found in the LIVE
    frame, given taught coordinates by back-projecting through that transform. Seeding
    the tracker from the taught points instead — the obvious reading of the spec — was
    measurably worse across 16 randomised poses: it turned a handful of poses into
    40-140 mm failures that this seeding brings under 11 mm, and cut mean abstentions
    from 21 to 14 frames out of 40.

    The reason is that a taught point is chosen for being *describable* and a tracked
    point has to be *followable*, and on a real object those sets barely overlap. Worse,
    :meth:`BindResult.seed_points` fills in taught points the matcher never found by
    transporting them through a 2D similarity — a planar assumption this pipeline
    deliberately removed from the servo. Lifting such a point to 3D reads the depth of
    whatever happens to lie at a fabricated pixel, which is how a rigid fit ends up
    reporting a 20% scale error and the loop abstains on 39 frames out of 40.

    Replenishment then keeps the team alive on the same principle: KLT loses points
    steadily as the marker moves and re-shades, and fresh corners are recruited from
    where the team currently IS (the bounding box of its own live points, never ground
    truth) and back-projected through the current fit.
    """

    def __init__(self, stage: Stage, team: str, rig: SimRig, rgb, binder: SiftBinder, mask):
        taught_xyz = stage.team_xyz(team)
        assert taught_xyz is not None, f"{team} team was compiled without 3D"
        result = binder.bind(rgb, stage, team, mask=mask)
        assert result.ok, f"{team} bind failed: {result.reason}"
        self.rig = rig
        self.min_keep = 8

        # Measured pairs only, and only to place the taught body in the live frame.
        seed, measured = result.seed_points(stage.team_uv(team))
        _rgb, depth = rig.render()
        z, has_depth = sample_depth(depth, seed)
        placed = ransac_fit_rigid(
            taught_xyz, rig.intrinsics.deproject(seed, z), measured & has_depth, inlier_m=0.008
        )
        assert placed.ok, f"{team} bind certified but no 3D pose agrees with it"

        fresh = shi_tomasi_points(rgb, mask, max_points=40, min_distance=6.0)
        fz, fok = sample_depth(depth, fresh)
        fresh = fresh[fok]
        assert len(fresh) >= self.min_keep, f"{team} offers only {len(fresh)} trackable corners"
        lifted = rig.intrinsics.deproject(fresh, fz[fok])
        self.taught_xyz = placed.transform.inverse().apply(lifted)
        self.tracker = KLTTracker()
        self.tracker.init(rgb, fresh)

    def fit(self, rgb, depth, *, first: bool = False):
        state = self.tracker.state if first else self.tracker.step(rgb)
        z, has_depth = sample_depth(depth, state.uv)
        live = self.rig.intrinsics.deproject(state.uv, z)
        usable = state.valid & has_depth
        result = ransac_fit_rigid(self.taught_xyz, live, usable, inlier_m=0.008)
        if result.ok and int(usable.sum()) < self.min_keep:
            self._replenish(rgb, depth, state.uv[usable], result)
        return result

    def _replenish(self, rgb, depth, live_uv, result) -> None:
        if len(live_uv) < 2:
            return
        # SHRINK, never pad. A padded box around the live points reaches past the
        # silhouette onto the table, and a static table point recruited into a moving
        # team is a lie the fit then has to outvote — invariant 3, violated by a
        # margin of four pixels.
        lo, hi = live_uv.min(axis=0), live_uv.max(axis=0)
        inset = 0.12 * (hi - lo)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        c0, r0 = np.maximum(np.ceil(lo + inset), 0).astype(int)
        c1, r1 = np.floor(hi - inset).astype(int)
        if c1 <= c0 or r1 <= r0:
            return
        mask[r0 : r1 + 1, c0 : c1 + 1] = True

        fresh = shi_tomasi_points(rgb, mask, max_points=10, min_distance=6.0, exclude=live_uv)
        if len(fresh) == 0:
            return
        z, ok = sample_depth(depth, fresh)
        fresh = fresh[ok]
        if len(fresh) == 0:
            return
        lifted = self.rig.intrinsics.deproject(fresh, z[ok])
        self.tracker.add(fresh)
        self.taught_xyz = np.vstack([self.taught_xyz, result.transform.inverse().apply(lifted)])


_BODY = {"target": "block", "held": "held"}


def _relative_pose(rig: SimRig) -> np.ndarray:
    """Ground truth, for assertions: the held body in the block's own frame."""
    bp, br = rig.pose_of("block")
    hp, _ = rig.pose_of("held")
    return br.T @ (hp - bp)


def _run(rig: SimRig, stage: Stage, start_pos, *, steps: int = 40, binder=None, silhouette: bool = False):
    binder = binder or SiftBinder()
    rgb, depth = rig.render()
    target = _Tracked(
        stage, "target", rig, rgb, binder, _designation_mask(rig, "block", silhouette=silhouette)
    )
    held = _Tracked(stage, "held", rig, rgb, binder, _designation_mask(rig, "held", silhouette=silhouette))

    def measure(first=False):
        rgb_, depth_ = rig.render()
        t_fit = target.fit(rgb_, depth_, first=first)
        h_fit = held.fit(rgb_, depth_, first=first)
        err = servo_error_3d(held.taught_xyz, t_fit, h_fit)
        current = h_fit.transform.apply(held.taught_xyz).mean(axis=0) if h_fit.ok else None
        return err, current

    # Probe: three small world-frame moves, and what each did to the measured position.
    # This is the empirical Jacobian — no hand-eye calibration is ever computed, and
    # nothing here knows where the camera is.
    est = JacobianEstimator(n_joints=3, m=3, damping=1e-3)
    pos = np.asarray(start_pos, dtype=np.float64)
    _, current = measure(first=True)
    assert current is not None, "held team unfittable at the start"

    dqs, des = [], []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 0.008
        pos = pos + step
        rig.place("held", pos)
        _, after = measure()
        assert after is not None, "held team lost during the probe"
        dqs.append(step)
        des.append(after - current)
        current = after
    est.seed_from_probe(np.array(dqs), np.array(des))

    # One render and one tracker step per iteration: every extra step is another round
    # of KLT attrition, so measuring twice per loop halves how long the team survives.
    pi = PIController(kp=0.6, v_max=0.014)
    trace = []
    for _ in range(steps):
        err, now = measure()
        if not err.ok:
            trace.append(np.nan)
            continue
        trace.append(err.norm)
        if current is not None and now is not None and len(trace) > 1:
            est.update(dq, now - current)  # noqa: F821 - set on the previous iteration
        current = now
        dq = est.solve(pi.step(err.e_t, dt=0.05))
        pos = pos + dq
        rig.place("held", pos)
    return trace


@pytest.fixture(scope="module")
def rig() -> SimRig:
    """One rig for the module: each SimRig builds its own EGL context, and several
    live contexts in one process break the renderer."""
    r = SimRig()
    r.self_check()
    return r


def test_the_whole_pipeline_converges_on_rendered_pixels(rig):
    """A block moved and rotated; the held end finds its taught relation from pixels."""
    stage = _teach(rig)
    taught_relative = _relative_pose(rig)

    # The block is somewhere else, turned 35 degrees, and the held end starts 6-7 cm off.
    rig.place("block", [0.055, -0.048, TAUGHT_BLOCK[2]], yaw=np.deg2rad(20.0))
    start = np.array([-0.02, 0.05, 0.17])
    rig.place("held", start)

    trace = _run(rig, stage, start)
    finite = [t for t in trace if np.isfinite(t)]
    assert len(finite) > 20, "the loop abstained on most frames"

    achieved = _relative_pose(rig)
    residual_mm = float(np.linalg.norm(achieved - taught_relative)) * 1000.0
    assert residual_mm < 8.0, f"converged to {residual_mm:.1f} mm from the taught relation"
    assert finite[-1] < finite[0], "the measured error never came down"


# Fixed poses spanning the workspace, not a random sample: the bench has to give the
# same answer twice, and a seed that happens to avoid the hard corners would flatter it.
D1_POSES = [
    (0.06, 0.03, -18.0),
    (-0.045, 0.055, 15.0),
    (0.055, -0.048, 20.0),
    (-0.05, -0.04, -25.0),
    (0.02, 0.06, 8.0),
    (-0.06, 0.02, 28.0),
]
D1_TOLERANCE_MM = 14.0
# Measured 4/6: 13.4, 3.6, 5.2, bind-refused, 2.2, bind-refused mm. Both failures are the
# binder refusing to certify, not the servo missing — the same SIFT weakness M0 measured
# on real video, and what the DINOv3 tier is expected to remove. The floor is pinned here
# so a servo regression is caught; raise it when the binder tier changes.
D1_MIN_SUCCESSES = 4


def test_it_converges_from_several_starts(rig):
    """Not one lucky offset: the D1 curve is success versus displacement.

    A success RATE rather than a per-pose threshold, because per-pose thresholds were
    measuring the wrong thing. Two poses used to gate this, one of them xfailed at
    ~50 mm; a 16-pose sweep showed the pass/fail of any single pose is dominated by
    whether its particular view binds, while the seeding policy moves the whole
    distribution. Gating on the distribution is what D1 actually asks for, and it stops
    a change that helps most poses from being blocked by one that it costs.

    The gate is deliberately loose against the real D1 bar (>=90% under a few mm): SIFT
    binding is the current ceiling and this pins the floor beneath it, so a regression
    is caught without pretending the pipeline is finished.
    """
    stage = _teach(rig)
    taught_relative = _relative_pose(rig)

    residuals = []
    for dx, dy, yaw_deg in D1_POSES:
        rig.place("block", [dx, dy, TAUGHT_BLOCK[2]], yaw=np.deg2rad(yaw_deg))
        start = np.array([dx - 0.025, dy + 0.03, 0.155])
        rig.place("held", start)
        try:
            _run(rig, stage, start)
        except AssertionError:
            # A refused bind is a pose the stage cannot start from — a failed pose, not a
            # broken bench. It has to be counted, because otherwise the one failure mode
            # the certificate is designed to expose would abort the measurement of all
            # the others. On this fixture SIFT refuses roughly one pose in four.
            residuals.append(float("inf"))
            continue
        residuals.append(float(np.linalg.norm(_relative_pose(rig) - taught_relative)) * 1000.0)

    successes = sum(r < D1_TOLERANCE_MM for r in residuals)
    detail = ", ".join(f"{p}: {r:.1f} mm" for p, r in zip(D1_POSES, residuals, strict=True))
    assert successes >= D1_MIN_SUCCESSES, f"only {successes}/{len(D1_POSES)} converged — {detail}"
