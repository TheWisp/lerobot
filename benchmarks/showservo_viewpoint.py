# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The viewpoint envelope: how far can the camera drift from the taught view?

A card is taught from ONE camera pose. Everything downstream of binding is
camera-invariant by construction — the servo error is relative, the Jacobian is
probed — so the only viewpoint-sensitive component is the bind. This measures it:
teach at the rig's home view, orbit the camera by increasing azimuth, and ask at
each offset (a) does the card bind, (b) does the whole loop still converge.

Two certificate gates, because the hypothesis under test is that the envelope is
set by the GATE, not by the descriptors:

* ``sim2`` — the shipped :class:`DinoBinder`: mutual matches gated by consensus on
  a 2D similarity. A viewpoint change is not a similarity, so honest matches stop
  agreeing with the model as the angle grows.
* ``rigid3d`` — the same mutual matches gated directly by RANSAC on the rigid 3D
  transform (the card carries 3D; the live side is depth-lifted). Perspective is
  not a distortion to this model, so if descriptors survive the angle, the gate
  should too.

Pre-registered predictions: sim2 collapses in the 20-40 deg band; rigid3d holds
meaningfully further; wherever a bind certifies, the servo converges.

Usage:
    MUJOCO_GL=egl PYTHONPATH=src <python-with-transformers> \\
        benchmarks/showservo_viewpoint.py --azimuths 0 10 20 30 45 60
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo.simrig import SimRig  # noqa: E402
from showservo.test_end_to_end_sim import (  # noqa: E402
    TAUGHT_BLOCK,
    TAUGHT_HELD,
    _designation_mask,
    _relative_pose,
    _run,
    _teach,
    _team,
)
from showservo_m0 import DinoTier  # noqa: E402
from showservo_sim_sweep import _BindOnlyTeam, _describe_with, make_poses  # noqa: E402

from lerobot.fewshot.registration import mutual_matches  # noqa: E402
from lerobot.showservo.card import Budget, GoalRelation, Stage, Termination  # noqa: E402
from lerobot.showservo.pose import ransac_fit_rigid, sample_depth  # noqa: E402

SUCCESS_MM = 8.0


class Rigid3DGateTeam(_BindOnlyTeam):
    """Bind-only team whose certificate is the rigid 3D fit itself.

    No intermediate 2D model: mutual matches go straight to depth-lifted RANSAC
    Kabsch, and the certificate is its inlier count plus scale plausibility. The
    parent's coast/abstain policy is inherited unchanged.
    """

    MIN_INLIERS = 6

    def fit(self, rgb, depth, *, first: bool = False):
        try:
            mask = _designation_mask(self.rig, self.body, silhouette=self.silhouette)
        except AssertionError:
            mask = None
        if mask is not None:
            try:
                uv, desc = self.tier.teach(rgb, mask)
            except AssertionError:
                uv, desc = np.zeros((0, 2)), np.zeros((0, 1))
            taught_desc = self.stage.team_descriptors(self.team)
            if len(uv) >= self.MIN_INLIERS and taught_desc is not None:
                ia, ib = mutual_matches(taught_desc, np.asarray(desc, dtype=np.float32))
                if len(ia) >= self.MIN_INLIERS:
                    z, ok = sample_depth(depth, uv[ib])
                    fit = ransac_fit_rigid(
                        self.taught_xyz[ia[ok]],
                        self.rig.intrinsics.deproject(uv[ib][ok], z[ok]),
                        inlier_m=0.008,
                    )
                    if fit.ok and fit.n_inliers >= self.MIN_INLIERS and fit.scale_is_plausible():
                        self.last_certified = fit
                        return fit
        if self.coast and self.last_certified is not None:
            return self.last_certified
        return ransac_fit_rigid(self.taught_xyz[:1], self.taught_xyz[:1], np.zeros(1, dtype=bool))


def teach_demo(texture, describe, block_yaw_deg: float) -> Stage:
    """One demonstration keyframe with the block at ``block_yaw_deg``.

    The held box is placed at the SAME relation in the block's frame as the home
    demo, so every demo teaches one relation seen from a different object view —
    which is exactly what an extra demo buys: view coverage, not a new goal.
    """
    rig = SimRig(texture=texture)
    th = np.deg2rad(block_yaw_deg)
    rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    rig.place("block", TAUGHT_BLOCK, yaw=th)
    rig.place("held", TAUGHT_BLOCK + rz @ (TAUGHT_HELD - TAUGHT_BLOCK), yaw=th)
    rgb, depth = rig.render()
    target = _team(rig, rgb, depth, "block", max_points=60, describe=describe, silhouette=True)
    held = _team(rig, rgb, depth, "held", max_points=40, describe=describe, silhouette=True)
    return Stage(
        name=f"demo-yaw-{block_yaw_deg:+.0f}",
        camera="rig",
        teams={"target": target, "held": held},
        goal_relation=GoalRelation(held_uv=np.stack([kp.uv for kp in held]), n_demos=1),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("pose_hold", {"tolerance_m": 0.003, "frames": 3}),
        budget=Budget(seconds=30.0, retries=2),
    )


class MultiDemoTarget:
    """Bind against EVERY demo's card; the best fresh certificate picks the demo.

    Sub-teams run with coasting off, so selection only compares fresh evidence;
    coasting happens here, at the multi level, remembering which demo it came from.
    ``selected`` is read by :class:`MultiDemoHeld` so both ends of the relation
    always come from the same demonstration — mixing demos would compare a live
    relation against a goal assembled from two different teaching frames.
    """

    def __init__(self, teams):
        assert teams, "at least one demo"
        self.teams = teams
        self.selected = 0
        self.last_certified = None

    @property
    def taught_xyz(self):
        return self.teams[self.selected].taught_xyz

    def fit(self, rgb, depth, *, first: bool = False):
        best_k, best_fit = None, None
        for k, team in enumerate(self.teams):
            f = team.fit(rgb, depth, first=first)
            if f.ok and (best_fit is None or f.n_inliers > best_fit.n_inliers):
                best_k, best_fit = k, f
        if best_fit is not None:
            self.selected = best_k
            self.last_certified = best_fit
            return best_fit
        if self.last_certified is not None:
            return self.last_certified
        return self.teams[0].fit(rgb, depth, first=first)  # a failed fit, honestly


class MultiDemoHeld:
    """The held end of whichever demo the target selected this frame."""

    def __init__(self, teams, target: MultiDemoTarget):
        self.teams = teams
        self.target = target

    @property
    def taught_xyz(self):
        return self.teams[self.target.selected].taught_xyz

    def fit(self, rgb, depth, *, first: bool = False):
        return self.teams[self.target.selected].fit(rgb, depth, first=first)


def teams_for(gate, stages, rig, binder, tier):
    """Teams over one or several demos. Post: (target, held) with matching demo choice."""

    def one(stage, team, body, coast):
        if gate == "sim2":
            return _BindOnlyTeam(stage, team, rig, binder, body, True, coast=coast)
        return Rigid3DGateTeam(stage, team, rig, binder, body, True, tier=tier, coast=coast)

    if len(stages) == 1:
        return one(stages[0], "target", "block", True), one(stages[0], "held", "held", False)
    target = MultiDemoTarget([one(s, "target", "block", False) for s in stages])
    held = MultiDemoHeld([one(s, "held", "held", False) for s in stages], target)
    return target, held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--azimuths", type=float, nargs="+", default=[0, 5, 10, 15, 20, 30, 45])
    ap.add_argument("--poses", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--texture", default="photo")
    ap.add_argument(
        "--demos",
        type=int,
        default=1,
        help="demonstrations to teach; >1 spreads the object yaw across demos, so the "
        "card carries several views of the same relation",
    )
    ap.add_argument("--gates", nargs="+", default=["sim2", "rigid3d"], choices=("sim2", "rigid3d"))
    ap.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tier = DinoTier(args.dino_model, device=args.device)
    describe = _describe_with(tier, max_points=400)

    # The card(s), taught at the home camera view. The same Stage objects are reused at
    # every azimuth — cards are data, and their portability is exactly what is under
    # test. Multiple demos vary the OBJECT's yaw, not the camera.
    if args.demos == 1:
        teach_rig = SimRig(texture=args.texture)
        stages = [_teach(teach_rig, describe=describe, silhouette=True)]
        del teach_rig
    else:
        yaws = np.linspace(-25.0, 25.0, args.demos)
        stages = [teach_demo(args.texture, describe, y) for y in yaws]

    # The taught relation, in the block's frame, straight from the teaching geometry.
    # Reading it off a rig via _relative_pose is only correct AFTER placing both bodies
    # at their taught poses — the first version of this bench read it before placing
    # anything and graded every run against the held box's XML parking spot, reporting
    # a constant ~141 mm "failure" for runs that had in fact converged.
    taught = TAUGHT_HELD - TAUGHT_BLOCK

    poses = make_poses(args.poses, args.seed)
    print(
        f"{args.demos} demo(s) taught at camera azimuth 0, {args.texture} texture; "
        f"{args.poses} poses per azimuth, success = final error < {SUCCESS_MM:.0f} mm\n"
    )
    print(f"  {'azimuth':>8} {'gate':>8} {'demos':>6} {'bound@f0':>9} {'success':>9} {'median mm':>10}")
    for az in args.azimuths:
        for gate in args.gates:
            bound, errs = 0, []
            for bx, by, yaw, start in poses:
                rig = SimRig(texture=args.texture)
                if az:
                    rig.orbit_camera(az)
                rig.place("block", [bx, by, TAUGHT_BLOCK[2]], yaw=np.deg2rad(yaw))
                rig.place("held", start)
                binder = tier.make_binder()
                target, held = teams_for(gate, stages, rig, binder, tier)

                rgb, depth = rig.render()
                bound += int(target.fit(rgb, depth).ok)
                try:
                    _run(rig, stages[0], start, binder=binder, silhouette=True, teams=(target, held))
                except AssertionError:
                    errs.append(None)
                    continue
                errs.append(float(np.linalg.norm(_relative_pose(rig) - taught)) * 1000.0)
            done = [e for e in errs if e is not None]
            ok = sum(1 for e in done if e < SUCCESS_MM)
            med = np.median(done) if done else float("nan")
            print(
                f"  {az:>8.0f} {gate:>8} {args.demos:>6d} {bound:>6d}/{args.poses:<2d} "
                f"{ok:>6d}/{args.poses:<2d} {med:>10.1f}"
            )


if __name__ == "__main__":
    main()
