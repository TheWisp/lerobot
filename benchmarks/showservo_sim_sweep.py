# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Success rate of the whole servo loop over many poses, per descriptor tier.

The committed sim test gates on six fixed poses and must stay GPU-free, so it can only
ever run the SIFT tier. This asks the question that test cannot: **does the binder tier
explain the failures?** M0 says SIFT cannot hold a low-texture object and DINOv3 can;
the sim test independently shows its only failures are the binder refusing to certify.
This joins the two claims on one substrate.

It also exists because two poses were, for a while, deciding an architectural question
by coin flip: a seeding change that fixed one pose and broke the other looked decisive
and was not — over 16 poses the two policies were indistinguishable. Any claim about
this loop needs a distribution, and this is where that distribution is measured.

Not a pytest test: it needs a GPU for the dense tiers, and its output is a measurement to
read rather than an assertion to pass.

Usage:
    MUJOCO_GL=egl PYTHONPATH=src <python-with-transformers> \\
        benchmarks/showservo_sim_sweep.py --poses 16 --binder sift dino
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# The sim rig and the loop under test live with the tests, and there must be exactly one
# of each — a benchmark that reimplemented them would drift and then disagree for reasons
# nobody could attribute. benchmarks/ is not a package, so the path is joined here.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo.simrig import TEXTURES, SimRig  # noqa: E402  (tests/showservo)
from showservo.test_end_to_end_sim import (  # noqa: E402
    TAUGHT_BLOCK,
    _designation_mask,
    _relative_pose,
    _run,
    _teach,
)
from showservo_m0 import DinoTier, SiftTier  # noqa: E402

from lerobot.fewshot.registration import mutual_matches  # noqa: E402
from lerobot.showservo.binder import _import_cv2  # noqa: E402
from lerobot.showservo.pose import ransac_fit_rigid, sample_depth  # noqa: E402

SUCCESS_MM = 8.0


def _describe_with(tier, max_points: int):
    """``(rgb, mask) -> (uv, desc)``, capped so a dense tier does not hand back a card of
    a thousand patches. Subsampling is evenly spaced rather than random: the card should
    cover the object, and the sweep has to give the same answer twice."""

    def describe(rgb, mask):
        uv, desc = tier.teach(rgb, mask)
        if len(uv) > max_points:
            keep = np.linspace(0, len(uv) - 1, max_points).astype(int)
            uv, desc = uv[keep], desc[keep]
        return uv, desc

    return describe


class _BindOnlyTeam:
    """One team with NO tracker: designate + bind + Kabsch, fresh, every frame.

    Exists to test whether the KLT tier is necessary at all. The spec's tracker exists
    on the assumption that binding is too expensive to run per frame; if per-frame
    binding also *converges better*, then teams, replenishment, decay and the rebind
    rung are machinery without a purpose.

    One piece of memory is allowed, and it is a certificate policy rather than a
    tracker: the last CERTIFIED fit of a team is held while later frames refuse to
    bind, because within a stage the target is static (the attachment monitor owns
    detecting when that stops being true) and the held object moves only when
    commanded. Without this, a marginal refusal is a deadlock: refused bind -> no
    command -> the identical frame -> the identical refusal, forever. Measured at one
    sweep pose: the held box's own cast shadow degraded the target's bind from ratio
    0.28 to 0.21 (gate: 0.25) and froze the loop 89 mm out. Coasting is bounded by the
    stage attempt, exactly like the invariant it leans on.
    """

    RECRUIT_DILATE_PX = 45  # annulus width around the designation; ~a quarter object width
    RECRUIT_MAX = 60

    def __init__(self, stage, team, rig, binder, body, silhouette, *, tier=None, movers=()):
        self.taught_xyz = stage.team_xyz(team)
        self.stage, self.team, self.rig, self.binder = stage, team, rig, binder
        self.body, self.silhouette = body, silhouette
        self.last_certified = None
        # Recruitment (the spec's "temporarily recruited ref points"), enabled by passing
        # the descriptor tier. Valid ONLY for a team whose object is static within the
        # stage: the surroundings co-move with a static target, but the ground a MOVING
        # object slides over does not co-move with it — so the held team must never
        # recruit from its surroundings (its one legitimate recruit, the gripper after a
        # grasp, does not exist in this rig).
        self.tier = tier
        self.movers = tuple(movers)
        self.recruit_xyz = None  # assigned coordinates in the taught frame
        self.recruit_desc = None  # runtime descriptors — the taught card is never consulted

    def _annulus(self, cv2) -> np.ndarray:
        """Where recruits may come from: a ring around the object, minus every mover.

        Built from the raw silhouette (not the eroded designation): recruits should sit
        OFF the object, and the mover subtraction — the held box, and on a real rig the
        arm — is what makes automating this safe. SAM3's negative concepts play this
        role outside sim.
        """
        sil = self.rig.silhouette(self.body)
        k = 2 * self.RECRUIT_DILATE_PX + 1
        ring = (cv2.dilate(sil.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & ~sil
        for mover in self.movers:
            ring &= ~self.rig.silhouette(mover)
        return ring

    def _try_recruit(self, rgb, depth, fit) -> None:
        """Expand the measurement set from the surroundings, at a certified moment.

        Each recruit gets coordinates in the TAUGHT FRAME via the certified fit — pure
        bookkeeping, no appearance claim: the surroundings need not have existed at
        teach time, and their descriptors are captured from the live frame now.
        """
        cv2 = _import_cv2()
        ring = self._annulus(cv2)
        if int(ring.sum()) < 2000:
            return  # occluded down to slivers; try again at the next certified frame
        try:
            uv, desc = self.tier.teach(rgb, ring)
        except AssertionError:
            return  # region too small for the extractor's grid
        if len(uv) < 8:
            return
        if len(uv) > self.RECRUIT_MAX:
            keep = np.linspace(0, len(uv) - 1, self.RECRUIT_MAX).astype(int)
            uv, desc = uv[keep], desc[keep]
        z, ok = sample_depth(depth, uv)
        if int(ok.sum()) < 8:
            return
        lifted = self.rig.intrinsics.deproject(uv[ok], z[ok])
        self.recruit_xyz = fit.transform.inverse().apply(lifted)
        self.recruit_desc = np.asarray(desc[ok], dtype=np.float32)

    def _recruit_pairs(self, rgb, depth, cv2):
        """Post: (src_xyz, dst_xyz) of currently re-matched recruits; empty if none."""
        empty = np.zeros((0, 3))
        ring = self._annulus(cv2)
        if int(ring.sum()) < 2000:
            return empty, empty
        try:
            uv, desc = self.tier.teach(rgb, ring)
        except AssertionError:
            return empty, empty
        if len(uv) == 0:
            return empty, empty
        ia, ib = mutual_matches(self.recruit_desc, np.asarray(desc, dtype=np.float32))
        if len(ia) == 0:
            return empty, empty
        z, ok = sample_depth(depth, uv[ib])
        return self.recruit_xyz[ia[ok]], self.rig.intrinsics.deproject(uv[ib][ok], z[ok])

    def fit(self, rgb, depth, *, first: bool = False):
        """The card is the AUTHORITY; recruits are a fallback, re-anchored whenever the
        card speaks.

        The first cut unioned card and recruit correspondences into one fit, and made
        two strong-card poses WORSE (5.0 -> 10.4 mm, 4.8 -> 15.1 mm) while rescuing the
        weak-card ones: ~60 recruits carrying their recruitment-moment bias outvoted
        the card's exact taught coordinates. Bias entrenchment, measured. Card-first
        fixes it structurally — recruit coordinates are refreshed through every
        certified card fit, so their error is always one frame of card truth away,
        never cumulative; and they only ever decide the answer when the card has
        nothing to say.
        """
        mask = _designation_mask(self.rig, self.body, silhouette=self.silhouette)
        r = self.binder.bind(rgb, self.stage, self.team, mask=mask)
        if r.ok:
            seed, measured = r.seed_points(self.stage.team_uv(self.team))
            z, has_depth = sample_depth(depth, seed)
            live = self.rig.intrinsics.deproject(seed, z)
            fit = ransac_fit_rigid(self.taught_xyz, live, measured & has_depth, inlier_m=0.008)
            if fit.ok:
                self.last_certified = fit
                if self.tier is not None:
                    self._try_recruit(rgb, depth, fit)  # refresh: coords through THIS fit
                return fit
        if self.recruit_xyz is not None:
            r_src, r_dst = self._recruit_pairs(rgb, depth, _import_cv2())
            if len(r_src) >= 8:
                fit = ransac_fit_rigid(r_src, r_dst, np.ones(len(r_src), dtype=bool), inlier_m=0.008)
                if fit.ok:
                    self.last_certified = fit
                    return fit
        if self.last_certified is not None:
            return self.last_certified
        # Nothing certified yet: a genuinely failed fit, so the loop abstains honestly.
        return ransac_fit_rigid(self.taught_xyz[:1], self.taught_xyz[:1], np.zeros(1, dtype=bool))


def sweep(
    tier,
    poses,
    *,
    steps: int = 40,
    silhouette: bool = False,
    texture: str = "speckle",
    mode: str = "track",
) -> list[float | None]:
    """Final error per pose in mm; None where the stage never started.

    A refused bind is None rather than a large number: it is a pose the loop declined to
    attempt, and averaging it in as if it were a bad servo would hide the one failure the
    certificate exists to make visible.
    """
    assert mode in ("track", "bindonly", "recruit"), mode
    # A dense tier's whole advantage is having a descriptor everywhere; capping its
    # card at a sparse tier's size throws that away and measures neither fairly. M0's
    # real-video binds carried 450+ inliers, not 60.
    describe = _describe_with(tier, max_points=400 if isinstance(tier, DinoTier) else 60)
    out: list[float | None] = []
    for bx, by, yaw, start in poses:
        rig = SimRig(texture=texture)
        stage = _teach(rig, describe=describe, silhouette=silhouette)
        taught = _relative_pose(rig)
        rig.place("block", [bx, by, TAUGHT_BLOCK[2]], yaw=np.deg2rad(yaw))
        rig.place("held", start)
        binder = tier.make_binder()
        teams = None
        if mode in ("bindonly", "recruit"):
            # Recruitment is target-only: the target is static within the stage, so its
            # surroundings co-move with it; the held object's surroundings do not.
            target_kw = {"tier": tier, "movers": ("held",)} if mode == "recruit" else {}
            teams = (
                _BindOnlyTeam(stage, "target", rig, binder, "block", silhouette, **target_kw),
                _BindOnlyTeam(stage, "held", rig, binder, "held", silhouette),
            )
        try:
            _run(rig, stage, start, steps=steps, binder=binder, silhouette=silhouette, teams=teams)
        except AssertionError:
            out.append(None)
            continue
        out.append(float(np.linalg.norm(_relative_pose(rig) - taught)) * 1000.0)
    return out


def make_poses(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return [
        (
            rng.uniform(-0.06, 0.06),
            rng.uniform(-0.06, 0.06),
            rng.uniform(-30, 30),
            np.array([rng.uniform(-0.08, 0.02), rng.uniform(0.03, 0.09), rng.uniform(0.14, 0.19)]),
        )
        for _ in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--binder", nargs="+", default=["sift"], choices=("sift", "dino"))
    ap.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--mode",
        default="track",
        choices=("track", "bindonly", "recruit"),
        help="'track' = bind once then KLT teams (the spec's design); 'bindonly' = no "
        "tracker, designate+bind+Kabsch every frame, last certified fit held through "
        "refusals; 'recruit' = bindonly plus temporarily recruited reference points "
        "from the target's static surroundings",
    )
    ap.add_argument(
        "--texture",
        default="speckle",
        choices=TEXTURES,
        help="object surface. 'speckle' is i.i.d. noise (the original fixture, locally "
        "unique and so ideal for a corner detector); 'photo' is multi-scale 1/f detail; "
        "'matte' is the real target's problem — almost no corners at all",
    )
    ap.add_argument(
        "--designation",
        default="box",
        choices=("box", "silhouette"),
        help="SAM3 stand-in. 'box' is what the committed test uses; 'silhouette' is the "
        "faithful one and the only fair mask for comparing a sparse tier against a dense "
        "one, since a box hands the dense tier a rectangle full of table",
    )
    args = ap.parse_args()

    silhouette = args.designation == "silhouette"
    poses = make_poses(args.poses, args.seed)
    results = {}
    for name in args.binder:
        tier = DinoTier(args.dino_model, device=args.device) if name == "dino" else SiftTier()
        print(f"running {tier.label} over {args.poses} poses ...", flush=True)
        results[tier.label] = sweep(tier, poses, silhouette=silhouette, texture=args.texture, mode=args.mode)

    print(
        f"\n{args.poses} poses (seed {args.seed}), {args.designation} designation, "
        f"{args.texture} texture, {args.mode} mode, "
        f"success = final error < {SUCCESS_MM:.0f} mm"
    )
    print(f"  {'tier':<42} {'success':>9} {'bind refused':>13} {'median mm':>10}")
    for label, errs in results.items():
        done = [e for e in errs if e is not None]
        ok = sum(1 for e in done if e < SUCCESS_MM)
        refused = sum(1 for e in errs if e is None)
        med = np.median(done) if done else float("nan")
        print(f"  {label:<42} {ok:>4d}/{args.poses:<4d} {refused:>13d} {med:>10.1f}")

    print("\nper pose (mm, 'refused' = the binder declined to start)")
    for label, errs in results.items():
        cells = "".join("refused " if e is None else f"{e:7.1f} " for e in errs)
        print(f"  {label[:28]:<28} {cells}")


if __name__ == "__main__":
    main()
