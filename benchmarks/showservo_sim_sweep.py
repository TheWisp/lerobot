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

from showservo.simrig import SimRig  # noqa: E402  (tests/showservo)
from showservo.test_end_to_end_sim import (  # noqa: E402
    TAUGHT_BLOCK,
    _relative_pose,
    _run,
    _teach,
)
from showservo_m0 import DinoTier, SiftTier  # noqa: E402

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


def sweep(tier, poses, *, steps: int = 40, silhouette: bool = False) -> list[float | None]:
    """Final error per pose in mm; None where the stage never started.

    A refused bind is None rather than a large number: it is a pose the loop declined to
    attempt, and averaging it in as if it were a bad servo would hide the one failure the
    certificate exists to make visible.
    """
    describe = _describe_with(tier, max_points=60)
    out: list[float | None] = []
    for bx, by, yaw, start in poses:
        rig = SimRig()
        stage = _teach(rig, describe=describe, silhouette=silhouette)
        taught = _relative_pose(rig)
        rig.place("block", [bx, by, TAUGHT_BLOCK[2]], yaw=np.deg2rad(yaw))
        rig.place("held", start)
        try:
            _run(rig, stage, start, steps=steps, binder=tier.make_binder(), silhouette=silhouette)
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
        results[tier.label] = sweep(tier, poses, silhouette=silhouette)

    print(
        f"\n{args.poses} poses (seed {args.seed}), {args.designation} designation, "
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
