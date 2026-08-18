# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""M0 — the input-quality gate, on real recorded video. No robot, no servo, no motion.

M0 asks one question: are the measurements the servo depends on trustworthy enough to
be worth servoing on? Two curves answer it.

* **Tracker point survival.** Seed KLT once, track forward, and record what fraction of
  points the forward-backward gate still believes after N frames. This sets how often
  a re-bind is needed, and therefore whether a stage can be crossed at all.
* **Bind inlier rate versus change.** Bind a taught frame's constellation against later
  frames, and against a different episode. This is the certificate the whole retry
  ladder keys on, measured where it actually degrades.

Deliberately run on REAL video rather than renders: a render has no motion blur, no
sensor noise and no rolling shutter, so it flatters both numbers. Nothing here is a
pytest test — it points at real recorded data, which tests must never do, and its
output is a measurement to be read rather than an assertion to be passed.

Both curves are measured over a DESIGNATED region, because both are meaningless over
an arbitrary one: a point on the static table survives forever, and a constellation
that is mostly background cannot correspond across a scene reset. Pass ``--concept``
to designate with SAM3, the way the real system does.

Usage:
    uv run python benchmarks/showservo_m0.py --repo-id thewisp/<dataset> --camera top

    # with real designation — needs an interpreter that has transformers + SAM3
    PYTHONPATH=src <python-with-transformers> benchmarks/showservo_m0.py \\
        --repo-id thewisp/<dataset> --camera front --concept "green ring" --binder dino
"""

from __future__ import annotations

import argparse

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.showservo.binder import BindGate, DinoBinder, SiftBinder, sift_keypoints
from lerobot.showservo.card import Budget, GoalRelation, Keypoint, Stage, Termination
from lerobot.showservo.tracker import KLTTracker

SURVIVAL_HORIZONS = (5, 10, 20, 40, 80, 150)
BIND_OFFSETS = (0, 5, 15, 30, 60, 120, 240)
MIN_TEAM = 8  # below this a team cannot be seeded at all, let alone fitted


def _frame(dataset: LeRobotDataset, index: int, key: str) -> np.ndarray:
    """Post: HxWx3 uint8 RGB."""
    img = dataset[index][key]
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class CentreRoi:
    """A crude stand-in for designation, kept only so the bench runs with no model.

    Binding over the WHOLE frame would be dominated by static background and would
    report a flattering inlier rate that says nothing about the target. The middle of
    the workspace is at least the region a card is usually about.

    Numbers from this designator are NOT interpretable and must not be reported as if
    they were: survival partly measures how much motionless background a view happens
    to frame, and a cross-episode constellation that is mostly table legitimately fails
    to correspond after a scene reset. Use ``--concept``.
    """

    label = "centre-roi (stand-in — numbers not interpretable)"

    def __init__(self, frac: float = 0.55):
        self.frac = frac

    def mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        m = np.zeros((h, w), dtype=bool)
        dh, dw = int(h * self.frac / 2), int(w * self.frac / 2)
        m[h // 2 - dh : h // 2 + dh, w // 2 - dw : w // 2 + dw] = True
        return m


class Sam3Concept:
    """Real designation: a text prompt through SAM3, one INDEPENDENT detection per frame.

    Temporal state is dropped before every call on purpose. The frames this bench asks
    about are non-consecutive (the bind offsets skip up to 8 s) or come from different
    episodes, so a memory bank conditioned on frames that never preceded them would
    propagate a mask from the wrong scene and quietly flatter every number.

    The mask is eroded slightly: SIFT prefers high-contrast corners, which puts a good
    share of its detections right on the silhouette, where the descriptor patch is half
    background. Those points are still object-attached, but their descriptors drift with
    whatever is behind the object — the same class of contamination designation exists
    to remove.
    """

    ERODE_PX = 3

    def __init__(self, concept: str, device: str = "cuda", resolution: int | None = None):
        from lerobot.overlays.adapters import Sam3TrackByDetectionAdapter

        self.concept = concept
        self.label = f"SAM3 {concept!r}"
        self.adapter = Sam3TrackByDetectionAdapter(device=device, resolution=resolution)
        self.adapter.set_control({"prompt": concept})
        self.adapter.set_camera("bench")
        self.misses = 0

    def mask(self, frame: np.ndarray) -> np.ndarray | None:
        """Post: HxW bool over the union of every designated concept, or None if the
        detector found nothing (counted in ``misses``; never silently widened)."""
        import cv2

        self.adapter.reset()  # the adapter's own "discontinuity" path: drop the memory bank
        masks = self.adapter.segment(np.ascontiguousarray(frame))
        if not masks:
            self.misses += 1
            return None
        union = np.zeros(frame.shape[:2], dtype=bool)
        for m in masks.values():
            union |= m
        k = 2 * self.ERODE_PX + 1
        eroded = cv2.erode(union.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        # Erosion of a thin object can empty it; a too-small designation is better than
        # a silently un-eroded one only when something survives.
        return eroded if eroded.any() else union


class SiftTier:
    """v0's cheap tier: SIFT detections inside the designated region, no GPU."""

    label = "SIFT"

    def __init__(self, max_points: int = 80):
        self.max_points = max_points

    def teach(self, frame, mask):
        return sift_keypoints(frame, mask, max_points=self.max_points)

    def make_binder(self):
        return SiftBinder(gate=BindGate())


class DinoTier:
    """Dense ViT patch features — the cross-instance tier.

    Descriptors sit on a regular patch grid rather than on detected corners, so unlike
    SIFT it cannot fail by finding nothing: every masked patch is a descriptor. That is
    the whole reason to measure it here, on an object whose texture SIFT cannot hold.
    """

    def __init__(self, model_id: str = "facebook/dinov2-small", device: str = "cuda"):
        from lerobot.fewshot.features import DinoPatchExtractor

        self.extractor = DinoPatchExtractor(model_id=model_id, device=device)
        self.label = f"{model_id} patches"

    def teach(self, frame, mask):
        return self.extractor.extract(frame, mask)

    def make_binder(self):
        return DinoBinder(self.extractor, gate=BindGate())


def _episode_starts(dataset: LeRobotDataset) -> list[int]:
    """Post: one start index per episode; falls back to uniform chunking."""
    for attr in ("episode_data_index",):
        idx = getattr(dataset, attr, None)
        if isinstance(idx, dict) and "from" in idx:
            return [int(v) for v in np.asarray(idx["from"]).ravel()]
    n_ep = max(int(dataset.meta.total_episodes), 1)
    per = max(int(dataset.meta.total_frames) // n_ep, 1)
    return [i * per for i in range(n_ep)]


def _episode_lengths(dataset: LeRobotDataset, starts: list[int]) -> list[int]:
    """Post: one length per episode, so a phase fraction can be turned into a frame."""
    ends = [*starts[1:], int(dataset.meta.total_frames)]
    return [max(e - s, 1) for s, e in zip(starts, ends, strict=True)]


def teach_frame(dataset, key, start: int, designator, tier, search: int = 120, step: int = 5):
    """The first frame of this episode a card could actually be taught from.

    Anchoring on frame 0 is arbitrary and, on real recordings, usually wrong: episodes
    open before the object is in view or in focus, so frame 0 measures whether the
    session happened to start pointed at the target. Teaching happens at a keyframe the
    demonstrator chooses. The FIRST usable frame is the closest unbiased stand-in —
    "first" rather than "best" so this cannot be read as cherry-picking the frame with
    the nicest features.

    Post: ``(index, frame, mask)``, or None if no frame in the search window designates
    a region with enough features to seed a team.
    """
    for offset in range(0, search + 1, step):
        frame = _frame(dataset, start + offset, key)
        mask = designator.mask(frame)
        if mask is None:
            continue
        uv, _ = tier.teach(frame, mask)
        if len(uv) >= MIN_TEAM:
            return start + offset, frame, mask
    return None


def survival(dataset, key, start: int, designator, n_points: int = 60) -> tuple[dict[int, float], int]:
    """Fraction of seeded points the tracker still believes, by horizon.

    Measures the TRACKER, so it is deliberately independent of the binder tier: it seeds
    from corners, because that is what KLT can follow, and picks its teach frame by the
    same criterion. Selecting the frame by the binder's tier instead made this row move
    when only the binder changed — dense patches are available on any masked frame, so
    they teach earlier, on a smaller and blurrier object, and the tracker then looked
    worse for a reason that had nothing to do with tracking.

    (Where a dense binder's tracker seeds should come from is a real open question — see
    the results write-up.)

    Post: ``({horizon: fraction}, n_seeded)``; ``({}, 0)`` when no frame in the episode
    yields a designation with enough features to seed a team.
    """
    taught = teach_frame(dataset, key, start, designator, SiftTier())
    if taught is None:
        return {}, 0
    index, first, mask = taught
    uv, _ = sift_keypoints(first, mask, max_points=n_points)
    if len(uv) < MIN_TEAM:
        return {}, len(uv)

    tracker = KLTTracker()
    tracker.init(first, uv)
    out, n0 = {}, len(uv)
    for offset in range(1, max(SURVIVAL_HORIZONS) + 1):
        state = tracker.step(_frame(dataset, index + offset, key))
        if offset in SURVIVAL_HORIZONS:
            out[offset] = state.n_valid / n0
    return out, n0


def _stage_from(frame: np.ndarray, roi, tier) -> Stage | None:
    uv, desc = tier.teach(frame, roi)
    if len(uv) < MIN_TEAM:
        return None
    return Stage(
        name="m0",
        camera="bench",
        teams={"target": [Keypoint(uv=p, descriptor=d) for p, d in zip(uv, desc, strict=True)]},
        goal_relation=GoalRelation(held_uv=uv[:1]),
        travel_dir=[0.0, 0.0, -1.0],
        termination=Termination("pose_hold"),
        budget=Budget(),
    )


def bind_curve(dataset, key, start: int, designator, tier) -> list[tuple[int, bool, int, float]]:
    """Post: (offset, certified, inliers, inlier_ratio) per horizon; [] if untaught.

    The live frame is re-designated rather than reusing the taught mask: at runtime the
    binder is handed the region SAM3 finds NOW, and the object has moved by then. A live
    frame with no designation counts as a bind failure — at runtime it is one, since
    there is nothing to bind against.
    """
    taught = teach_frame(dataset, key, start, designator, tier)
    if taught is None:
        return []
    index, first, roi = taught
    stage = _stage_from(first, roi, tier)
    if stage is None:
        return []

    binder = tier.make_binder()
    rows = []
    for offset in BIND_OFFSETS:
        live = _frame(dataset, index + offset, key)
        live_roi = designator.mask(live)
        if live_roi is None:
            rows.append((offset, False, 0, 0.0))
            continue
        r = binder.bind(live, stage, "target", mask=live_roi)
        rows.append((offset, r.ok, r.n_inliers, r.inlier_ratio))
    return rows


def cross_episode(dataset, key, starts, lengths, designator, tier, phase: float = 0.15, limit: int = 6):
    """Bind one episode's constellation against OTHER episodes' scenes.

    The hardest honest case available without a rig: a genuinely re-placed scene, with
    whatever lighting and background drift the session carried.

    Every episode is entered at the SAME fraction of its length, because the episodes
    are repetitions of one task and the comparison is only meaningful between frames at
    the same point in it. Entering each episode at its first teachable frame instead
    compared a close-up of a grasped ring against a distant one still on the table, and
    scored the binder on a scale change the design never asks it to survive.

    Post: (episode, certified, inliers, inlier_ratio) per compared episode; ``certified``
    is None when that episode never designates (nothing was asked of the binder).
    """

    def enter(i):
        return teach_frame(dataset, key, starts[i] + int(phase * lengths[i]), designator, tier)

    taught = enter(0)
    if taught is None:
        return []
    _index, first, roi = taught
    stage = _stage_from(first, roi, tier)
    if stage is None:
        return []

    binder = tier.make_binder()
    rows = []
    for ep in range(1, min(limit, len(starts) - 1) + 1):
        live = enter(ep)
        if live is None:
            rows.append((ep, None, 0, 0.0))
            continue
        _i, live_frame, live_roi = live
        r = binder.bind(live_frame, stage, "target", mask=live_roi)
        rows.append((ep, r.ok, r.n_inliers, r.inlier_ratio))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--camera", default="top")
    ap.add_argument("--episodes", type=int, default=4, help="episodes to average survival/bind over")
    ap.add_argument(
        "--concept",
        default=None,
        help="SAM3 text designation, e.g. 'green ring' ('.'-separated for several). "
        "Without it the bench falls back to a centre-of-frame ROI whose numbers are "
        "confounded by static background and must not be reported as results.",
    )
    ap.add_argument("--resolution", type=int, default=None, help="SAM3 inference resolution")
    ap.add_argument("--binder", default="sift", choices=("sift", "dino"), help="descriptor tier")
    ap.add_argument(
        "--dino-model",
        default="facebook/dinov2-small",
        help="patch backbone for --binder dino, e.g. facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    ap.add_argument(
        "--phase",
        type=float,
        default=0.15,
        help="fraction into each episode at which the cross-episode comparison enters, so "
        "both sides sit at the same point in the task",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dataset = LeRobotDataset(args.repo_id)
    key = f"observation.images.{args.camera}"
    assert key in dataset.meta.camera_keys, f"{key} not in {list(dataset.meta.camera_keys)}"
    starts = _episode_starts(dataset)
    lengths = _episode_lengths(dataset, starts)
    designator = (
        Sam3Concept(args.concept, device=args.device, resolution=args.resolution)
        if args.concept
        else CentreRoi()
    )
    tier = DinoTier(args.dino_model, device=args.device) if args.binder == "dino" else SiftTier()
    print(
        f"{args.repo_id}  {dataset.meta.total_episodes} episodes  {dataset.meta.fps} fps  "
        f"camera={args.camera}\ndesignation: {designator.label}   binder: {tier.label}\n"
    )

    results = [survival(dataset, key, s, designator) for s in starts[: args.episodes]]
    surv = [s for s, _ in results if s]
    seeded = [n for s, n in results if s]
    print("TRACKER POINT SURVIVAL (fraction still believed after N frames)")
    print(f"  seeded:  {np.mean(seeded):.0f} points/team over {len(surv)}/{args.episodes} episodes")
    print("  frames:  " + "".join(f"{h:>8d}" for h in SURVIVAL_HORIZONS))
    print("  seconds: " + "".join(f"{h / dataset.meta.fps:>8.1f}" for h in SURVIVAL_HORIZONS))
    print("  survival:" + "".join(f"{np.mean([s[h] for s in surv]):>8.2f}" for h in SURVIVAL_HORIZONS))

    print("\nBIND INLIER RATE vs FRAMES SINCE TEACHING (same episode)")
    print(f"  {'offset':>7} {'sec':>6} {'certified':>10} {'inliers':>8} {'ratio':>7}")
    rows = [bind_curve(dataset, key, s, designator, tier) for s in starts[: args.episodes]]
    rows = [r for r in rows if r]
    for i, offset in enumerate(BIND_OFFSETS):
        ok = np.mean([r[i][1] for r in rows])
        inl = np.mean([r[i][2] for r in rows])
        ratio = np.mean([r[i][3] for r in rows])
        print(f"  {offset:>7d} {offset / dataset.meta.fps:>6.1f} {ok:>10.0%} {inl:>8.1f} {ratio:>7.2f}")

    print(f"\nBIND ACROSS EPISODES (episode 0 vs others, all entered at {args.phase:.0%} of the episode)")
    print(f"  {'episode':>7} {'certified':>10} {'inliers':>8} {'ratio':>7}")
    for ep, ok, inl, ratio in cross_episode(
        dataset, key, starts, lengths, designator, tier, phase=args.phase
    ):
        # "no-mask" is a designation failure, not a binder verdict — the binder was
        # never asked. Printed distinctly so it cannot be counted as either outcome.
        print(f"  {ep:>7d} {('no-mask' if ok is None else str(ok)):>10} {inl:>8d} {ratio:>7.2f}")

    misses = getattr(designator, "misses", 0)
    if misses:
        # A miss is a designation failure, not a tracker or binder failure. Kept separate
        # so it can never be read as either.
        print(f"\ndesignation found nothing on {misses} frame(s) — counted as bind failures")


if __name__ == "__main__":
    main()
