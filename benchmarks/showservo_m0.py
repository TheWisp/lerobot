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

Usage:
    uv run python benchmarks/showservo_m0.py --repo-id thewisp/<dataset> --camera top
"""

from __future__ import annotations

import argparse

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.showservo.binder import BindGate, SiftBinder, sift_keypoints
from lerobot.showservo.card import Budget, GoalRelation, Keypoint, Stage, Termination
from lerobot.showservo.tracker import KLTTracker

SURVIVAL_HORIZONS = (5, 10, 20, 40, 80, 150)
BIND_OFFSETS = (0, 5, 15, 30, 60, 120, 240)


def _frame(dataset: LeRobotDataset, index: int, key: str) -> np.ndarray:
    """Post: HxWx3 uint8 RGB."""
    img = dataset[index][key]
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _centre_roi(shape, frac: float = 0.55) -> np.ndarray:
    """A crude stand-in for SAM3 designation.

    Binding over the WHOLE frame would be dominated by static background and would
    report a flattering inlier rate that says nothing about the target. Restricting to
    the middle of the workspace is coarse, but it at least measures the region a card
    would actually be about. Replace with the real mask once designation is wired.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    dh, dw = int(h * frac / 2), int(w * frac / 2)
    mask[h // 2 - dh : h // 2 + dh, w // 2 - dw : w // 2 + dw] = True
    return mask


def _episode_starts(dataset: LeRobotDataset) -> list[int]:
    """Post: one start index per episode; falls back to uniform chunking."""
    for attr in ("episode_data_index",):
        idx = getattr(dataset, attr, None)
        if isinstance(idx, dict) and "from" in idx:
            return [int(v) for v in np.asarray(idx["from"]).ravel()]
    n_ep = max(int(dataset.meta.total_episodes), 1)
    per = max(int(dataset.meta.total_frames) // n_ep, 1)
    return [i * per for i in range(n_ep)]


def survival(dataset, key, start: int, n_points: int = 60) -> dict[int, float]:
    """Fraction of seeded points the tracker still believes, by horizon."""
    first = _frame(dataset, start, key)
    roi = _centre_roi(first.shape)
    uv, _ = sift_keypoints(first, roi, max_points=n_points)
    if len(uv) < 8:
        return {}

    tracker = KLTTracker()
    tracker.init(first, uv)
    out, n0 = {}, len(uv)
    for offset in range(1, max(SURVIVAL_HORIZONS) + 1):
        state = tracker.step(_frame(dataset, start + offset, key))
        if offset in SURVIVAL_HORIZONS:
            out[offset] = state.n_valid / n0
    return out


def _stage_from(frame: np.ndarray, roi, max_points: int = 80) -> Stage | None:
    uv, desc = sift_keypoints(frame, roi, max_points=max_points)
    if len(uv) < 8:
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


def bind_curve(dataset, key, start: int) -> list[tuple[int, bool, int, float]]:
    """Post: (offset, certified, inliers, inlier_ratio) per horizon."""
    first = _frame(dataset, start, key)
    roi = _centre_roi(first.shape)
    stage = _stage_from(first, roi)
    if stage is None:
        return []

    binder = SiftBinder(gate=BindGate())
    rows = []
    for offset in BIND_OFFSETS:
        live = _frame(dataset, start + offset, key)
        r = binder.bind(live, stage, "target", mask=roi)
        rows.append((offset, r.ok, r.n_inliers, r.inlier_ratio))
    return rows


def cross_episode(dataset, key, starts: list[int], limit: int = 6) -> list[tuple[int, bool, int, float]]:
    """Bind episode 0's constellation against other episodes' opening frames.

    The hardest honest case available without a rig: a genuinely re-placed scene, with
    whatever lighting and background drift the session carried.
    """
    first = _frame(dataset, starts[0], key)
    roi = _centre_roi(first.shape)
    stage = _stage_from(first, roi)
    if stage is None:
        return []

    binder = SiftBinder(gate=BindGate())
    rows = []
    for ep, s in enumerate(starts[1 : 1 + limit], start=1):
        r = binder.bind(_frame(dataset, s, key), stage, "target", mask=roi)
        rows.append((ep, r.ok, r.n_inliers, r.inlier_ratio))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--camera", default="top")
    ap.add_argument("--episodes", type=int, default=4, help="episodes to average survival/bind over")
    args = ap.parse_args()

    dataset = LeRobotDataset(args.repo_id)
    key = f"observation.images.{args.camera}"
    assert key in dataset.meta.camera_keys, f"{key} not in {list(dataset.meta.camera_keys)}"
    starts = _episode_starts(dataset)
    print(
        f"{args.repo_id}  {dataset.meta.total_episodes} episodes  {dataset.meta.fps} fps  camera={args.camera}\n"
    )

    surv = [survival(dataset, key, s) for s in starts[: args.episodes]]
    surv = [s for s in surv if s]
    print("TRACKER POINT SURVIVAL (fraction still believed after N frames)")
    print("  frames:  " + "".join(f"{h:>8d}" for h in SURVIVAL_HORIZONS))
    print("  seconds: " + "".join(f"{h / dataset.meta.fps:>8.1f}" for h in SURVIVAL_HORIZONS))
    print("  survival:" + "".join(f"{np.mean([s[h] for s in surv]):>8.2f}" for h in SURVIVAL_HORIZONS))

    print("\nBIND INLIER RATE vs FRAMES SINCE TEACHING (same episode)")
    print(f"  {'offset':>7} {'sec':>6} {'certified':>10} {'inliers':>8} {'ratio':>7}")
    rows = [bind_curve(dataset, key, s) for s in starts[: args.episodes]]
    rows = [r for r in rows if r]
    for i, offset in enumerate(BIND_OFFSETS):
        ok = np.mean([r[i][1] for r in rows])
        inl = np.mean([r[i][2] for r in rows])
        ratio = np.mean([r[i][3] for r in rows])
        print(f"  {offset:>7d} {offset / dataset.meta.fps:>6.1f} {ok:>10.0%} {inl:>8.1f} {ratio:>7.2f}")

    print("\nBIND ACROSS EPISODES (episode 0's constellation vs other episodes' first frame)")
    print(f"  {'episode':>7} {'certified':>10} {'inliers':>8} {'ratio':>7}")
    for ep, ok, inl, ratio in cross_episode(dataset, key, starts):
        print(f"  {ep:>7d} {str(ok):>10} {inl:>8d} {ratio:>7.2f}")


if __name__ == "__main__":
    main()
