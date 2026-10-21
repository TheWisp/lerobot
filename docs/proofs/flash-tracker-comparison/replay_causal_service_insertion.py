"""Replay the production causal scene estimator on recorded insertion RGB-D.

Direct Point2Pose/BundleSDF poses are source-frame candidates. The pose worker
may skip source frames and return after a configured number of camera frames;
the causal point/scene workers still consume every frame. This is not the
current live appearance backend or an independent pose ground-truth test.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from run_causal_points_demo import read_frame
from simulate_existing_surface_policy import reference_points

from lerobot.policies.flash.causal_service import CausalPoseService, CausalTrackingConfig, _Anchor
from lerobot.policies.flash.perception import LatestPoseStore
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anchor-delay-frames", type=int, default=0)
    parser.add_argument("--proposal-stride", type=int, default=1)
    args = parser.parse_args()
    if args.anchor_delay_frames < 0 or args.proposal_stride < 1:
        parser.error("delay must be nonnegative and proposal stride must be positive")
    root = Path(__file__).parent
    traces = np.load(root / "pose_traces.npz")
    p2p = np.load(root / "p2p_dowel_internal_control_100_550.npz")
    tracking = CausalTrackingConfig.from_config(json.loads(args.config.read_text())["causal_tracking"])
    models = {
        role: reference_points(args.dataset, args.masks, role, traces["intrinsics"])
        for role in ("dowel", "ring")
    }
    store = LatestPoseStore()
    service = CausalPoseService(
        store,
        **asdict(tracking),
        object_points_by_role=models,
        object_centers_by_role={role: points.mean(axis=0) for role, points in models.items()},
    )
    service.set_attached_role("dowel")
    delayed: deque[tuple[int, _Anchor]] = deque()
    rows = []
    decision_rows = []

    def deliver(anchor: _Anchor) -> float:
        before = {
            role: sum(service.direct_point_disagreements_by_role.get(role, {}).values())
            for role in ("dowel", "ring")
        }
        started = time.perf_counter()
        service._anchor(anchor)
        elapsed_ms = (time.perf_counter() - started) * 1000
        while service.pending:
            role = next(
                (name for name in service.pending if name == "dowel" or name not in service.relationships),
                None,
            )
            if role is None:
                break
            service._replay_one(role)
        for role in ("dowel", "ring"):
            after = sum(service.direct_point_disagreements_by_role.get(role, {}).values())
            if after > before[role]:
                decision_rows.append(
                    (anchor.frame_id, role, service.last_direct_point_comparison[role].verdict.value)
                )
        return elapsed_ms

    for frame_id in range(430, 505):
        read_started = time.perf_counter()
        frame = read_frame(args.dataset, frame_id, traces["intrinsics"])
        read_ms = (time.perf_counter() - read_started) * 1000
        started = time.perf_counter()
        service._frame(frame)
        frame_ms = (time.perf_counter() - started) * 1000
        estimates = []
        masks = {}
        if (frame_id - 430) % args.proposal_stride == 0:
            for role in ("dowel", "ring"):
                lost = (
                    p2p["lost"][frame_id - 100]
                    if role == "dowel"
                    else traces["bundlesdf_ring_failed"][frame_id - 100]
                )
                if lost:
                    continue
                pose = (
                    p2p["poses"][frame_id - 100]
                    if role == "dowel"
                    else traces["bundlesdf_ring_delta"][frame_id - 100]
                )
                estimates.append(
                    PoseEstimate(
                        role,
                        pose,
                        frame.source_time_s,
                        time.perf_counter(),
                        Evidence.OBSERVED,
                        f"{role}-v1",
                        (True,) * 3,
                        (True,) * 3,
                        estimator="offline_direct",
                    )
                )
                mask = cv2.imread(str(args.masks / role / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"missing {role} mask at frame {frame_id}")
                masks[role] = mask > 0
            delayed.append(
                (
                    frame_id + args.anchor_delay_frames,
                    _Anchor(
                        frame_id, tuple(estimates), masks, frame.source_time_s, frame.calibration_version
                    ),
                )
            )
        anchor_ms = 0.0
        while delayed and delayed[0][0] <= frame_id:
            anchor_ms += deliver(delayed.popleft()[1])
        accepted = store.snapshot()[1]
        rows.append(
            (
                frame_id,
                read_ms,
                frame_ms,
                anchor_ms,
                *(accepted[role].estimator if role in accepted else "missing" for role in ("dowel", "ring")),
                *(
                    accepted[role].evidence.value if role in accepted else "missing"
                    for role in ("dowel", "ring")
                ),
            )
        )
    while delayed:
        deliver(delayed.popleft()[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "camera_frame",
                "disk_read_ms",
                "causal_frame_ms",
                "delivered_anchor_ms",
                "dowel_estimator",
                "ring_estimator",
                "dowel_evidence",
                "ring_evidence",
            )
        )
        writer.writerows(rows)
    with args.output.with_name(args.output.stem + "_decisions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("source_frame", "role", "direct_point_verdict"))
        writer.writerows(decision_rows)
    for label, index in (("causal frame", 2), ("delivered anchor", 3)):
        samples = np.array([row[index] for row in rows])
        print(
            label,
            "median/p95 ms",
            round(float(np.median(samples)), 2),
            round(float(np.quantile(samples, 0.95)), 2),
        )
    for role in ("dowel", "ring"):
        print(role, "outcomes", service.anchor_outcomes_by_role.get(role, {}))
        print(role, "point comparison", service.direct_point_disagreements_by_role.get(role, {}))
    print("verified groups", sorted(service.relationships))


if __name__ == "__main__":
    main()
