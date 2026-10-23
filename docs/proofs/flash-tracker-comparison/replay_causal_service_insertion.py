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
    parser.add_argument(
        "--dowel-trace", type=Path, default=Path(__file__).with_name("p2p_dowel_internal_control_100_550.npz")
    )
    parser.add_argument("--first-frame", type=int, default=430)
    parser.add_argument("--last-frame", type=int, default=504)
    parser.add_argument("--attach-frame", type=int, default=430)
    parser.add_argument("--anchor-delay-frames", type=int, default=0)
    parser.add_argument("--proposal-stride", type=int, default=1)
    args = parser.parse_args()
    if args.anchor_delay_frames < 0 or args.proposal_stride < 1:
        parser.error("delay must be nonnegative and proposal stride must be positive")
    if not 100 <= args.first_frame <= args.last_frame <= 550:
        parser.error("requested frames must lie in the saved pose traces")
    root = Path(__file__).parent
    traces = np.load(root / "pose_traces.npz")
    p2p = np.load(args.dowel_trace)
    if not np.array_equal(p2p["frames"], traces["frames"]):
        raise ValueError("dowel direct trace must cover the recorded frames")
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
    service.set_attached_role("dowel" if args.attach_frame <= args.first_frame else None)
    delayed: deque[tuple[int, _Anchor]] = deque()
    rows = []
    decision_rows = []
    support_rows = []
    pose_rows: dict[str, list[np.ndarray]] = {"dowel": [], "ring": []}

    def deliver(anchor: _Anchor) -> float:
        for estimate in anchor.estimates:
            active = service.active.get(estimate.role)
            recorded = next(
                (
                    (motion, pixels)
                    for frame_id, motion, pixels in service.motion_history.get(estimate.role, ())
                    if frame_id == anchor.frame_id
                ),
                None,
            )
            mask = anchor.masks.get(estimate.role)
            overlapping = (
                int(mask[recorded[1][:, 1], recorded[1][:, 0]].sum())
                if recorded is not None and mask is not None
                else 0
            )
            support_rows.append(
                (
                    anchor.frame_id,
                    estimate.role,
                    active is not None,
                    recorded is not None,
                    recorded is not None and recorded[0].base_motion_from_seed is not None,
                    len(recorded[1]) if recorded is not None else 0,
                    overlapping,
                    active[0].min_inliers if active is not None else 0,
                )
            )
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

    for frame_id in range(args.first_frame, args.last_frame + 1):
        if frame_id == args.attach_frame:
            service.set_attached_role("dowel")
        read_started = time.perf_counter()
        frame = read_frame(args.dataset, frame_id, traces["intrinsics"])
        read_ms = (time.perf_counter() - read_started) * 1000
        started = time.perf_counter()
        service._frame(frame)
        frame_ms = (time.perf_counter() - started) * 1000
        estimates = []
        masks = {}
        if (frame_id - args.first_frame) % args.proposal_stride == 0:
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
        for role in pose_rows:
            estimate = accepted.get(role)
            pose_rows[role].append(
                np.asarray(estimate.pose).copy()
                if estimate is not None and estimate.pose is not None
                else np.full((4, 4), np.nan)
            )
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
    with args.output.with_name(args.output.stem + "_point_support.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "source_frame",
                "role",
                "active_track",
                "source_motion_recorded",
                "source_motion_fit",
                "tracked_pixels",
                "pixels_in_source_mask",
                "required_inliers",
            )
        )
        writer.writerows(support_rows)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        frames=np.asarray([row[0] for row in rows]),
        **{f"{role}_poses": np.stack(poses) for role, poses in pose_rows.items()},
    )
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
        print(role, "direct/depth diagnostic", service.direct_free_space_counts_by_role.get(role, {}))
    print("verified groups", sorted(service.relationships))


if __name__ == "__main__":
    main()
