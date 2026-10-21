"""Score several causal pose traces against the same current RGB-D observation.

Each trace was generated causally from frames no later than the scored frame.
Sources can share masks, RGB-D or initial anchors and are not independent votes.
The current mask is used only as same-frame evidence. Pairwise Pareto results
are conflicts or preferences, not independently labeled pose truth.

Example:
    python analyze_pose_hypothesis_bank.py --dataset /path/to/recording \
        --masks /path/to/sam3_masks --role dowel --first-frame 300 \
        --last-frame 500 --candidate sealed=p2p_dowel_internal_control_100_550.npz \
        --candidate initial=p2p_dowel_sam3_initial_100_550.npz \
        --candidate perframe=p2p_dowel_sam3_100_550.npz --output /tmp/bank.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import frame_points, move, surface_p90_mm
from analyze_ray_consistency import reference_points, score_rays

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import free_space_support, surface_support


@dataclass(frozen=True)
class Candidate:
    name: str
    frames: np.ndarray
    poses: np.ndarray
    lost: np.ndarray

    def pose_at(self, frame: int) -> np.ndarray | None:
        index = frame - int(self.frames[0])
        if index < 0 or index >= len(self.frames) or int(self.frames[index]) != frame or self.lost[index]:
            return None
        return self.poses[index]


def load_candidate(specification: str) -> Candidate:
    name, separator, source = specification.partition("=")
    if not separator or not name or not source:
        raise ValueError("candidate must be NAME=PATH[:POSES_KEY:LOST_KEY]")
    pieces = source.rsplit(":", 2)
    if len(pieces) == 1:
        path, pose_key, lost_key = pieces[0], "poses", "lost"
    elif len(pieces) == 3:
        path, pose_key, lost_key = pieces
    else:
        raise ValueError("candidate must be NAME=PATH[:POSES_KEY:LOST_KEY]")
    with np.load(path) as trace:
        return Candidate(name, trace["frames"].copy(), trace[pose_key].copy(), trace[lost_key].copy())


def pareto(a: dict[str, float], b: dict[str, float], *, ray_evidence: bool) -> str:
    """Compare same-frame evidence without weighting unrelated units."""
    fields = (
        (
            ("surface_p90_mm", False),
            ("ray_support_fraction", True),
            ("ray_contradiction_fraction", False),
        )
        if ray_evidence
        else (("surface_p90_mm", False), ("projected_mask_fraction", True))
    )
    values = [value for field, _ in fields for value in (a[field], b[field])]
    if not np.isfinite(values).all():
        return "unscored"
    a_better = all(a[field] >= b[field] if higher else a[field] <= b[field] for field, higher in fields)
    b_better = all(b[field] >= a[field] if higher else b[field] <= a[field] for field, higher in fields)
    if a_better and not b_better:
        return "first_favored"
    if b_better and not a_better:
        return "second_favored"
    return "tradeoff"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--candidate", required=True, action="append")
    parser.add_argument("--first-frame", required=True, type=int)
    parser.add_argument("--last-frame", required=True, type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[3] / "src/lerobot/policies/flash/docs/trial_config.example.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidates = [load_candidate(specification) for specification in args.candidate]
    if len(candidates) < 2 or len({candidate.name for candidate in candidates}) != len(candidates):
        parser.error("supply at least two candidates with distinct names")
    if args.last_frame < args.first_frame:
        parser.error("last frame must follow first frame")
    config = json.loads(args.config.read_text())["causal_tracking"]
    fit_threshold_m = float(config["fit_threshold_m"])
    disagreement_m = float(config["max_pose_disagreement_translation_m"])
    min_points = int(config["min_inliers"])
    intrinsics = np.load(Path(__file__).with_name("pose_traces.npz"))["intrinsics"]
    reference = reference_points(args.dataset, args.masks, args.role, intrinsics)
    rows = []
    summaries: Counter[str] = Counter()
    for frame_id in range(args.first_frame, args.last_frame + 1):
        raw_depth = cv2.imread(str(args.dataset / "depth" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        raw_mask = cv2.imread(str(args.masks / args.role / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if raw_depth is None or raw_mask is None or raw_depth.shape != raw_mask.shape:
            raise ValueError(f"missing aligned depth/mask at frame {frame_id}")
        mask = raw_mask > 0
        depth_m = raw_depth.astype(np.float32) * 0.001
        neighborhood = np.ones((3, 3), dtype=np.uint8)
        lower = cv2.erode(depth_m, neighborhood)
        upper = cv2.dilate(depth_m, neighborhood)
        valid_depth = (depth_m > 0) & (lower > 0) & (upper - lower <= disagreement_m)
        live, live_count = frame_points(
            args.dataset,
            args.masks,
            args.role,
            frame_id,
            intrinsics,
            edge_limit_m=disagreement_m,
        )
        rgbd = RGBDFrame(
            frame_id,
            np.zeros((*raw_depth.shape, 3), dtype=np.uint8),
            raw_depth,
            intrinsics,
            0.001,
            np.eye(4),
            frame_id / 30,
            frame_id / 30,
            "recorded-camera",
        )
        scored = {}
        for candidate in candidates:
            pose = candidate.pose_at(frame_id)
            if pose is None:
                rows.append((frame_id, candidate.name, "lost", live_count, *([None] * 10)))
                continue
            surface = surface_support(rgbd, mask, reference, pose)
            free = free_space_support(
                rgbd,
                reference,
                pose,
                depth_tolerance_m=fit_threshold_m,
                max_local_depth_span_m=disagreement_m,
            )
            _, rays, supported, contradicted, _, _ = score_rays(
                reference,
                pose,
                depth_m,
                valid_depth,
                mask,
                intrinsics,
                depth_tolerance_m=fit_threshold_m,
            )
            metrics = {
                "surface_p90_mm": surface_p90_mm(live, move(reference, pose)),
                "projected_mask_fraction": surface.projected_point_mask_fraction,
                "projected_conditional_p90_mm": surface.depth_residual_p90_m * 1000,
                "signed_gap_mm": free.median_signed_gap_m * 1000
                if free.median_signed_gap_m is not None
                else None,
                "free_space_fraction": free.contradiction_fraction,
                "ray_support_fraction": supported / rays if rays else float("nan"),
                "ray_contradiction_fraction": contradicted / rays if rays else float("nan"),
            }
            status = "insufficient_depth" if len(live) < min_points else "scored"
            if (
                free.valid_depth_pixels >= min_points
                and free.median_signed_gap_m is not None
                and free.median_signed_gap_m > disagreement_m
            ):
                status = "free_space_contradicted"
            scored[candidate.name] = metrics if status != "insufficient_depth" else None
            rows.append(
                (
                    frame_id,
                    candidate.name,
                    status,
                    live_count,
                    metrics["surface_p90_mm"],
                    metrics["projected_mask_fraction"],
                    metrics["projected_conditional_p90_mm"],
                    free.projected_pixels,
                    free.valid_depth_pixels,
                    metrics["signed_gap_mm"],
                    metrics["free_space_fraction"],
                    surface.masked_depth_points,
                    metrics["ray_support_fraction"],
                    metrics["ray_contradiction_fraction"],
                )
            )
        for i, first in enumerate(candidates):
            for second in candidates[i + 1 :]:
                a, b = scored.get(first.name), scored.get(second.name)
                for evidence in ("mask", "ray"):
                    verdict = (
                        pareto(a, b, ray_evidence=evidence == "ray")
                        if a is not None and b is not None
                        else "unavailable"
                    )
                    summaries[f"{evidence}:{first.name}/{second.name}:{verdict}"] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "candidate",
                "status",
                "observed_depth_pixels",
                "observed_surface_p90_mm",
                "projected_mask_fraction",
                "projected_conditional_depth_p90_mm",
                "projected_pixels",
                "valid_projected_depth_pixels",
                "median_signed_gap_mm",
                "free_space_fraction",
                "projected_masked_depth_points",
                "ray_support_fraction",
                "ray_contradiction_fraction",
            )
        )
        writer.writerows(rows)
    for name, count in sorted(summaries.items()):
        print(name, count)


if __name__ == "__main__":
    main()
