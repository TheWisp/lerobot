"""Summarize saved FLASH RGB-D evidence without treating pose availability as accuracy.

This reads the same per-frame journal used by the replay UI. The source videos
and masks remain in the local LeRobot cache, outside git.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarize(root: Path, evidence_name: str = "observation-evidence") -> dict:
    flash = root / "flash" / "v1"
    source = json.loads((flash / "source.json").read_text())
    evidence_root = flash / evidence_name / "episode_000000"
    metadata = json.loads((evidence_root / "observation.json").read_text())
    entries = [
        json.loads(gzip.decompress(path.read_bytes())) for path in sorted(evidence_root.glob("*.json.gz"))
    ]
    if len(entries) != metadata["frames"]:
        raise ValueError(f"{root}: incomplete evidence journal")
    dataset = LeRobotDataset(source["repo_id"], root=root, video_backend="pyav")
    roles = metadata["roles"]
    stats = {
        role: {
            "state": Counter(),
            "estimator": Counter(),
            "failure": Counter(),
            "mask_frames": 0,
            "estimated_mask_frames": 0,
            "mask_iou": [],
            "depth_fractions": [],
            "lost_streak": 0,
            "longest_lost_streak": 0,
            "first_lost_frame": None,
            "first_observed_frame": None,
            "initial_reference": None,
            "lost_with_raw_candidate": 0,
            "lost_raw_depth_p90_mm": [],
            "observed_depth_p90_mm": [],
        }
        for role in roles
    }
    for frame_id, entry in enumerate(entries):
        if entry["frame_index"] != frame_id:
            raise ValueError(f"{root}: evidence frame order changed")
        row = dataset[metadata["source_row_indices"][frame_id]]
        depth = row[source["depth_feature"]].numpy().squeeze()
        for role in roles:
            item = stats[role]
            mask_path = flash / "imported-masks" / role / f"{frame_id:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != depth.shape:
                raise ValueError(f"{root}: missing or misaligned mask {mask_path}")
            visible = bool(np.any(mask))
            if not visible:
                item["lost_streak"] = 0
                continue
            item["mask_frames"] += 1
            estimated_mask = np.zeros(mask.shape, dtype=bool)
            for start, count in entry["masks_rle"].get(role, []):
                estimated_mask.flat[start : start + count] = True
            item["mask_iou"].append(
                float(
                    np.count_nonzero(estimated_mask & (mask > 0))
                    / np.count_nonzero(estimated_mask | (mask > 0))
                )
            )
            if np.any(estimated_mask):
                item["estimated_mask_frames"] += 1
                readings = depth[estimated_mask]
                item["depth_fractions"].append(float(np.mean(np.isfinite(readings) & (readings > 0))))
            estimate = entry["estimates"][role]
            proposal = entry["proposals"][role]
            state = estimate["evidence"]
            item["state"][state] += 1
            item["estimator"][estimate["estimator"]] += 1
            if state == "observed" and item["first_observed_frame"] is None:
                support = proposal.get("visible_support") or {}
                item["first_observed_frame"] = frame_id
                item["initial_reference"] = {
                    "raw_mask_depth_p90_mm": (
                        1000 * float(support["depth_residual_p90_m"])
                        if support.get("depth_residual_p90_m") is not None
                        else None
                    ),
                    "mask_pixels": support.get("mask_pixels"),
                    "filtered_depth_points": support.get("depth_points"),
                }
            if state == "observed":
                support = proposal.get("visible_support") or {}
                residual = support.get("depth_residual_p90_m")
                if residual is not None:
                    item["observed_depth_p90_mm"].append(1000 * float(residual))
            if state == "lost":
                item["lost_streak"] += 1
                item["longest_lost_streak"] = max(item["lost_streak"], item["longest_lost_streak"])
                if item["first_lost_frame"] is None:
                    item["first_lost_frame"] = frame_id
                item["failure"][proposal.get("failure") or "unspecified"] += 1
                if proposal.get("raw_pose") is not None:
                    item["lost_with_raw_candidate"] += 1
                    support = proposal.get("visible_support") or {}
                    distance = support.get("depth_residual_p90_m")
                    if distance is not None:
                        item["lost_raw_depth_p90_mm"].append(1000 * float(distance))
            else:
                item["lost_streak"] = 0
    output_roles = {}
    for role, item in stats.items():
        visible = item["mask_frames"]
        output_roles[role] = {
            "mask_visible_frames": visible,
            "estimated_mask_present_frames_when_source_visible": item["estimated_mask_frames"],
            "mask_iou_median_vs_supplied": _median(item["mask_iou"]),
            "observed": item["state"]["observed"],
            "predicted": item["state"]["predicted"],
            "lost": item["state"]["lost"],
            "available_fraction_when_mask_visible": (
                (item["state"]["observed"] + item["state"]["predicted"]) / visible if visible else None
            ),
            "depth_valid_fraction_median_within_mask": _median(item["depth_fractions"]),
            "longest_consecutive_lost_while_visible": item["longest_lost_streak"],
            "first_lost_while_visible": item["first_lost_frame"],
            "first_observed_frame": item["first_observed_frame"],
            "initial_reference": item["initial_reference"],
            "lost_with_raw_pose_candidate": item["lost_with_raw_candidate"],
            "lost_candidate_raw_mask_depth_p90_mm_median": _median(item["lost_raw_depth_p90_mm"]),
            "accepted_surface_depth_p90_mm_median": _median(item["observed_depth_p90_mm"]),
            "estimators": dict(item["estimator"]),
            "rejection_reasons": dict(item["failure"]),
        }
    return {
        "dataset": str(root),
        "source": source["source"],
        "frames": len(entries),
        "mask_source": metadata["mask_source"],
        "evidence_name": evidence_name,
        "evidence_identity": metadata["evidence_identity"],
        "config_bundle": metadata["config_bundle"],
        "pose_accuracy_ground_truth_available": False,
        "roles": output_roles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-name", default="observation-evidence")
    args = parser.parse_args()
    results = [summarize(root, args.evidence_name) for root in args.datasets]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
