"""Audit geometry-goal transfer over an ordered FLASH skill, without robot motion.

The two target registrations use clear demo views, while each held-goal surface
comes from its own demo sample.  A later keyframe is never evaluated as an
action from the reset scene: between keyframes the actual live observation and
gripper attachment would be needed.  This script also measures whether adjacent
demo held surfaces support the motion recorded by the robot encoders.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from flash_geometry_goal import candidates, fit_metrics, measured_points, read_depth, read_mask, sample
from flash_rgbd_registration import apply, projection_evidence, trimmed_icp
from scipy.spatial.transform import Rotation


def transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def surface_path(mask_root: Path, frame: int) -> Path:
    return mask_root / f"{frame:06}.png"


def sampled_surface(
    depth_path: Path,
    mask_path: Path,
    intrinsics: np.ndarray,
    depth_scale: float,
    interior_px: int,
    max_points: int,
) -> np.ndarray | None:
    mask = read_mask(mask_path)
    if not np.any(mask):
        return None
    try:
        return sample(
            measured_points(read_depth(depth_path), mask, intrinsics, depth_scale, interior_px),
            max_points,
        )
    except ValueError:
        return None


def camera_motion_from_ee(before: dict, after: dict, base_from_camera: np.ndarray) -> np.ndarray:
    base_from_ee_before = np.asarray(before["demo_ee"], dtype=np.float64)
    base_from_ee_after = np.asarray(after["demo_ee"], dtype=np.float64)
    camera_from_base = np.linalg.inv(base_from_camera)
    return camera_from_base @ base_from_ee_after @ np.linalg.inv(base_from_ee_before) @ base_from_camera


def register_reference(
    demo_cloud: np.ndarray,
    live_cloud: np.ndarray,
    live_mask: np.ndarray,
    live_depth: np.ndarray,
    live_k: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    fits = candidates(
        demo_cloud,
        live_cloud,
        radius_m=args.diagnostic_radius_m,
        iterations=args.icp_iterations,
        trim_fraction=args.trim_fraction,
    )
    for fit in fits:
        evidence, _, _ = projection_evidence(
            apply(demo_cloud, np.asarray(fit["rotation"]), np.asarray(fit["translation_m"])),
            live_mask,
            live_depth,
            live_k,
            args.diagnostic_radius_m,
        )
        fit["pixel_evidence"] = evidence
    fits.sort(
        key=lambda fit: (
            -fit["balanced_coverage"],
            -fit["pixel_evidence"]["projection_mask_and_depth_fraction"],
            fit["pixel_evidence"]["projection_front_of_depth_fraction"],
        )
    )
    return {"best": fits[0], "alternatives": fits[1 : args.report_candidates]}


def run(args: argparse.Namespace) -> dict:
    skill = json.loads((args.prepared_skill / "skill.json").read_text())
    annotation = json.loads((args.prepared_skill / "annotation.json").read_text())
    held_role, target_role = annotation["held_role"], annotation["target_role"]
    rows = [json.loads(line) for line in (args.live_trace / "frames.jsonl").read_text().splitlines()]
    live = rows[args.live_index]
    live_frame = live["frame_id"]
    live_k = np.asarray(live["intrinsics"], dtype=np.float64)
    demo_k = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
    calibration = json.loads(Path(annotation["calibration_path"]).read_text())
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=np.float64)
    live_depth = read_depth(args.live_trace / "depth" / f"{live_frame:06}.png")
    episode = pq.read_table(args.demo_data_parquet, columns=["frame_index", "tip.pose"])
    frame_numbers = episode["frame_index"].to_pylist()
    poses = episode["tip.pose"].to_pylist()
    ee_by_frame = {
        int(frame): {"demo_ee": np.asarray(pose, dtype=np.float64).reshape(4, 4)}
        for frame, pose in zip(frame_numbers, poses, strict=True)
    }

    @lru_cache(maxsize=160)
    def demo_surface(role: str, frame: int) -> np.ndarray | None:
        return sampled_surface(
            args.demo_recording / "depth" / f"{frame:06}.png",
            surface_path(args.demo_held_masks if role == held_role else args.demo_target_masks, frame),
            demo_k,
            args.demo_depth_scale,
            args.mask_interior_px,
            args.max_points,
        )

    live_sources = {}
    registrations = {}
    for role, live_role in ((held_role, args.live_held_role), (target_role, args.live_target_role)):
        mask = read_mask(args.live_trace / "masks" / live_role / f"{live_frame:06}.png")
        live_cloud = sampled_surface(
            args.live_trace / "depth" / f"{live_frame:06}.png",
            args.live_trace / "masks" / live_role / f"{live_frame:06}.png",
            live_k,
            live["depth_m_per_unit"],
            args.mask_interior_px,
            args.max_points,
        )
        demo_cloud = demo_surface(role, args.demo_reference_frame)
        if live_cloud is None or demo_cloud is None:
            raise ValueError(f"{role} has no measured reference surface")
        live_sources[role] = live_cloud
        registrations[role] = register_reference(demo_cloud, live_cloud, mask, live_depth, live_k, args)

    intervals = []
    held_pairs = []
    first_held_frame = next(
        interval["samples"][0]["source_frame_index"]
        for interval in skill["intervals"]
        if interval["relation"] == "held_to_target"
    )
    first_held_surface = demo_surface(held_role, first_held_frame)
    cumulative_surface_motion = np.eye(4)
    cumulative_valid = first_held_surface is not None
    for index, interval in enumerate(skill["intervals"]):
        samples = interval["samples"]
        reference_role = interval["reference_role"]
        mapped_role = reference_role if reference_role in registrations else None
        selected = samples if interval["mode"] == "demonstrated" else [samples[-1]]
        item = {
            "index": index,
            "name": interval["name"],
            "mode": interval["mode"],
            "relation": interval["relation"],
            "reference_role": reference_role,
            "held_role": interval["held_role"],
            "start_frame": samples[0]["source_frame_index"],
            "end_frame": samples[-1]["source_frame_index"],
            "ordered_sample_count": len(samples),
            "evaluated_sample_count": len(selected),
            "endpoint_frame": selected[-1]["source_frame_index"],
            "held_measured_points_at_endpoint": len(endpoint_surface)
            if (endpoint_surface := demo_surface(held_role, selected[-1]["source_frame_index"])) is not None
            else 0,
        }
        if mapped_role is not None:
            match = registrations[mapped_role]["best"]
            cam_live_from_cam_demo = transform_matrix(
                np.asarray(match["rotation"]), np.asarray(match["translation_m"])
            )
            sample = selected[-1]
            # This maps the demonstrated gripper goal through the observed
            # reference surface, without assigning a canonical object frame.
            base_from_ee_goal = (
                base_from_camera
                @ cam_live_from_cam_demo
                @ np.linalg.inv(base_from_camera)
                @ np.asarray(sample["demo_ee"])
            )
            item["endpoint_ee_goal_base"] = base_from_ee_goal.tolist()
        if interval["relation"] == "held_to_target":
            counts = [
                len(surf)
                if (surf := demo_surface(held_role, sample["source_frame_index"])) is not None
                else 0
                for sample in selected
            ]
            item["held_visible_points_at_samples"] = {
                "minimum": min(counts),
                "median": float(np.median(counts)),
                "maximum": max(counts),
            }
            # The connection intervals retain only their endpoint keyframes
            # in the skill, but the underlying demo has intervening RGB-D and
            # encoder frames. Check those one by one; a direct keyframe jump
            # would confuse appearance change with tracking failure.
            interval_frames = [
                frame for frame in range(item["start_frame"], item["end_frame"] + 1) if frame in ee_by_frame
            ]
            for before_frame, after_frame in zip(interval_frames, interval_frames[1:], strict=False):
                first = demo_surface(held_role, before_frame)
                second = demo_surface(held_role, after_frame)
                pair = {
                    "interval": index,
                    "from_frame": before_frame,
                    "to_frame": after_frame,
                }
                if first is None or second is None:
                    pair["status"] = "no_measured_surface"
                    cumulative_valid = False
                else:
                    prior = camera_motion_from_ee(
                        ee_by_frame[before_frame], ee_by_frame[after_frame], base_from_camera
                    )
                    expected = apply(first, prior[:3, :3], prior[:3, 3])
                    pair["encoder_motion_surface_fit"] = fit_metrics(
                        first, second, expected, args.diagnostic_radius_m
                    )
                    refined_r, refined_t = trimmed_icp(
                        first,
                        second,
                        prior[:3, :3],
                        prior[:3, 3],
                        iterations=args.icp_iterations,
                        trim_fraction=args.trim_fraction,
                    )
                    refined = apply(first, refined_r, refined_t)
                    pair["refined_surface_fit"] = fit_metrics(
                        first, second, refined, args.diagnostic_radius_m
                    )
                    correction = transform_matrix(refined_r, refined_t) @ np.linalg.inv(prior)
                    # A rigid transform's translation depends on the chosen
                    # origin. Measure disagreement at the actual object points.
                    pair["correction_from_encoder_mm"] = float(
                        np.sqrt(np.mean(np.sum((refined - expected) ** 2, axis=1))) * 1000
                    )
                    pair["correction_from_encoder_deg"] = float(
                        np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude())
                    )
                    if cumulative_valid and first_held_surface is not None:
                        cumulative_surface_motion = (
                            transform_matrix(refined_r, refined_t) @ cumulative_surface_motion
                        )
                        whole_encoder_motion = camera_motion_from_ee(
                            ee_by_frame[first_held_frame], ee_by_frame[after_frame], base_from_camera
                        )
                        chained = apply(
                            first_held_surface,
                            cumulative_surface_motion[:3, :3],
                            cumulative_surface_motion[:3, 3],
                        )
                        encoder = apply(
                            first_held_surface,
                            whole_encoder_motion[:3, :3],
                            whole_encoder_motion[:3, 3],
                        )
                        pair["cumulative_error_at_held_surface_mm"] = float(
                            np.sqrt(np.mean(np.sum((chained - encoder) ** 2, axis=1))) * 1000
                        )
                        error_rotation = cumulative_surface_motion[:3, :3] @ whole_encoder_motion[:3, :3].T
                        pair["cumulative_rotation_error_deg"] = float(
                            np.degrees(Rotation.from_matrix(error_rotation).magnitude())
                        )
                    pair["status"] = "measured"
                held_pairs.append(pair)
        intervals.append(item)
    result = {
        "purpose": "ordered geometry transfer audit; no robot command and no skipped live stage",
        "live_frame_id": live_frame,
        "demo_reference_frame": args.demo_reference_frame,
        "roles": {"held": held_role, "target": target_role},
        "reference_registrations": registrations,
        "intervals": intervals,
        "held_adjacent_demo_pairs": held_pairs,
        "limitation": "Later live goals cannot be evaluated from the reset frame; each needs observation after the preceding action.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-skill", type=Path, required=True)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-data-parquet", type=Path, required=True)
    parser.add_argument("--demo-held-masks", type=Path, required=True)
    parser.add_argument("--demo-target-masks", type=Path, required=True)
    parser.add_argument("--demo-reference-frame", type=int, required=True)
    parser.add_argument("--live-trace", type=Path, required=True)
    parser.add_argument("--live-index", type=int, default=0)
    parser.add_argument("--live-held-role", required=True)
    parser.add_argument("--live-target-role", required=True)
    parser.add_argument("--demo-depth-scale", type=float, default=0.001)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--mask-interior-px", type=int, default=2)
    parser.add_argument("--diagnostic-radius-m", type=float, default=0.006)
    parser.add_argument("--icp-iterations", type=int, default=15)
    parser.add_argument("--trim-fraction", type=float, default=0.8)
    parser.add_argument("--report-candidates", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    measured = [p for p in result["held_adjacent_demo_pairs"] if p["status"] == "measured"]
    print(
        json.dumps(
            {
                "intervals": len(result["intervals"]),
                "ordered_samples": sum(i["ordered_sample_count"] for i in result["intervals"]),
                "adjacent_held_pairs": len(measured),
                "held_pair_median_encoder_coverage": float(
                    np.median([p["encoder_motion_surface_fit"]["balanced_coverage"] for p in measured])
                )
                if measured
                else None,
                "held_pair_median_correction_mm": float(
                    np.median([p["correction_from_encoder_mm"] for p in measured])
                )
                if measured
                else None,
                "result": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
