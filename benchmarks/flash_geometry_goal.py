"""Offline test of a demo surface goal against a live RGB-D tracking trace.

No robot command is issued. A target surface from the demo is registered to
the live target, carrying the demo's held-object goal surface along with it.
The measured live held surface is then registered to that goal. The resulting
rigid displacement is a diagnostic, not a verified manipulation command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from flash_rgbd_registration import apply, pca_hypotheses, projection_evidence, trimmed_icp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def measured_points(
    depth: np.ndarray, mask: np.ndarray, k: np.ndarray, depth_scale: float, interior_px: int
) -> np.ndarray:
    if interior_px:
        kernel = np.ones((2 * interior_px + 1, 2 * interior_px + 1), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
    ys, xs = np.nonzero(mask & (depth > 0))
    if len(xs) < 3:
        raise ValueError("fewer than three measured object points")
    z = depth[ys, xs].astype(np.float64) * depth_scale
    return np.column_stack(((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z))


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 0


def read_depth(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def sample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    # Deterministic spatial sampling; don't favor one image scan line.
    from lerobot.policies.flash.surface_observation import spatial_sample

    return points[spatial_sample(points, limit)]


def fit_metrics(source: np.ndarray, target: np.ndarray, moved: np.ndarray, radius_m: float) -> dict:
    a = cKDTree(target).query(moved)[0]
    b = cKDTree(moved).query(target)[0]
    return {
        "balanced_coverage": float(min(np.mean(a < radius_m), np.mean(b < radius_m))),
        "source_to_target_median_mm": float(np.median(a) * 1000),
        "target_to_source_median_mm": float(np.median(b) * 1000),
        "source_to_target_p90_mm": float(np.percentile(a, 90) * 1000),
        "target_to_source_p90_mm": float(np.percentile(b, 90) * 1000),
    }


def candidates(
    source: np.ndarray,
    target: np.ndarray,
    *,
    radius_m: float,
    iterations: int,
    trim_fraction: float,
    include_nearby: bool = False,
) -> list[dict]:
    seeds = list(pca_hypotheses(source, target))
    if include_nearby:
        seeds.insert(0, (np.eye(3), target.mean(axis=0) - source.mean(axis=0)))
    fits = []
    for index, (rotation, translation) in enumerate(seeds):
        rotation, translation = trimmed_icp(
            source,
            target,
            rotation,
            translation,
            iterations=iterations,
            trim_fraction=trim_fraction,
        )
        moved = apply(source, rotation, translation)
        displacement = moved - source
        fits.append(
            {
                "seed": index,
                "rotation": rotation.tolist(),
                "translation_m": translation.tolist(),
                "rotation_deg": float(np.degrees(Rotation.from_matrix(rotation).magnitude())),
                "centroid_motion_mm": float(np.linalg.norm(moved.mean(axis=0) - source.mean(axis=0)) * 1000),
                "point_motion_rms_mm": float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))) * 1000),
                **fit_metrics(source, target, moved, radius_m),
            }
        )
    fits.sort(key=lambda fit: (-fit["balanced_coverage"], fit["point_motion_rms_mm"]))
    return fits


def transformed(points: np.ndarray, fit: dict) -> np.ndarray:
    return apply(points, np.asarray(fit["rotation"]), np.asarray(fit["translation_m"]))


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return points[:, :2] / points[:, 2, None] @ k[:2, :2].T + k[:2, 2]


def render(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    target: np.ndarray,
    goal: np.ndarray,
    moved: np.ndarray,
    k: np.ndarray,
    output: Path,
) -> None:
    image = rgb.copy()
    for label, color in (("held", (255, 0, 255)), ("target", (255, 255, 0))):
        contours, _ = cv2.findContours(
            masks[label].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, 2)
    for points, color in ((target, (0, 255, 255)), (goal, (0, 140, 255)), (moved, (0, 255, 0))):
        visible = points[points[:, 2] > 0]
        uv = np.rint(project(visible, k)).astype(int)
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])]
        for x, y in uv[::5]:
            cv2.circle(image, (int(x), int(y)), 1, color, -1)
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        image,
        "Cyan: target fit   Orange: demo goal   Green: moved held surface",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
    )
    if not cv2.imwrite(str(output), image):
        raise OSError(f"could not write {output}")


def run(args: argparse.Namespace) -> dict:
    demo = args.demo_recording
    trace = args.live_trace
    rows = [json.loads(line) for line in (trace / "frames.jsonl").read_text().splitlines()]
    row = rows[args.live_index]
    frame_id = row["frame_id"]
    k_demo = np.loadtxt(demo / "cam_K.txt").reshape(3, 3)
    k_live = np.asarray(row["intrinsics"], dtype=float)
    depth_live = read_depth(trace / "depth" / f"{frame_id:06}.png")
    rgb_live = cv2.imread(str(trace / "rgb" / f"{frame_id:06}.png"))
    masks_live = {
        "held": read_mask(trace / "masks" / args.live_held_role / f"{frame_id:06}.png"),
        "target": read_mask(trace / "masks" / args.live_target_role / f"{frame_id:06}.png"),
    }
    demo_target_depth = read_depth(demo / "depth" / f"{args.demo_target_frame:06}.png")
    demo_goal_depth = read_depth(demo / "depth" / f"{args.demo_goal_frame:06}.png")
    demo_target_mask = read_mask(args.demo_target_masks / f"{args.demo_target_frame:06}.png")
    demo_goal_mask = read_mask(args.demo_held_masks / f"{args.demo_goal_frame:06}.png")
    surface = {
        "demo_target": sample(
            measured_points(
                demo_target_depth, demo_target_mask, k_demo, args.demo_depth_scale, args.mask_interior_px
            ),
            args.max_points,
        ),
        "demo_goal": sample(
            measured_points(
                demo_goal_depth, demo_goal_mask, k_demo, args.demo_depth_scale, args.mask_interior_px
            ),
            args.max_points,
        ),
        "live_target": sample(
            measured_points(
                depth_live, masks_live["target"], k_live, row["depth_m_per_unit"], args.mask_interior_px
            ),
            args.max_points,
        ),
        "live_held": sample(
            measured_points(
                depth_live, masks_live["held"], k_live, row["depth_m_per_unit"], args.mask_interior_px
            ),
            args.max_points,
        ),
    }
    target_fits = candidates(
        surface["demo_target"],
        surface["live_target"],
        radius_m=args.diagnostic_radius_m,
        iterations=args.icp_iterations,
        trim_fraction=args.trim_fraction,
    )
    for target_fit in target_fits:
        transferred_target = transformed(surface["demo_target"], target_fit)
        evidence, _, _ = projection_evidence(
            transferred_target,
            masks_live["target"],
            depth_live,
            k_live,
            args.diagnostic_radius_m,
        )
        target_fit["pixel_evidence"] = evidence
    target_fits.sort(
        key=lambda fit: (
            -fit["balanced_coverage"],
            -fit["pixel_evidence"]["projection_mask_and_depth_fraction"],
            fit["pixel_evidence"]["projection_front_of_depth_fraction"],
        )
    )
    for target_fit in target_fits[: args.report_target_candidates]:
        goal = transformed(surface["demo_goal"], target_fit)
        actions = candidates(
            surface["live_held"],
            goal,
            radius_m=args.diagnostic_radius_m,
            iterations=args.icp_iterations,
            trim_fraction=args.trim_fraction,
            include_nearby=True,
        )
        target_fit["action_candidates"] = actions[: args.report_action_candidates]
    args.output.mkdir(parents=True, exist_ok=True)
    for rank, target_fit in enumerate(target_fits[: args.report_target_candidates]):
        action = target_fit["action_candidates"][0]
        render(
            rgb_live,
            masks_live,
            transformed(surface["demo_target"], target_fit),
            transformed(surface["demo_goal"], target_fit),
            transformed(surface["live_held"], action),
            k_live,
            args.output / f"overlay-target-{rank:02}.png",
        )
    result = {
        "purpose": "offline geometry-goal diagnostic; no robot command",
        "demo_target_frame": args.demo_target_frame,
        "demo_goal_frame": args.demo_goal_frame,
        "live_frame_id": frame_id,
        "settings": {
            "max_points": args.max_points,
            "mask_interior_px": args.mask_interior_px,
            "diagnostic_radius_m": args.diagnostic_radius_m,
            "icp_iterations": args.icp_iterations,
            "trim_fraction": args.trim_fraction,
        },
        "point_counts": {name: len(points) for name, points in surface.items()},
        "target_candidates": target_fits,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-target-masks", type=Path, required=True)
    parser.add_argument("--demo-held-masks", type=Path, required=True)
    parser.add_argument("--demo-target-frame", type=int, required=True)
    parser.add_argument("--demo-goal-frame", type=int, required=True)
    parser.add_argument("--demo-depth-scale", type=float, default=0.001)
    parser.add_argument("--live-trace", type=Path, required=True)
    parser.add_argument("--live-index", type=int, default=0)
    parser.add_argument("--live-held-role", required=True)
    parser.add_argument("--live-target-role", required=True)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--mask-interior-px", type=int, default=2)
    parser.add_argument("--diagnostic-radius-m", type=float, default=0.006)
    parser.add_argument("--icp-iterations", type=int, default=15)
    parser.add_argument("--trim-fraction", type=float, default=0.8)
    parser.add_argument("--report-target-candidates", type=int, default=8)
    parser.add_argument("--report-action-candidates", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    best = result["target_candidates"][0]
    action = best["action_candidates"][0]
    print(
        json.dumps(
            {
                "live_frame": result["live_frame_id"],
                "target_coverage": best["balanced_coverage"],
                "target_depth_support": best["pixel_evidence"]["projection_mask_and_depth_fraction"],
                "target_front_of_depth": best["pixel_evidence"]["projection_front_of_depth_fraction"],
                "goal_coverage": action["balanced_coverage"],
                "required_point_motion_mm": action["point_motion_rms_mm"],
                "required_rotation_deg": action["rotation_deg"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
