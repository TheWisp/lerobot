"""Score causal point motion with different observed-pose anchors on the demo.

A rigid point fit measures motion since its seed. Its absolute object pose is
the seed pose composed with that motion; a bad seed can remain wrong despite
many successful point fits. Visible-surface scores are not hidden-tip labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from analyze_multiview import frame_points, move, surface_p90_mm
from analyze_p2p_mask_ablation import projected_axis_error_deg

CASES = (
    ("dowel", 380, 500, 0.5, "p2p"),
    ("dowel", 380, 500, 1, "p2p"),
    ("ring", 300, 500, 0.5, "bundle"),
    ("ring", 300, 500, 1, "bundle"),
    ("ring", 340, 500, 0.5, "bundle"),
    ("ring", 340, 500, 1, "bundle"),
)


def point_trace(root: Path, role: str, seed: int, last: int, scale: float) -> np.lib.npyio.NpzFile:
    return np.load(root / f"points_{role}_{seed}_{last}_scale{scale}.npz")


def plot_scores(rows: list[tuple], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for axis, role in zip(axes, ("dowel", "ring"), strict=True):
        subset = [row for row in rows if row[0] == role and 400 <= row[5] <= 500]
        baseline = {}
        for row in subset:
            baseline[row[5]] = row[9]
        axis.plot(
            sorted(baseline),
            [baseline[frame] for frame in sorted(baseline)],
            "k-",
            label="direct anchor source",
        )
        variants = (
            ((380, 0.5, "#edaa18"), (380, 1, "#1677a8"))
            if role == "dowel"
            else ((300, 1, "#d74391"), (340, 0.5, "#edaa18"), (340, 1, "#1677a8"))
        )
        for seed, scale, color in variants:
            selected = [row for row in subset if row[1] == seed and row[3] == scale]
            axis.plot(
                [row[5] for row in selected],
                [row[8] for row in selected],
                color=color,
                label=f"points seeded {seed}, scale {scale:g}",
            )
        axis.set(title=role.capitalize(), ylabel="Current surface p90 (mm)", ylim=(0, 20))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set(xlabel="Demo frame", xlim=(400, 500))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    traces = np.load(root / "pose_traces.npz")
    intrinsics = traces["intrinsics"]
    center = traces["dowel_reference_center_camera_m"]
    axis = traces["dowel_reference_shaft_camera"]
    near_end = traces["dowel_reference_end_offsets_m"][0]
    contact_end = center + traces["dowel_reference_end_offsets_m"][1] * axis
    dowel_direct = np.load(root / "p2p_dowel_internal_control_100_550.npz")
    direct_sources = {
        "dowel": (dowel_direct["poses"], dowel_direct["lost"]),
        "ring": (traces["bundlesdf_ring_delta"], traces["bundlesdf_ring_failed"]),
    }
    reference = {
        role: frame_points(args.dataset, args.masks, role, 100, intrinsics, max_points=4000)[0]
        for role in ("dowel", "ring")
    }
    rows = []
    for role, seed, last, scale, anchor in CASES:
        point = point_trace(root, role, seed, last, scale)
        source, source_lost = direct_sources[role]
        if source_lost[seed - 100]:
            raise ValueError(f"{anchor} anchor unavailable at {role} frame {seed}")
        for frame, motion, inliers, step_ms in zip(
            point["frames"], point["motions"], point["inliers"], point["step_ms"], strict=True
        ):
            frame = int(frame)
            if not np.isfinite(motion).all():
                continue
            pose = motion @ source[seed - 100]
            live, _ = frame_points(args.dataset, args.masks, role, frame, intrinsics, edge_limit_m=0.01)
            score = surface_p90_mm(live, move(reference[role], pose))
            direct_available = not bool(source_lost[frame - 100])
            direct_score = (
                surface_p90_mm(live, move(reference[role], source[frame - 100]))
                if direct_available
                else np.nan
            )
            projected_angle = np.nan
            direct_angle = np.nan
            near_end_residual = np.nan
            direct_near_end_residual = np.nan
            contact_end_gap = np.nan
            if role == "dowel":
                mask = cv2.imread(str(args.masks / role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
                projected_angle = projected_axis_error_deg(mask, pose, intrinsics, center, axis)[0]
                if direct_available:
                    direct_angle = projected_axis_error_deg(
                        mask, source[frame - 100], intrinsics, center, axis
                    )[0]
                visible, _ = frame_points(args.dataset, args.masks, role, frame, intrinsics)
                candidates = [(pose, "points")]
                if direct_available:
                    candidates.append((source[frame - 100], "direct"))
                for candidate, target in candidates:
                    measured = np.quantile(
                        (visible - move(center[None], candidate)[0]) @ (candidate[:3, :3] @ axis),
                        0.05,
                    )
                    if target == "points":
                        near_end_residual = (measured - near_end) * 1000
                    else:
                        direct_near_end_residual = (measured - near_end) * 1000
                if direct_available:
                    contact_end_gap = float(
                        np.linalg.norm(
                            move(contact_end[None], pose)[0] - move(contact_end[None], source[frame - 100])[0]
                        )
                        * 1000
                    )
            rows.append(
                (
                    role,
                    seed,
                    last,
                    scale,
                    anchor,
                    frame,
                    not direct_available,
                    int(inliers),
                    score,
                    direct_score,
                    projected_angle,
                    direct_angle,
                    near_end_residual,
                    direct_near_end_residual,
                    contact_end_gap,
                    float(step_ms),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "role",
                "seed_frame",
                "last_frame",
                "image_scale",
                "anchor_source",
                "frame",
                "direct_lost",
                "inliers",
                "points_surface_p90_mm",
                "direct_surface_p90_mm",
                "points_mask_axis_error_deg",
                "direct_mask_axis_error_deg",
                "points_visible_end_axial_residual_mm",
                "direct_visible_end_axial_residual_mm",
                "contact_end_difference_mm",
                "tracker_step_ms",
            )
        )
        writer.writerows(rows)
    plot_scores(rows, args.figure)
    for role, seed, last, scale, anchor in CASES:
        selected = [
            row for row in rows if row[:5] == (role, seed, last, scale, anchor) and 400 <= row[5] <= 500
        ]
        print(
            role,
            seed,
            scale,
            "insertion fits",
            len(selected),
            "surface p90 mm",
            round(float(np.median([row[8] for row in selected])), 2),
            "tracker step ms",
            round(float(np.median([row[15] for row in selected])), 2),
        )


if __name__ == "__main__":
    main()
