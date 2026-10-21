"""Score saved pose candidates with FLASH's existing live surface diagnostic.

Run under the repository virtual environment, which imports the FLASH package.
This is a replay check of the existing diagnostic, not a live estimator change.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import SurfaceSupport, surface_support


def reference_points(dataset: Path, masks: Path, intrinsics: np.ndarray) -> np.ndarray:
    depth = cv2.imread(str(dataset / "depth" / "000100.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(masks / "dowel" / "000100.png"), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None or depth.shape != mask.shape:
        raise ValueError("missing or misaligned frame-100 dowel RGB-D/mask")
    depth_m = depth.astype(float) * 0.001
    eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8)) > 0
    y, x = np.nonzero(eroded & (depth_m > 0))
    if len(x) > 4000:
        selected = np.linspace(0, len(x) - 1, 4000, dtype=int)
        x, y = x[selected], y[selected]
    z = depth_m[y, x]
    return np.column_stack(
        ((x - intrinsics[0, 2]) * z / intrinsics[0, 0], (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z)
    )


def verdict(sam2: SurfaceSupport, sam3: SurfaceSupport) -> str:
    if (
        sam2.projected_point_mask_fraction >= sam3.projected_point_mask_fraction
        and sam2.depth_residual_p90_m <= sam3.depth_residual_p90_m
    ):
        return "sam2_dominates"
    if (
        sam3.projected_point_mask_fraction >= sam2.projected_point_mask_fraction
        and sam3.depth_residual_p90_m <= sam2.depth_residual_p90_m
    ):
        return "sam3_dominates"
    return "tradeoff"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    intrinsics = np.load(root / "pose_traces.npz")["intrinsics"]
    reference = reference_points(args.dataset, args.masks, intrinsics)
    traces = {
        "sam2": np.load(root / "p2p_dowel_internal_control_100_550.npz"),
        "sam3": np.load(root / "p2p_dowel_sam3_100_550.npz"),
    }
    rows = []
    counts: dict[str, Counter[str]] = {"reacquisition": Counter(), "carry": Counter(), "insertion": Counter()}
    for frame_id in range(249, 501):
        phase = "reacquisition" if frame_id <= 280 else "carry" if frame_id < 400 else "insertion"
        index = frame_id - 100
        if any(trace["lost"][index] for trace in traces.values()):
            counts[phase]["missing_candidate"] += 1
            continue
        stem = f"{frame_id:06d}"
        bgr = cv2.imread(str(args.dataset / "rgb" / f"{stem}.jpg"))
        depth = cv2.imread(str(args.dataset / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(args.masks / "dowel" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if (
            bgr is None
            or depth is None
            or mask is None
            or bgr.shape[:2] != depth.shape
            or mask.shape != depth.shape
        ):
            raise ValueError(f"missing or misaligned RGB-D/mask frame {frame_id}")
        frame = RGBDFrame(
            frame_id,
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            depth,
            intrinsics,
            0.001,
            np.eye(4),
            frame_id / 30,
            frame_id / 30,
            "offline replay",
        )
        support = {
            backend: surface_support(frame, mask > 0, reference, trace["poses"][index])
            for backend, trace in traces.items()
        }
        decision = verdict(support["sam2"], support["sam3"])
        counts[phase][decision] += 1
        rows.append(
            (
                frame_id,
                phase,
                decision,
                support["sam2"].projected_point_mask_fraction,
                support["sam2"].depth_residual_p90_m,
                support["sam3"].projected_point_mask_fraction,
                support["sam3"].depth_residual_p90_m,
            )
        )
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "phase",
                "pareto_verdict",
                "sam2_projected_mask_fraction",
                "sam2_depth_residual_p90_m",
                "sam3_projected_mask_fraction",
                "sam3_depth_residual_p90_m",
            )
        )
        writer.writerows(rows)
    for phase, outcomes in counts.items():
        print(phase, dict(outcomes))


if __name__ == "__main__":
    main()
