"""Render labelled full-pose benchmark results and one source-frame audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from benchmarks.flash_full_pose_score import source_points

METHODS = (
    ("p2p_internal_masks", "P2P\nSAM2"),
    ("p2p_oracle_masks", "P2P\nGT masks"),
    ("litetracker_common_fit", "Lite\n+ fit"),
    ("tapnext_common_fit", "TAPNext\n+ fit"),
    ("lk_common_fit", "Optical flow\n+ fit"),
)


def heatmap(report: dict, output: Path) -> None:
    cases = report["cases"]
    data = np.ma.masked_all((len(cases), len(METHODS)))
    labels: list[list[str]] = []
    for i, case in enumerate(cases):
        row = []
        for j, (method, _) in enumerate(METHODS):
            summary = case["methods"][method]["summary"]
            if summary["available"]:
                data[i, j] = summary["cloud_mm_median"]
                row.append(f"{summary['cloud_mm_median']:.1f} mm\n{summary['available']}/30")
            else:
                row.append("init failed\n0/30")
        labels.append(row)
    fig, ax = plt.subplots(figsize=(10.5, 9), dpi=150)
    palette = plt.get_cmap("RdYlGn_r").copy()
    palette.set_bad("#dddde2")
    plotted = ax.imshow(data, cmap=palette, norm=Normalize(0, 30), aspect="auto")
    ax.set_xticks(range(len(METHODS)), [label for _, label in METHODS], fontsize=9)
    ax.set_yticks(range(len(cases)), [case["case"].replace("ycb-", "") for case in cases], fontsize=9)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    for i in range(len(cases)):
        for j in range(len(METHODS)):
            color = "white" if not data.mask[i, j] and data[i, j] >= 21 else "#151515"
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8.3, color=color)
    ax.set_xticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cases), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(plotted, ax=ax, shrink=0.65, pad=0.02, label="Median transported source-cloud error (mm)")
    fig.suptitle("Complete pose comparison on annotated RGB-D motion", fontsize=14)
    fig.text(
        0.02,
        0.015,
        "Cell shows median error on available frames and coverage out of 30. Grey = no pose.\n"
        "P2P is the complete causal system; other trackers use a shared 10-mm rigid fit."
        " GT masks are dataset annotations.\nNo contact-tip ground truth.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def projection(pts: np.ndarray, pose: np.ndarray, k: np.ndarray) -> np.ndarray:
    current = (pose[:3, :3] @ pts.T).T + pose[:3, 3]
    return np.column_stack(
        (k[0, 0] * current[:, 0] / current[:, 2] + k[0, 2], k[1, 1] * current[:, 1] / current[:, 2] + k[1, 2])
    )


def example(case: Path, pose_root: Path, frame: int, output: Path) -> None:
    manifest = json.loads((case / "manifest.json").read_text())
    with np.load(case / "queries.npz", allow_pickle=False) as reference:
        source = source_points({key: reference[key] for key in reference.files})
        k = reference["intrinsics"].copy()
    paths = sorted((Path(manifest["source"]) / "annotated_poses").glob("*.txt"))
    first = np.loadtxt(paths[manifest["source_first_index"]]).reshape(4, 4)
    truth = np.loadtxt(paths[manifest["source_first_index"] + frame]).reshape(4, 4) @ np.linalg.inv(first)
    with np.load(pose_root / case.name / "internal_masks" / "p2p_object_pose.npz", allow_pickle=False) as run:
        p2p = run["poses"][frame].copy()
        if run["lost"][frame]:
            raise ValueError(f"Point2Pose has no pose at frame {frame}")
    with np.load(case / "fits-10mm" / "litetracker.npz", allow_pickle=False) as fit:
        lite = fit["transforms"][frame].copy()
        if not fit["available"][frame]:
            raise ValueError(f"LiteTracker has no fitted pose at frame {frame}")
    image = cv2.imread(str(Path(manifest["recording"]) / "rgb" / f"{frame:06d}.jpg"))
    mask = cv2.imread(str(Path(manifest["mask_dir"]) / f"{frame:06d}.png"), 0)
    if image is None or mask is None:
        raise ValueError("source frame or mask missing")
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_y, mask_x = np.nonzero(mask)
    projected = [projection(source, pose, k) for pose in (truth, p2p, lite)]
    all_x = np.concatenate(([mask_x.min(), mask_x.max()], *[uv[:, 0] for uv in projected]))
    all_y = np.concatenate(([mask_y.min(), mask_y.max()], *[uv[:, 1] for uv in projected]))
    span = max(float(np.ptp(all_x)), float(np.ptp(all_y)))
    pad = max(16.0, span * 0.2)
    crop_width = min(image.shape[1], max(np.ptp(all_x) + 2 * pad, (np.ptp(all_y) + 2 * pad) * 4 / 3))
    crop_height = min(image.shape[0], max(np.ptp(all_y) + 2 * pad, crop_width * 3 / 4))
    center_x, center_y = (
        float(np.mean((all_x.min(), all_x.max()))),
        float(np.mean((all_y.min(), all_y.max()))),
    )
    x0 = int(np.clip(center_x - crop_width / 2, 0, image.shape[1] - crop_width))
    y0 = int(np.clip(center_y - crop_height / 2, 0, image.shape[0] - crop_height))
    x1, y1 = int(x0 + crop_width), int(y0 + crop_height)
    panels = []
    for title, uv, color in (
        ("Annotated motion", projected[0], (230, 80, 240)),
        ("Point2Pose + SAM2", projected[1], (240, 235, 40)),
        ("LiteTracker + common fit", projected[2], (100, 245, 100)),
    ):
        panel = image.copy()
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 2)
        for u, v in uv:
            if np.isfinite([u, v]).all() and 0 <= u < panel.shape[1] and 0 <= v < panel.shape[0]:
                cv2.circle(panel, (round(u), round(v)), 2, color, -1)
        panel = cv2.resize(panel[y0:y1, x0:x1], (750, 555))
        heading = np.full((65, 750, 3), 24, np.uint8)
        cv2.putText(heading, title, (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        panels.append(np.vstack((heading, panel)))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", required=True, type=Path)
    parser.add_argument("--heatmap", required=True, type=Path)
    parser.add_argument("--example-case", type=Path)
    parser.add_argument("--example-frame", type=int, default=30)
    parser.add_argument("--pose-root", type=Path)
    parser.add_argument("--example-output", type=Path)
    args = parser.parse_args()
    heatmap(json.loads(args.score.read_text()), args.heatmap)
    if args.example_case is not None:
        if args.pose_root is None or args.example_output is None:
            parser.error("--example-case needs --pose-root and --example-output")
        example(args.example_case, args.pose_root, args.example_frame, args.example_output)


if __name__ == "__main__":
    main()
