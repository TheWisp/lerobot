"""Plot two causal dowel pose lineages over insertion RGB and object masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plot_dowel_candidate_axes import project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, nargs=4, default=(403, 451, 491, 499))
    parser.add_argument("--candidate", action="append", default=[], help="NAME=POSE_TRACE.npz; exactly two")
    args = parser.parse_args()
    root = Path(__file__).parent
    with np.load(root / "pose_traces.npz") as reference:
        intrinsics = reference["intrinsics"].copy()
        center = reference["dowel_reference_center_camera_m"].copy()
        shaft = reference["dowel_reference_shaft_camera"].copy()
        offsets = reference["dowel_reference_end_offsets_m"].copy()
    if args.candidate:
        if len(args.candidate) != 2:
            parser.error("provide exactly two pose candidates")
        specifications = []
        for entry in args.candidate:
            name, separator, filename = entry.partition("=")
            if not separator or not name or not filename:
                parser.error("each candidate must be NAME=POSE_TRACE.npz")
            specifications.append((name, Path(filename)))
    else:
        specifications = [
            (f"seed {seed}", root / f"cotracker3_shared_dowel_{seed}_550_rebased_pose.npz")
            for seed in (340, 380)
        ]
    traces = {}
    for (name, path), color in zip(specifications, ("#f97316", "#b91cdb"), strict=True):
        with np.load(path) as source:
            traces[name] = (source["frames"].copy(), source["poses"].copy(), source["lost"].copy(), color)
    ends = [center + offset * shaft for offset in offsets]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, frame in zip(axes.flat, args.frames, strict=True):
        image = cv2.imread(str(args.dataset / "rgb" / f"{frame:06d}.jpg"))
        masks = {}
        for role in ("dowel", "ring"):
            masks[role] = cv2.imread(str(args.masks / role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or any(mask is None for mask in masks.values()):
            raise ValueError(f"missing RGB or masks at frame {frame}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        union = masks["dowel"] | masks["ring"]
        rows, columns = np.nonzero(union)
        if not len(rows):
            raise ValueError(f"no object mask at frame {frame}")
        pad = 45
        left, right = max(0, int(columns.min()) - pad), min(image.shape[1], int(columns.max()) + pad)
        top, bottom = max(0, int(rows.min()) - pad), min(image.shape[0], int(rows.max()) + pad)
        crop = image[top:bottom, left:right].copy()
        for role, color in (("dowel", (0, 240, 245)), ("ring", (245, 235, 0))):
            edges = cv2.morphologyEx(masks[role], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            edge_rows, edge_columns = np.nonzero(edges[top:bottom, left:right])
            crop[edge_rows, edge_columns] = color
        axis.imshow(crop)
        contact_ends = []
        for name, (frames, poses, lost, color) in traces.items():
            index = np.flatnonzero(frames == frame)
            if len(index) != 1 or lost[index[0]]:
                continue
            pose = poses[index[0]]
            contact_ends.append(pose[:3, :3] @ ends[1] + pose[:3, 3])
            projected = [project(pose[:3, :3] @ point + pose[:3, 3], intrinsics) for point in ends]
            pixels = np.stack(projected) - np.array((left, top))
            axis.plot(pixels[:, 0], pixels[:, 1], "-o", color=color, linewidth=2, markersize=4, label=name)
            axis.plot(pixels[1, 0], pixels[1, 1], "o", color=color, markersize=10, fillstyle="none")
        if len(contact_ends) != 2:
            raise ValueError(f"both pose lineages must be valid at frame {frame}")
        gap = float(np.linalg.norm(contact_ends[0] - contact_ends[1]) * 1000)
        axis.set_title(f"Frame {frame}  |  predicted end gap: {gap:.1f} mm in 3D", fontsize=11)
        axis.set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle(
        "Two causal pose lineages; large ring-side circles mark predicted contact ends", fontsize=14
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
