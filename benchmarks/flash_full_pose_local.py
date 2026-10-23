"""Unlabelled local pose diagnostics against current mask and measured depth.

These scores cannot establish pose accuracy or the hidden contact-point pose.
Every candidate projects the same first-frame measured surface points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_full_pose_score import source_points
from benchmarks.flash_surface_rigid_fit import robust_fit


def point_fit(case: Path, name: str, source: np.ndarray, frames: np.ndarray) -> list[np.ndarray | None]:
    with np.load(case / f"{name}.npz", allow_pickle=False) as prediction:
        if not np.array_equal(frames, prediction["frames"]):
            raise ValueError(f"frame mismatch for {name}")
        xyz, visible = prediction["xyz"].copy(), prediction["visible"].copy()
    rng = np.random.default_rng(20261023)
    poses: list[np.ndarray | None] = []
    for current, claimed in zip(xyz, visible, strict=True):
        usable = claimed & np.isfinite(current).all(axis=1)
        fit, _, _ = robust_fit(source[usable], current[usable], 0.01, 0.99, 0.5, rng)
        poses.append(fit)
    return poses


def score_case(case: Path, pose_root: Path) -> dict:
    manifest = json.loads((case / "manifest.json").read_text())
    with np.load(case / "queries.npz", allow_pickle=False) as query:
        source = source_points({key: query[key] for key in query.files})
        intrinsics, frames = query["intrinsics"].copy(), query["frames"].copy()
    methods: dict[str, list[np.ndarray | None]] = {}
    for mode in ("oracle_masks", "internal_masks"):
        path = pose_root / case.name / mode / "p2p_object_pose.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as trace:
            if not np.array_equal(frames, trace["frames"]):
                raise ValueError(f"frame mismatch for {mode}")
            methods["p2p_" + ("supplied_sam3" if mode == "oracle_masks" else "internal_sam2")] = [
                None if lost else pose for pose, lost in zip(trace["poses"], trace["lost"], strict=True)
            ]
    for name in ("lk", "litetracker", "tapnext"):
        if (case / f"{name}.npz").exists():
            methods[name + "_common_fit"] = point_fit(case, name, source, frames)
    report: dict = {"case": case.name, "frames": frames.tolist(), "methods": {}}
    for name, poses in methods.items():
        rows = []
        for frame, pose in zip(frames, poses, strict=True):
            if pose is None:
                rows.append(None)
                continue
            stem = f"{frame:06d}"
            mask = cv2.imread(str(Path(manifest["mask_dir"]) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            depth = cv2.imread(
                str(Path(manifest["recording"]) / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED
            )
            if mask is None or depth is None or mask.shape != depth.shape:
                raise ValueError(f"missing aligned mask/depth at {stem}")
            points = (pose[:3, :3] @ source.T).T + pose[:3, 3]
            positive = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
            u = np.zeros(len(points), int)
            v = np.zeros(len(points), int)
            u[positive] = np.rint(
                intrinsics[0, 0] * points[positive, 0] / points[positive, 2] + intrinsics[0, 2]
            ).astype(int)
            v[positive] = np.rint(
                intrinsics[1, 1] * points[positive, 1] / points[positive, 2] + intrinsics[1, 2]
            ).astype(int)
            in_image = positive & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
            ids = np.flatnonzero(in_image)
            if len(ids) == 0:
                rows.append({"mask_fraction": 0.0, "surface_fraction": 0.0, "front_fraction": 0.0})
                continue
            measured = depth[v[ids], u[ids]].astype(float) * 0.001
            mask_here = mask[v[ids], u[ids]] > 0
            valid = measured > 0
            residual = measured - points[ids, 2]
            rows.append(
                {
                    "mask_fraction": float(np.mean(mask_here)),
                    "surface_fraction": float(np.mean(mask_here & valid & (np.abs(residual) <= 0.01))),
                    "front_fraction": float(np.mean(valid & (residual > 0.01))),
                }
            )
        available = [row for row in rows[1:] if row is not None]
        report["methods"][name] = {
            "available": len(available),
            "total": len(rows) - 1,
            "surface_fraction_median": float(np.median([row["surface_fraction"] for row in available]))
            if available
            else None,
            "front_fraction_median": float(np.median([row["front_fraction"] for row in available]))
            if available
            else None,
            "rows": rows,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True, type=Path)
    parser.add_argument("--pose-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "schema": "flash-full-pose-local-diagnostic-v1",
        "cases": [score_case(p, args.pose_root) for p in args.cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
