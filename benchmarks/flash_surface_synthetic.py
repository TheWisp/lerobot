"""Generate exact-motion RGB-D plane controls for point-adapter validation.

These rendered clips have exact camera/object transforms and deterministic
material-point labels. They test coordinate conventions and obvious drift, not
real sensor performance or the generality of a pretrained tracker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _digest, prepare_local


def _pose(progress: float, motion: str) -> np.ndarray:
    if motion == "translation":
        angle_y, angle_z, move_x = 0.0, 0.0, 0.08 * progress
    elif motion == "out_of_plane_rotation":
        angle_y, angle_z, move_x = np.deg2rad(50 * progress), 0.0, 0.025 * progress
    elif motion == "in_plane_rotation":
        angle_y, angle_z, move_x = 0.0, np.deg2rad(90 * progress), 0.0
    else:
        raise ValueError(f"unknown motion {motion}")
    cy, sy, cz, sz = np.cos(angle_y), np.sin(angle_y), np.cos(angle_z), np.sin(angle_z)
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = rotation_z @ rotation_y
    pose[:3, 3] = [move_x, 0, 0.58]
    return pose


def _project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    return xyz[:, :2] / xyz[:, 2:3] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]


def build(output: Path, motion: str, texture: str, frames: int = 31, points: int = 64) -> dict:
    if frames < 2 or texture not in {"textured", "smooth"}:
        raise ValueError("need at least two frames and a supported texture")
    h, w = 240, 320
    k = np.array([[290, 0, w / 2], [0, 290, h / 2], [0, 0, 1]], float)
    recording, masks = output / "recording", output / "masks"
    for directory in (recording / "rgb", recording / "depth", masks):
        directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(recording / "cam_K.txt", k)
    np.savetxt(recording / "times.txt", np.arange(frames) / 30, fmt="%.9f")
    rng = np.random.default_rng(20261023)
    background = rng.integers(50, 90, (h, w, 3), dtype=np.uint8)
    texture_rgb = np.full((160, 240, 3), (45, 170, 220), np.uint8)
    if texture == "textured":
        for y in range(0, 160, 20):
            for x in range(0, 240, 20):
                if (x // 20 + y // 20) % 2:
                    texture_rgb[y : y + 20, x : x + 20] = (170, 55, 210)
        for _ in range(80):
            cx, cy = rng.integers(0, 240), rng.integers(0, 160)
            cv2.circle(texture_rgb, (int(cx), int(cy)), 2, tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    texture_mask = np.full((160, 240), 255, np.uint8)
    corner_local = np.array([[-0.075, -0.05, 0], [0.075, -0.05, 0], [0.075, 0.05, 0], [-0.075, 0.05, 0]])
    corner_tex = np.float32([[0, 0], [239, 0], [239, 159], [0, 159]])
    poses = []
    for index in range(frames):
        pose = _pose(index / (frames - 1), motion)
        poses.append(pose)
        corners_cam = (pose[:3, :3] @ corner_local.T).T + pose[:3, 3]
        homography = cv2.getPerspectiveTransform(corner_tex, _project(corners_cam, k).astype(np.float32))
        mask = cv2.warpPerspective(texture_mask, homography, (w, h), flags=cv2.INTER_NEAREST)
        object_rgb = cv2.warpPerspective(texture_rgb, homography, (w, h), flags=cv2.INTER_LINEAR)
        rgb = background.copy()
        rgb[mask > 0] = object_rgb[mask > 0]
        depth = np.full((h, w), 1200, np.uint16)
        yy, xx = np.nonzero(mask)
        rays = np.column_stack(((xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones(len(xx))))
        normal = pose[:3, 2]
        plane_depth = (normal @ pose[:3, 3]) / (rays @ normal)
        depth[yy, xx] = np.rint(plane_depth * 1000).astype(np.uint16)
        cv2.imwrite(str(recording / "rgb" / f"{index:06d}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(recording / "depth" / f"{index:06d}.png"), depth)
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    manifest = prepare_local(recording, masks, 0, frames - 1, points, output)
    query_path = output / "queries.npz"
    with np.load(query_path, allow_pickle=False) as saved:
        reference = {key: saved[key] for key in saved.files}
    q, z = reference["queries"], reference["initial_depth_m"]
    source_xyz = np.column_stack(((q[:, 1] - k[0, 2]) * z / k[0, 0], (q[:, 2] - k[1, 2]) * z / k[1, 1], z))
    local = (np.linalg.inv(poses[0]) @ np.column_stack((source_xyz, np.ones(points))).T).T[:, :3]
    truth_xyz = np.array([(pose[:3, :3] @ local.T).T + pose[:3, 3] for pose in poses])
    truth_xy = np.array([_project(xyz, k) for xyz in truth_xyz])
    truth_visible = np.zeros((frames, points), bool)
    depth_gap = np.full((frames, points), np.nan)
    for index in range(frames):
        mask = cv2.imread(str(masks / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(recording / "depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED) * 0.001
        x, y = np.rint(truth_xy[index]).astype(int).T
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        ids = np.flatnonzero(inside)
        depth_gap[index, ids] = np.abs(depth[y[ids], x[ids]] - truth_xyz[index, ids, 2])
        truth_visible[index, ids] = (mask[y[ids], x[ids]] > 0) & (depth_gap[index, ids] <= 0.002)
    if not truth_visible[0].all():
        raise ValueError("rendered source queries do not lie on the plane")
    np.savez_compressed(
        query_path,
        **reference,
        truth_xy=truth_xy.astype(np.float32),
        truth_xyz=truth_xyz.astype(np.float32),
        truth_visible=truth_visible,
        reference_depth_gap_m=depth_gap.astype(np.float32),
    )
    manifest.update(
        {
            "queries_sha256": _digest(query_path),
            "ground_truth": "analytic material-point transport on a rendered rigid plane",
            "motion": motion,
            "texture": texture,
            "labelled_visible_fraction": float(truth_visible.mean()),
            "reference_depth_gap_m_median_on_visible": float(np.median(depth_gap[truth_visible])),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion", choices=("translation", "out_of_plane_rotation", "in_plane_rotation"), required=True
    )
    parser.add_argument("--texture", choices=("textured", "smooth"), required=True)
    parser.add_argument("--frames", type=int, default=31)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.motion, args.texture, args.frames, args.points), indent=2))


if __name__ == "__main__":
    main()
