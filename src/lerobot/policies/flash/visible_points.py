"""Measured visible-surface support for point tracks on aligned RGB-D frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_mask(mask_dir: Path, frame: int, shape: tuple[int, int]) -> np.ndarray | None:
    path = mask_dir / f"{frame:06d}.png"
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        raise ValueError(f"mask does not align with RGB-D frame {frame}: {path}")
    return mask > 0


def surface_support(
    points: np.ndarray,
    visibility: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    z = np.where(finite, points[:, 2], 1.0)
    x = np.rint(intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2]).astype(int)
    y = np.rint(intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2]).astype(int)
    supported = finite & (visibility > 0) & (x >= 0) & (x < mask.shape[1])
    supported &= (y >= 0) & (y < mask.shape[0])
    indices = np.flatnonzero(supported)
    supported[indices] = mask[y[indices], x[indices]] & (depth[y[indices], x[indices]] > 0)
    indices = np.flatnonzero(supported)
    supported[indices] &= np.abs(depth[y[indices], x[indices]] - points[indices, 2]) <= depth_tolerance_m
    return supported, x, y


def select_growth_view(
    frames: np.ndarray,
    points: np.ndarray,
    visibility: np.ndarray,
    births: np.ndarray,
    depths: np.ndarray,
    mask_dir: Path,
    intrinsics: np.ndarray,
    depth_tolerance_m: float,
    points_per_view: int,
) -> tuple[int, np.ndarray] | None:
    """Find the later view with the most measured mask area uncovered by current points."""
    best: tuple[int, int, np.ndarray] | None = None
    for index, frame in enumerate(frames):
        if index <= births.max():
            continue
        mask = read_mask(mask_dir, int(frame), depths[index].shape)
        if mask is None:
            continue
        valid = mask & np.isfinite(depths[index]) & (depths[index] > 0)
        if not valid.any():
            continue
        supported, x, y = surface_support(
            points[index],
            np.where(births <= index, visibility[index], -np.inf),
            mask,
            depths[index],
            intrinsics,
            depth_tolerance_m,
        )
        occupied = np.zeros(mask.shape, dtype=np.uint8)
        occupied[y[supported], x[supported]] = 1
        spacing_px = np.sqrt(valid.sum() / points_per_view)
        distance = cv2.distanceTransform(1 - occupied, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        novel = valid & (distance > spacing_px)
        score = int(novel.sum())
        if score and (best is None or score > best[0]):
            best = score, index, novel
    return None if best is None else (best[1], best[2])
