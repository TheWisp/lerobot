"""Measured partial RGB-D surfaces, shared by the offline experiments and live debugger.

These filters withhold boundary/depth discontinuities; they do not certify object ownership.
Defaults reproduce the connected-surface experiment, not a calibrated sensor noise model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceObservationConfig:
    mask_margin_px: int = 2
    depth_span_m: float = 0.01
    depth_neighborhood_px: int = 3
    normal_bilateral_diameter_px: int = 5
    normal_bilateral_sigma_px: float = 2.0
    huber_m: float = 0.005
    surface_spacing_m: float = 0.003
    component_gap_m: float | None = 0.01
    validation_tile_px: int = 4


def move(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def unproject(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    y, x = np.indices(depth.shape)
    return np.stack(
        (
            (x - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )


def voxel_sample(points: np.ndarray, spacing: float) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=int)
    _, selected = np.unique(np.floor(points / spacing).astype(np.int32), axis=0, return_index=True)
    return np.sort(selected)


def largest_connected_surface(points: np.ndarray, gap_m: float) -> np.ndarray:
    """Study ablation: use one connected patch, withholding all detached patches.

    This can discard legitimate visible pieces across an occluder. It is not an
    object-ownership oracle, and connected contamination can still survive.
    """
    if not len(points):
        return np.zeros(0, dtype=bool)
    pairs = cKDTree(points).query_pairs(gap_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels == np.argmax(np.bincount(labels))


def observation(depth: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray, config: dict) -> dict:
    xyz = unproject(depth, intrinsics)
    raw = mask & (depth > 0) & np.isfinite(depth)
    margin = int(config["mask_margin_px"])
    interior = cv2.erode(mask.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    neighborhood = np.ones((config["depth_neighborhood_px"],) * 2, np.uint8)
    lower = cv2.erode(depth, neighborhood)
    upper = cv2.dilate(depth, neighborhood)
    reliable = (lower > 0) & (upper - lower <= config["depth_span_m"])
    # Smoothed local gradients define tangent planes only; fitted XYZ remains measured depth.
    smoothed = cv2.bilateralFilter(
        depth, config["normal_bilateral_diameter_px"], config["huber_m"], config["normal_bilateral_sigma_px"]
    )
    normal_xyz = unproject(smoothed, intrinsics)
    horizontal = np.zeros_like(xyz)
    vertical = np.zeros_like(xyz)
    horizontal[:, 1:-1] = normal_xyz[:, 2:] - normal_xyz[:, :-2]
    vertical[1:-1] = normal_xyz[2:] - normal_xyz[:-2]
    normals = np.cross(horizontal, vertical)
    norm = np.linalg.norm(normals, axis=-1)
    valid = raw & interior & reliable & (norm > np.finfo(float).eps)
    normals /= np.maximum(norm[..., None], np.finfo(float).eps)
    normals[np.sum(normals * xyz, axis=-1) > 0] *= -1
    y, x = np.nonzero(valid)
    selected = voxel_sample(xyz[y, x], config["surface_spacing_m"])
    y, x = y[selected], x[selected]
    detached = 0
    if config.get("component_gap_m") is not None:
        keep = largest_connected_surface(xyz[y, x], config["component_gap_m"])
        detached = int((~keep).sum())
        y, x = y[keep], x[keep]
    # Spatially interleaved held-out pixels, not independent ground-truth labels.
    tile = config["validation_tile_px"]
    fit = ((x // tile + y // tile) % 2) == 0
    return {
        "points": xyz[y, x],
        "normals": normals[y, x],
        "pixels": np.column_stack((x, y)),
        "fit": fit,
        "valid_image": valid,
        "depth_reliable": reliable,
        "raw_count": int(raw.sum()),
        "boundary_count": int((raw & ~interior).sum()),
        "edge_count": int((raw & interior & ~reliable).sum()),
        "clean_count": int(valid.sum()),
        "detached_count": detached,
    }


def initial_landmarks(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    basis = basis.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    first = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    second = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))
    axis = points[second] - points[first]
    axis /= np.linalg.norm(axis)
    residual = points - points[first]
    residual -= (residual @ axis)[:, None] * axis
    third = int(np.argmax(np.linalg.norm(residual, axis=1)))
    return points[[first, second, third]], basis
