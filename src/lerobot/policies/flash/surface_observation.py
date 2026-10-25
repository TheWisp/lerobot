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
    # Keep small-object detail without making large surfaces equally expensive.
    # Older experiment dictionaries without these fields keep fixed spacing.
    samples_across_object: int = 64
    min_surface_spacing_m: float = 0.001
    component_gap_m: float | None = 0.01
    validation_tile_px: int = 4


def move(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def unproject(
    depth: np.ndarray, intrinsics: np.ndarray, *, pixel_origin: tuple[int, int] = (0, 0)
) -> np.ndarray:
    y, x = np.indices(depth.shape)
    x += pixel_origin[0]
    y += pixel_origin[1]
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
    cells = np.floor(points / spacing).astype(np.int64)
    cells -= cells.min(axis=0)
    dimensions = cells.max(axis=0) + 1
    # Scalar cell IDs avoid sorting structured XYZ records. Same voxels, same
    # first measured pixel in each voxel; no averaging or invented depth.
    if np.prod(dimensions.astype(object)) <= np.iinfo(np.int64).max:
        keys = np.ravel_multi_index(cells.T, tuple(dimensions))
        _, selected = np.unique(keys, return_index=True)
    else:
        _, selected = np.unique(cells, axis=0, return_index=True)
    return np.sort(selected)


def surface_spacing(points: np.ndarray, config: dict) -> float:
    """Adapt resolution to visible extent, with explicit sensor/work limits."""
    coarse = config["surface_spacing_m"]
    across = config.get("samples_across_object")
    if not len(points) or not across:
        return coarse
    return float(
        np.clip(np.max(np.ptp(points, axis=0)) / across, config.get("min_surface_spacing_m", coarse), coarse)
    )


def spatial_sample(points: np.ndarray, budget: int) -> np.ndarray:
    """Bound residual queries, retaining spatial coverage instead of a prefix.

    The full cloud still grounds nearest-neighbour distances. This only picks
    which measured points contribute residuals; it does not simplify geometry.
    """
    if len(points) <= budget:
        return np.arange(len(points))
    extent = np.ptp(points, axis=0)
    if not np.any(extent):
        return np.array([0])
    spacing = float(np.max(extent) / np.sqrt(budget))
    local = points - points.min(axis=0)
    selected = np.arange(len(points))
    # Increase cells until the occupied-cell count fits the computational budget.
    while len(selected) > budget:
        # Anchor the grid at the cloud bounds: a grid straddling world zero
        # cannot merge its octants even when spacing grows without bound.
        selected = voxel_sample(local, spacing)
        spacing *= max(1.1, np.sqrt(len(selected) / budget))
    return selected


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


def _surface_region(mask: np.ndarray) -> tuple[slice, slice]:
    """Mask bounds plus the one-pixel halo used by central-difference normals."""
    x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
    if not width:
        return slice(0, 0), slice(0, 0)
    return (
        slice(max(0, y - 1), min(mask.shape[0], y + height + 1)),
        slice(max(0, x - 1), min(mask.shape[1], x + width + 1)),
    )


def observation(depth: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray, config: dict) -> dict:
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
    # Keep the inexpensive image filters above in full-frame coordinates:
    # depth_reliable is also consumed outside the mask by the research tools,
    # and OpenCV's float bilateral lookup table depends on the image depth range.
    # Only the expensive XYZ/normal arrays need the object's bounding region.
    region = _surface_region(mask)
    origin = (region[1].start, region[0].start)
    xyz = unproject(depth[region], intrinsics, pixel_origin=origin)
    normal_xyz = unproject(smoothed[region], intrinsics, pixel_origin=origin)
    horizontal = np.zeros_like(xyz)
    vertical = np.zeros_like(xyz)
    horizontal[:, 1:-1] = normal_xyz[:, 2:] - normal_xyz[:, :-2]
    vertical[1:-1] = normal_xyz[2:] - normal_xyz[:-2]
    normals = np.cross(horizontal, vertical)
    norm = np.linalg.norm(normals, axis=-1)
    local_valid = raw[region] & interior[region] & reliable[region] & (norm > np.finfo(float).eps)
    valid = np.zeros(mask.shape, dtype=bool)
    valid[region] = local_valid
    normals /= np.maximum(norm[..., None], np.finfo(float).eps)
    normals[np.sum(normals * xyz, axis=-1) > 0] *= -1
    y, x = np.nonzero(local_valid)
    spacing = surface_spacing(xyz[y, x], config)
    selected = voxel_sample(xyz[y, x], spacing)
    y, x = y[selected], x[selected]
    detached = 0
    if config.get("component_gap_m") is not None:
        keep = largest_connected_surface(xyz[y, x], config["component_gap_m"])
        detached = int((~keep).sum())
        y, x = y[keep], x[keep]
    # Spatially interleaved held-out pixels, not independent ground-truth labels.
    tile = config["validation_tile_px"]
    pixels = np.column_stack((x + origin[0], y + origin[1]))
    fit = ((pixels[:, 0] // tile + pixels[:, 1] // tile) % 2) == 0
    return {
        "points": xyz[y, x],
        "normals": normals[y, x],
        "pixels": pixels,
        "fit": fit,
        "valid_image": valid,
        "depth_reliable": reliable,
        "raw_count": int(raw.sum()),
        "boundary_count": int((raw & ~interior).sum()),
        "edge_count": int((raw & interior & ~reliable).sum()),
        "clean_count": int(valid.sum()),
        "detached_count": detached,
        "spacing_m": spacing,
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
