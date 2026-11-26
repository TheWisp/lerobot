"""Optional exact CPU accelerators for the shared visible-surface fitter.

The equations match surface_grounding; sampling and solver decisions stay there.
The backend owns the search pool; Numba kernels are warmed once per process.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np


class SearchPool:
    """Reusable threads over exact, single-threaded cKDTree queries."""

    def __init__(self, workers):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="flash-surface-search")
        self.closed = False

    def query(self, tree, points, *, workers=1):
        if self.closed:
            raise RuntimeError("search pool is closed")
        count = min(len(points), self.workers, workers)
        if count <= 1:
            return tree.query(points, workers=1)
        pieces = np.array_split(points, count)
        answers = list(self.pool.map(lambda piece: tree.query(piece, workers=1), pieces))
        return tuple(np.concatenate([answer[i] for answer in answers]) for i in (0, 1))

    def close(self):
        self.pool.shutdown(wait=True)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class PooledTree:
    def __init__(self, tree, pool):
        self.tree, self.pool = tree, pool

    def query(self, points, *, workers=1):
        return self.pool.query(self.tree, points, workers=workers)


def depth_blocks(
    predicted,
    model_distance,
    target_distance,
    mask_distance,
    depth,
    mask,
    uv,
    tolerance,
    maximum,
    margin,
    fx_camera,
):
    """Same four depth rays and operation order; compile without fastmath."""
    count, target_count = len(predicted), len(target_distance)
    result = np.empty(count * 2 + target_count, np.float64)
    height, width = depth.shape
    for i in range(count):
        u, v = uv[i, 0], uv[i, 1]
        inside = 0 <= u <= width - 1 and 0 <= v <= height - 1
        x, y = min(max(u, 0.0), width - 1.0), min(max(v, 0.0), height - 1.0)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        fx, fy = x - x0, y - y0
        weights = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
        rays = (depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1])
        object_rays = (mask[y0, x0], mask[y0, x1], mask[y1, x0], mask[y1, x1])
        visible, free_space = 0.0, 0.0
        for j in range(4):
            known = rays[j] > 0 and inside
            visibility = min(max((rays[j] - predicted[i, 2] + tolerance) / tolerance, 0.0), 1.0)
            visible += weights[j] * known * object_rays[j] * visibility
            free_space += (
                weights[j] * known * min(max((rays[j] - predicted[i, 2] - tolerance) / tolerance, 0.0), 1.0)
            )
        result[i] = np.sqrt(visible / count) * min(model_distance[i], maximum) / tolerance
        silhouette = max(mask_distance[i] - margin, 0.0) * predicted[i, 2] / fx_camera
        result[count + target_count + i] = np.sqrt(free_space / count) * silhouette / tolerance
    for i in range(target_count):
        result[count + i] = min(target_distance[i], maximum) / (tolerance * np.sqrt(target_count))
    return result


def projection_evidence(predicted, uv, mask_distance, depth, mask, tolerance, margin, fx_camera):
    """Four measured rays per point, without intermediate N×4 arrays."""
    count = len(predicted)
    visible = np.zeros(count, np.float64)
    empty = np.zeros(count, np.float64)
    silhouette = np.empty(count, np.float64)
    height, width = depth.shape
    for i in range(count):
        u, v = uv[i, 0], uv[i, 1]
        inside = 0 <= u <= width - 1 and 0 <= v <= height - 1
        x, y = min(max(u, 0.0), width - 1.0), min(max(v, 0.0), height - 1.0)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        fx, fy = x - x0, y - y0
        weights = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
        xs, ys = (x0, x1, x0, x1), (y0, y0, y1, y1)
        for j in range(4):
            ray = depth[ys[j], xs[j]]
            known = ray > 0 and inside
            v = min(max((ray - predicted[i, 2] + tolerance) / tolerance, 0.0), 1.0)
            f = min(max((ray - predicted[i, 2] - tolerance) / tolerance, 0.0), 1.0)
            visible[i] += weights[j] * known * mask[ys[j], xs[j]] * v
            empty[i] += weights[j] * known * f
        silhouette[i] = max(mask_distance[i] - margin, 0.0) * predicted[i, 2] / fx_camera
    return visible, empty, silhouette


def scene_evidence(predicted, uv, depth, tolerance):
    """Evaluate empty space and occlusion on the same conservative depth rays."""
    count = len(predicted)
    free = np.zeros(count, np.float64)
    exposed = np.zeros(count, np.float64)
    contradicted = np.zeros(count, np.float64)
    height, width = depth.shape
    for i in range(count):
        u, v = uv[i, 0], uv[i, 1]
        valid = (
            np.isfinite(u)
            and np.isfinite(v)
            and predicted[i, 2] > 0
            and 0 <= u <= width - 1
            and 0 <= v <= height - 1
        )
        x = min(max(0.0 if np.isnan(u) else u, 0.0), width - 1.0)
        y = min(max(0.0 if np.isnan(v) else v, 0.0), height - 1.0)
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        fx, fy = x - x0, y - y0
        weights = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
        xs, ys = (x0, x1, x0, x1), (y0, y0, y1, y1)
        for j in range(4):
            ray = depth[ys[j], xs[j]]
            known = ray > 0 and valid
            gap = ray - predicted[i, 2]
            exposed[i] += weights[j] * known * min(max((gap + tolerance) / tolerance, 0.0), 1.0)
            contradicted[i] += weights[j] * known * (gap > tolerance)
            free[i] += weights[j] * known * max(gap - tolerance, 0.0) ** 2
        free[i] = np.sqrt(free[i])
    return free, exposed, contradicted


def outline_distances(outline, polygon, margin):
    result = np.empty(len(outline), np.float64)
    for i in range(len(outline)):
        best = np.inf
        positive, negative = True, True
        for j in range(len(polygon)):
            end = (j + 1) % len(polygon)
            ex, ey = polygon[end, 0] - polygon[j, 0], polygon[end, 1] - polygon[j, 1]
            dx, dy = outline[i, 0] - polygon[j, 0], outline[i, 1] - polygon[j, 1]
            along = min(max((dx * ex + dy * ey) / (ex * ex + ey * ey), 0.0), 1.0)
            rx, ry = dx - along * ex, dy - along * ey
            best = min(best, np.sqrt(rx * rx + ry * ry))
            cross = ex * dy - ey * dx
            positive = positive and cross >= 0
            negative = negative and cross <= 0
        if positive or negative:
            best = 0.0
        result[i] = max(best - margin, 0.0)
    return result


@lru_cache(maxsize=1)
def compiled_kernels():
    # Optional dependency: the unchanged NumPy calculation remains available.
    try:
        from numba import njit
    except ImportError:
        logging.getLogger(__name__).warning("Numba unavailable; using NumPy surface fitting")
        return None

    # Preserve sequential arithmetic; nogil lets independent object fits run
    # concurrently without acquiring the interpreter lock for these loops.
    kernels = (
        njit(nogil=True, fastmath=False, cache=True)(depth_blocks),
        njit(nogil=True, fastmath=False, cache=True)(outline_distances),
        njit(nogil=True, fastmath=False, cache=True)(projection_evidence),
        njit(nogil=True, fastmath=False, cache=True)(scene_evidence),
    )

    # Compile before processing frames; float32 is metric_depth's live dtype.
    # Also warm float64 for offline callers without changing their precision.
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]])
    distances = np.zeros(3)
    uv = np.zeros((3, 2))
    for dtype in (np.float32, np.float64):
        kernels[0](
            points,
            distances,
            distances,
            distances,
            np.ones((2, 2), dtype),
            np.ones((2, 2), bool),
            uv,
            0.005,
            0.03,
            2.0,
            1.0,
        )
        kernels[2](points, uv, distances, np.ones((2, 2), dtype), np.ones((2, 2), bool), 0.005, 2.0, 1.0)
        kernels[3](points, uv, np.ones((2, 2), dtype), 0.005)
    kernels[1](uv, points[:, :2].copy(), 2.0)
    return kernels
