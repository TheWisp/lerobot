"""Research candidates; none is selected by the live FLASH configuration.

Keep the production solver, samples, objective weights and branch selection.
The reference implementation is the independent numerical oracle in tests.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull

from lerobot.policies.flash.surface_grounding import WholeSurfaceFit, project
from lerobot.policies.flash.surface_observation import move


class SearchPool:
    """Reusable threads over exact, single-threaded cKDTree queries."""

    def __init__(self, workers):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="flash-search-experiment")
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
        visible, unknown = 0.0, 0.0
        for j in range(4):
            known = rays[j] > 0 and inside
            visibility = min(max((rays[j] - predicted[i, 2] + tolerance) / tolerance, 0.0), 1.0)
            visible += weights[j] * known * visibility
            unknown += weights[j] * (not known)
        result[i] = np.sqrt(visible / count) * min(model_distance[i], maximum) / tolerance
        silhouette = max(mask_distance[i] - margin, 0.0) * predicted[i, 2] / fx_camera
        result[count + target_count + i] = np.sqrt((visible + unknown) / count) * silhouette / tolerance
    for i in range(target_count):
        result[count + i] = min(target_distance[i], maximum) / (tolerance * np.sqrt(target_count))
    return result


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


def compiled_kernels():
    # Optional experiment dependency, never imported by the deployed tracker.
    from numba import njit

    # The serial compiled version isolates removal of Python/temporary arrays.
    # Parallel numerical candidates can call these concurrently because nogil
    # releases the interpreter lock; concurrent candidates are measured too.
    return (
        njit(nogil=True, fastmath=False)(depth_blocks),
        njit(nogil=True, fastmath=False)(outline_distances),
    )


def fitter_type(*, search_pool=None, kernels=None, candidate_pool=None):
    class CandidateFit(WholeSurfaceFit):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if search_pool is not None:
                self.reference_tree = PooledTree(self.reference_tree, search_pool)
                self.target_tree = PooledTree(self.target_tree, search_pool)
            if candidate_pool is not None:
                self.reference_query_workers = self.target_query_workers = 1

        def _residual_with_distances(self, pose, predicted, model_distance, target_distance):
            if kernels is None:
                return super()._residual_with_distances(pose, predicted, model_distance, target_distance)
            uv = project(predicted, self.k)
            height, width = self.observation["depth"].shape
            coordinates = (np.clip(uv[:, 1], 0, height - 1), np.clip(uv[:, 0], 0, width - 1))
            mask_distance = map_coordinates(self.outside_mask, coordinates, order=1, mode="nearest")
            config = self.config
            measured = kernels[0](
                predicted,
                model_distance,
                target_distance,
                mask_distance,
                self.observation["depth"],
                uv,
                config.depth_tolerance_m,
                config.max_surface_distance_m,
                config.mask_boundary_tolerance_px,
                self.k[0, 0],
            )
            envelope = project(move(self.envelope, pose), self.k)
            polygon = envelope[ConvexHull(envelope).vertices]
            outline = kernels[1](self.mask_outline, polygon, config.mask_boundary_tolerance_px)
            outline = (
                outline
                * np.median(predicted[:, 2])
                / (self.k[0, 0] * config.depth_tolerance_m * np.sqrt(len(self.mask_outline)))
            )
            return np.concatenate((measured, outline))

        def batch_residuals(self, poses):
            if candidate_pool is None:
                return super().batch_residuals(poses)
            # No nested nearest-search workers when whole candidates run together.
            return list(candidate_pool.map(self.residual, poses))

    return CandidateFit


def predict_pairs(model, images_a, images_b, *, device="cuda"):
    """One neural call for independent same-resolution crops, preserving order."""
    import torch

    if len(images_a) != len(images_b):
        raise ValueError("source and target batch sizes differ")
    if not images_a:
        return []
    with torch.inference_mode():
        result = model.predict_correspondences_batched(
            torch.from_numpy(np.stack(images_a)).to(device),
            torch.from_numpy(np.stack(images_b)).to(device),
        )
    flows = result.flow.flow_output.detach().float().cpu().numpy()
    visible = result.covisibility.mask.detach().float().cpu().numpy()
    if len(flows) != len(images_a) or len(visible) != len(images_a):
        raise ValueError("model output changed batch cardinality")
    return list(zip(flows, visible, strict=True))
