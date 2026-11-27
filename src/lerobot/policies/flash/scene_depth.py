"""Current-camera depth evidence shared by groups, pose checks and surface cleanup."""

from dataclasses import dataclass

import cv2
import numpy as np

from .surface_observation import move


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return points[:, :2] / points[:, 2, None] @ k[:2, :2].T + k[:2, 2]


def scene_depth_evidence(predicted, intrinsics, scene_nearest_depth, tolerance_m):
    """Per-point free-space evidence shared by pose checking and map cleanup.

    A point is fully contradicted only when all four interpolation corners have
    reliable depth farther than the predicted surface by the pose tolerance.
    Missing depth and a nearer occluder provide no such evidence.
    """
    uv = project(predicted, intrinsics)
    h, w = scene_nearest_depth.shape
    valid = (
        np.isfinite(uv).all(1)
        & (predicted[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= w - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= h - 1)
    )
    x, y = np.nan_to_num(uv[:, 0]).clip(0, w - 1), np.nan_to_num(uv[:, 1]).clip(0, h - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    weights = np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
    near = np.column_stack(
        (
            scene_nearest_depth[y0, x0],
            scene_nearest_depth[y0, x1],
            scene_nearest_depth[y1, x0],
            scene_nearest_depth[y1, x1],
        )
    )
    known = (near > 0) & valid[:, None]
    gap = near - predicted[:, 2, None]
    exposed = np.sum(weights * known * np.clip((gap + tolerance_m) / tolerance_m, 0, 1), axis=1)
    contradicted = np.sum(weights * known * (gap > tolerance_m), axis=1)
    free = np.sqrt(np.sum(weights * known * np.maximum(gap - tolerance_m, 0) ** 2, axis=1))
    return free, exposed, contradicted


@dataclass(frozen=True)
class SceneDepth:
    """Mask-independent current depth for checking carried robot-base points.

    The frame owner supplies metric depth and the already computed reliability
    mask. A nearer surface or a hole anywhere in the pixel neighborhood protects
    a possible occlusion; only fully supported farther depth rejects a point.
    """

    nearest_depth: np.ndarray
    intrinsics: np.ndarray
    camera_from_base: np.ndarray
    tolerance_m: float

    @classmethod
    def from_frame(cls, frame, filters, *, tolerance_m, radius_px):
        valid = filters.reliable & np.isfinite(filters.depth) & (filters.depth > 0)
        clean = np.where(valid, filters.depth, 0).astype(np.float32, copy=False)
        radius = int(np.ceil(radius_px))
        nearest = cv2.erode(
            clean,
            np.ones((2 * radius + 1,) * 2, np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return cls(nearest, frame.intrinsics, np.linalg.inv(frame.base_from_camera), tolerance_m)

    def contradicted(self, base_xyz):
        camera = move(base_xyz, self.camera_from_base)
        _, _, bad = scene_depth_evidence(camera, self.intrinsics, self.nearest_depth, self.tolerance_m)
        return bad >= 1 - 1e-6
