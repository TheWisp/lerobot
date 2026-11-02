"""UFM RGB-D motion proposal shared by the experiment and live debugger.

The fixed 320px crop and matching/RANSAC defaults reproduce the benchmark recipe.
The final pose is corrected by surface_grounding; these matches are not confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .image_matching import UFMMatchConfig, prepare_image_pair
from .surface_observation import move


@dataclass(frozen=True)
class SurfaceMotionConfig:
    # Defaults reproduce the UFM comparison; these are tunable heuristics.
    crop_min_px: int = 160
    crop_max_px: int = 480
    crop_extent_multiplier: float = 2.5
    input_size_px: int = 320
    grid_step_px: int = 4
    covisibility_min: float = 0.5
    min_ransac_pairs: int = 6
    ransac_distance_m: float = 0.006
    ransac_confidence: float = 0.99
    refinement_steps: int = 2
    connected_dilation_px: int = 7


_DEFAULT = SurfaceMotionConfig()


def crop_pair(
    a: dict, b: dict, config: SurfaceMotionConfig = _DEFAULT
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    coords = np.concatenate((np.argwhere(a["mask"]), np.argwhere(b["mask"])))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    size = max(config.crop_min_px, int(max(y1 - y0, x1 - x0) * config.crop_extent_multiplier))
    # The context limit must never crop away part of a large selected object.
    # Network input size stays fixed, so this does not increase GPU work.
    size = max(int(max(y1 - y0, x1 - x0)) + 1, min(size, config.crop_max_px))
    x = (x0 + x1) / 2 - size / 2
    y = (y0 + y1) / 2 - size / 2
    resolution = config.input_size_px
    matrix = np.array(
        [[resolution / size, 0, -x * resolution / size], [0, resolution / size, -y * resolution / size]],
        np.float32,
    )
    images = [
        cv2.warpAffine(
            item["rgb"],
            matrix,
            (resolution, resolution),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        for item in (a, b)
    ]
    return images[0], images[1], (x, y, size)


def full_pixels(
    crop_pixels: np.ndarray, crop: tuple[float, float, float], config: SurfaceMotionConfig = _DEFAULT
) -> np.ndarray:
    x, y, size = crop
    # crop_pair uses warpAffine, whose integer pixel centres map by this exact
    # affine transform. A resize-style half-pixel correction is not its inverse.
    return crop_pixels * size / config.input_size_px + np.array([x, y])


def measured_pairs(
    a: dict, b: dict, uv_a: np.ndarray, uv_b: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ua = np.rint(uv_a).astype(np.int32)
    ub = np.rint(uv_b).astype(np.int32)
    height, width = a["depth"].shape
    bounds = (
        (ua[:, 0] >= 0)
        & (ua[:, 0] < width)
        & (ua[:, 1] >= 0)
        & (ua[:, 1] < height)
        & (ub[:, 0] >= 0)
        & (ub[:, 0] < width)
        & (ub[:, 1] >= 0)
        & (ub[:, 1] < height)
    )
    ua, ub = ua[bounds], ub[bounds]
    good = a["usable"][ua[:, 1], ua[:, 0]] & b["usable"][ub[:, 1], ub[:, 0]]
    ua, ub = ua[good], ub[good]
    z_a, z_b = a["depth"][ua[:, 1], ua[:, 0]], b["depth"][ub[:, 1], ub[:, 0]]

    def lift(uv: np.ndarray, z: np.ndarray, camera: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                (uv[:, 0] - camera[0, 2]) * z / camera[0, 0],
                (uv[:, 1] - camera[1, 2]) * z / camera[1, 1],
                z,
            )
        )

    return lift(ua, z_a, a.get("intrinsics", intrinsics)), lift(ub, z_b, b.get("intrinsics", intrinsics))


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x, y = source.mean(axis=0), target.mean(axis=0)
    u, _, vh = np.linalg.svd((source - x).T @ (target - y))
    fix = np.diag([1, 1, np.linalg.det(u @ vh)])
    rotation = (u @ fix @ vh).T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = y - rotation @ x
    return transform


def robust_fit(
    source: np.ndarray, target: np.ndarray, config: SurfaceMotionConfig = _DEFAULT
) -> tuple[np.ndarray | None, int, float]:
    if len(source) < config.min_ransac_pairs:
        return None, 0, float("nan")
    ok, _, inliers = cv2.estimateAffine3D(
        source, target, ransacThreshold=config.ransac_distance_m, confidence=config.ransac_confidence
    )
    if not ok or inliers is None or int(inliers.sum()) < config.min_ransac_pairs:
        return None, 0, float("nan")
    active = inliers.ravel() > 0
    pose = rigid_fit(source[active], target[active])
    for _ in range(config.refinement_steps):
        residuals = np.linalg.norm(move(source, pose) - target, axis=1)
        active = residuals < config.ransac_distance_m
        if active.sum() < config.min_ransac_pairs:
            return None, int(active.sum()), float("nan")
        pose = rigid_fit(source[active], target[active])
    return (
        pose,
        int(active.sum()),
        float(np.median(np.linalg.norm(move(source[active], pose) - target[active], axis=1))),
    )


def ufm_pairs(
    model,
    image_a: np.ndarray,
    image_b: np.ndarray,
    crop,
    a: dict,
    b: dict,
    intrinsics: np.ndarray,
    config: SurfaceMotionConfig = _DEFAULT,
    *,
    evidence: dict | None = None,
):
    pair = prepare_image_pair(image_a, image_b, UFMMatchConfig(config.covisibility_min), model=model)
    step, end = config.grid_step_px, config.input_size_px - config.grid_step_px
    ys, xs = np.mgrid[step:end:step, step:end:step]
    ys, xs = ys.ravel(), xs.ravel()
    matches = pair.match(np.column_stack((xs, ys)).astype(float))
    keep = matches.accepted
    uv_a, uv_b = matches.source_xy[keep], matches.target_xy[keep]
    source_uv, target_uv = full_pixels(uv_a, crop, config), full_pixels(uv_b, crop, config)
    source, target = measured_pairs(a, b, source_uv, target_uv, intrinsics)
    if evidence is not None:
        # Observe the existing matcher without changing sampling, rejection or fit.
        # Include low-visibility predictions, but only source pixels on this object.
        all_a, all_b = matches.source_xy, matches.target_xy
        all_a, all_b = full_pixels(all_a, crop, config), full_pixels(all_b, crop, config)
        pa, pb = np.rint(all_a).astype(int), np.rint(all_b).astype(int)
        h, w = a["depth"].shape
        inside_a = (pa[:, 0] >= 0) & (pa[:, 0] < w) & (pa[:, 1] >= 0) & (pa[:, 1] < h)
        on_object = np.zeros(len(pa), dtype=bool)
        on_object[inside_a] = a["mask"][pa[inside_a, 1], pa[inside_a, 0]]
        indices = np.flatnonzero(on_object & np.isfinite(all_b).all(axis=1))
        pa, pb = pa[indices], pb[indices]
        # Stages: 0 low co-visibility, 1 outside target mask, 2 depth/cleanup,
        # 3 usable 3D pair, 4 also agrees with the resulting rigid motion.
        stage = np.zeros(len(indices), dtype=np.uint8)
        covisible = keep[indices]
        stage[covisible] = 1
        inside_b = (pb[:, 0] >= 0) & (pb[:, 0] < w) & (pb[:, 1] >= 0) & (pb[:, 1] < h)
        rows = np.flatnonzero(covisible & inside_b)
        rows = rows[b["mask"][pb[rows, 1], pb[rows, 0]]]
        stage[rows] = 2
        usable = a["usable"][pa[rows, 1], pa[rows, 0]] & b["usable"][pb[rows, 1], pb[rows, 0]]
        stage[rows[usable]] = 3
        evidence.update(
            source_uv=all_a[indices],
            target_uv=all_b[indices],
            stage=stage,
            # Only optional diagnostic recording uses this full field. Fitting
            # still receives the same sampled pairs. Do not serialize the array.
            flow=pair.flow,
            crop=crop,
        )
    return source, target
