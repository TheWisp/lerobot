"""Replay-only filters for testing SDF depth inputs (not production policy)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import minimum_filter


def inside_mask_margin(mask: np.ndarray, margin_px: float) -> np.ndarray:
    """Keep pixels whose centres are more than ``margin_px`` inside the mask."""
    if margin_px < 0:
        raise ValueError("mask margin must be nonnegative")
    if margin_px == 0:
        return np.asarray(mask, dtype=bool)
    distance = cv2.distanceTransform(np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance > margin_px


def old_surface_depth_image(
    volume,
    object_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    shape: tuple[int, int],
    surface_tsdf_max: float,
):
    """Project already fused observed zero-crossings into this camera frame.

    The image stores the nearest previously observed surface at each pixel;
    unknown pixels stay infinite. This is a sparse occlusion diagnostic, not a
    complete object model or a visibility oracle.
    """
    if volume is None:
        return np.full(shape, np.inf, dtype=np.float32)
    if volume.gpu_mode:
        raise RuntimeError("this replay-only filter requires the CPU TSDF backend")
    tsdf = volume._tsdf_vol_cpu
    weight = volume._weight_vol_cpu
    ijk = np.argwhere((weight > 0) & (np.abs(tsdf) < surface_tsdf_max))
    predicted = np.full(shape, np.inf, dtype=np.float32)
    if not len(ijk):
        return predicted
    points = volume._vol_origin + ijk * volume._voxel_size
    pose = np.asarray(object_to_camera)
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    valid_z = camera[:, 2] > 0
    camera = camera[valid_z]
    k = np.asarray(intrinsics)
    px = np.rint(camera[:, 0] * k[0, 0] / camera[:, 2] + k[0, 2]).astype(np.int32)
    py = np.rint(camera[:, 1] * k[1, 1] / camera[:, 2] + k[1, 2]).astype(np.int32)
    valid = (px >= 0) & (px < shape[1]) & (py >= 0) & (py < shape[0])
    np.minimum.at(predicted, (py[valid], px[valid]), camera[valid, 2])
    return predicted


def reject_nearer_than_old_surface(
    depth_m: np.ndarray,
    old_surface_depth_m: np.ndarray,
    radius_px: int,
    gap_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag measured pixels clearly in front of the prior mapped surface.

    An old surface within +/- radius_px in both image axes supplies evidence.
    A truly newly exposed front surface can also be flagged, so this is an
    ablation to measure, not a rule automatically applied to live tracking.
    """
    if radius_px < 0 or gap_m < 0:
        raise ValueError("occlusion radius and gap must be nonnegative")
    if radius_px:
        nearby = minimum_filter(old_surface_depth_m, size=2 * radius_px + 1, mode="constant", cval=np.inf)
    else:
        nearby = old_surface_depth_m
    comparable = np.isfinite(nearby) & (depth_m > 0)
    rejected = comparable & (depth_m < nearby - gap_m)
    return rejected, comparable


def old_voxels_in_front_of_new_depth(
    volume,
    accepted_depth_m: np.ndarray,
    object_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    radius_px: int,
    gap_m: float,
    surface_tsdf_max: float,
) -> tuple[np.ndarray, int]:
    """Find old weighted voxels contradicted by a deeper current object pixel.

    Only the *nearest observed zero-crossing at the exact projected pixel* is
    eligible. A farther, opposite surface cannot be selected on that ray.
    All measured object pixels in the image neighbourhood must be deeper by
    at least ``gap_m``.
    """
    if radius_px < 0 or gap_m < 0:
        raise ValueError("occlusion radius and gap must be nonnegative")
    if volume is None:
        return np.empty((0, 3), dtype=np.int32), 0
    if volume.gpu_mode:
        raise RuntimeError("this replay-only filter requires the CPU TSDF backend")
    surface = (volume._weight_vol_cpu > 0) & (np.abs(volume._tsdf_vol_cpu) < surface_tsdf_max)
    indices = np.argwhere(surface)
    if not len(indices):
        return indices, 0
    points = volume._vol_origin + indices * volume._voxel_size
    pose = np.asarray(object_to_camera)
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    k = np.asarray(intrinsics)
    positive = camera[:, 2] > 0
    px = np.zeros(len(indices), dtype=np.int32)
    py = np.zeros(len(indices), dtype=np.int32)
    px[positive] = np.rint(camera[positive, 0] * k[0, 0] / camera[positive, 2] + k[0, 2]).astype(np.int32)
    py[positive] = np.rint(camera[positive, 1] * k[1, 1] / camera[positive, 2] + k[1, 2]).astype(np.int32)
    height, width = accepted_depth_m.shape
    inside = positive & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    depth_or_infinity = np.where(accepted_depth_m > 0, accepted_depth_m, np.inf)
    if radius_px:
        nearby_new = minimum_filter(depth_or_infinity, size=2 * radius_px + 1, mode="constant", cval=np.inf)
    else:
        nearby_new = depth_or_infinity
    local_new = np.full(len(indices), np.inf, dtype=np.float32)
    local_new[inside] = nearby_new[py[inside], px[inside]]
    front_depth = old_surface_depth_image(
        volume, object_to_camera, intrinsics, accepted_depth_m.shape, surface_tsdf_max
    )
    local_front = np.full(len(indices), np.inf, dtype=np.float32)
    local_front[inside] = front_depth[py[inside], px[inside]]
    on_front_layer = np.isfinite(local_front) & (camera[:, 2] <= local_front + 1e-6)
    contradicted = inside & on_front_layer & np.isfinite(local_new) & (camera[:, 2] < local_new - gap_m)
    selected = indices[contradicted]
    return selected, len(selected)
