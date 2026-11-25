"""Interchangeable 2D matching of a prepared image pair.

Coordinates are pixels in the supplied images (after any caller-owned resize or
crop). Rows retain query order, including rejected predictions. Neither backend
knows about SAM, depth, object ownership, point IDs, or rigid motion. Acceptance
is a backend check, not calibrated confidence or proof of a physical match.

Preparing UFM runs inference once; querying that pair again reuses its field.
LK computes sparse flow on each query. Callers still own frame selection, image
preparation, persistent identities, and conversion back to sensor coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class LKMatchConfig:
    window_size_px: int = 21
    pyramid_levels: int = 3
    forward_backward_limit_px: float = 1.0

    def __post_init__(self):
        if (
            type(self.window_size_px) is not int
            or self.window_size_px < 3
            or self.window_size_px % 2 != 1
            or type(self.pyramid_levels) is not int
            or self.pyramid_levels < 0
            or not np.isfinite(self.forward_backward_limit_px)
            or self.forward_backward_limit_px <= 0
        ):
            raise ValueError("invalid LK matching configuration")


@dataclass(frozen=True)
class UFMMatchConfig:
    covisibility_min: float = 0.5

    def __post_init__(self):
        if not np.isfinite(self.covisibility_min) or not 0 <= self.covisibility_min <= 1:
            raise ValueError("UFM co-visibility threshold must be between zero and one")


@dataclass(frozen=True)
class ImageMatches:
    source_xy: np.ndarray
    target_xy: np.ndarray
    accepted: np.ndarray
    # Backend diagnostics have different meanings and must not be compared as
    # one confidence score. FB error uses the supplied image's pixel units.
    forward_backward_error_px: np.ndarray | None = None
    covisibility: np.ndarray | None = None


class PreparedImagePair(Protocol):
    def match(self, source_xy: np.ndarray) -> ImageMatches: ...


def _queries(points, *, dtype):
    points = np.asarray(points, dtype=dtype)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("matching queries must be finite Nx2 image pixels")
    return points


@dataclass(frozen=True)
class LKImagePair:
    source: np.ndarray
    target: np.ndarray
    config: LKMatchConfig

    def match(self, source_xy: np.ndarray) -> ImageMatches:
        source = _queries(source_xy, dtype=np.float32)
        if not len(source):
            return ImageMatches(source, source.copy(), np.zeros(0, bool), np.empty(0, np.float32))
        before = source.reshape(-1, 1, 2)
        window = (self.config.window_size_px,) * 2
        after, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
            self.source, self.target, before, None, winSize=window, maxLevel=self.config.pyramid_levels
        )
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            self.target, self.source, after, None, winSize=window, maxLevel=self.config.pyramid_levels
        )
        xy = after.reshape(-1, 2)
        error = np.linalg.norm((before - backward).reshape(-1, 2), axis=1)
        height, width = self.target.shape
        good = (
            (forward_ok.ravel() == 1)
            & (backward_ok.ravel() == 1)
            & (error < self.config.forward_backward_limit_px)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        return ImageMatches(source, xy, good, forward_backward_error_px=error)


@dataclass(frozen=True)
class UFMImagePair:
    flow: np.ndarray  # 2,H,W; also consumed by the existing trail renderer
    visibility: np.ndarray  # H,W
    config: UFMMatchConfig

    def match(self, source_xy: np.ndarray) -> ImageMatches:
        source = _queries(source_xy, dtype=float)
        height, width = self.visibility.shape
        inside = (
            (source[:, 0] >= 0)
            & (source[:, 0] <= width - 1)
            & (source[:, 1] >= 0)
            & (source[:, 1] <= height - 1)
        )
        target = np.full_like(source, np.nan)
        visibility = np.full(len(source), np.nan, dtype=self.visibility.dtype)
        points = source[inside]
        # Integer queries use exact indexing, preserving the historical grid.
        # Fractional queries interpolate the same field for persistent tracks.
        lo = np.floor(points).astype(int)
        if np.array_equal(points, lo):
            target[inside] = points + self.flow[:, lo[:, 1], lo[:, 0]].T
            visibility[inside] = self.visibility[lo[:, 1], lo[:, 0]]
            return ImageMatches(
                source,
                target,
                inside & (visibility >= self.config.covisibility_min),
                covisibility=visibility,
            )
        hi = np.minimum(lo + 1, [width - 1, height - 1])
        fraction = points - lo

        def sample(field):
            a = field[..., lo[:, 1], lo[:, 0]]
            b = field[..., lo[:, 1], hi[:, 0]]
            c = field[..., hi[:, 1], lo[:, 0]]
            d = field[..., hi[:, 1], hi[:, 0]]
            x, y = fraction.T
            return (a * (1 - x) + b * x) * (1 - y) + (c * (1 - x) + d * x) * y

        target[inside] = points + sample(self.flow).T
        visibility[inside] = sample(self.visibility)
        accepted = inside & (visibility >= self.config.covisibility_min)
        return ImageMatches(source, target, accepted, covisibility=visibility)


def prepare_image_pair(
    source: np.ndarray,
    target: np.ndarray,
    config: LKMatchConfig | UFMMatchConfig,
    *,
    model=None,
) -> PreparedImagePair:
    """Select a backend explicitly. Both accept RGB uint8 image pairs.

    LK also accepts caller-prepared grayscale. No resize, crop, mask, or frame
    retry is implicit here. The caller retains the images until queries finish.
    UFM model loading and lifetime remain with the existing inference worker.
    """
    if source.shape != target.shape or source.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("matching requires equally shaped uint8 images")
    if isinstance(config, LKMatchConfig):
        if source.ndim == 3 and source.shape[2] == 3:
            source, target = (cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (source, target))
        if source.ndim != 2:
            raise ValueError("LK requires grayscale or RGB images")
        return LKImagePair(source, target, config)
    if not isinstance(config, UFMMatchConfig):
        raise TypeError("unknown image matcher configuration")
    if source.ndim != 3 or source.shape[2] != 3 or model is None:
        raise ValueError("UFM requires RGB images and a loaded model")
    import torch

    with torch.inference_mode():
        output = model.predict_correspondences_batched(
            torch.from_numpy(source.copy()).cuda(), torch.from_numpy(target.copy()).cuda()
        )
    return UFMImagePair(
        output.flow.flow_output[0].detach().float().cpu().numpy(),
        output.covisibility.mask[0].detach().float().cpu().numpy(),
        config,
    )
