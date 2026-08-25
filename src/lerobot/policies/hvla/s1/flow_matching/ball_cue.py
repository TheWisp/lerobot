# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Turn a ball segmentation into what the policy consumes.

One implementation, called from both sides: training reads the mask from the
dataset's cached RLE column, inference gets it from SAM3 in the observation
path. Only the SOURCE of the mask differs -- everything downstream is this
file, so the two paths cannot drift into computing subtly different cues.

Why both a coordinate and an image: at the 224x224 training size the ball is
~12 px across against DINOv2's 14 px patch (measured on the experiment
dataset), so it is sub-patch and effectively unresolvable in any rendered
view. The coordinate sidesteps resolution entirely; the rendered view keeps
the ball spatially grounded in the scene. Which matters is the experiment.
"""

from __future__ import annotations

import numpy as np

# What a frame with no detection reports. The coordinates are outside [0, 1]
# so they cannot be confused with a real position, and `visible` is the flag a
# model should actually branch on. Measured miss rate on the experiment
# dataset: 3.5% of frames.
NOT_VISIBLE = (-1.0, -1.0, 0.0)

# The rendered view's key. Fixed rather than derived because it names THIS
# feature's output, not a camera the setup happens to have -- every consumer
# must agree on it, and a checkpoint records it verbatim.
BALL_VIEW_KEY = "observation.images.ball_view"  # hardcode-ok: synthetic input this feature defines


def ball_cue(mask: np.ndarray | None) -> tuple[float, float, float]:
    """``(x, y, visible)`` from a boolean mask, normalised to [0, 1].

    Preconditions: ``mask`` is a 2-D boolean array in image orientation, or
    None when nothing was detected. Postcondition: x and y are the mask's
    centroid as a fraction of width and height, or NOT_VISIBLE.

    Normalised rather than pixel coordinates so the cue is independent of the
    capture resolution -- a pixel value would silently change meaning the day a
    camera or a resize does.
    """
    if mask is None or not mask.any():
        return NOT_VISIBLE
    ys, xs = np.nonzero(mask)
    h, w = mask.shape[:2]
    return float(xs.mean() / w), float(ys.mean() / h), 1.0


def render_ball_view(frame_rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """The frame with everything outside the mask blacked out.

    Black rather than noise: a constant is one value the encoder can learn to
    ignore, where noise would occupy capacity and vary per frame. A frame with
    no detection renders all black, which is the same thing the policy sees
    whenever the ball is genuinely absent -- so the miss case is in
    distribution rather than a surprise at inference.
    """
    out = np.zeros_like(frame_rgb)
    if mask is None or not mask.any():
        return out
    if mask.shape[:2] != frame_rgb.shape[:2]:
        raise ValueError(
            f"mask {mask.shape[:2]} does not match frame {frame_rgb.shape[:2]}; "
            "the cue must be computed against the frame it will be shown with"
        )
    out[mask] = frame_rgb[mask]
    return out
