# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Interaction-event detection from the object's mask track.

The demo needs one timestamp found automatically: the moment the robot first MOVES
the object. Everything before it is approach (replayable relative to the object);
everything after is manipulation (replayable relative to the end-effector). The
mask track gives it without any force sensing: the object's centroid is still,
then it moves — sustained, to reject jitter, and gated on mask area so a partial
occlusion (the gripper crossing in front) is not mistaken for motion.
"""

from __future__ import annotations

import numpy as np


def detect_interaction(
    centroids: np.ndarray,
    areas: np.ndarray,
    *,
    still_frames: int = 10,
    move_px: float = 4.0,
    sustain: int = 3,
    occlusion_drop: float = 0.6,
) -> int | None:
    """First frame index where the object starts moving. ``None`` if it never does.

    Pre: ``centroids`` (T, 2) float, ``areas`` (T,) — NaN centroid / zero area for
    frames with no mask; T > still_frames. Post: returned index is > still_frames
    and motion is sustained for ``sustain`` consecutive valid frames.

    Occlusion handling: a frame whose area falls below ``occlusion_drop`` x the
    still-phase median is IGNORED (its centroid is the visible fragment's, not the
    object's). SAM under partial occlusion shrinks the mask; the centroid of what
    remains shifts even though the object has not moved — those frames must not
    fire the event.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    assert centroids.ndim == 2 and centroids.shape[1] == 2 and len(areas) == len(centroids)
    n_frames = len(centroids)
    assert still_frames < n_frames, "need a still baseline before motion can be declared"

    base_valid = ~np.isnan(centroids[:still_frames]).any(axis=1)
    assert base_valid.sum() >= 3, "no usable baseline: object not visible at demo start"
    ref = np.nanmedian(centroids[:still_frames][base_valid], axis=0)
    area_ref = float(np.nanmedian(areas[:still_frames][base_valid]))
    assert area_ref > 0

    run, run_start = 0, -1
    for t in range(still_frames, n_frames):
        if np.isnan(centroids[t]).any() or areas[t] < occlusion_drop * area_ref:
            continue  # occluded or lost: not evidence of motion, not evidence of stillness
        if np.linalg.norm(centroids[t] - ref) > move_px:
            if run == 0:
                run_start = t  # occlusions may sit inside the run; report its true start
            run += 1
            if run >= sustain:
                return run_start
        else:
            run = 0
    return None
