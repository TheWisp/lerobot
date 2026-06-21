# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Reprojection overlay: the zero-hardware proof. Draw the reconstructed object
grasp point and the fitted table plane back onto the RGB. If the markers land on
the object and the table grid lies flat on the surface, the reconstruction +
projection math is correct — with no motor command issued.
"""

from __future__ import annotations

import cv2
import numpy as np

from .calib import project_points


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes for a unit ``normal``."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return e1, e2


def draw_overlay(
    rgb: np.ndarray, recon: dict, intr: dict, object_mask: np.ndarray | None = None
) -> np.ndarray:
    """Annotate one RGB frame with the camera-frame reconstruction.

    Draws: object mask contour, object centroid (cyan) + top/grasp point
    (magenta), a table-plane grid (green), and base-frame sanity text.
    """
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    if object_mask is not None:
        cnts, _ = cv2.findContours(object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 200, 255), 2)

    # Table grid (camera frame): a 9x9 lattice on the plane around the object.
    normal, d = recon["table_cam"]["normal"], recon["table_cam"]["d"]
    centroid = recon["object_cam"]["centroid"]
    foot = centroid - (normal @ centroid + d) * normal
    e1, e2 = _plane_basis(normal)
    ticks = np.linspace(-0.08, 0.08, 9)
    grid = np.array([foot + a * e1 + b * e2 for a in ticks for b in ticks])
    gpx = project_points(grid, intr).astype(int)
    for px in gpx:
        cv2.circle(img, tuple(px), 1, (0, 220, 0), -1, cv2.LINE_AA)

    cpx = project_points(centroid[None], intr)[0].astype(int)
    tpx = project_points(recon["object_cam"]["top"][None], intr)[0].astype(int)
    cv2.circle(img, tuple(cpx), 6, (255, 220, 0), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(tpx), 6, (255, 0, 220), -1, cv2.LINE_AA)

    cb = recon["object_centroid"]
    up = recon["table_normal"]
    lines = [
        f"obj base xyz: [{cb[0]:+.3f} {cb[1]:+.3f} {cb[2]:+.3f}] m",
        f"obj height over table: {recon['object_height'] * 1000:+.0f} mm",
        f"table normal . base_up(+z): {up[2]:+.2f}",
    ]
    for i, line in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_image(path: str, bgr: np.ndarray) -> None:
    cv2.imwrite(path, bgr)
