"""Batch projection preserves the overlay's path, including clipped points."""

import cv2
import numpy as np
import pytest

from lerobot.gui.api.manipulation import camera, rendering


@pytest.mark.parametrize("count", [0, 1, 200, 500])
def test_batched_path_matches_individual_projection(count):
    rng = np.random.default_rng(8)
    tips = np.repeat(np.eye(4)[None], count, axis=0)
    tips[:, :3, 3] = rng.uniform([-0.2, -0.1, -0.1], [0.2, 0.1, 0.5], (count, 3))
    t_bc = np.eye(4)
    t_bc[:3, 3] = [0.01, -0.01, 0.02]
    intr = {"fx": 600, "fy": 600, "cx": 424, "cy": 240}
    expected = np.zeros((480, 848, 3), np.uint8)
    actual = expected.copy()
    colour = (255, 200, 0)
    pts = [
        p
        for tip in tips[:: max(1, count // 200)]
        if (p := camera._project(t_bc, intr, tip[:3, 3])) is not None
    ]
    for a, b in zip(pts, pts[1:], strict=False):
        cv2.line(expected, a, b, colour, 2)
    if pts:
        cv2.circle(expected, pts[0], 5, colour, -1)
        x, y = pts[-1]
        cv2.rectangle(expected, (x - 4, y - 4), (x + 4, y + 4), colour, -1)
    rendering._draw_path(actual, t_bc, intr, tips)
    np.testing.assert_array_equal(actual, expected)
