"""Depth holes affect metric evidence, not RGB point identity."""

import numpy as np
import pytest

from lerobot.policies.flash.depth_quality import depth_support_mask


def test_depth_support_requires_an_observed_neighborhood():
    depth = np.full((7, 7), 1000, dtype=np.uint16)
    depth[3, 3] = 0
    center_only = depth_support_mask(depth, radius_px=0)
    local = depth_support_mask(depth, radius_px=1)
    assert center_only[3, 4]
    assert not local[3, 4]
    assert local[1, 1]
    assert not local[0, 0]  # The image has no measurements beyond its border.


@pytest.mark.parametrize("radius", [-1, 0.5, True])
def test_depth_support_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        depth_support_mask(np.ones((3, 3), dtype=np.uint16), radius_px=radius)
