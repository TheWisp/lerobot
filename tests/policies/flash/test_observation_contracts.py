"""RGB-D observation contracts."""

import numpy as np
import pytest

from lerobot.policies.flash.observations import RGBDFrame


def pose(x=0.0, y=0.0, z=0.0):
    value = np.eye(4)
    value[:3, 3] = x, y, z
    return value


def test_rgbd_frame_requires_aligned_depth_and_converts_it_to_metres():
    frame = RGBDFrame(
        1,
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.full((2, 3), 1000, dtype=np.uint16),
        np.eye(3),
        0.001,
        pose(),
        1.0,
        1.05,
        "cal1",
    )
    np.testing.assert_allclose(frame.metric_depth(), 1.0)
    with pytest.raises(ValueError, match="aligned"):
        RGBDFrame(2, frame.rgb, np.zeros((3, 2), dtype=np.uint16), np.eye(3), 0.001, pose(), 1.0, 1.1, "cal1")
