"""A tracker-independent matcher can publish a metric pose or explicit loss."""

import numpy as np
import pytest

from lerobot.policies.flash.matched_rgbd import (
    MatchedRGBDPoseBackend,
    ObjectPointModel,
    PixelMatches,
    ReferencePixels,
    build_object_point_model,
)
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.trajectory import Evidence

POINTS = np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, 0.02, 0], [0, 0, 0]])


class Provider:
    def __init__(self, matches):
        self.items = matches

    def matches(self, frame):
        return self.items


def setup():
    model = ObjectPointModel("tool", "tool-v1", POINTS)
    pixels = np.column_stack((10 + (POINTS[:, 0] + 0.02) * 100, 10 + POINTS[:, 1] * 100))
    match = PixelMatches("tool", "tool-v1", np.arange(len(POINTS)), pixels)
    depth = np.zeros((24, 24), dtype=np.uint16)
    depth[np.rint(pixels[:, 1]).astype(int), np.rint(pixels[:, 0]).astype(int)] = 1000
    intrinsics = np.array([[100.0, 0, 10], [0, 100.0, 10], [0, 0, 1.0]])
    frame = RGBDFrame(
        0, np.zeros((24, 24, 3), dtype=np.uint8), depth, intrinsics, 0.001, np.eye(4), 5.0, 5.1, "camera-v1"
    )
    snippet = ReferenceSnippet("demo/data", 0, 0, "tool", np.eye(4), "mask", "camera-v1")
    reference = ObjectReference("tool", "tool-v1", (snippet,))
    return model, match, frame, reference


def test_registered_pose_uses_valid_aligned_depth_and_version():
    model, match, frame, reference = setup()
    frame.depth[10, 12] = 0  # one depth hole must not erase the other correspondences
    backend = MatchedRGBDPoseBackend(
        (model,), Provider((match,)), threshold_m=0.002, min_inliers=3, trials=20, clock=lambda: 5.2
    )
    backend.initialize((reference,))
    (estimate,) = backend.update(frame)
    assert estimate.evidence == Evidence.OBSERVED
    assert estimate.reference_version == "tool-v1"
    assert estimate.pose[:3, 3] == pytest.approx([0.02, 0, 1], abs=1e-8)
    assert estimate.available_time_s == pytest.approx(5.2)


def test_failed_match_publishes_loss_without_stale_pose():
    model, match, frame, reference = setup()
    provider = Provider((match,))
    backend = MatchedRGBDPoseBackend(
        (model,), provider, threshold_m=0.002, min_inliers=3, trials=20, clock=lambda: 5.2
    )
    backend.initialize((reference,))
    assert backend.update(frame)[0].evidence == Evidence.OBSERVED
    provider.items = ()
    (lost,) = backend.update(frame)
    assert lost.evidence == Evidence.LOST and lost.pose is None
    assert backend.last_failure["tool"] == "no matched pixels"


def test_loss_explains_match_shortage_separately_from_missing_depth():
    model, match, frame, reference = setup()
    provider = Provider((PixelMatches("tool", "tool-v1", match.model_indices[:2], match.live_xy[:2]),))
    backend = MatchedRGBDPoseBackend(
        (model,), provider, threshold_m=0.002, min_inliers=3, trials=20, clock=lambda: 5.2
    )
    backend.initialize((reference,))
    assert backend.update(frame)[0].evidence == Evidence.LOST
    assert backend.last_failure["tool"] == "too few pixel correspondences (2/3)"

    provider.items = (match,)
    for x, y in match.live_xy[:3]:
        frame.depth[round(y), round(x)] = 0
    assert backend.update(frame)[0].evidence == Evidence.LOST
    assert backend.last_failure["tool"] == "too few correspondences with aligned depth (2/5)"


def test_multiple_demo_snippets_share_one_object_frame():
    _, match, frame, _ = setup()
    first_pose = np.eye(4)
    first_pose[:3, 3] = [-0.02, 0, -1]
    second_pose = np.eye(4)
    second_pose[:3, 3] = [-0.01, 0, -1]
    first = ReferenceSnippet("demo/data", 0, 0, "tool", first_pose, "mask-0", "camera-v1")
    second = ReferenceSnippet("demo/data", 0, 1, "tool", second_pose, "mask-1", "camera-v1")
    intrinsics = frame.intrinsics.copy()
    intrinsics[0, 2] += 1
    frame2 = RGBDFrame(1, frame.rgb, frame.depth, intrinsics, 0.001, np.eye(4), 5.0, 5.1, "camera-v1")
    reference = ObjectReference("tool", "tool-v1", (first, second))
    model = build_object_point_model(
        reference,
        (
            ReferencePixels(first, frame, match.live_xy[:3]),
            ReferencePixels(second, frame2, match.live_xy[3:]),
        ),
    )
    assert model.points_object_m == pytest.approx(POINTS)
