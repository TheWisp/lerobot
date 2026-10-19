"""A pose-frame origin may move while the physical object stays still."""

import numpy as np

from lerobot.policies.flash.continuity import PoseCoastLimit, PoseContinuityBackend
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def _frame(index: int) -> RGBDFrame:
    return RGBDFrame(
        index,
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        float(index),
        float(index),
        "cal",
    )


def test_continuity_checks_object_center_not_arbitrary_pose_origin():
    center = np.array([1.0, 0.0, 0.0])
    rotated = np.eye(4)
    angle = np.deg2rad(8)
    rotated[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    rotated[:3, 3] = center - rotated[:3, :3] @ center

    class Backend:
        def initialize(self, references):
            pass

        def update(self, frame):
            pose = np.eye(4) if frame.frame_id == 0 else rotated
            return (
                PoseEstimate(
                    "object",
                    pose,
                    frame.source_time_s,
                    frame.available_time_s,
                    Evidence.OBSERVED,
                    "v1",
                    (True,) * 3,
                    (True,) * 3,
                ),
            )

    reference = ObjectReference(
        "object", "v1", (ReferenceSnippet("source", 0, 0, "object", np.eye(4), "mask.png", "cal"),)
    )
    backend = PoseContinuityBackend(Backend(), (PoseCoastLimit("object", 3, 0.01, np.deg2rad(10), center),))
    backend.initialize((reference,))
    assert backend.update(_frame(0))[0].evidence == Evidence.OBSERVED
    assert np.linalg.norm(rotated[:3, 3]) > 0.1
    assert backend.update(_frame(1))[0].evidence == Evidence.OBSERVED
