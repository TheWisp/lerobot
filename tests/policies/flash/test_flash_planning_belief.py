"""The planner sees one committed scene and held-object relation revision."""

import numpy as np

from lerobot.policies.flash.planning_belief import PlanningBeliefStore
from lerobot.policies.flash.trajectory import AttachmentState, Evidence, PoseEstimate


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def _scene(x: float) -> PoseEstimate:
    return PoseEstimate("target", _pose(x), 0, 0, Evidence.OBSERVED, "target-v1", (True,) * 3, (True,) * 3)


def test_planning_belief_publishes_matching_scene_and_attachment_revisions():
    belief = PlanningBeliefStore(max_attachment_age_s=2)
    first = AttachmentState(_pose(0.1), 0, 0, "held-v1")
    second = AttachmentState(_pose(0.2), 0.5, 0.5, "held-v1")
    belief.publish_scene(1, {"target": _scene(1)}, 1, "held", first)
    version, poses, attached = belief.snapshot(0.6, "held")
    assert (version, poses["target"].pose[0, 3], attached.gripper_from_object[0, 3]) == (1, 1, 0.1)

    belief.publish_scene(2, {"target": _scene(2)}, 2, "held", second)
    belief.publish_scene(1, {"target": _scene(99)}, 1, "held", first)
    version, poses, attached = belief.snapshot(0.6, "held")
    assert (version, poses["target"].pose[0, 3], attached.gripper_from_object[0, 3]) == (2, 2, 0.2)
    assert belief.snapshot(0.6, "other")[2] is None
    assert belief.snapshot(2.6, "held")[2] is None

    belief.publish_attachment(3, None, None)
    version, poses, attached = belief.snapshot(0.6, "held")
    assert version == 2 and poses["target"].pose[0, 3] == 2 and attached is None
