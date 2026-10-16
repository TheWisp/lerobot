"""A common measurement feeds attachment math without using prediction as evidence."""

from types import SimpleNamespace

import numpy as np

from lerobot.gui.api.manipulation import held_views, pose_updates, runtime


def test_pose_source_is_role_independent_and_rejects_stale_or_predicted(monkeypatch):
    demo = SimpleNamespace(objects={"a": {"frame": 1}, "b": {"frame": 2}})
    tr = SimpleNamespace(on=True, model={"capture_mono": 12.0, "t": 100.0, "delta_cam": np.eye(4)})
    teach = SimpleNamespace(keypoints={"ref": {"delta": np.eye(4)}})
    target = SimpleNamespace(last={}, hold_views=[(100.0, np.eye(4))])
    monkeypatch.setattr(runtime._state, "demo", demo)
    monkeypatch.setattr(runtime._state, "track", tr)
    monkeypatch.setattr(runtime._state, "teach", teach)
    monkeypatch.setattr(runtime._state, "targets", {"b": target})
    monkeypatch.setattr(pose_updates.planning, "_session_object", lambda d: "a")
    for name in ["a", "b"]:
        observation = pose_updates.latest_verified(demo, name)
        assert observation.object == name and observation.capture_mono == 12.0
        np.testing.assert_array_equal(observation.reference_to_camera, np.eye(4))
    # A measured patch belongs to the teach reference, while the published
    # observation is in the canonical demo reference. Transform both together.
    teach.keypoints["ref"]["delta"][0, 3] = 0.1
    tr.model["surface_constraint"] = {"reference_point": [0.2, 0.3, 0.4]}
    obs = pose_updates.latest_verified(demo, "a")
    np.testing.assert_allclose(obs.reference_point_cam, [0.1, 0.3, 0.4])
    np.testing.assert_allclose(
        (obs.reference_to_camera @ np.r_[obs.reference_point_cam, 1])[:3], [0.2, 0.3, 0.4]
    )
    tr.model["surface_constraint"] = None
    assert pose_updates.latest_verified(demo, "a").reference_point_cam is None
    tr.model["delta_cam"] = None
    tr.model["held_delta_cam"] = np.eye(4)
    target.hold_views = [(99.0, np.eye(4))]
    assert pose_updates.latest_verified(demo, "a") is None
    assert pose_updates.latest_verified(demo, "b") is None
    tr.model["capture_mono"] = None
    assert pose_updates.latest_verified(demo, "a") is None


def test_attachment_adapter_cancels_gripper_motion_and_preserves_ambiguous_rotation():
    t_bc = np.eye(4)
    t_bc[:3, 3] = [0.4, 0.1, 0.2]
    desired = np.eye(4)
    desired[:3, 3] = [0.02, 0, 0]
    live = np.eye(4)
    live[:3, 3] = [0.01, 0, 0]
    samples = []
    for i in range(5):
        tip = np.eye(4)
        tip[:3, 3] = [i * 0.01, 0.05, 0.08]
        pose = np.linalg.inv(t_bc) @ tip @ live @ t_bc
        observation = pose_updates.PoseObservation("object", 1, float(i), pose)
        samples.append(pose_updates.attachment_sample(observation, tip, t_bc, desired, np.array([0, 0, 0.1])))
    result = held_views.consensus(samples, np.eye(4))
    np.testing.assert_allclose(result["fix"][:3, 3], [0.01, 0, 0], atol=1e-12)
    assert result["orientation"] == "predicted"  # one reference cannot resolve symmetric roll


async def _exercise_regular_observer(monkeypatch):
    import asyncio
    import time

    from lerobot.gui.api import jog

    class Kin:
        def forward_kinematics(self, q):
            return np.eye(4)

    monkeypatch.setattr(jog, "kinematics", lambda: Kin())
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    monkeypatch.setattr(jog, "joints_at", lambda *args: dict.fromkeys(MOTOR_NAMES, 0))
    monkeypatch.setattr(held_views.planning, "_object_points", lambda *args: np.zeros((3, 3)))
    monkeypatch.setattr(runtime, "ACT_TICK_S", 0.001)
    delta = np.eye(4)
    delta[0, 3] = 0.01
    monkeypatch.setattr(
        pose_updates,
        "latest_verified",
        lambda *args: pose_updates.PoseObservation(
            "object", 1, time.perf_counter(), delta, reference_point_cam=np.zeros(3)
        ),
    )
    received = []
    bank = asyncio.get_running_loop().create_future()
    bank.set_result([])
    await asyncio.wait_for(
        held_views.watch(
            object(),
            "object",
            np.eye(4),
            lambda: np.eye(4),
            lambda: bool(received),
            received.append,
            lambda *args: None,
            prepared_references=bank,
            demo_attachment=np.eye(4),
        ),
        timeout=2,
    )
    assert received[0]["source"] == "verified tracking"
    np.testing.assert_allclose(received[0]["fix"][:3, 3], [-0.01, 0, 0])


def test_normal_pose_updates_correct_attachment_without_any_registration_job(monkeypatch):
    import asyncio

    asyncio.run(_exercise_regular_observer(monkeypatch))
