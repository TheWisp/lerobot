"""A delayed pose anchor must replay against its own RGB-D frame."""

import itertools
import time
from threading import Event, Thread

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.causal_points import CausalRigidPointTracker
from lerobot.policies.flash.causal_service import (
    CausalPoseService,
    CausalTrackingConfig,
    _Anchor,
    _AttachmentEvent,
)
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.scene_groups import ScenePointTracker, SceneTracks
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def frame(image, frame_id):
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    return RGBDFrame(
        frame_id,
        rgb,
        np.full(image.shape, 1000, dtype=np.uint16),
        np.array([[100.0, 0, 48.0], [0, 100.0, 48.0], [0, 0, 1]]),
        0.001,
        np.eye(4),
        frame_id / 30,
        frame_id / 30,
        "camera-v1",
    )


def test_slow_camera_processing_records_stage_and_source_frame(monkeypatch):
    service = CausalPoseService(LatestPoseStore(), max_frame_gap_s=0.01)
    update = service.scene.update

    def slow_update(item):
        scene = update(item)
        time.sleep(0.02)
        return scene

    monkeypatch.setattr(service.scene, "update", slow_update)
    service.start()
    service.submit_frame(frame(np.zeros((96, 96), dtype=np.uint8), 4))
    service.close()

    assert service.error is None
    assert service.slow_work_count == 1
    event = service.slow_work[0]
    assert event["kind"] == "frame"
    assert event["thread_cpu_s"] >= 0
    assert event["environment_stages"] == {}
    assert event["frame_id"] == 4
    assert event["stages_s"]["scene_update"] >= 0.02
    assert event["duration_s"] >= event["stages_s"]["scene_update"]


def test_planning_snapshot_excludes_an_unfinished_scene_event(monkeypatch):
    store = LatestPoseStore()
    service = CausalPoseService(store)
    entered = Event()
    release = Event()
    update = service.scene.update
    estimate = PoseEstimate(
        "object", np.eye(4), 0, 0, Evidence.OBSERVED, "reference-v1", (True,) * 3, (True,) * 3
    )

    def paused_update(item):
        store.publish((estimate,))
        entered.set()
        assert release.wait(2)
        return update(item)

    monkeypatch.setattr(service.scene, "update", paused_update)
    service.start()
    try:
        service.submit_frame(frame(np.zeros((96, 96), dtype=np.uint8), 0))
        assert entered.wait(2)
        assert store.snapshot()[0] == 1
        assert service.committed_snapshot()[0] == 0
        assert service.completed_sequence == 0
        release.set()
        deadline = time.monotonic() + 2
        while service.completed_sequence < 1 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.committed_snapshot()[0] == 1
    finally:
        release.set()
        service.close()


def test_environment_timing_distinguishes_group_fitting_from_worker_wait():
    rng = np.random.default_rng(65)
    points = rng.uniform((-0.08, -0.08, 0.98), (0.08, 0.08, 1.02), (32, 3))
    pixels = np.column_stack((np.arange(32) + 20, np.arange(32) + 30))
    first = SceneTracks(0, 0.0, np.arange(32), pixels, points)
    second = SceneTracks(1, 1 / 30, np.arange(32), pixels, points.copy())
    pose = np.eye(4)
    pose[2, 3] = 1.0
    estimate = PoseEstimate(
        "object", pose, 0.0, 0.0, Evidence.OBSERVED, "reference-v1", (True,) * 3, (True,) * 3
    )
    mask = np.zeros((96, 96), dtype=bool)
    service = CausalPoseService(LatestPoseStore())
    timings = {}

    service._observe_environment(estimate, first, {"object": mask}, timings=timings)
    assert "seed_relationship" in timings
    assert timings["seed_relationship"]["wall_s"] >= 0
    assert timings["seed_relationship"]["thread_cpu_s"] >= 0
    assert "record_observation" in timings

    service.candidate_relationships.clear()
    service._observe_environment(estimate, second, {"object": mask}, timings=timings)
    assert "establish_relationship" in timings
    assert timings["establish_relationship"]["wall_s"] >= 0
    assert timings["establish_relationship"]["thread_cpu_s"] >= 0


def test_queued_camera_frames_keep_the_attachment_state_at_capture_time(monkeypatch):
    service = CausalPoseService(LatestPoseStore())
    entered = Event()
    release = Event()
    update = service.scene.update

    def paused_update(item):
        if item.frame_id == 0:
            entered.set()
            assert release.wait(2)
        return update(item)

    monkeypatch.setattr(service.scene, "update", paused_update)
    image = np.zeros((96, 96), dtype=np.uint8)
    service.start()
    try:
        service.submit_frame(frame(image, 0))
        assert entered.wait(2)
        for frame_id in range(1, 6):
            service.submit_frame(frame(image, frame_id))
        # The motor has already grasped and released while camera processing
        # is behind. Neither event may be applied to earlier queued images.
        service.set_attached_role("object", source_time_s=2.5 / 30)
        service.set_attached_role(None, source_time_s=4.5 / 30)
        release.set()
        deadline = time.monotonic() + 2
        while service.frames < 6 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.frames == 6
        assert [service.attachment_by_frame[index] for index in range(6)] == [
            None,
            None,
            None,
            "object",
            "object",
            None,
        ]
    finally:
        release.set()
        service.close()


def test_late_object_pose_replays_grasp_and_release_in_camera_source_order():
    image = np.random.default_rng(58).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(3)]
    mask = np.ones(image.shape, dtype=bool)

    def anchor(role: str, source: RGBDFrame) -> _Anchor:
        estimate = PoseEstimate(
            role,
            np.eye(4),
            source.source_time_s,
            source.available_time_s,
            Evidence.OBSERVED,
            "reference-v1",
            (True,) * 3,
            (True,) * 3,
        )
        return _Anchor(source.frame_id, (estimate,), {role: mask})

    def run(*, late: bool):
        store = LatestPoseStore()
        revisions = []
        service = CausalPoseService(
            store,
            on_source_order_replay=lambda *args: revisions.append((*args, store.snapshot()[0])),
        )
        service._frame(frames[0])
        if not late:
            service._anchor(anchor("object", frames[0]))
            service._drain_pending()
        service.set_attached_role("object", source_time_s=0.5 / 30)
        service._frame(frames[1])
        if not late:
            service._anchor(anchor("object", frames[1]))
            service._drain_pending()
        service.set_attached_role(None, source_time_s=1.5 / 30)
        service._frame(frames[2])
        if late:
            service._anchor(anchor("object", frames[0]))
            service._anchor(anchor("object", frames[1]))
        service._drain_pending()
        return service, store.snapshot()[1]["object"], revisions

    _, expected, _ = run(late=False)
    service, actual, revisions = run(late=True)
    assert actual.source_time_s == expected.source_time_s
    assert actual.evidence == expected.evidence
    if expected.pose is None:
        assert actual.pose is None
    else:
        np.testing.assert_allclose(actual.pose, expected.pose, atol=0.005)
    assert service.anchor_decisions_by_source[(0, "object")].outcome.startswith("replayed_")
    assert service.anchor_decisions_by_source[(1, "object")].outcome.startswith("replayed_")
    assert revisions
    assert revisions[-1][0] == "object"
    assert revisions[-1][1:3] == (frames[0].source_time_s, frames[-1].source_time_s)
    assert all(
        pose.role == "object" and pose.source_time_s == source.source_time_s
        for pose, source, _mask in revisions[-1][3]
    )
    assert revisions[-1][4] == revisions[-1][5] + 1
    assert revisions[-1][4] <= service.store.snapshot()[0]

    # An unrelated late observation does not need event replay.
    service._anchor(anchor("target", frames[0]))
    assert service.anchor_decisions_by_source[(0, "target")].outcome == "accepted"
    assert service.store.snapshot()[1]["target"].pose is not None


@pytest.mark.parametrize("release_before_next_camera", [False, True])
def test_late_pregrasp_image_replays_exact_attachment_events_without_current_scene_pose(
    release_before_next_camera,
):
    image = np.random.default_rng(61).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(3)]
    mask = np.zeros(image.shape, dtype=bool)
    mask[47:49, 47:49] = True  # insufficient support for a new independent point track
    decisions = []
    store = LatestPoseStore()
    service = CausalPoseService(store, on_anchor_decision=decisions.append)
    service._frame(frames[0])
    service.set_attached_role("object", source_time_s=0.25 / 30)
    if release_before_next_camera:
        service.set_attached_role(None, source_time_s=0.75 / 30)
    service._frame(frames[1])
    service._frame(frames[2])

    estimate = PoseEstimate(
        "object", np.eye(4), 0, 0, Evidence.OBSERVED, "reference-v1", (True,) * 3, (True,) * 3
    )
    service._anchor(_Anchor(0, (estimate,), {"object": mask}))
    assert service.anchor_decisions_by_source[(0, "object")].outcome == "event_replay_pending"
    service._drain_pending()

    current = store.snapshot()[1]["object"]
    assert current.evidence == Evidence.LOST
    assert current.pose is None
    assert current.source_frame_id == frames[1].frame_id
    assert current.estimator == "attachment_transition_unknown"
    assert service.anchor_decisions_by_source[(0, "object")].outcome == "replayed_accepted"
    assert decisions[-1].accepted is not None
    assert decisions[-1].accepted.source_frame_id == frames[0].frame_id
    assert "object" not in service.observed
    assert "object" not in service.relationships


@pytest.mark.parametrize("release_before_last_camera", [False, True])
def test_late_pregrasp_image_replays_gripper_prediction_at_each_camera_time(
    release_before_last_camera,
):
    image = np.zeros((96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(3)]
    mask = np.zeros(image.shape, dtype=bool)
    mask[47:49, 47:49] = True

    def predict(role, source):
        pose = np.eye(4)
        pose[0, 3] = source.frame_id * 0.01
        return PoseEstimate(
            role,
            pose,
            source.source_time_s,
            source.available_time_s,
            Evidence.PREDICTED,
            "reference-v1",
            (True,) * 3,
            (True,) * 3,
            estimator="gripper_motion",
            source_frame_id=source.frame_id,
        )

    seed = PoseEstimate(
        "object", np.eye(4), 0, 0, Evidence.OBSERVED, "reference-v1", (True,) * 3, (True,) * 3
    )

    def run(late):
        service = CausalPoseService(LatestPoseStore(), kinematic_prediction_at=predict)
        service._frame(frames[0])
        if not late:
            service._anchor(_Anchor(0, (seed,), {"object": mask}))
        service.set_attached_role("object", source_time_s=0.5 / 30)
        service._frame(frames[1])
        if release_before_last_camera:
            service.set_attached_role(None, source_time_s=1.5 / 30)
        service._frame(frames[2])
        if late:
            service._anchor(_Anchor(0, (seed,), {"object": mask}))
            service._drain_pending()
        return service, service.store.snapshot()[1]["object"]

    _, source_order = run(late=False)
    delayed_service, delayed = run(late=True)
    assert delayed_service.event_replays == 1
    assert delayed.source_frame_id == source_order.source_frame_id == 2
    if release_before_last_camera:
        assert delayed.estimator == source_order.estimator == "attachment_transition_unknown"
        assert delayed.evidence == source_order.evidence == Evidence.LOST
        assert delayed.pose is source_order.pose is None
    else:
        assert delayed.estimator == source_order.estimator == "gripper_motion"
        assert delayed.evidence == source_order.evidence == Evidence.PREDICTED
        np.testing.assert_allclose(delayed.pose, source_order.pose)


def test_attachment_transition_withdraws_old_scene_pose_until_new_evidence():
    image = np.random.default_rng(59).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(3)]
    mask = np.zeros(image.shape, dtype=bool)
    mask[47:49, 47:49] = True  # too little surface to seed an independent point track
    store = LatestPoseStore()
    service = CausalPoseService(store)
    service._frame(frames[0])
    estimate = PoseEstimate(
        "object", np.eye(4), 0, 0, Evidence.OBSERVED, "reference-v1", (True,) * 3, (True,) * 3
    )
    service._anchor(_Anchor(0, (estimate,), {"object": mask}))
    assert store.snapshot()[1]["object"].evidence == Evidence.OBSERVED

    service.set_attached_role("object", source_time_s=0.5 / 30)
    service._frame(frames[1])
    during_hold = store.snapshot()[1]["object"]
    assert during_hold.evidence == Evidence.LOST
    assert during_hold.source_frame_id == frames[1].frame_id

    service.set_attached_role(None, source_time_s=1.5 / 30)
    service._frame(frames[2])
    after_release = store.snapshot()[1]["object"]
    assert after_release.evidence == Evidence.LOST
    assert after_release.source_frame_id == frames[2].frame_id

    # A current image can establish the object again after release.
    new_observation = PoseEstimate(
        "object",
        np.eye(4),
        frames[2].source_time_s,
        frames[2].available_time_s,
        Evidence.OBSERVED,
        "reference-v1",
        (True,) * 3,
        (True,) * 3,
    )
    service._anchor(_Anchor(2, (new_observation,), {"object": mask}))
    assert store.snapshot()[1]["object"].evidence == Evidence.OBSERVED


def test_gripper_prediction_fills_occluded_held_scene_pose_but_stops_at_release():
    image = np.zeros((96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(4)]
    calls = []

    def predict(role, source):
        calls.append((role, source.frame_id))
        pose = np.eye(4)
        pose[0, 3] = source.frame_id * 0.01
        return PoseEstimate(
            role,
            pose,
            source.source_time_s,
            source.available_time_s,
            Evidence.PREDICTED,
            "reference-v1",
            (True,) * 3,
            (True,) * 3,
            estimator="gripper_motion",
            last_direct_observation_s=0.0,
            source_frame_id=source.frame_id,
        )

    store = LatestPoseStore()
    service = CausalPoseService(store, kinematic_prediction_at=predict)
    service._frame(frames[0])
    service.set_attached_role("object", source_time_s=0.5 / 30)
    service._frame(frames[1])
    during_hold = store.snapshot()[1]["object"]
    assert during_hold.evidence == Evidence.PREDICTED
    assert during_hold.estimator == "gripper_motion"
    assert during_hold.pose[0, 3] == pytest.approx(0.01)
    assert calls == [("object", 1)]

    mask = np.zeros(image.shape, dtype=bool)
    mask[47:49, 47:49] = True
    measured = np.eye(4)
    measured[0, 3] = 0.03
    observation = PoseEstimate(
        "object",
        measured,
        frames[1].source_time_s,
        frames[1].available_time_s,
        Evidence.OBSERVED,
        "reference-v1",
        (True,) * 3,
        (True,) * 3,
    )
    service._anchor(_Anchor(1, (observation,), {"object": mask}))
    assert store.snapshot()[1]["object"].evidence == Evidence.OBSERVED
    assert store.snapshot()[1]["object"].pose[0, 3] == pytest.approx(0.03)

    service._frame(frames[2])
    assert store.snapshot()[1]["object"].pose[0, 3] == pytest.approx(0.02)
    service.set_attached_role(None, source_time_s=2.5 / 30)
    service._frame(frames[3])
    after_release = store.snapshot()[1]["object"]
    assert after_release.evidence == Evidence.LOST
    assert after_release.pose is None
    assert calls == [("object", 1), ("object", 2)]


def test_grasp_and_release_between_camera_frames_clear_old_scene_support():
    image = np.random.default_rng(60).integers(0, 255, (96, 96), dtype=np.uint8)
    store = LatestPoseStore()
    service = CausalPoseService(store)
    first = frame(image, 0)
    service._frame(first)
    observed = PoseEstimate(
        "object", np.eye(4), 0, 0, Evidence.OBSERVED, "reference-v1", (True,) * 3, (True,) * 3
    )
    service.observed["object"] = (observed, service.scene_history[-1], ())
    store.publish((observed,))

    service.set_attached_role("object", source_time_s=0.25 / 30)
    service.set_attached_role(None, source_time_s=0.75 / 30)
    service._frame(frame(image, 1))

    assert "object" not in service.observed
    assert store.snapshot()[1]["object"].evidence == Evidence.LOST


def test_motor_attachment_arriving_after_its_camera_frame_withdraws_old_scene_pose():
    image = np.random.default_rng(62).integers(0, 255, (96, 96), dtype=np.uint8)
    first, second = frame(image, 0), frame(image, 1)
    mask = np.zeros(image.shape, dtype=bool)
    mask[47:49, 47:49] = True
    observed = PoseEstimate(
        "object", np.eye(4), 0, 0, Evidence.OBSERVED, "reference-v1", (True,) * 3, (True,) * 3
    )
    target = PoseEstimate(
        "target", np.eye(4), 0, 0, Evidence.OBSERVED, "reference-v1", (True,) * 3, (True,) * 3
    )
    store = LatestPoseStore()
    service = CausalPoseService(store)
    service.slow_work_threshold_s = 0
    service.start()
    try:
        service.submit_frame(first)
        service.submit_anchor(0, (observed, target), {"object": mask, "target": mask})
        deadline = time.monotonic() + 2
        while "object" not in store.snapshot()[1] and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.snapshot()[1]["object"].evidence == Evidence.OBSERVED

        service.submit_frame(second)
        while service.frames < 2 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.frames == 2
        current_image_pose = PoseEstimate(
            "object",
            np.eye(4),
            second.source_time_s,
            second.available_time_s,
            Evidence.OBSERVED,
            "reference-v1",
            (True,) * 3,
            (True,) * 3,
        )
        service.submit_anchor(1, (current_image_pose,), {"object": mask})
        while (
            store.snapshot()[1]["object"].source_time_s < second.source_time_s
            and service.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert store.snapshot()[1]["object"].evidence == Evidence.OBSERVED
        # The command was dispatched at this image time, but its callback
        # reached the scene worker just after that image was processed.
        service.set_attached_role("object", source_time_s=second.source_time_s)
        while (
            store.snapshot()[1]["object"].evidence != Evidence.LOST
            and service.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        current = store.snapshot()[1]["object"]
        assert service.error is None
        assert current.evidence == Evidence.LOST
        assert current.source_frame_id == second.frame_id
        assert "object" not in service.observed
        assert store.snapshot()[1]["target"].evidence == Evidence.OBSERVED
        assert service.late_attachment_invalidations == 1
        assert any(item["kind"] == "attachment" for item in service.slow_work)
    finally:
        service.close()


def test_new_attachment_event_restarts_a_delayed_source_order_replay():
    image = np.random.default_rng(63).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(3)]
    mask = np.zeros(image.shape, dtype=bool)
    mask[47:49, 47:49] = True
    store = LatestPoseStore()
    service = CausalPoseService(store)
    for item in frames:
        service._frame(item)

    def observation(index: int) -> _Anchor:
        estimate = PoseEstimate(
            "object",
            np.eye(4),
            frames[index].source_time_s,
            frames[index].available_time_s,
            Evidence.OBSERVED,
            "reference-v1",
            (True,) * 3,
            (True,) * 3,
        )
        return _Anchor(index, (estimate,), {"object": mask})

    service._anchor(observation(1))
    service._anchor(observation(0))
    assert service._event_replay_step("object")  # Its first image used the old event journal.
    service.set_attached_role("object", source_time_s=1.5 / 30)
    service._drain_pending()

    current = store.snapshot()[1]["object"]
    assert current.evidence == Evidence.LOST
    assert current.source_frame_id == frames[2].frame_id
    assert service.anchor_decisions_by_source[(0, "object")].outcome == "replayed_accepted"


def test_motor_event_does_not_erase_a_camera_pose_already_processed_with_that_event(monkeypatch):
    image = np.random.default_rng(64).integers(0, 255, (96, 96), dtype=np.uint8)
    first, second = frame(image, 0), frame(image, 1)
    mask = np.zeros(image.shape, dtype=bool)
    mask[47:49, 47:49] = True
    store = LatestPoseStore()
    service = CausalPoseService(store)
    event_appended = Event()
    enqueue_event = Event()
    submit = service._submit

    def delayed_event_submit(item):
        if isinstance(item, _AttachmentEvent):
            event_appended.set()
            assert enqueue_event.wait(2)
        submit(item)

    monkeypatch.setattr(service, "_submit", delayed_event_submit)
    service.start()
    writer = Thread(target=lambda: service.set_attached_role("object", source_time_s=second.source_time_s))
    try:
        service.submit_frame(first)
        deadline = time.monotonic() + 2
        while service.frames < 1 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        writer.start()
        assert event_appended.wait(2)

        service.submit_frame(second)
        observed = PoseEstimate(
            "object",
            np.eye(4),
            second.source_time_s,
            second.available_time_s,
            Evidence.OBSERVED,
            "reference-v1",
            (True,) * 3,
            (True,) * 3,
        )
        service.submit_anchor(1, (observed,), {"object": mask})
        while "object" not in store.snapshot()[1] and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.snapshot()[1]["object"].evidence == Evidence.OBSERVED

        enqueue_event.set()
        writer.join(2)
        assert not writer.is_alive()
        # A barrier frame proves the queued motor event has been consumed.
        service.submit_frame(frame(image, 2))
        while service.frames < 3 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert store.snapshot()[1]["object"].evidence == Evidence.OBSERVED
        assert service.late_attachment_invalidations == 0
    finally:
        enqueue_event.set()
        if writer.ident is not None:
            writer.join(2)
        service.close()


def test_causal_config_rejects_unbounded_or_invalid_tracking():
    valid = {
        "history_frames": 60,
        "queue_frames": 60,
        "min_inliers": 12,
        "max_frame_gap_s": 0.1,
        "fit_threshold_m": 0.005,
    }
    CausalTrackingConfig(**valid, image_scale=0.5)
    with pytest.raises(ValueError, match="invalid causal tracking configuration"):
        CausalTrackingConfig(**{**valid, "image_scale": 0.0})
    complete = {
        **valid,
        "image_scale": 0.5,
        "scene_max_points": 800,
        "environment_context_radius_m": 0.25,
        "environment_min_motion_m": 0.008,
        "environment_prediction_max_s": 1.0,
        "environment_min_inliers": 12,
        "static_prediction_max_s": None,
        "max_pose_disagreement_translation_m": 0.01,
        "max_pose_disagreement_rotation_rad": 0.15,
    }
    assert CausalTrackingConfig.from_config(complete).scene_max_points == 800
    assert CausalTrackingConfig.from_config(complete).depth_support_radius_px == 1
    assert (
        CausalTrackingConfig.from_config({**complete, "depth_support_radius_px": 0}).depth_support_radius_px
        == 0
    )
    with pytest.raises(ValueError, match="causal tracking keys differ"):
        CausalTrackingConfig.from_config(
            {key: value for key, value in complete.items() if key != "scene_max_points"}
        )
    with pytest.raises(ValueError, match="invalid causal tracking configuration"):
        CausalTrackingConfig.from_config({**complete, "max_pose_disagreement_translation_m": 0})


def test_delayed_anchor_uses_retained_frame_and_all_intervening_motion():
    image = np.random.default_rng(4).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * i], [0, 1, 0]]), (96, 96)), i) for i in range(3)
    ]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    store = LatestPoseStore()
    service = CausalPoseService(store)
    service.start()
    try:
        for item in frames:
            service.submit_frame(item)
        anchor = PoseEstimate(
            "object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(0, (anchor,), {"object": mask})
        deadline = time.monotonic() + 2
        while service.published < 2 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.anchors == 1
        pose = store.snapshot()[1]["object"]
        assert pose.source_time_s == frames[2].source_time_s
        assert pose.evidence == Evidence.PREDICTED
        center = pose.pose @ np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(center[:3], [0.04, 0.0, 1.0], atol=0.005)
    finally:
        service.close()


def test_later_observation_rebases_pending_track_without_restarting_replay():
    image = np.random.default_rng(14).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * i], [0, 1, 0]]), (96, 96)), i) for i in range(5)
    ]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    shifted_mask = np.zeros_like(mask)
    shifted_mask[20:76, 24:80] = True
    store = LatestPoseStore()
    accepted_direct = []
    service = CausalPoseService(store, on_direct_observation=accepted_direct.append)
    service.start()
    try:
        for item in frames:
            service.submit_frame(item)
        first = PoseEstimate(
            "object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(0, (first,), {"object": mask})
        second_pose = np.eye(4)
        second_pose[0, 3] = 0.05  # frame 2 plus a 1 cm absolute correction
        second = PoseEstimate(
            "object", second_pose, 2 / 30, 2 / 30, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(2, (second,), {"object": shifted_mask})
        deadline = time.monotonic() + 2
        while (
            service.anchor_outcomes_by_role.get("object", {}).get("point_replay_caught_up", 0) < 1
            and service.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert service.error is None
        assert service.anchors == 1
        assert service.anchor_outcomes_by_role["object"]["pending_point_track"] == 1
        assert service.anchor_outcomes_by_role["object"]["point_replay_caught_up"] == 1
        result = store.snapshot()[1]["object"]
        assert result.source_time_s == frames[-1].source_time_s
        assert result.last_direct_observation_s == frames[2].source_time_s
        np.testing.assert_allclose(result.pose[0, 3], 0.09, atol=0.005)
        assert [estimate.source_time_s for estimate in accepted_direct] == [
            frames[0].source_time_s,
            frames[2].source_time_s,
        ]
    finally:
        service.close()


def test_reseeding_pending_point_track_retires_old_active_track():
    """Two seed frames must not write into one shared motion history."""
    image = np.random.default_rng(57).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * i], [0, 1, 0]]), (96, 96)), i) for i in range(4)
    ]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    store = LatestPoseStore()
    service = CausalPoseService(store)
    for item in frames[:3]:
        service._frame(item)
    first = PoseEstimate(
        "object",
        np.eye(4),
        frames[0].source_time_s,
        frames[0].available_time_s,
        Evidence.OBSERVED,
        "first-view",
        (True,) * 3,
        (True,) * 3,
    )
    service._anchor(_Anchor(0, (first,), {"object": mask}))
    service._drain_pending()
    assert "object" in service.active
    revised = PoseEstimate(
        "object",
        np.eye(4),
        frames[2].source_time_s,
        frames[2].available_time_s,
        Evidence.OBSERVED,
        "new-view",
        (True,) * 3,
        (True,) * 3,
    )
    service._anchor(_Anchor(2, (revised,), {"object": mask}))
    assert "object" in service.pending
    assert "object" not in service.active
    assert [item[0] for item in service.motion_history["object"]] == [2]
    service._frame(frames[3])
    service._drain_pending()
    assert service.error is None
    result = store.snapshot()[1]["object"]
    assert result.source_time_s == frames[3].source_time_s
    assert result.last_direct_observation_s == frames[2].source_time_s


@pytest.mark.parametrize("last_frame", [2, 3])
def test_late_older_observation_rechecks_a_newer_direct_fit_against_its_point_history(last_frame):
    """Arrival order must not make a disputed later pose become the final belief."""
    image = np.random.default_rng(4).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * i], [0, 1, 0]]), (96, 96)), i) for i in range(4)
    ]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    points = np.array([[x, y, 1.0] for x in np.linspace(-0.2, 0.2, 15) for y in np.linspace(-0.2, 0.2, 15)])

    def run(events):
        store = LatestPoseStore()
        decisions = []
        service = CausalPoseService(
            store, object_points_by_role={"object": points}, on_anchor_decision=decisions.append
        )
        for kind, index in events:
            if kind == "frame":
                service._frame(frames[index])
            else:
                pose = np.eye(4)
                if index == 2:
                    pose[0, 3] = 0.1  # The image and point motion support only 0.04 m.
                estimate = PoseEstimate(
                    "object", pose, index / 30, index / 30, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3
                )
                service._anchor(_Anchor(index, (estimate,), {"object": mask}))
                if index == 0 and service.history[-1].frame_id > 0:
                    assert service.event_replays == 0
                    assert "object" in service.pending_event_replays
                    service.queue.put_nowait(frames[-1])
                    service._drain_pending(while_queue_empty=True)
                    assert service.pending_event_replays["object"].last_frame_id is None
                    service.queue.get_nowait()
                service._drain_pending()
        return store.snapshot()[1]["object"], decisions

    source_events = [("frame", 0), ("anchor", 0), ("frame", 1), ("frame", 2), ("anchor", 2)]
    late_events = [("frame", 0), ("frame", 1), ("frame", 2), ("anchor", 2)]
    if last_frame == 3:
        source_events.append(("frame", 3))
        late_events.append(("frame", 3))
    late_events.append(("anchor", 0))
    source_order, ordered_decisions = run(source_events)
    late_order, late_decisions = run(late_events)
    assert ordered_decisions[-1].outcome == "direct_point_unresolved"
    np.testing.assert_allclose(late_order.pose, source_order.pose, atol=0.005)
    assert late_order.last_direct_observation_s == source_order.last_direct_observation_s
    assert late_decisions[-1].frame_id == 2
    assert late_decisions[-1].outcome == "replayed_direct_point_unresolved"


def test_multiple_delayed_pose_jobs_converge_to_the_same_camera_order_belief():
    """Job completion order cannot decide which of several source-frame fits wins."""
    image = np.random.default_rng(61).integers(0, 256, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * index], [0, 1, 0]]), (96, 96)), index)
        for index in range(8)
    ]
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:76, 20:76] = True
    points = np.array([[x, y, 1.0] for x in np.linspace(-0.2, 0.2, 15) for y in np.linspace(-0.2, 0.2, 15)])
    anchors = {}
    for index, error_m in ((0, 0.0), (2, 0.002), (4, 0.06), (6, 0.003)):
        pose = np.eye(4)
        pose[0, 3] = 0.02 * index + error_m
        anchors[index] = _Anchor(
            index,
            (
                PoseEstimate(
                    "object", pose, index / 30, index / 30, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3
                ),
            ),
            {"object": mask},
        )

    def run(order):
        store = LatestPoseStore()
        service = CausalPoseService(store, history_frames=20, object_points_by_role={"object": points})
        if order is None:
            for item in frames:
                service._frame(item)
                if item.frame_id in anchors:
                    service._anchor(anchors[item.frame_id])
                    service._drain_pending()
        else:
            for item in frames:
                service._frame(item)
            for index in order:
                service._anchor(anchors[index])
                service._drain_pending()
        return store.snapshot()[1]["object"], service

    expected, source_order = run(None)
    assert source_order.anchor_decisions_by_source[(4, "object")].outcome == "direct_point_unresolved"
    assert expected.pose[0, 3] == pytest.approx(0.14, abs=0.01)
    for order in itertools.permutations(anchors):
        actual, service = run(order)
        np.testing.assert_allclose(actual.pose, expected.pose, atol=0.005, err_msg=str(order))
        assert actual.source_time_s == expected.source_time_s
        assert actual.evidence == expected.evidence
        assert service.error is None


def test_event_replay_keeps_current_belief_if_its_source_frame_expires():
    image = np.random.default_rng(5).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(4)]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    store = LatestPoseStore()
    decisions = []
    service = CausalPoseService(store, history_frames=3, on_anchor_decision=decisions.append)
    for item in frames[:3]:
        service._frame(item)
    newer = PoseEstimate(
        "object", np.eye(4), 2 / 30, 2 / 30, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3
    )
    service._anchor(_Anchor(2, (newer,), {"object": mask}))
    service._drain_pending()
    older = PoseEstimate("object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3)
    service._anchor(_Anchor(0, (older,), {"object": mask}))
    assert "object" in service.pending_event_replays
    service._frame(frames[3])
    before = store.snapshot()[1]["object"]
    assert "object" not in service.pending_event_replays
    service._drain_pending()
    after = store.snapshot()[1]["object"]
    np.testing.assert_array_equal(after.pose, before.pose)
    assert service.event_replays == 0
    assert service.anchor_outcomes_by_role["object"]["event_replay_history_expired"] == 1
    assert decisions[-1].frame_id == 0
    assert decisions[-1].outcome == "event_replay_history_expired"
    assert set(service.anchor_events) == {2}


def test_event_replay_refuses_to_skip_an_evicted_intermediate_frame():
    image = np.random.default_rng(4).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * index], [0, 1, 0]]), (96, 96)), index)
        for index in range(6)
    ]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    store = LatestPoseStore()
    decisions = []
    service = CausalPoseService(store, history_frames=4, on_anchor_decision=decisions.append)
    for item in frames[:4]:
        service._frame(item)
    newer = PoseEstimate(
        "object", np.eye(4), 3 / 30, 3 / 30, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3
    )
    service._anchor(_Anchor(3, (newer,), {"object": mask}))
    service._drain_pending()
    older = PoseEstimate("object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3)
    service._anchor(_Anchor(0, (older,), {"object": mask}))
    assert service._event_replay_step("object")  # Frame 0 is already in the replay.
    assert service.pending_event_replays["object"].remaining_frame_ids[0] == 1
    service._frame(frames[4])
    assert "object" in service.pending_event_replays
    service._frame(frames[5])  # Frame 1 is evicted before the replay consumes it.
    assert "object" not in service.pending_event_replays
    assert decisions[-1].outcome == "event_replay_history_expired"
    assert service.event_replays == 0


def test_old_point_replay_yields_to_a_new_camera_frame(monkeypatch):
    image = np.random.default_rng(24).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(22)]
    mask = np.ones(image.shape, dtype=bool)
    first_replay = Event()
    allow_replay = Event()
    new_scene_frame = Event()
    allow_scene = Event()
    replay_calls = []
    point_update = CausalRigidPointTracker.update
    scene_update = ScenePointTracker.update

    def tracked_update(self, item):
        replay_calls.append(item.frame_id)
        if len(replay_calls) == 1:
            first_replay.set()
            assert allow_replay.wait(2)
        return point_update(self, item)

    def scene_frame(self, item):
        if item.frame_id == 21:
            new_scene_frame.set()
            assert allow_scene.wait(2)
        return scene_update(self, item)

    monkeypatch.setattr(CausalRigidPointTracker, "update", tracked_update)
    monkeypatch.setattr(ScenePointTracker, "update", scene_frame)
    service = CausalPoseService(LatestPoseStore(), history_frames=60)
    service.start()
    try:
        for item in frames[:21]:
            service.submit_frame(item)
        anchor = PoseEstimate(
            "object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(0, (anchor,), {"object": mask})
        assert first_replay.wait(2)
        service.submit_frame(frames[21])
        allow_replay.set()
        assert new_scene_frame.wait(2)
        assert replay_calls == [1]
    finally:
        allow_replay.set()
        allow_scene.set()
        service.close()
    assert service.error is None


def test_anchor_requires_the_recorded_capture_time_and_calibration():
    image = np.random.default_rng(9).integers(0, 255, (96, 96), dtype=np.uint8)
    captured = frame(image, 1)
    mask = np.ones(image.shape, dtype=bool)
    estimate = PoseEstimate(
        "object",
        np.eye(4),
        captured.source_time_s,
        captured.available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    store = LatestPoseStore()
    service = CausalPoseService(store)
    service.start()
    try:
        service.submit_frame(captured)
        service.submit_anchor(
            1,
            (estimate,),
            {"object": mask},
            source_time_s=captured.source_time_s,
            calibration_version="different-calibration",
        )
        service.submit_anchor(
            1,
            (estimate,),
            {"object": mask},
            source_time_s=captured.source_time_s + 0.01,
            calibration_version=captured.calibration_version,
        )
        deadline = time.monotonic() + 2
        while service.missed_anchors < 2 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.missed_anchors == 2
        assert store.snapshot() == (0, {})
    finally:
        service.close()


def test_accepted_live_pose_is_available_at_scene_decision_not_backend_completion():
    image = np.random.default_rng(51).integers(0, 255, (96, 96), dtype=np.uint8)
    captured = frame(image, 0)
    candidate = PoseEstimate(
        "object",
        np.eye(4),
        captured.source_time_s,
        captured.available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    decisions = []
    store = LatestPoseStore()
    service = CausalPoseService(store, on_anchor_decision=decisions.append)
    service.start()
    try:
        service.submit_frame(captured)
        service.submit_anchor(0, (candidate,), {"object": np.ones(image.shape, dtype=bool)})
        deadline = time.monotonic() + 2
        while not decisions and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.outcome == "accepted"
        assert decision.candidate.available_time_s == captured.available_time_s
        assert decision.accepted.available_time_s == decision.decided_at_s
        assert store.snapshot()[1]["object"].available_time_s >= decision.decided_at_s
    finally:
        service.close()


def test_offline_source_order_pose_keeps_recorded_availability_time():
    image = np.random.default_rng(52).integers(0, 255, (96, 96), dtype=np.uint8)
    captured = frame(image, 0)
    candidate = PoseEstimate(
        "object",
        np.eye(4),
        captured.source_time_s,
        captured.available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    store = LatestPoseStore()
    service = CausalPoseService(store)
    result = service.process_offline(captured, (candidate,), {"object": np.ones(image.shape, dtype=bool)})
    assert result["object"].available_time_s == captured.available_time_s


def test_delayed_symmetry_proposal_does_not_use_later_pose_as_its_reference():
    image = np.random.default_rng(53).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(2)]
    references = []

    class RecordingSymmetry:
        role = "object"

        def canonicalize(self, pose, *, reference_pose=None):
            references.append(reference_pose)
            return pose

    store = LatestPoseStore()
    service = CausalPoseService(store, axial_symmetries=(RecordingSymmetry(),))
    for item in frames:
        service._frame(item)
    newer = PoseEstimate(
        "object",
        np.eye(4),
        frames[1].source_time_s,
        frames[1].available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    service.active["object"] = (CausalRigidPointTracker(), newer)
    store.publish((newer,))
    older = PoseEstimate(
        "object",
        np.eye(4),
        frames[0].source_time_s,
        frames[0].available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    service._anchor(_Anchor(0, (older,), {"object": np.ones(image.shape, dtype=bool)}))
    assert references == [None]
    assert service.superseded_proposals == 1
