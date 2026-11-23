"""FLASH recording adapters preserve the shared recorder's lifecycle and data."""

import copy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.stereo_recording import add_stereo_features, capture_stereo
from lerobot.policies.flash.record import prepare_recording
from lerobot.utils.feature_utils import build_dataset_frame
from tests.datasets.test_stereo_recording import camera


@pytest.fixture(autouse=True)
def no_hub_access(monkeypatch):
    import httpx
    import requests

    def reject(*args, **kwargs):
        pytest.fail("A local recording test attempted network access")

    monkeypatch.setattr(httpx.Client, "send", reject)
    monkeypatch.setattr(requests.Session, "request", reject)


def _record_one_frame(run_loop, robot, dataset, tmp_path):
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    teleop = MockTeleop(
        MockTeleopConfig(
            n_motors=1, random_values=False, static_values=[0.0], calibration_dir=tmp_path / "teleop"
        )
    )
    events = {"exit_early": False}

    def action_processor(values):
        events["exit_early"] = True
        return values[0]

    def process(observation):
        assert set(observation) == {"scene"}
        return observation | {"scene": np.full_like(observation["scene"], 42)}

    teleop.connect()
    try:
        run_loop(
            robot=robot,
            events=events,
            fps=30,
            teleop_action_processor=action_processor,
            robot_action_processor=lambda values: values[0],
            robot_observation_processor=process,
            dataset=dataset,
            teleop=teleop,
            control_time_s=1,
            single_task="move",
        )
    finally:
        teleop.disconnect()


@pytest.mark.parametrize("adapted", [False, True])
def test_record_lifecycle_uses_existing_robot_and_dataset_contracts(tmp_path, monkeypatch, adapted):
    from lerobot.scripts import lerobot_record
    from tests.mocks.mock_robot import MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    cam = camera()
    cam.config.enable_infrared = adapted
    stereo = {"T_right_left": np.eye(4).tolist()}
    cam.stereo_calibration = lambda: stereo
    measurements = {
        "robot_id": "probe",
        "cameras": {
            "scene": {
                "identity": {"type": "intelrealsense", "serial_number_or_name": "123456789"},
                "T_base_cam": np.eye(4).tolist(),
            }
        },
    }
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(measurements))
    make_robot = lerobot_record.make_robot_from_config

    def make_test_robot(config):
        robot = make_robot(config)
        robot.cameras = {"scene": cam}
        robot.config.cameras = {"scene": cam.config}
        get_observation = robot.get_observation

        def observe():
            cam.latest_timestamp = time.perf_counter()
            return get_observation() | {"scene": cam.latest_color_frame}

        robot.get_observation = observe
        return robot

    monkeypatch.setattr(lerobot_record, "make_robot_from_config", make_test_robot)
    monkeypatch.setattr(
        lerobot_record,
        "init_keyboard_listener",
        lambda: (None, {"exit_early": False, "rerecord_episode": False, "stop_recording": False}),
    )
    config = lerobot_record.RecordConfig(
        robot=MockRobotConfig(
            id="probe",
            n_motors=1,
            random_values=False,
            static_values=[4.0],
            calibration_dir=tmp_path / "robot",
            geometry_path=geometry_path if adapted else None,
        ),
        teleop=MockTeleopConfig(
            n_motors=1, random_values=False, static_values=[9.0], calibration_dir=tmp_path / "teleop"
        ),
        dataset=lerobot_record.DatasetRecordConfig(
            repo_id="test/record",
            root=tmp_path / "dataset",
            single_task="move",
            fps=30,
            episode_time_s=0.05,
            reset_time_s=0,
            num_episodes=1,
            video=False,
            push_to_hub=False,
            num_image_writer_processes=0,
        ),
        display_data=False,
        play_sounds=False,
        latency_output_dir=tmp_path / "logs",
    )
    dataset = lerobot_record.record(config, prepare_recording=prepare_recording if adapted else None)
    reopened = LeRobotDataset(dataset.repo_id, root=dataset.root)
    assert reopened.num_episodes == 1
    assert len(reopened) > 0
    np.testing.assert_array_equal(reopened[0]["observation.state"], [4.0])
    np.testing.assert_array_equal(reopened[0]["action"], [9.0])
    assert reopened.meta.info.robot_geometry == (measurements if adapted else None)
    assert ("observation.images.scene_infrared_left" in reopened.features) == adapted
    if adapted:
        assert (
            reopened.features["observation.images.scene"]["info"]["stereo"]["T_right_left"]
            == stereo["T_right_left"]
        )


def test_stereo_adapter_resumes_and_keeps_extra_channels_out_of_processing(tmp_path):
    cam = camera()
    robot = SimpleNamespace(
        name="probe",
        robot_type="probe",
        geometry=None,
        cameras={"scene": cam},
        get_observation=lambda: {"scene": cam.latest_color_frame},
        send_action=lambda action: action,
    )
    features = {
        "observation.images.scene": {
            "dtype": "image",
            "shape": (480, 848, 3),
            "names": ["height", "width", "channels"],
        }
    }
    root = tmp_path / "recording"
    legacy_root = tmp_path / "legacy"
    legacy_features = add_stereo_features(copy.deepcopy(features), robot)
    timestamps = []
    for resume in (False, True):
        adapted_robot, factory, run_loop = prepare_recording(robot, features)
        dataset = (
            factory.resume("test/stereo-resume", root=root)
            if resume
            else factory.create(
                "test/stereo-resume",
                fps=30,
                root=root,
                robot_type="probe",
                features=features,
                use_videos=False,
            )
        )
        legacy = (
            LeRobotDataset.resume("test/stereo-legacy", root=legacy_root)
            if resume
            else LeRobotDataset.create(
                "test/stereo-legacy",
                fps=30,
                root=legacy_root,
                robot_type="probe",
                features=legacy_features,
                use_videos=False,
            )
        )
        cam.latest_timestamp = time.perf_counter()
        timestamps.append(round(cam.latest_timestamp * 1e9))
        # Reconstruct the previous inline recording rule and feed both paths
        # the same camera snapshot and observation-processor output.
        observation = robot.get_observation()
        images, timing = capture_stereo(robot)
        observation.update({key: value for key, value in images.items() if key in observation})
        processed = observation | {"scene": np.full_like(observation["scene"], 42)}
        legacy_frame = build_dataset_frame(
            legacy.features,
            {**processed, **{key: value for key, value in images.items() if key not in observation}},
            "observation",
        )
        legacy_frame.update(timing)
        legacy_frame["task"] = "move"
        _record_one_frame(run_loop, adapted_robot, dataset, tmp_path)
        legacy.add_frame(legacy_frame)
        legacy.save_episode()
        legacy.finalize()
        dataset.save_episode()
        dataset.finalize()
    reopened = LeRobotDataset("test/stereo-resume", root=root)
    legacy = LeRobotDataset("test/stereo-legacy", root=legacy_root)
    assert reopened.features == legacy.features
    assert reopened.meta.stats.keys() == legacy.meta.stats.keys()
    for key in reopened.meta.stats:
        assert reopened.meta.stats[key].keys() == legacy.meta.stats[key].keys()
        for stat in reopened.meta.stats[key]:
            np.testing.assert_array_equal(reopened.meta.stats[key][stat], legacy.meta.stats[key][stat])
    assert reopened.num_episodes == 2
    for index, timestamp in enumerate(timestamps):
        frame = reopened[index]
        legacy_frame = legacy[index]
        assert frame.keys() == legacy_frame.keys()
        for key, value in frame.items():
            np.testing.assert_array_equal(value, legacy_frame[key])
        np.testing.assert_allclose(frame["observation.images.scene"], 42 / 255)
        np.testing.assert_allclose(frame["observation.images.scene_infrared_left"], 7 / 255)
        np.testing.assert_allclose(frame["observation.images.scene_infrared_right"], 8 / 255)
        np.testing.assert_array_equal(frame["observation.images.scene_depth"], 407)
        assert frame["observation.cameras.scene.capture_time_ns"].item() == timestamp
        assert frame["observation.cameras.scene.frame_sequence"].item() == 7
    cam.config.enable_infrared = False
    _, incompatible, _ = prepare_recording(robot, features)
    with pytest.raises(ValueError, match="compatibility"):
        incompatible.resume("test/stereo-resume", root=root)


def test_optional_stereo_is_required_for_recording_but_not_reset(tmp_path):
    cam = camera()
    sent = []
    robot = SimpleNamespace(
        name="probe",
        robot_type="probe",
        geometry=None,
        cameras={"scene": cam},
        get_observation=lambda: {"scene": cam.latest_color_frame},
        send_action=lambda action: sent.append(action) or action,
    )
    features = {
        "observation.images.scene": {
            "dtype": "image",
            "shape": (480, 848, 3),
            "names": ["height", "width", "channels"],
        }
    }
    adapted_robot, factory, run_loop = prepare_recording(robot, features)
    dataset = factory.create(
        "test/reset",
        fps=30,
        root=tmp_path / "dataset",
        robot_type="probe",
        features=features,
        use_videos=False,
    )
    cam.latest_infrared_pair = None
    _record_one_frame(run_loop, adapted_robot, None, tmp_path)
    assert len(sent) == 1
    sent.clear()
    cam.latest_timestamp = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="complete stereo frameset"):
            _record_one_frame(run_loop, adapted_robot, dataset, tmp_path)
        assert not sent
    finally:
        dataset.finalize()


def test_recording_adapter_attaches_geometry_and_stereo_to_both_datasets(tmp_path):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.robots.geometry import RobotGeometry

    cam = camera()
    stereo = {"T_right_left": np.eye(4).tolist()}
    cam.stereo_calibration = lambda: stereo
    measurements = {
        "robot_id": "probe",
        "cameras": {
            "scene": {
                "identity": {"type": "intelrealsense", "serial_number_or_name": "123456789"},
                "T_base_cam": np.eye(4).tolist(),
            }
        },
    }
    connected = []

    def unsupported_model():
        raise NotImplementedError

    robot = SimpleNamespace(
        name="probe",
        robot_type="probe",
        cameras={"scene": cam},
        geometry=RobotGeometry(measurements, robot_id="probe"),
        cartesian_model=unsupported_model,
        connect=lambda: connected.append(True),
        get_observation=lambda: {"scene": cam.latest_color_frame},
        send_action=lambda action: action,
    )
    features = {
        "observation.images.scene": {
            "dtype": "image",
            "shape": (480, 848, 3),
            "names": ["height", "width", "channels"],
        }
    }
    adapted_robot, factory, run_loop = prepare_recording(robot, features)
    datasets = [
        factory.create(
            f"test/{name}",
            fps=30,
            root=tmp_path / name,
            robot_type="probe",
            features=features,
            use_videos=False,
        )
        for name in ("main", "intervention")
    ]
    assert not connected
    adapted_robot.connect()
    assert connected == [True]
    for dataset in datasets:
        cam.latest_timestamp = time.perf_counter()
        _record_one_frame(run_loop, adapted_robot, dataset, tmp_path)
        dataset.save_episode()
        dataset.finalize()
        metadata = LeRobotDatasetMetadata(dataset.repo_id, root=dataset.root)
        assert metadata.info.robot_geometry == measurements
        assert (
            metadata.features["observation.images.scene"]["info"]["stereo"]["T_right_left"]
            == stereo["T_right_left"]
        )
    robot.geometry = None
    with pytest.raises(ValueError, match="requires its recorded robot geometry"):
        factory.resume("test/main", root=tmp_path / "main")
    assert connected == [True]
