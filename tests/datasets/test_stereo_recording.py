import copy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.datasets.stereo_recording import add_stereo_features, capture_stereo


def camera():
    cam = RealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name="123456789",
            width=848,
            height=480,
            fps=30,
            use_depth=True,
            enable_infrared=True,
        )
    )
    cam.rs_pipeline = object()
    cam.rs_profile = object()
    cam.thread = SimpleNamespace(is_alive=lambda: True)
    cam.latest_color_frame = np.full((480, 848, 3), 7, np.uint8)
    cam.latest_depth_frame = np.full((480, 848, 1), 407, np.uint16)
    cam.latest_infrared_pair = (np.full((480, 848), 7, np.uint8), np.full((480, 848), 8, np.uint8))
    cam.latest_timestamp = time.perf_counter()
    cam.latest_frame_sequence = 7
    return cam


def test_complete_snapshot_and_prefixed_camera_keys():
    cam = camera()
    robot = SimpleNamespace(cameras={"scene": cam}, camera_feature_names={"scene": "left_scene"})
    features = {"observation.images.left_scene": {"dtype": "video", "shape": (480, 848, 3)}}
    add_stereo_features(features, robot)
    images, timing = capture_stereo(robot)
    assert images["left_scene"] is cam.latest_color_frame
    assert images["left_scene_depth"] is cam.latest_depth_frame
    assert np.all(images["left_scene_infrared_left"] == 7)
    assert np.all(images["left_scene_infrared_right"] == 8)
    assert timing["observation.cameras.left_scene.capture_time_ns"][0] == round(cam.latest_timestamp * 1e9)
    assert timing["observation.cameras.left_scene.frame_sequence"][0] == 7
    assert features["observation.images.left_scene_depth"]["dtype"] == "image"
    assert features["observation.images.left_scene_infrared_left"]["info"]["is_depth_map"] is False
    # Peeking must not consume a frame event needed by a different consumer.
    cam.new_frame_event.set()
    cam.read_latest_stereo()
    assert cam.new_frame_event.is_set()


@pytest.mark.parametrize("fault", ["missing_ir", "stale", "consumed_depth", "disabled"])
def test_incomplete_stereo_never_silently_records_rgb_only(fault):
    cam = camera()
    if fault == "missing_ir":
        cam.latest_infrared_pair = None
    elif fault == "stale":
        cam.latest_timestamp -= 1
    elif fault == "consumed_depth":
        cam.post_grab_processor = object()
    else:
        cam.config.enable_infrared = False
    with pytest.raises((RuntimeError, TimeoutError)):
        cam.read_latest_stereo()


def test_snapshot_cannot_mix_two_publications():
    cam = camera()
    acquired = threading.Event()

    def publish():
        with cam.frame_lock:
            acquired.set()
            cam.latest_color_frame = cam.latest_color_frame + 1
            time.sleep(0.01)  # Reader must wait for the complete publication.
            cam.latest_depth_frame = cam.latest_depth_frame + 1
            cam.latest_infrared_pair = tuple(x + 1 for x in cam.latest_infrared_pair)
            cam.latest_frame_sequence += 1

    writer = threading.Thread(target=publish)
    writer.start()
    assert acquired.wait(1)
    rgb, depth, left, right, _, sequence = cam.read_latest_stereo()
    writer.join()
    assert sequence == 8
    assert rgb[0, 0, 0] == left[0, 0] == sequence
    assert depth[0, 0, 0] == 400 + sequence
    assert right[0, 0] == sequence + 1


def test_disabled_stereo_does_not_change_other_recorders():
    robot = SimpleNamespace(cameras={"scene": SimpleNamespace(config=SimpleNamespace())})
    features = {"observation.images.scene": {"dtype": "video", "shape": (480, 848, 3)}}
    original = copy.deepcopy(features)
    assert add_stereo_features(features, robot) == original
    assert capture_stereo(robot) == ({}, {})


def test_calibration_preserves_sdk_transform_direction_and_units():
    import pyrealsense2 as rs

    cam = camera()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = [0.01, 0.02, 0.03]
    intrinsic = SimpleNamespace(
        width=848, height=480, fx=450, fy=451, ppx=420, ppy=240, model="none", coeffs=[0] * 5
    )
    right, color = object(), object()
    left = SimpleNamespace(
        get_intrinsics=lambda: intrinsic,
        get_extrinsics_to=lambda target: SimpleNamespace(
            rotation=rotation.flatten(order="F").tolist()
            if target is color
            else np.eye(3).flatten().tolist(),
            translation=translation if target is color else [-0.05, 0, 0],
        ),
    )

    def stream(kind, index=0):
        if kind == rs.stream.color:
            return color
        return SimpleNamespace(as_video_stream_profile=lambda: left if index == 1 else right)

    right = SimpleNamespace(get_intrinsics=lambda: intrinsic)
    cam.rs_profile = SimpleNamespace(
        get_stream=stream,
        get_device=lambda: SimpleNamespace(
            first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001)
        ),
    )
    result = cam.stereo_calibration()
    np.testing.assert_array_equal(np.array(result["color_from_left"])[:3, :3], rotation)
    np.testing.assert_array_equal(np.array(result["color_from_left"])[:3, 3], translation)
    assert result["baseline_m"] == 0.05
    assert result["depth_m_per_unit"] == 0.001
    assert result["left_intrinsics"]["fx"] == 450


def test_stereo_dataset_roundtrip_preserves_pixels_channels_stats_and_timing(tmp_path):
    from PIL import Image

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame

    cam = camera()
    robot = SimpleNamespace(cameras={"scene": cam})
    features = {
        "observation.images.scene": {
            "dtype": "image",
            "shape": (480, 848, 3),
            "names": ["height", "width", "channels"],
        }
    }
    add_stereo_features(features, robot)
    images, timing = capture_stereo(robot)
    dataset = LeRobotDataset.create(
        repo_id="test/stereo", root=tmp_path / "episode", fps=30, features=features, use_videos=False
    )
    dataset.add_frame(build_dataset_frame(features, images, "observation") | timing | {"task": "stereo"})
    for suffix, mode, format_name in (
        ("", "RGB", "PNG"),
        ("_depth", "I;16", "TIFF"),
        ("_infrared_left", "L", "TIFF"),
        ("_infrared_right", "L", "TIFF"),
    ):
        key = f"observation.images.scene{suffix}"
        path = dataset.writer._get_image_file_path(0, key, 0)
        with Image.open(path) as saved:
            assert saved.mode == mode
            assert saved.format == format_name
            np.testing.assert_array_equal(np.asarray(saved), images[f"scene{suffix}"].squeeze())
    dataset.save_episode()
    dataset.finalize()
    reopened = LeRobotDataset("test/stereo", root=dataset.root)
    row = reopened[0]
    for suffix in ("", "_depth", "_infrared_left", "_infrared_right"):
        key = f"observation.images.scene{suffix}"
        expected = images[f"scene{suffix}"].transpose(2, 0, 1).astype(np.float32)
        if suffix != "_depth":
            expected /= 255
        np.testing.assert_array_equal(row[key].numpy(), expected)
        for stat in ("min", "max", "mean", "std"):
            assert reopened.meta.stats[key][stat].shape == (expected.shape[0], 1, 1)
        np.testing.assert_allclose(reopened.meta.stats[key]["mean"], expected[:, :1, :1])
    for key, value in timing.items():
        np.testing.assert_array_equal(row[key].numpy(), value)


@pytest.mark.parametrize("stereo", [False, True])
def test_record_loop_keeps_processed_rgb_with_optional_stereo(tmp_path, stereo):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.lerobot_record import record_loop
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    cam = camera()
    cam.config.enable_infrared = stereo
    features = {
        "observation.images.scene": {
            "dtype": "image",
            "shape": (480, 848, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (1,), "names": ["motor_1.pos"]},
    }
    robot = SimpleNamespace(
        name="probe",
        cameras={"scene": cam},
        action_features={"motor_1.pos": float},
        get_observation=lambda: {"scene": cam.latest_color_frame, "motor_1.pos": 0.0},
        send_action=lambda action: action,
    )
    add_stereo_features(features, robot)
    dataset = LeRobotDataset.create(
        repo_id="test/processed", root=tmp_path / "episode", fps=30, features=features, use_videos=False
    )
    teleop = MockTeleop(
        MockTeleopConfig(
            n_motors=1, random_values=False, static_values=[0.0], calibration_dir=tmp_path / "teleop"
        )
    )
    events = {"exit_early": False}
    processed = []

    def process(observation):
        processed.append(observation["scene"].copy())
        return observation | {"scene": np.full_like(observation["scene"], 42)}

    def action_processor(values):
        events["exit_early"] = True  # Finish this frame, then exit the real record loop.
        return values[0]

    teleop.connect()
    try:
        cam.latest_timestamp = time.perf_counter()
        record_loop(
            robot,
            events,
            30,
            action_processor,
            lambda values: values[0],
            process,
            dataset=dataset,
            teleop=teleop,
            control_time_s=1,
            single_task="processed",
        )
    finally:
        teleop.disconnect()
    dataset.save_episode()
    dataset.finalize()
    reopened = LeRobotDataset("test/processed", root=dataset.root)
    assert len(processed) == 1 and np.all(processed[0] == 7)
    np.testing.assert_allclose(reopened[0]["observation.images.scene"], 42 / 255)
    if stereo:
        np.testing.assert_allclose(reopened[0]["observation.images.scene_infrared_left"], 7 / 255)
